"""Blender avg-8 sweep pipeline at n=8 (canonical literature protocol).

Produces, for each of the 7 missing scenes (chair / drums / ficus / hotdog
/ materials / mic / ship), the same two runs we already have for lego:

    blender_<scene>_n8                  — baseline 3DGS, no SSL
    blender_<scene>_n8_ssl_mv_dav2s     — DAv2-Small + multiview_photo

Stages per scene::

    1.  download <scene>.zip from huggingface (nerfbaselines/nerfbaselines-data)
    2.  unzip into data/nerf_synthetic/<scene>/
    3.  precompute DAv2-Small disparity cache (train split, 100 frames, GPU)
    4.  train baseline      (subprocess, ~2-3 min on 16 GB cards)
    5.  train DAv2 + mv     (subprocess, ~6-8 min)

Parallelism strategy
--------------------
- A background **producer thread** runs stages 1-2 (network + disk IO,
  no GPU) for all 7 scenes back-to-back. While the GPU is busy training
  scene K, scene K+1's data can be coming down the wire.
- The **main thread** is the GPU consumer: as soon as a scene reaches
  the "unzipped" state it runs stages 3-5 serially, *then* moves on.
  Each train is a fresh subprocess to side-step the cudaErrorUnknown
  context-rot we saw on the LLFF sweep + the lego full-eval.
- Stage 3 (DAv2 cache) takes the GPU for ~30-60 s per scene; producer
  cannot overlap with it on the same GPU, but stages 1-2 of the *next*
  scene still can. So worst case the GPU is busy ~95% of the time and
  download wait is ~0 after scene 0.

State / recovery
----------------
- Every stage writes a sentinel marker file under
  ``outputs/_blender_n8_state/<scene>/{downloaded,unzipped,cached,trained_baseline,trained_dav2s}``.
- Restart-safe: if the script is killed and re-run, it skips any stage
  whose marker exists and just continues.
- A unified log lives at ``outputs/_blender_n8_pipeline.log``.

Usage
-----

    # default: all 7 scenes (chair drums ficus hotdog materials mic ship)
    python scripts/run_blender_n8_pipeline.py

    # subset of scenes
    python scripts/run_blender_n8_pipeline.py --scenes chair drums

    # dry-run (just prints the plan)
    python scripts/run_blender_n8_pipeline.py --dry-run

The script is meant to be launched once and left running in the
background; on Windows we already have a "detached launcher" pattern
in scripts/launch_llff_sweep_detached.py if you want truly fire-and-forget.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import List, Optional

# ---- bootstrap so we can run as either ``python scripts/...`` or ``-m`` ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import _bootstrap  # noqa: E402,F401

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"
_DATA_ROOT = _ROOT / "data" / "nerf_synthetic"
_CFG_DIR = _ROOT / "configs" / "_blender_n8"
_STATE_DIR = _OUT / "_blender_n8_state"
_LOG_DIR = _OUT / "_blender_n8_logs"
_PIPE_LOG = _OUT / "_blender_n8_pipeline.log"
_HF_REPO = "nerfbaselines/nerfbaselines-data"
_HF_REPO_TYPE = "dataset"
_DAV2_TAG = "depth_anything_v2_small"
# Local DAv2-Small snapshot. snapshot_download to a real local dir works
# around Windows hub-cache snapshot folders being empty (symlink quirk),
# and lets transformers 5.x load via AutoModelForDepthEstimation cleanly.
_DAV2_MODEL = str(_ROOT / "outputs" / "_models" / "Depth-Anything-V2-Small-hf")

_DEFAULT_SCENES = ["chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]


# ----------------------------------------------------------------------
# logging — stdout AND _PIPE_LOG, thread-safe-ish (single python GIL writes)
# ----------------------------------------------------------------------
_LOCK = threading.Lock()


def _log(msg: str, *, tag: str = "main") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    with _LOCK:
        print(line, flush=True)
        try:
            _PIPE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(_PIPE_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ----------------------------------------------------------------------
# state markers
# ----------------------------------------------------------------------
def _state_dir(scene: str) -> Path:
    return _STATE_DIR / scene


def _marker(scene: str, name: str) -> Path:
    return _state_dir(scene) / name


def _has(scene: str, name: str) -> bool:
    return _marker(scene, name).exists()


def _mark(scene: str, name: str, payload: str = "") -> None:
    p = _marker(scene, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload + ("\n" if payload and not payload.endswith("\n") else ""),
                 encoding="utf-8")


# ----------------------------------------------------------------------
# stage 1: download <scene>.zip from HF
# ----------------------------------------------------------------------
def _download_scene(scene: str) -> Path:
    """Download blender/<scene>.zip via huggingface_hub. Returns local zip path."""
    if _has(scene, "downloaded"):
        zip_path = _state_dir(scene) / f"{scene}.zip"
        if zip_path.exists():
            return zip_path
        # marker stale, redo
        _log(f"{scene}: marker says downloaded but zip missing, redoing", tag="dl")

    from huggingface_hub import hf_hub_download

    _state_dir(scene).mkdir(parents=True, exist_ok=True)
    _log(f"{scene}: downloading blender/{scene}.zip from HF ...", tag="dl")
    t0 = time.time()
    p = hf_hub_download(
        repo_id=_HF_REPO,
        repo_type=_HF_REPO_TYPE,
        filename=f"blender/{scene}.zip",
        local_dir=str(_state_dir(scene)),
    )
    p = Path(p)
    # nerfbaselines layout puts the zip at <state>/<scene>/blender/<scene>.zip
    dt = time.time() - t0
    sz = p.stat().st_size / (1024 * 1024)
    _log(f"{scene}: downloaded {sz:.1f} MB in {dt:.1f}s -> {p}", tag="dl")
    _mark(scene, "downloaded", payload=str(p))
    return p


# ----------------------------------------------------------------------
# stage 2: unzip into data/nerf_synthetic/<scene>/
# ----------------------------------------------------------------------
def _unzip_scene(scene: str, zip_path: Path) -> Path:
    """Unzip into ``data/nerf_synthetic/<scene>/``. Idempotent."""
    target = _DATA_ROOT / scene
    if _has(scene, "unzipped") and (target / "transforms_train.json").exists():
        return target

    _log(f"{scene}: unzipping {zip_path.name} -> {target} ...", tag="dl")
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # nerfbaselines packs as <scene>/{train,test,val,transforms_*.json}
        # extract to data/nerf_synthetic/  -> appears at data/nerf_synthetic/<scene>/
        zf.extractall(_DATA_ROOT)
    dt = time.time() - t0
    if not (target / "transforms_train.json").exists():
        # maybe the zip put files at top-level. detect that case.
        # find any transforms_train.json that just landed near _DATA_ROOT
        cands = list(_DATA_ROOT.glob("**/transforms_train.json"))
        cands = [c for c in cands if scene in c.as_posix().lower()]
        if not cands:
            raise FileNotFoundError(
                f"{scene}: unzipped but transforms_train.json not found anywhere; "
                f"zip names sample: {names[:5]}"
            )
        _log(f"{scene}: layout differs; transforms_train.json found at {cands[0]}",
             tag="dl")
    _log(f"{scene}: unzipped in {dt:.1f}s", tag="dl")
    _mark(scene, "unzipped", payload=str(target))
    return target


# ----------------------------------------------------------------------
# stage 3: DAv2 cache (subprocess so it doesn't load HF model into pipeline proc)
# ----------------------------------------------------------------------
def _cache_dav2(scene: str) -> None:
    if _has(scene, "cached"):
        # also sanity-check at least 1 npz exists
        cache_dir = _DATA_ROOT / scene / "_depth_cache" / _DAV2_TAG / "train"
        if cache_dir.exists() and any(cache_dir.glob("*.npz")):
            return
        _log(f"{scene}: marker says cached but cache empty, redoing", tag="cache")

    _log(f"{scene}: precomputing DAv2-Small cache (GPU) ...", tag="cache")
    log = _LOG_DIR / f"{scene}_cache.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "precompute_depth.py"),
        "--scene", scene,
        "--backend", "transformers",
        "--model", _DAV2_MODEL,
        "--tag", _DAV2_TAG,
        "--device", "cuda",
        "--splits", "train",
    ]
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    creationflags = 0x08000000 if os.name == "nt" else 0
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=str(_ROOT),
            creationflags=creationflags, env=env,
        )
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{scene}: dav2 cache failed rc={rc}, see {log}")
    _log(f"{scene}: dav2 cache done in {dt:.1f}s", tag="cache")
    _mark(scene, "cached")


# ----------------------------------------------------------------------
# stage 4 / 5: train (subprocess — fresh CUDA context per run)
# ----------------------------------------------------------------------
def _train(scene: str, variant: str) -> None:
    """variant ∈ {'baseline', 'dav2s'}."""
    marker = f"trained_{variant}"
    run_name = (f"blender_{scene}_n8" if variant == "baseline"
                else f"blender_{scene}_n8_ssl_mv_dav2s")
    cfg_name = f"blender_{scene}_n8_{variant}.yaml"
    cfg_path = _CFG_DIR / cfg_name
    out_dir = _OUT / run_name
    metrics = out_dir / "metrics.json"

    if _has(scene, marker) and metrics.exists():
        return
    if metrics.exists():
        # have metrics from a prior run; refresh marker and skip
        _mark(scene, marker, payload=str(metrics))
        _log(f"{scene}/{variant}: metrics already present, skipping", tag="train")
        return
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    # if a half-baked dir exists from a prior crash, blow it away
    if out_dir.exists():
        try:
            shutil.rmtree(out_dir)
            _log(f"{scene}/{variant}: cleaned partial dir {out_dir.name}", tag="train")
        except Exception as e:
            _log(f"{scene}/{variant}: WARN failed to clean {out_dir}: {e}", tag="train")

    log = _LOG_DIR / f"{run_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_ROOT / "scripts" / "train.py"),
           "--config", str(cfg_path)]
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    creationflags = 0x08000000 if os.name == "nt" else 0

    _log(f"{scene}/{variant}: training -> {run_name} ...", tag="train")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=str(_ROOT),
            creationflags=creationflags, env=env,
        )
    dt = time.time() - t0
    if rc != 0 or not metrics.exists():
        raise RuntimeError(
            f"{scene}/{variant}: train failed rc={rc} dt={dt:.1f}s, see {log}"
        )
    try:
        d = json.loads(metrics.read_text(encoding="utf-8"))
        psnr = d.get("metrics", {}).get("test/psnr")
        ssim = d.get("metrics", {}).get("test/ssim")
        lpips = d.get("metrics", {}).get("test/lpips")
        _log(f"{scene}/{variant}: PSNR={psnr:.3f} SSIM={ssim:.3f} "
             f"LPIPS={lpips:.3f}  ({dt:.1f}s)", tag="train")
    except Exception:
        _log(f"{scene}/{variant}: metrics.json present but unparseable", tag="train")
    _mark(scene, marker, payload=str(metrics))


# ----------------------------------------------------------------------
# producer (download+unzip queue)
# ----------------------------------------------------------------------
class _Producer(threading.Thread):
    """Walks the scene list, downloads + unzips each, signals the consumer."""

    def __init__(self, scenes: List[str], ready_q: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self.scenes = scenes
        self.ready_q = ready_q
        self.errors: List[str] = []

    def run(self) -> None:
        for scene in self.scenes:
            try:
                if _has(scene, "unzipped") and (
                    (_DATA_ROOT / scene / "transforms_train.json").exists()
                ):
                    _log(f"{scene}: already unzipped, queuing", tag="dl")
                    self.ready_q.put(scene)
                    continue
                zip_path = _download_scene(scene)
                _unzip_scene(scene, zip_path)
                self.ready_q.put(scene)
            except Exception as e:
                msg = f"{scene}: producer FAILED: {e}\n{traceback.format_exc()}"
                _log(msg, tag="dl")
                self.errors.append(msg)
                # do NOT enqueue; consumer will skip this scene
        self.ready_q.put("__DONE__")
        _log("producer finished", tag="dl")


# ----------------------------------------------------------------------
# consumer (GPU pipeline)
# ----------------------------------------------------------------------
def _consume(ready_q: "queue.Queue[str]") -> List[str]:
    failures: List[str] = []
    seen_done = False
    while not seen_done:
        scene = ready_q.get()
        if scene == "__DONE__":
            seen_done = True
            continue
        try:
            _cache_dav2(scene)
            _train(scene, "baseline")
            _train(scene, "dav2s")
            _mark(scene, "all_done")
            _log(f"{scene}: ALL STAGES DONE", tag="main")
        except Exception as e:
            msg = f"{scene}: consumer FAILED: {e}\n{traceback.format_exc()}"
            _log(msg, tag="main")
            failures.append(msg)
    return failures


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=_DEFAULT_SCENES,
                    choices=_DEFAULT_SCENES,
                    help="subset to run; default = all 7 missing Blender scenes")
    ap.add_argument("--dry-run", action="store_true",
                    help="show plan and per-scene state, do not run")
    args = ap.parse_args()

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"=== Blender avg-8 @ n=8 pipeline ===  scenes={args.scenes}", tag="main")

    # state report
    _log("current state:", tag="main")
    for s in args.scenes:
        flags = []
        for st in ["downloaded", "unzipped", "cached",
                   "trained_baseline", "trained_dav2s", "all_done"]:
            flags.append(f"{st}={'Y' if _has(s, st) else '.'}")
        _log(f"  {s:<10} {' '.join(flags)}", tag="main")

    if args.dry_run:
        _log("dry-run; exiting", tag="main")
        return 0

    # ensure configs exist (idempotent)
    if not _CFG_DIR.exists() or not list(_CFG_DIR.glob("*.yaml")):
        _log("generating configs/_blender_n8/*.yaml ...", tag="main")
        subprocess.check_call(
            [sys.executable, str(_ROOT / "scripts" / "_gen_blender_n8_configs.py")],
            cwd=str(_ROOT),
        )

    ready_q: "queue.Queue[str]" = queue.Queue()
    producer = _Producer(args.scenes, ready_q)
    producer.start()
    failures = _consume(ready_q)
    producer.join(timeout=10)

    if producer.errors or failures:
        _log(f"=== pipeline done WITH ERRORS  "
             f"producer={len(producer.errors)}  consumer={len(failures)} ===",
             tag="main")
        return 1
    _log("=== pipeline done CLEAN ===", tag="main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
