"""Retry any failed / missing runs from the LLFF sweep.

Usage::

    python scripts/retry_failed_runs.py                 # retry all missing
    python scripts/retry_failed_runs.py --max-retries 2 # retry each up to 2x
    python scripts/retry_failed_runs.py --dry-run       # only show what would run

Behaviour:
- Scans every expected run dir ``outputs/llff_<scene>_n3_<variant>/`` for the
  8 LLFF scenes x {baseline, dav2s}.
- A run is considered "failed" iff it has no ``metrics.json``.
- Each failed run is retried up to ``--max-retries`` times. Between retries we
  wait a few seconds and clean up any partial output dir, so the next attempt
  starts from a fresh state.
- Children are spawned with ``CREATE_NO_WINDOW`` on Windows -- no popup at all.
- The script can be safely launched detached (e.g. by
  ``launch_llff_sweep_detached.py`` after the main sweep ends).

This is intentionally separate from ``run_llff_sweep.py``: that script's only
job is to enumerate-and-train; this one's only job is to *recover*.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _ROOT / "configs" / "_llff_sweep"
_OUT_DIR = _ROOT / "outputs"
_LOG_DIR = _OUT_DIR / "_llff_sweep_logs"
_DAEMON_LOG = _OUT_DIR / "_llff_sweep_retry.log"

_SCENES = ["fern", "flower", "fortress", "horns",
           "leaves", "orchids", "room", "trex"]
_VARIANTS = ["baseline", "dav2s"]


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        _DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DAEMON_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _run_name(scene: str, variant: str) -> str:
    return f"llff_{scene}_n3_{variant}"


def _missing(scenes: List[str], variants: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in scenes:
        for v in variants:
            m = _OUT_DIR / _run_name(s, v) / "metrics.json"
            if not m.exists():
                out.append((s, v))
    return out


def _cleanup_partial(scene: str, variant: str) -> None:
    d = _OUT_DIR / _run_name(scene, variant)
    if d.exists():
        try:
            shutil.rmtree(d)
            _log(f"  cleaned partial dir: {d.name}")
        except Exception as e:
            _log(f"  WARN: failed to clean {d}: {e}")


def _archive_log(scene: str, variant: str, attempt: int) -> None:
    log = _LOG_DIR / f"{_run_name(scene, variant)}.log"
    if log.exists():
        archive = _LOG_DIR / f"{_run_name(scene, variant)}.fail{attempt}.log"
        try:
            log.replace(archive)
            _log(f"  archived prior log -> {archive.name}")
        except Exception as e:
            _log(f"  WARN: failed to archive {log}: {e}")


def _train_one(scene: str, variant: str) -> int:
    cfg = _CFG_DIR / f"{_run_name(scene, variant)}.yaml"
    log = _LOG_DIR / f"{_run_name(scene, variant)}.log"
    if not cfg.exists():
        _log(f"  ERROR: missing config {cfg}")
        return -1
    log.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    cmd = [py, str(_ROOT / "scripts" / "train.py"),
           "--config", str(cfg)]
    creationflags = 0
    if os.name == "nt":
        creationflags = 0x08000000  # CREATE_NO_WINDOW
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(
            cmd,
            stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(_ROOT),
            creationflags=creationflags,
            env=env,
        )
    dt = time.time() - t0
    _log(f"  train rc={rc} elapsed={dt:.1f}s")
    return rc


def _verify_metrics(scene: str, variant: str) -> bool:
    m = _OUT_DIR / _run_name(scene, variant) / "metrics.json"
    if not m.exists():
        return False
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
        psnr = data.get("metrics", {}).get("test/psnr")
        if psnr is None:
            return False
        _log(f"  metrics OK: test/psnr = {psnr:.3f}")
        return True
    except Exception as e:
        _log(f"  metrics.json unreadable: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=_SCENES,
                    choices=_SCENES,
                    help=f"subset of scenes; default = all 8")
    ap.add_argument("--variants", nargs="+", default=_VARIANTS,
                    choices=_VARIANTS,
                    help="subset of variants; default = both")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="how many times to retry each missing run "
                         "(default: 2)")
    ap.add_argument("--cooldown-sec", type=float, default=8.0,
                    help="sleep this many seconds between attempts "
                         "(let CUDA driver settle); default: 8")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print what would be retried; do not train")
    args = ap.parse_args()

    _log(f"=== retry daemon start  scenes={args.scenes}  "
         f"variants={args.variants}  max_retries={args.max_retries} ===")

    missing = _missing(args.scenes, args.variants)
    if not missing:
        _log("no missing runs detected; nothing to do.")
        return 0
    _log(f"found {len(missing)} missing runs: "
         f"{[f'{s}/{v}' for s,v in missing]}")
    if args.dry_run:
        _log("dry-run mode; exit without training.")
        return 0

    failures: List[Tuple[str, str]] = []
    for scene, variant in missing:
        ok = False
        for attempt in range(1, args.max_retries + 1):
            _log(f"--- retry {scene}/{variant}  attempt {attempt}/"
                 f"{args.max_retries} ---")
            _archive_log(scene, variant, attempt)
            _cleanup_partial(scene, variant)
            rc = _train_one(scene, variant)
            if rc == 0 and _verify_metrics(scene, variant):
                ok = True
                break
            _log(f"  attempt {attempt} did not produce metrics.json; "
                 f"sleeping {args.cooldown_sec:.0f}s before next try")
            time.sleep(args.cooldown_sec)
        if not ok:
            failures.append((scene, variant))
            _log(f"  GIVING UP on {scene}/{variant} after "
                 f"{args.max_retries} retries")

    if failures:
        _log(f"=== retry daemon done  unrecovered={len(failures)}: "
             f"{[f'{s}/{v}' for s,v in failures]} ===")
        return 1
    _log("=== retry daemon done  all runs recovered ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
