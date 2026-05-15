"""Pre-compute monocular depth priors for the training views of a scene.

Saves one ``<image_name>.npz`` per view under
``<scene>/_depth_cache/<tag>/<split>/``. The trainer can then read them
back via ``cfg.data.depth_prior = {enabled: true, backend: cache, tag: <tag>}``.

Backends
--------
``alpha`` (default; trivially derived, no model)
    Just dumps ``alpha`` from RGBA as a pseudo-disparity. Equivalent to
    using ``backend: alpha`` directly at training time, but materialising
    the cache lets you compare runs on a level playing field with the
    "real" priors below. **NeRF-Synthetic only** (LLFF has no alpha).

``transformers`` (requires ``transformers`` and an internet connection on
first run; the model is cached by HuggingFace Hub afterwards)
    Runs a HF depth-estimation pipeline on every train image. Default
    model: ``depth-anything/Depth-Anything-V2-Small-hf`` (~25 M params,
    runs on a single consumer GPU in seconds). Works on any RGB input,
    so use this for LLFF / DTU / real scenes.

Datasets
--------
``--dataset nerf_synthetic`` (default)
    Walk ``transforms_<split>.json``. Frames have ``file_path`` entries
    like ``./train/r_0`` (no extension). Cache files are named
    ``<image_name_stem>.npz`` (e.g. ``r_0.npz``).

``--dataset llff``
    Walk the LLFF ``images/`` (or ``images_<downsample>/``) directory.
    Treat the whole directory as the ``train`` split — sparse-view
    selection happens later inside the trainer's ``LLFFDataset``, so we
    just dump a depth map for every available view here. This matches
    the ``Camera.image_name`` produced by ``LLFFDataset`` so cache lookup
    via ``utils.depth_prior.cache_path_for`` finds it.

Usage
-----
    python scripts/precompute_depth.py --scene lego --backend alpha
    python scripts/precompute_depth.py --scene lego --backend transformers \\
        --model depth-anything/Depth-Anything-V2-Small-hf
    python scripts/precompute_depth.py --dataset llff --scene fern \\
        --data-root data/nerf_llff_data --image-downsample 4 \\
        --backend transformers --model depth-anything/Depth-Anything-V2-Small-hf

Then in your config:

    data:
      depth_prior:
        enabled: true
        backend: cache
        tag: alpha                                          # or e.g. depth_anything_v2_small_hf
        kind: disparity                                     # almost always "disparity" for HF DPT
"""

from __future__ import annotations

# ---- bootstrap so we can run as either ``python scripts/precompute_depth.py``
# or ``python -m scripts.precompute_depth``. Mirrors train.py.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap                                          # noqa: E402,F401

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
from pathlib import Path                                                 # noqa: E402
from typing import List, Optional, Tuple                                  # noqa: E402

import numpy as np                                                       # noqa: E402

from sparse_gs.utils.io import load_image_rgba                           # noqa: E402

_ROOT = _bootstrap.PROJECT_ROOT


# =====================================================================
# frame enumeration: returns a list of (image_path, image_name_for_cache)
#
# ``image_name_for_cache`` is what ``Camera.image_name`` will be at
# training time, so cache lookup uses ``Path(name).stem`` (see
# ``utils.depth_prior.cache_path_for``).
# =====================================================================
def _enumerate_nerf_synthetic(scene_root: Path, split: str) -> List[Tuple[Path, str]]:
    j = scene_root / f"transforms_{split}.json"
    with open(j, "r", encoding="utf-8") as f:
        meta = json.load(f)
    out: List[Tuple[Path, str]] = []
    for fr in meta["frames"]:
        rel = fr["file_path"]
        ip = _resolve_image_path_ns(scene_root, rel)
        # NerfSyntheticDataset uses Path(rel).name as image_name (e.g. "r_0")
        out.append((ip, Path(rel).name))
    return out


def _resolve_image_path_ns(scene_root: Path, rel: str) -> Path:
    for cand in (
        scene_root / (rel + ".png"),
        scene_root / rel,
        scene_root / Path(rel).with_suffix(".png"),
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(rel)


def _enumerate_llff(scene_root: Path, image_downsample: int) -> List[Tuple[Path, str]]:
    """Mirror LLFFDataset's image-folder selection so cache names line up."""
    if image_downsample > 1:
        pre = scene_root / f"images_{image_downsample}"
        images_dir = pre if pre.is_dir() else (scene_root / "images")
    else:
        images_dir = scene_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    files = sorted(
        [p for p in images_dir.iterdir() if p.suffix in exts],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"no images under {images_dir}")
    # LLFFDataset uses fpath.name (with extension) as image_name.
    return [(p, p.name) for p in files]


def _save_npz(out: Path, depth: np.ndarray, kind: str, source: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out),
             depth=depth.astype(np.float32),
             meta=np.array(json.dumps({"kind": kind, "source": source})))


# ---------------------------------------------------------------------
# backend: alpha (NeRF-Synthetic only)
# ---------------------------------------------------------------------
def _run_alpha(frames: List[Tuple[Path, str]], split: str,
               out_root: Path) -> int:
    n = 0
    for ip, name in frames:
        img = load_image_rgba(ip)
        if img.ndim == 2 or img.shape[-1] != 4:
            print(f"[alpha] {ip.name}: no alpha channel, skipping")
            continue
        alpha = img[..., 3].astype(np.float32)                # (H, W) in [0,1]
        # Same convention as utils.depth_prior._alpha_to_disparity:
        # disparity = alpha (foreground=1, background=0).
        disp = alpha.copy()
        out = out_root / split / f"{Path(name).stem}.npz"
        _save_npz(out, disp, kind="disparity", source=f"alpha:{ip.name}")
        n += 1
    return n


# ---------------------------------------------------------------------
# backend: transformers (HF Depth-Anything-V2 / DPT / ...)
# ---------------------------------------------------------------------
def _run_transformers(
    frames: List[Tuple[Path, str]],
    split: str,
    out_root: Path,
    model_name: str,
    device: str,
    composite_white: bool,
) -> int:
    try:
        import torch
        from PIL import Image
    except Exception as e:                                     # noqa: BLE001
        raise RuntimeError(
            f"backend=transformers requires `transformers` and PIL. "
            f"Import failed: {e}"
        )

    # Try the pipeline path first (works on transformers <5). On transformers
    # 5.x the depth-anything `pipeline()` constructor errors with
    # "Unrecognized model in <repo>" because the auto-config registry has
    # moved; in that case we fall back to AutoImageProcessor +
    # AutoModelForDepthEstimation, which still works.
    pipe = None
    proc = None
    mdl = None
    use_pipeline = True
    try:
        from transformers import pipeline
        print(f"[hf] loading pipeline {model_name} on {device} ...")
        pipe = pipeline("depth-estimation", model=model_name,
                        device=0 if device == "cuda" else -1)
    except Exception as e_pipe:
        use_pipeline = False
        print(f"[hf] pipeline() failed ({type(e_pipe).__name__}: {e_pipe}); "
              f"falling back to AutoModelForDepthEstimation")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(model_name)
        mdl = AutoModelForDepthEstimation.from_pretrained(model_name)
        if device == "cuda" and torch.cuda.is_available():
            mdl = mdl.to("cuda")
        mdl.eval()

    n = 0
    for ip, name in frames:
        img = load_image_rgba(ip)                              # H,W,{3,4} float [0,1]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            rgb, a = img[..., :3], img[..., 3:4]
            if composite_white:
                rgb = rgb * a + (1.0 - a)
            img = rgb
        pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        if use_pipeline:
            with torch.no_grad():
                out = pipe(pil)
            # pipe returns {'depth': PIL.Image (uint8 visualisation),
            #               'predicted_depth': torch.Tensor (1,h,w) raw}
            if "predicted_depth" in out:
                d = out["predicted_depth"]
                if hasattr(d, "detach"):
                    d = d.detach().float().cpu().numpy()
                d = np.asarray(d)
                if d.ndim == 3:
                    d = d[0]
            else:
                vis = np.asarray(out["depth"]).astype(np.float32) / 255.0
                d = vis if vis.ndim == 2 else vis[..., 0]
        else:
            inp = proc(pil, return_tensors="pt")
            if device == "cuda" and torch.cuda.is_available():
                inp = {k: v.to("cuda") for k, v in inp.items()}
            with torch.no_grad():
                out = mdl(**inp)
            d = out.predicted_depth
            if hasattr(d, "detach"):
                d = d.detach().float().cpu().numpy()
            d = np.asarray(d)
            if d.ndim == 3:
                d = d[0]
        # Resize to source resolution if needed
        H, W = img.shape[:2]
        if d.shape[0] != H or d.shape[1] != W:
            d_t = torch.from_numpy(d.astype(np.float32))[None, None]
            d_t = torch.nn.functional.interpolate(
                d_t, size=(H, W), mode="bilinear", align_corners=False,
            )
            d = d_t[0, 0].numpy()

        # Depth-Anything-V2-* (HF) returns a *disparity* (larger = closer).
        # We persist that as-is and tag kind=disparity.
        out_path = out_root / split / f"{Path(name).stem}.npz"
        _save_npz(out_path, d.astype(np.float32),
                  kind="disparity", source=model_name)
        n += 1
        if n % 10 == 0:
            print(f"  ... {n} done")
    return n


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nerf_synthetic", "llff"],
                    default="nerf_synthetic")
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--data-root", default=None,
                    help="defaults to data/nerf_synthetic or data/nerf_llff_data "
                         "depending on --dataset")
    ap.add_argument("--splits", default="train",
                    help="comma-separated, e.g. 'train' or 'train,val'. "
                         "LLFF only supports 'train' (the dataset has no JSON splits; "
                         "we cache every image and let the dataloader pick).")
    ap.add_argument("--backend", choices=["alpha", "transformers"], default="alpha")
    ap.add_argument("--tag", default=None,
                    help="cache sub-folder name; default depends on backend")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF model id for backend=transformers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--composite-white", action="store_true", default=True,
                    help="composite RGBA on white before feeding the model "
                         "(matches the trainer's data pipeline; LLFF: no-op)")
    ap.add_argument("--image-downsample", type=int, default=1,
                    help="LLFF only: pick images_<N>/ if available so cache "
                         "matches the trainer's resolution.")
    args = ap.parse_args()

    if args.data_root is None:
        args.data_root = str(
            _ROOT / "data"
            / ("nerf_synthetic" if args.dataset == "nerf_synthetic" else "nerf_llff_data")
        )
    scene_root = Path(args.data_root) / args.scene
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)

    if args.tag is None:
        if args.backend == "alpha":
            args.tag = "alpha"
        else:
            # e.g. depth-anything/Depth-Anything-V2-Small-hf -> depth_anything_v2_small_hf
            args.tag = (args.model.split("/")[-1]
                        .replace("-", "_").lower())

    out_root = scene_root / "_depth_cache" / args.tag
    print(f"[run] dataset={args.dataset}  scene={args.scene}  "
          f"backend={args.backend}  tag={args.tag}")
    print(f"      writing -> {out_root}")

    # ---- enumerate frames per split ----
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.dataset == "llff":
        # LLFF has no JSON splits; we ignore the "split" axis and dump
        # everything under split=train (matches LLFFDataset).
        if splits != ["train"]:
            print(f"[warn] dataset=llff: forcing splits=['train'] (was {splits})")
        frames_by_split = {
            "train": _enumerate_llff(scene_root, args.image_downsample)
        }
    else:
        frames_by_split = {
            s: _enumerate_nerf_synthetic(scene_root, s) for s in splits
        }

    # ---- run backend ----
    total = 0
    for split, frames in frames_by_split.items():
        if args.backend == "alpha":
            if args.dataset == "llff":
                raise SystemExit(
                    "[error] backend=alpha is NeRF-Synthetic-only (no alpha on LLFF). "
                    "Use --backend transformers instead."
                )
            n = _run_alpha(frames, split, out_root)
        else:
            n = _run_transformers(frames, split, out_root,
                                  model_name=args.model,
                                  device=args.device,
                                  composite_white=args.composite_white)
        print(f"[done] split={split}: wrote {n} files")
        total += n

    print(f"[done] total: {total}")


if __name__ == "__main__":
    main()
