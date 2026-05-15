"""Evaluate a finished training run on the FULL test split.

Why this exists
---------------
The trainer caps eval to ``cfg.eval.num_test_renders`` (default 8) for speed
during the training loop. When we want a paper-quality number we need PSNR /
SSIM / LPIPS over **all** test cameras. This script:

  1. Loads ``<run_dir>/config.yaml`` and ``<run_dir>/ckpts/last.pt``.
  2. Re-builds the dataset *exactly* as during training (same seed, same
     train_view_ids, same downsample, same depth_prior cache).
  3. Renders every test camera and reports PSNR / SSIM / LPIPS.
  4. Writes ``<run_dir>/metrics.json`` (and ``<run_dir>/metrics_full.json``
     if a previous metrics.json from training already exists).

Usage
-----
    python scripts/eval_ckpt.py --run outputs/lego_n9_ssl_mv_dav2s
    python scripts/eval_ckpt.py --run outputs/lego_n6_ssl_mv_v3 --max-views 100
    # Sweep:
    python scripts/eval_ckpt.py --glob "outputs/lego_*"

Memory notes
------------
On a 24 GB Laptop 5090 with 800x800 + ~500K Gaussians + LPIPS-VGG, peak
eval memory is ~12 GB. We render one view at a time and free LPIPS feature
maps between views; OOM should not occur. If it does, pass ``--no-lpips``.
"""
from __future__ import annotations

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap                                    # noqa: E402,F401

import argparse                                                   # noqa: E402
import gc                                                         # noqa: E402
import glob                                                       # noqa: E402
import json                                                       # noqa: E402
import time                                                       # noqa: E402
from pathlib import Path                                          # noqa: E402
from typing import Any, Dict, List, Optional                      # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from sparse_gs.datasets.nerf_synthetic import (                   # noqa: E402
    Camera, NerfSyntheticDataset,
)
from sparse_gs.datasets.llff import LLFFDataset                   # noqa: E402
from sparse_gs.models.gaussians import GaussianModel              # noqa: E402
from sparse_gs.rendering.gsplat_renderer import GSplatRenderer    # noqa: E402
from sparse_gs.utils.config import load_config                    # noqa: E402
from sparse_gs.utils.io import save_image                         # noqa: E402
from sparse_gs.utils.metrics import (                             # noqa: E402
    LPIPSMetric, hwc_to_bchw, psnr, ssim,
)

_ROOT = _bootstrap.PROJECT_ROOT


def _build_dataset(cfg: Dict[str, Any]):
    d = cfg["data"]
    dtype = d.get("type", "nerf_synthetic")
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    # Make root absolute (cfg in run dir often already is, but be safe).
    root = Path(d["root"])
    if not root.is_absolute():
        root = (_ROOT / root).resolve()
    if dtype == "nerf_synthetic":
        return NerfSyntheticDataset(
            root=str(root), scene=d["scene"],
            n_train_views=int(d.get("n_train_views", 6)),
            train_view_ids=d.get("train_view_ids"),
            image_downsample=int(d.get("image_downsample", 1)),
            white_background=bool(d.get("white_background", True)),
            seed=seed, depth_prior=d.get("depth_prior"),
        )
    elif dtype == "llff":
        return LLFFDataset(
            root=str(root), scene=d["scene"],
            n_train_views=int(d.get("n_train_views", 3)),
            train_view_ids=d.get("train_view_ids"),
            image_downsample=int(d.get("image_downsample", 4)),
            white_background=bool(d.get("white_background", False)),
            recenter=bool(d.get("recenter", True)),
            rescale=bool(d.get("rescale", True)),
            seed=seed,
            sparse_mode=str(d.get("sparse_mode", "uniform")),
            depth_prior=d.get("depth_prior"),
        )
    raise ValueError(f"unknown dataset type: {dtype!r}")


def _white_or_zero_bg(cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    use_white = bool(cfg["data"].get("white_background", True))
    bg = torch.tensor([1.0, 1.0, 1.0] if use_white else [0.0, 0.0, 0.0], device=device)
    return bg


@torch.no_grad()
def evaluate_run(
    run_dir: Path,
    ckpt_name: str = "last.pt",
    max_views: Optional[int] = None,
    use_lpips: bool = True,
    save_renders: bool = False,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print(f"\n=== {run_dir.name} ===")
    print(f"  scene={cfg['data']['scene']}  n_views={cfg['data'].get('n_train_views')}  "
          f"type={cfg['data'].get('type', 'nerf_synthetic')}")

    dataset = _build_dataset(cfg)
    ckpt_path = run_dir / "ckpts" / ckpt_name
    if not ckpt_path.exists():
        # Fall back to last iter ckpt if present.
        candidates = sorted((run_dir / "ckpts").glob("iter_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no ckpt under {run_dir/'ckpts'}")
        ckpt_path = candidates[-1]
    print(f"  ckpt={ckpt_path.name}")

    gaussians = GaussianModel(sh_degree=int(cfg.get("model", {}).get("sh_degree", 3)))
    gaussians.load(str(ckpt_path), device=str(device))
    n_g = gaussians.num_points
    print(f"  #G={n_g}  test_views={len(dataset.test)}")

    rcfg = cfg.get("renderer", {})
    renderer = GSplatRenderer(
        sh_degree=int(cfg.get("model", {}).get("sh_degree", 3)),
        near_plane=float(rcfg.get("near_plane", 0.01)),
        far_plane=float(rcfg.get("far_plane", 1.0e10)),
        rasterize_mode=str(rcfg.get("rasterize_mode", "antialiased")),
        packed=bool(rcfg.get("packed", False)),
        absgrad=bool(rcfg.get("absgrad", True)),
        render_mode=str(rcfg.get("render_mode", "RGB+ED")),
    )
    bg = _white_or_zero_bg(cfg, device)

    lpips_metric = None
    if use_lpips:
        try:
            net = str(cfg.get("eval", {}).get("lpips_net", "vgg"))
            lpips_metric = LPIPSMetric(net=net, device=device)
            if not lpips_metric.available:
                lpips_metric = None
        except Exception as e:  # pragma: no cover
            print(f"  [warn] LPIPS unavailable: {e}")
            lpips_metric = None

    cams: List[Camera] = list(dataset.test)
    if max_views is not None and max_views > 0:
        cams = cams[:max_views]
    psnrs: List[float] = []
    ssims: List[float] = []
    lpipss: List[float] = []

    render_dir = run_dir / "renders_eval"
    if save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, cam in enumerate(cams):
        cam = cam.to(device)
        out = renderer.render(
            gaussians,
            viewmat=cam.viewmat, K=cam.K,
            width=cam.width, height=cam.height,
            active_sh_degree=gaussians.sh_degree,
            background=bg,
        )
        pred = out["rgb"].clamp(0, 1)
        gt = cam.image
        psnrs.append(float(psnr(pred, gt).item()))
        ssims.append(float(ssim(hwc_to_bchw(pred), hwc_to_bchw(gt)).item()))
        if lpips_metric is not None:
            lp = lpips_metric(pred, gt)
            if lp is not None:
                lpipss.append(float(lp))
        if save_renders:
            save_image(pred, render_dir / f"test_{i:03d}_{cam.image_name}.png")
        # release intermediate tensors
        del out, pred, gt
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    metrics = {
        "test/psnr": float(np.mean(psnrs)),
        "test/ssim": float(np.mean(ssims)),
    }
    if lpipss:
        metrics["test/lpips"] = float(np.mean(lpipss))
    print(f"  -> {' | '.join(f'{k}={v:.4f}' for k, v in metrics.items())} "
          f"({elapsed:.1f}s, {len(cams)} views)")

    payload = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "scene": cfg.get("data", {}).get("scene"),
        "type": cfg.get("data", {}).get("type", "nerf_synthetic"),
        "n_train_views": int(cfg.get("data", {}).get("n_train_views", 0)),
        "ckpt": ckpt_path.name,
        "num_gaussians": int(n_g),
        "num_test_views_used": int(len(cams)),
        "num_test_views_total": int(len(dataset.test)),
        "wall_clock_eval_sec": float(elapsed),
        "train_view_ids": list(dataset.train_view_ids),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    # Don't trample an existing metrics.json from training (which may have
    # extra fields like wall_clock_sec). Append _full to distinguish.
    out_name = "metrics.json"
    existing = run_dir / out_name
    if existing.exists():
        out_name = "metrics_full.json"
    with open(run_dir / out_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {run_dir/out_name}")

    # Aggressive cleanup so multi-run sweeps don't leak.
    del gaussians, renderer, dataset
    if lpips_metric is not None:
        del lpips_metric
    gc.collect()
    torch.cuda.empty_cache()

    return payload


def _expand_runs(args) -> List[Path]:
    runs: List[Path] = []
    if args.run:
        runs.append(Path(args.run))
    if args.glob:
        for p in sorted(glob.glob(args.glob)):
            pp = Path(p)
            if pp.is_dir() and (pp / "config.yaml").exists():
                runs.append(pp)
    # Dedup, keep order.
    seen = set()
    unique: List[Path] = []
    for r in runs:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            unique.append(r)
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Path to a single run dir (must contain config.yaml + ckpts/last.pt).")
    ap.add_argument("--glob", type=str, default=None,
                    help='Glob pattern, e.g. "outputs/lego_*" to evaluate many runs.')
    ap.add_argument("--ckpt", type=str, default="last.pt",
                    help="Checkpoint filename within ckpts/ (default: last.pt).")
    ap.add_argument("--max-views", type=int, default=None,
                    help="Cap number of test cameras (default: all).")
    ap.add_argument("--no-lpips", action="store_true",
                    help="Skip LPIPS to save memory.")
    ap.add_argument("--save-renders", action="store_true",
                    help="Save full-test renders under renders_eval/.")
    args = ap.parse_args()

    runs = _expand_runs(args)
    if not runs:
        ap.error("nothing to evaluate; pass --run or --glob")

    print(f"[eval_ckpt] evaluating {len(runs)} run(s)")
    summary = []
    for r in runs:
        try:
            payload = evaluate_run(
                r, ckpt_name=args.ckpt, max_views=args.max_views,
                use_lpips=not args.no_lpips, save_renders=args.save_renders,
            )
            summary.append(payload)
        except Exception as e:
            print(f"  [error] {r.name}: {type(e).__name__}: {e}")

    if len(summary) > 1:
        print("\n=== summary ===")
        print(f"{'experiment':<40} {'n':>3} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'#G':>9}")
        for p in summary:
            m = p["metrics"]
            print(f"{(p['experiment'] or p.get('scene','?'))[:40]:<40} "
                  f"{p['n_train_views']:>3} "
                  f"{m.get('test/psnr', float('nan')):>8.4f} "
                  f"{m.get('test/ssim', float('nan')):>7.4f} "
                  f"{m.get('test/lpips', float('nan')):>7.4f} "
                  f"{p['num_gaussians']:>9d}")


if __name__ == "__main__":
    main()
