"""Render the test split from a checkpoint and report PSNR/SSIM.

Usage:

    python -m scripts.render --config configs/sparse_view.yaml \
                             --ckpt outputs/lego_n6/ckpts/last.pt
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap                                    # noqa: E402,F401

import argparse                                                   # noqa: E402
from pathlib import Path                                          # noqa: E402

import torch                                                      # noqa: E402

from sparse_gs.datasets.nerf_synthetic import NerfSyntheticDataset
from sparse_gs.models.gaussians import GaussianModel
from sparse_gs.rendering.gsplat_renderer import GSplatRenderer
from sparse_gs.utils.config import load_config
from sparse_gs.utils.io import save_image
from sparse_gs.utils.metrics import psnr, ssim, hwc_to_bchw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None,
                    help="render dir (default: <ckpt parent>/../renders)")
    ap.add_argument("--max_views", default=None, type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["experiment"].get("device", "cuda"))

    # Resolve data root relative to the project root.
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        cfg["data"]["root"] = str((_bootstrap.PROJECT_ROOT / data_root).resolve())

    d = cfg["data"]
    dataset = NerfSyntheticDataset(
        root=d["root"], scene=d["scene"],
        n_train_views=int(d.get("n_train_views", 6)),
        train_view_ids=d.get("train_view_ids"),
        image_downsample=int(d.get("image_downsample", 1)),
        white_background=bool(d.get("white_background", True)),
        seed=int(cfg["experiment"].get("seed", 42)),
    )

    g = GaussianModel(sh_degree=int(cfg["model"]["sh_degree"])).to(device)
    g.load(args.ckpt, device=device)

    r = cfg["renderer"]
    renderer = GSplatRenderer(
        sh_degree=int(cfg["model"]["sh_degree"]),
        near_plane=float(r.get("near_plane", 0.01)),
        far_plane=float(r.get("far_plane", 1.0e10)),
        rasterize_mode=str(r.get("rasterize_mode", "antialiased")),
        packed=bool(r.get("packed", True)),
        absgrad=bool(r.get("absgrad", True)),
        render_mode=str(r.get("render_mode", "RGB+ED")),
    )

    out_dir = Path(args.out) if args.out else (Path(args.ckpt).parent.parent / "renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if d.get("white_background", True) else [0.0, 0.0, 0.0],
        device=device,
    )

    cams = dataset.test
    if args.max_views is not None:
        cams = cams[: args.max_views]

    psnrs, ssims = [], []
    with torch.no_grad():
        for i, cam in enumerate(cams):
            cam = cam.to(device)
            out = renderer.render(
                g, viewmat=cam.viewmat, K=cam.K,
                width=cam.width, height=cam.height,
                active_sh_degree=g.sh_degree, background=bg,
            )
            pred = out["rgb"].clamp(0, 1)
            psnrs.append(float(psnr(pred, cam.image).item()))
            ssims.append(float(ssim(hwc_to_bchw(pred), hwc_to_bchw(cam.image)).item()))
            save_image(pred, out_dir / f"test_{i:03d}_{cam.image_name}.png")

    if psnrs:
        print(f"[render] N={len(psnrs)}  PSNR={sum(psnrs)/len(psnrs):.3f}  "
              f"SSIM={sum(ssims)/len(ssims):.4f}")
        print(f"[render] images -> {out_dir}")


if __name__ == "__main__":
    main()
