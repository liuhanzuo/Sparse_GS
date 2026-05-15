"""B-stage diagnostic plot: train vs test camera coverage on the unit sphere.

Generates a 4-panel azimuth-elevation scatter plot for each scene showing:
  * test views colored by PSNR (red=bad, green=good)
  * train views as black stars

Writes outputs/_b_diag/<scene>_coverage.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nerf_synthetic"
DIAG = ROOT / "outputs" / "_b_diag"

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def cart_to_az_el(p: np.ndarray):
    x, y, z = p
    r = np.linalg.norm(p)
    el = math.asin(z / max(r, 1e-9))
    az = math.atan2(y, x)
    return math.degrees(az), math.degrees(el)


def build_coverage_panel(ax, scene: str):
    train_d = json.loads((DATA / scene / "transforms_train.json").read_text(encoding="utf-8"))
    test_d = json.loads((DATA / scene / "transforms_test.json").read_text(encoding="utf-8"))
    diag = json.loads((DIAG / f"{scene}.json").read_text(encoding="utf-8"))
    train_ids = diag["train_view_ids"]

    train_pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in train_d["frames"]])
    test_pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in test_d["frames"]])
    sel_train_pos = train_pos[train_ids]

    # Build a per-test-view PSNR map from worst5+best5+global (or recompute average)
    # We already saved per_view but trimmed; use diag's per_view stats and infer
    # via computing again: attach 'orig' from worst5/best5 only is too sparse.
    # Easier: re-read per-view by listing render dir size — but we only saved
    # mean/min/p10. So we recompute PSNR on the fly using PIL.
    from PIL import Image
    render_dir = ROOT / "outputs" / (
        f"blender_{scene}_n8_ssl_mv_dav2s_depthv2_prune"
        if scene != "lego" else "lego_n8_ssl_mv_dav2s_depthv2_prune"
    ) / "renders"
    gt_dir = DATA / scene / "test"
    psnr_map = {}
    for p in sorted(render_dir.glob("test_*_r_*.png")):
        orig = int(p.stem.split("_")[-1])
        gt_p = gt_dir / f"r_{orig}.png"
        if not gt_p.exists():
            continue
        rend = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        gt_rgba = np.asarray(Image.open(gt_p).convert("RGBA"), dtype=np.float32) / 255.0
        a = gt_rgba[..., 3:4]
        gt = gt_rgba[..., :3] * a + (1.0 - a) * 1.0
        if rend.shape != gt.shape:
            continue
        mse = float(np.mean((rend - gt) ** 2))
        psnr_map[orig] = 10.0 * math.log10(1.0 / max(mse, 1e-12))

    test_az_el = np.array([cart_to_az_el(p) for p in test_pos])
    train_az_el = np.array([cart_to_az_el(p) for p in sel_train_pos])

    # Build a color array for test views by PSNR
    test_psnr = np.array([psnr_map.get(i, np.nan) for i in range(len(test_pos))])
    valid = ~np.isnan(test_psnr)

    sc = ax.scatter(
        test_az_el[valid, 0], test_az_el[valid, 1], c=test_psnr[valid],
        s=18, cmap="RdYlGn", vmin=12.0, vmax=30.0, alpha=0.85, edgecolor="none",
    )
    ax.scatter(
        train_az_el[:, 0], train_az_el[:, 1],
        marker="*", s=240, color="black", edgecolor="white", linewidths=1.0, zorder=5,
        label="train view",
    )
    ax.set_title(f"{scene} (global PSNR={diag['test_psnr_global']:.2f})", fontsize=10)
    ax.set_xlabel("azimuth (deg)", fontsize=8)
    ax.set_ylabel("elevation (deg)", fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    return sc


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    sc = None
    for ax, scene in zip(axes.flat, SCENES):
        sc = build_coverage_panel(ax, scene)
    fig.suptitle(
        "W2 train-view (★) vs test-view (○ colored by per-view PSNR) coverage on camera sphere\n"
        "Red = test view rendered poorly; isolated red clusters = blind spots due to sparse train view selection",
        fontsize=12,
    )
    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.05, aspect=80)
    cbar.set_label("test view PSNR (clipped to [12, 30])", fontsize=10)
    out = DIAG / "all_scenes_coverage.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
