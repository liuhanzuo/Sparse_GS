"""Diagnose why drums/materials are extremely low PSNR after W3 dual-GS.

Side-by-side compares pred vs GT(white bg) for several test views, save
triptych and print quantitative stats. Run:

    python scripts/diag_w3_failed_scenes.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "nerf_synthetic"
OUT_ROOT = ROOT / "outputs"
DIAG_OUT = OUT_ROOT / "_diag_w3_failed"
DIAG_OUT.mkdir(parents=True, exist_ok=True)

SCENES = {
    "drums":     "blender_drums_n8_w3_dual_gs",
    "materials": "blender_materials_n8_w3_dual_gs",
    "chair":     "blender_chair_n8_w3_dual_gs",   # 对照（好场景）
}

# 选几个均匀分布的 test 视角
VIEW_IDS: List[int] = [0, 50, 100, 150]


def _load_gt_white_with_alpha(scene: str, view: int):
    p = DATA_ROOT / scene / "test" / f"r_{view}.png"
    im = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255.0
    rgb, a = im[..., :3], im[..., 3:4]
    white = np.clip(rgb * a + (1.0 - a), 0.0, 1.0)
    return white, a[..., 0]


def _load_pred(out_name: str, view: int) -> np.ndarray:
    cand = OUT_ROOT / out_name / "renders" / f"test_{view:03d}_r_{view}.png"
    if not cand.exists():
        ms = list((OUT_ROOT / out_name / "renders").glob(f"test_*_r_{view}.png"))
        cand = ms[0]
    im = np.asarray(Image.open(cand).convert("RGB")).astype(np.float32) / 255.0
    return im


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def main() -> None:
    rows = []
    for scene, out_name in SCENES.items():
        for v in VIEW_IDS:
            try:
                gt, alpha = _load_gt_white_with_alpha(scene, v)
                pr = _load_pred(out_name, v)
            except Exception as e:
                print(f"[skip] {scene} v={v}: {e}")
                continue
            if pr.shape != gt.shape:
                gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((pr.shape[1], pr.shape[0]))).astype(np.float32) / 255.0
                alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((pr.shape[1], pr.shape[0]))).astype(np.float32) / 255.0

            psnr_full = _psnr(pr, gt)

            # 1) 前景 mask（GT alpha > 0.5）
            fg = alpha > 0.5
            bg = ~fg
            err = (pr - gt) ** 2
            mse_fg = float(err[fg].mean()) if fg.any() else 0.0
            mse_bg = float(err[bg].mean()) if bg.any() else 0.0
            psnr_fg = -10.0 * np.log10(mse_fg + 1e-12)
            psnr_bg = -10.0 * np.log10(mse_bg + 1e-12)

            # 2) 「该是白底但被涂色」的像素比例
            #    GT 背景（alpha<0.05），但 pred 偏离白色超过 0.05
            bg_strict = alpha < 0.05
            pr_offwhite = (np.abs(pr - 1.0).mean(-1) > 0.05)
            floater_pix = float((bg_strict & pr_offwhite).sum()) / float(bg_strict.sum() + 1e-9)

            # 3) 颜色偏置：背景区域 pred 平均色
            if bg_strict.any():
                bg_pred_mean = pr[bg_strict].mean(0)
            else:
                bg_pred_mean = np.array([1, 1, 1])

            # 三联保存（pred | GT | diff）
            diff = np.clip(np.abs(pr - gt) * 5.0, 0.0, 1.0)
            tri = np.concatenate([pr, gt, diff], axis=1)
            outp = DIAG_OUT / f"{scene}_v{v:03d}.png"
            Image.fromarray((tri * 255).astype(np.uint8)).save(outp)

            rows.append((scene, v, psnr_full, psnr_fg, psnr_bg, floater_pix,
                         tuple(bg_pred_mean.round(3))))
            print(f"{scene:10s} v={v:3d} | full={psnr_full:6.2f} | fg={psnr_fg:6.2f} | bg={psnr_bg:6.2f} | "
                  f"floater_in_bg={floater_pix*100:5.1f}% | bg_pred_mean(R,G,B)={tuple(bg_pred_mean.round(3))}")

    md = ["# W3 dual-GS failed-scene diagnosis (foreground / background split)",
          "",
          "| scene | view | full PSNR | fg PSNR | bg PSNR | floater%(bg) | bg pred mean(RGB) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]*100:.1f}% | {r[6]} |")
    (DIAG_OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[diag] saved -> {DIAG_OUT}")


if __name__ == "__main__":
    main()
