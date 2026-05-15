"""B-stage diagnostic: per-view PSNR + train-view coverage analysis.

For each of the 8 scenes, compute:
  * per test-view PSNR / SSIM / LPIPS (already-rendered images vs GT)
  * train-view camera positions (unit-sphere coords)
  * average geodesic distance from each test-view to nearest train-view
  * worst-K test-views (for visual inspection later)

Writes outputs/_b_diag/<scene>.json with everything needed for the
follow-up report.

Usage:
  python scripts/b_diagnose.py [--scenes hotdog drums ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nerf_synthetic"
OUT = ROOT / "outputs"
DIAG = OUT / "_b_diag"
DIAG.mkdir(parents=True, exist_ok=True)

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def expname(scene: str) -> str:
    if scene == "lego":
        return "lego_n8_ssl_mv_dav2s_depthv2_prune"
    return f"blender_{scene}_n8_ssl_mv_dav2s_depthv2_prune"


def load_pose(transforms_json: Path, idx: int) -> np.ndarray:
    """Return a 4x4 c2w matrix for frame `idx`."""
    d = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = d["frames"]
    # NeRF-Synthetic test/train transforms are listed in r_0..r_N order.
    return np.asarray(frames[idx]["transform_matrix"], dtype=np.float64)


def cam_position(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3]


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) between camera positions on the sphere centered at origin."""
    na, nb = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
    cos_t = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return math.acos(cos_t)


def per_view_psnr_ssim(scene: str) -> List[Dict]:
    """Compute per-view PSNR / SSIM by reading rendered PNGs and GT.

    Note: SSIM via skimage to avoid depending on torch loading here.
    """
    from PIL import Image
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:  # pragma: no cover
        ssim = None

    render_dir = OUT / expname(scene) / "renders"
    gt_dir = DATA / scene / "test"
    rendered = sorted(render_dir.glob("test_*_r_*.png"))
    out = []
    for p in rendered:
        # filename pattern: test_<seq>_r_<orig>.png
        stem = p.stem  # test_000_r_0
        parts = stem.split("_")
        orig = int(parts[-1])  # the r_<orig> id
        gt_path = gt_dir / f"r_{orig}.png"
        if not gt_path.exists():
            out.append({"orig": orig, "psnr": None, "ssim": None, "note": "no_gt"})
            continue
        rend = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        # Composite GT against white background to match training (white_background=true).
        gt_rgba = np.asarray(Image.open(gt_path).convert("RGBA"), dtype=np.float32) / 255.0
        a = gt_rgba[..., 3:4]
        gt_white = gt_rgba[..., :3] * a + (1.0 - a) * 1.0  # white bg
        gt = gt_white
        if rend.shape != gt.shape:
            # downsample / crop mismatch; skip
            out.append({"orig": orig, "psnr": None, "ssim": None, "note": "shape_mismatch"})
            continue
        mse = float(np.mean((rend - gt) ** 2))
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim_v = None
        if ssim is not None:
            ssim_v = float(ssim(rend, gt, channel_axis=2, data_range=1.0))
        out.append({"orig": orig, "psnr": psnr, "ssim": ssim_v, "note": "ok"})
    return out


def analyze_scene(scene: str) -> Dict:
    info: Dict = {"scene": scene}

    # train view ids
    metrics_p = OUT / expname(scene) / "metrics.json"
    md = json.loads(metrics_p.read_text(encoding="utf-8")) if metrics_p.exists() else {}
    train_ids = md.get("train_view_ids", [])
    info["train_view_ids"] = train_ids
    info["test_psnr_global"] = (md.get("metrics") or {}).get("test/psnr")
    info["test_ssim_global"] = (md.get("metrics") or {}).get("test/ssim")
    info["test_lpips_global"] = (md.get("metrics") or {}).get("test/lpips")
    info["num_gaussians"] = md.get("num_gaussians")

    # train + test poses
    train_json = DATA / scene / "transforms_train.json"
    test_json = DATA / scene / "transforms_test.json"
    train_d = json.loads(train_json.read_text(encoding="utf-8"))
    test_d = json.loads(test_json.read_text(encoding="utf-8"))

    train_positions = np.array(
        [np.asarray(f["transform_matrix"])[:3, 3] for f in train_d["frames"]]
    )
    test_positions = np.array(
        [np.asarray(f["transform_matrix"])[:3, 3] for f in test_d["frames"]]
    )
    sel_train = train_positions[train_ids]

    # geodesic angles to nearest selected-train for each test view
    def nearest_angle_deg(p: np.ndarray) -> float:
        return min(angle_between(p, s) for s in sel_train) * 180.0 / math.pi

    nearest_deg = np.array([nearest_angle_deg(p) for p in test_positions])
    info["nearest_train_angle_deg"] = {
        "mean": float(nearest_deg.mean()),
        "median": float(np.median(nearest_deg)),
        "max": float(nearest_deg.max()),
        "p90": float(np.percentile(nearest_deg, 90)),
    }

    # also pairwise min angle between selected train views (= "spread")
    pairs = []
    for i in range(len(sel_train)):
        for j in range(i + 1, len(sel_train)):
            pairs.append(angle_between(sel_train[i], sel_train[j]) * 180.0 / math.pi)
    info["train_pair_angles_deg"] = {
        "min": float(min(pairs)),
        "median": float(np.median(pairs)),
        "max": float(max(pairs)),
    }

    # per-view PSNR
    pv = per_view_psnr_ssim(scene)
    valid = [x for x in pv if x["psnr"] is not None]
    if valid:
        psnrs = np.array([x["psnr"] for x in valid])
        info["per_view"] = {
            "n": len(valid),
            "psnr_mean": float(psnrs.mean()),
            "psnr_min": float(psnrs.min()),
            "psnr_max": float(psnrs.max()),
            "psnr_p10": float(np.percentile(psnrs, 10)),
            "psnr_p90": float(np.percentile(psnrs, 90)),
        }
        # worst-K and best-K test views
        sorted_idx = np.argsort(psnrs)
        worst = [valid[int(i)] for i in sorted_idx[:5]]
        best = [valid[int(i)] for i in sorted_idx[-5:]]
        # attach nearest-train-angle to each picked view
        for x in worst + best:
            test_pos = test_positions[x["orig"]]
            x["nearest_train_angle_deg"] = nearest_angle_deg(test_pos)
        info["worst5"] = worst
        info["best5"] = best
    else:
        info["per_view"] = None

    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    summary = []
    for s in args.scenes:
        print(f"-- diagnosing {s} ...", flush=True)
        try:
            info = analyze_scene(s)
        except Exception as e:
            print(f"   ERROR on {s}: {e}", file=sys.stderr)
            info = {"scene": s, "error": str(e)}
        out_p = DIAG / f"{s}.json"
        out_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"   wrote {out_p}", flush=True)
        summary.append(info)

    # one-line table to stdout
    print("\nscene      | global PSNR | per-view PSNR (mean/min/p10) | nearest-train (median/max) | train-pair (min/median)")
    for info in summary:
        if "error" in info:
            print(f"  {info['scene']:10s} ERROR {info['error']}")
            continue
        pv = info.get("per_view") or {}
        nt = info["nearest_train_angle_deg"]
        tp = info["train_pair_angles_deg"]
        print(
            f"  {info['scene']:10s} | {info['test_psnr_global']:.3f} | "
            f"{pv.get('psnr_mean', 0):.3f} / {pv.get('psnr_min', 0):.3f} / {pv.get('psnr_p10', 0):.3f} | "
            f"{nt['median']:.1f} / {nt['max']:.1f} | {tp['min']:.1f} / {tp['median']:.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
