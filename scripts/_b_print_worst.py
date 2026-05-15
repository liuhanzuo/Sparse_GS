"""Print worst5 / best5 per anomaly scene from B-diag JSONs."""
import json
import pathlib
for s in ["hotdog", "drums", "materials", "ship"]:
    d = json.loads(pathlib.Path(f"outputs/_b_diag/{s}.json").read_text(encoding="utf-8"))
    print(f"=== {s} (global PSNR {d['test_psnr_global']:.3f}) ===")
    def _fmt(x):
        ssim_s = f"{x['ssim']:.4f}" if x.get('ssim') is not None else "  -  "
        return f"r_{x['orig']:3d}: PSNR={x['psnr']:.3f} SSIM={ssim_s} nearest_train_angle={x['nearest_train_angle_deg']:.1f}deg"
    print("  worst5:")
    for x in d["worst5"]:
        print("    " + _fmt(x))
    print("  best5:")
    for x in d["best5"]:
        print("    " + _fmt(x))
