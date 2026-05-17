import json, glob, os
patterns = [
    r"F:\Sparse_GS\outputs\ablation_*\metrics.json",
    r"F:\Sparse_GS\outputs\blender_*v8_hard*\metrics.json",
    r"F:\Sparse_GS\outputs\blender_ship*v7*x45k*\metrics.json",
]
seen = set()
files = []
for pat in patterns:
    for p in sorted(glob.glob(pat)):
        if p in seen:
            continue
        seen.add(p)
        files.append(p)
for p in files:
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    m = j.get("metrics", {})
    name = os.path.basename(os.path.dirname(p))
    print(
        f"{name:<55s} PSNR={m.get('test/psnr',0):.3f}  "
        f"SSIM={m.get('test/ssim',0):.4f}  LPIPS={m.get('test/lpips',0):.4f}  "
        f"N={j.get('num_gaussians',0):>8d}  iters={j.get('iterations',0):>5d}  "
        f"wall={j.get('wall_clock_sec',0)/3600:.2f}h"
    )
