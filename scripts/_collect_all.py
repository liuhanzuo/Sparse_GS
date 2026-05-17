import json, glob, os
ps = sorted(glob.glob(r"F:\Sparse_GS\outputs\*\metrics.json"))
rows = []
for p in ps:
    try:
        j = json.load(open(p, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {p}: {e}")
        continue
    m = j.get("metrics", {})
    rows.append({
        "name": os.path.basename(os.path.dirname(p)),
        "psnr": m.get("test/psnr", 0.0),
        "ssim": m.get("test/ssim", 0.0),
        "lpips": m.get("test/lpips", 0.0),
        "N": j.get("num_gaussians", 0),
        "iters": j.get("iterations", 0),
        "wall_h": j.get("wall_clock_sec", 0) / 3600.0,
    })
print(f"{'name':<60s} | {'PSNR':>6} | {'SSIM':>6} | {'LPIPS':>6} | {'N':>9} | {'iters':>6} | {'wall':>7}")
print("-" * 120)
for r in rows:
    print(f"{r['name']:<60s} | {r['psnr']:6.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['N']:>9d} | {r['iters']:>6d} | {r['wall_h']:5.2f}h")
