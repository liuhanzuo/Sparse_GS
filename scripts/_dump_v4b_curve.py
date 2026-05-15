"""ad-hoc：打印 v4b 两个 run 的 val 曲线，对比 peak vs final。"""
import json, sys, os

PATHS = [
    "outputs/blender_drums_n8_w3_aggrprune_long_v4_pvd/eval_log.jsonl",
    "outputs/blender_drums_n8_w3_aggrprune_long_v4b_pvd/eval_log.jsonl",
    "outputs/blender_hotdog_n8_w3_aggrprune_long_v4_pvd/eval_log.jsonl",
    "outputs/blender_hotdog_n8_w3_aggrprune_long_v4b_pvd/eval_log.jsonl",
]

for p in PATHS:
    if not os.path.exists(p):
        print(f"[skip] {p} not found")
        continue
    print("===", p)
    rows = [json.loads(l) for l in open(p, "r", encoding="utf-8")]
    best = None
    for r in rows:
        psnr = r.get("val/psnr")
        if psnr is None:
            continue
        print(f"  step={r['step']:>5}  phase={r['phase']:<22}  N={r['num_gaussians']:>7}  val={psnr:.3f}  ssim={r.get('val/ssim',float('nan')):.4f}")
        if best is None or psnr > best[1]:
            best = (r["step"], psnr, r["phase"])
    final = next((r for r in reversed(rows) if r.get("val/psnr") is not None), None)
    if best and final:
        print(f"  >> peak  val: step={best[0]} psnr={best[1]:.3f} ({best[2]})")
        print(f"  >> final val: step={final['step']} psnr={final['val/psnr']:.3f} ({final['phase']})")
        print(f"  >> peak - final = {best[1]-final['val/psnr']:+.3f} dB")
