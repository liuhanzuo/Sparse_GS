"""Quick aggregate of W3 dual_gs vs aggrprune across the four scenes
where aggrprune was actually run (drums, materials, hotdog, ship).

Plus pulls out the [eval @ N] checkpoints from the training log to show
the dip-then-recover pattern user is asking about.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ["drums", "materials", "hotdog", "ship"]


def load_metrics(name: str):
    p = os.path.join(ROOT, "outputs", name, "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, "r", encoding="utf-8"))["metrics"]
    except Exception:
        return None


def parse_eval_curve(log_path: str):
    """Extract [eval @ STEP] val/psnr from a training log."""
    if not os.path.exists(log_path):
        return []
    out = []
    pat = re.compile(r"\[eval @ (\d+)\] val/psnr=([\d.]+)")
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    for m in pat.finditer(text):
        out.append((int(m.group(1)), float(m.group(2))))
    return out


print("=" * 72)
print(f"{'scene':10s}  {'dual_gs':>10s}  {'aggrprune':>10s}  {'delta':>8s}   LPIPS dual->aggr")
print("-" * 72)
for s in SCENES:
    base = load_metrics(f"blender_{s}_n8_w3_dual_gs")
    aggr = load_metrics(f"blender_{s}_n8_w3_aggrprune")
    if not base or not aggr:
        print(f"{s:10s}  missing")
        continue
    bp = base.get("test/psnr")
    ap = aggr.get("test/psnr")
    bl = base.get("test/lpips", float("nan"))
    al = aggr.get("test/lpips", float("nan"))
    print(
        f"{s:10s}  {bp:>10.3f}  {ap:>10.3f}  {ap - bp:>+8.3f}   "
        f"{bl:.3f} -> {al:.3f}"
    )
print("=" * 72)

print("\n--- val/psnr trajectory inside aggrprune training (dip-then-recover) ---")
for s in SCENES:
    log = os.path.join(ROOT, "outputs", "logs", f"w3aggr_{s}.log")
    curve = parse_eval_curve(log)
    if not curve:
        print(f"  {s:10s}  no eval ckpts in log")
        continue
    parts = "  ".join(f"@{step}={psnr:.2f}" for step, psnr in curve)
    print(f"  {s:10s}  {parts}")
