"""Pull the full eval/prune trajectory from the aggrprune_long run and
print a side-by-side compact summary. Used to test the user's
hypothesis that the val/psnr is following a 25 -> dip -> 24 -> dip -> 26
ratchet, but the last prune happens too late to fully recover.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "outputs", "logs", "w3hotdog_long_aggr.log")

text = open(LOG, "r", encoding="utf-8", errors="replace").read()

print("=" * 76)
print("aggrprune_long — eval & prune trajectory")
print("=" * 76)

# eval checkpoints
eval_pat = re.compile(
    r"\[eval @ (\d+)\] val/psnr=([\d.]+) \| val/ssim=([\d.]+) \| val/lpips=([\d.]+)"
)
evals = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
         for m in eval_pat.finditer(text)]

# floater prune events (executed)
fpat = re.compile(r"\[w2-prune\] floater pruned N=(\d+) \(step=(\d+), remain=(\d+)\)")
floater_events = [(int(m.group(2)), int(m.group(1)), int(m.group(3))) for m in fpat.finditer(text)]

# floater prune skipped events
spat = re.compile(r"floater-prune over-aggressive \(N_candidate=(\d+), ratio=([\d.]+),[^)]*\); skipping at step=(\d+)")
skip_events = [(int(m.group(3)), int(m.group(1)), float(m.group(2))) for m in spat.finditer(text)]

# unseen prune events
upat = re.compile(r"\[w2-prune\] unseen pruned N=(\d+) \(step=(\d+), remain=(\d+)\)")
unseen_events = [(int(m.group(2)), int(m.group(1)), int(m.group(3))) for m in upat.finditer(text)]

# test
tpat = re.compile(r"\[test\] test/psnr=([\d.]+) \| test/ssim=([\d.]+) \| test/lpips=([\d.]+)[^#]*#G=(\d+)")
tmatch = tpat.search(text)

print("\n--- eval checkpoints ---")
print(f"{'step':>6}  {'val_psnr':>9}  {'val_ssim':>9}  {'val_lpips':>9}")
prev = None
for step, p, s, lp in evals:
    delta = "" if prev is None else f"  Δ={p-prev:+.3f}"
    print(f"{step:>6d}  {p:>9.3f}  {s:>9.4f}  {lp:>9.4f}{delta}")
    prev = p

print("\n--- floater-prune events (executed) ---")
print(f"{'step':>6}  {'pruned':>10}  {'remain':>10}")
for step, n, r in floater_events:
    print(f"{step:>6d}  {n:>10d}  {r:>10d}")

print("\n--- floater-prune events (skipped, over safety_max_ratio=0.25) ---")
print(f"{'step':>6}  {'cand':>10}  {'ratio':>7}")
for step, c, r in skip_events:
    print(f"{step:>6d}  {c:>10d}  {r:>7.3f}")

print("\n--- unseen-prune events ---")
print(f"{'step':>6}  {'pruned':>10}  {'remain':>10}")
for step, n, r in unseen_events:
    print(f"{step:>6d}  {n:>10d}  {r:>10d}")

if tmatch:
    print("\n--- final test (15000) ---")
    print(f"  PSNR={float(tmatch.group(1)):.3f}  SSIM={float(tmatch.group(2)):.4f}  "
          f"LPIPS={float(tmatch.group(3)):.4f}  #G={int(tmatch.group(4))}")
