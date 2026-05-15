"""One-shot dump: collect every metrics*.json under outputs/ for the
blender 8-view runs and print a comparison table grouped by scene."""
import glob, json, os, re

ROOT = "outputs"
ROWS = []  # (scene, method, psnr, ssim, lpips, N, src)

PAT = re.compile(r"^blender_(\w+?)_n8(?:_(.+))?$")

for d in sorted(os.listdir(ROOT)):
    full = os.path.join(ROOT, d)
    if not os.path.isdir(full):
        continue
    m = PAT.match(d)
    if not m:
        continue
    scene = m.group(1)
    method = m.group(2) or "vanilla"
    # prefer metrics_full.json, then metrics.json, then last test entry of eval_log.jsonl
    cand = None
    for nm in ("metrics_full.json", "metrics.json"):
        p = os.path.join(full, nm)
        if os.path.exists(p):
            cand = p; break
    if cand is not None:
        try:
            with open(cand, "r", encoding="utf-8") as f:
                D = json.load(f)
            M = D.get("metrics", {})  # nested in our metrics.json schema
            psnr = M.get("test/psnr", D.get("psnr", D.get("test/psnr")))
            ssim = M.get("test/ssim", D.get("ssim", D.get("test/ssim")))
            lpips = M.get("test/lpips", D.get("lpips", D.get("test/lpips")))
            N = D.get("num_gaussians", D.get("N", 0))
            ROWS.append((scene, method, psnr, ssim, lpips, N, os.path.basename(cand)))
            continue
        except Exception:
            pass
    # fallback: read last "test" line in eval_log.jsonl
    p = os.path.join(full, "eval_log.jsonl")
    if os.path.exists(p):
        last = None
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("phase") == "test":
                    last = r
        if last is not None:
            ROWS.append((scene, method,
                         last.get("test/psnr"),
                         last.get("test/ssim"),
                         last.get("test/lpips"),
                         last.get("num_gaussians", 0),
                         "eval_log.jsonl"))

# Print grouped by scene
ROWS.sort(key=lambda r: (r[0], r[1]))
last_scene = None
print(f"{'scene':<10} {'method':<45} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'#G':>9}  src")
print("-" * 100)
for scene, method, psnr, ssim, lpips, N, src in ROWS:
    if scene != last_scene:
        print()
        last_scene = scene
    def f(x, w, p): return f"{x:>{w}.{p}f}" if isinstance(x, (int,float)) else f"{'-':>{w}}"
    print(f"{scene:<10} {method:<45} {f(psnr,7,3)} {f(ssim,7,4)} {f(lpips,7,3)} {N:>9}  {src}")
