"""List all LLFF run metrics for sweep progress check."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent.parent / "outputs"
rows = []
for m in sorted(root.glob("llff_*/metrics.json")):
    try:
        d = json.loads(m.read_text())
        rows.append((m.parent.name, d.get("psnr"), d.get("ssim"), d.get("lpips")))
    except Exception as e:
        rows.append((m.parent.name, f"ERR:{e}", None, None))

print(f"{'run':<36} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
for name, psnr, ssim, lpips in rows:
    p = f"{psnr:.3f}" if isinstance(psnr, float) else str(psnr)
    s = f"{ssim:.4f}" if isinstance(ssim, float) else "-"
    l = f"{lpips:.4f}" if isinstance(lpips, float) else "-"
    print(f"{name:<36} {p:>7} {s:>7} {l:>7}")

# List what is missing from the standard 8-scene x 2-variant grid
scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
variants = ["baseline", "dav2s"]
done = {r[0] for r in rows}
missing = [f"llff_{s}_n3_{v}" for s in scenes for v in variants if f"llff_{s}_n3_{v}" not in done]
print(f"\nMissing ({len(missing)}):")
for m in missing:
    print(f"  {m}")
