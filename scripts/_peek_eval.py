"""扫描指定 eval_log.jsonl，按 val/psnr 倒序打印前 N 行 + final test。"""
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
val_rows = [r for r in rows if "val/psnr" in r]
val_rows.sort(key=lambda r: -r["val/psnr"])
print(f"=== {p} ===")
print(f"total_eval_records = {len(rows)} (with val/psnr: {len(val_rows)})")
print(f"\n--- Top-10 val/psnr peaks ---")
for r in val_rows[:10]:
    print(f"step={r['step']:>5d}  phase={r['phase']:<24s}  "
          f"psnr={r['val/psnr']:.3f}  ssim={r.get('val/ssim',0):.4f}  N={r.get('num_gaussians',0)}")
print(f"\n--- Final test ---")
for r in rows:
    if r.get("phase") == "test":
        print(json.dumps(r, indent=2, ensure_ascii=False))
