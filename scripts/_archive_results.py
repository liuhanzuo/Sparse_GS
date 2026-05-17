"""Copy the small 'metrics.json + eval_log.jsonl + config.yaml' triplet
from each outputs/<run>/ into results/experiments/<run>/ so the numbers
in README §1bis are reproducible from the committed tree.

Heavy artefacts (*.pt checkpoints, renders/) are NOT copied — they live
only in outputs/ which is gitignored.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs"
DST = ROOT / "results" / "experiments"
KEEP = ("metrics.json", "eval_log.jsonl", "config.yaml")

DST.mkdir(parents=True, exist_ok=True)
copied = 0
skipped = 0
for run in sorted(SRC.iterdir()):
    if not run.is_dir():
        continue
    m = run / "metrics.json"
    if not m.exists():
        skipped += 1
        continue
    out = DST / run.name
    out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for fname in KEEP:
        s = run / fname
        if s.exists():
            shutil.copy2(s, out / fname)
            n_files += 1
    print(f"  {run.name:<60s}  +{n_files} files")
    copied += 1

print(f"\nDone. {copied} runs archived, {skipped} skipped (no metrics.json).")
print(f"Target dir: {DST}")
