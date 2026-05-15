"""Print a human-readable trajectory of one (or several) training runs.

Reads the ``eval_log.jsonl`` files produced by ``Trainer._log_eval`` and
emits a table with one row per eval, plus a summary line:

    === blender_hotdog_n8_w3_aggrprune_long_v2 ===
    log: outputs/.../eval_log.jsonl  (43 entries)
       step | phase                 |   psnr |   ssim | lpips |       N | extra
       2000 | regular               | 25.020 | 0.9322 | 0.126 |  244000 |
       4000 | pre_floater_prune     | 26.103 | 0.9402 | 0.118 |  282369 | will=49546 tb=0.050
       4000 | post_floater_prune    | 19.339 | 0.9043 | 0.155 |  232823 | n=49546 tb=0.050
       4001 | post_floater_offset   | 19.450 | ...    | ...   |  ...    | anchor=4000 off=1
       ...
    peak val_psnr = 26.103 @ step 4000 (pre_floater_prune)
    final test    = 24.07

Usage
-----
    python scripts/log_summary.py outputs/blender_hotdog_n8_w3_aggrprune_long_v2
    python scripts/log_summary.py outputs/blender_*_w3_aggrprune_long*

If the argument is a directory we look for ``<dir>/eval_log.jsonl``; if it
is a glob pattern we expand it; if it is a file we read it directly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List


def _resolve(arg: str) -> List[Path]:
    # Glob first (so ``outputs/foo*`` works on PowerShell where the shell
    # does not expand globs by itself).
    matches = glob.glob(arg)
    if not matches and os.path.exists(arg):
        matches = [arg]
    out: List[Path] = []
    for m in matches:
        p = Path(m)
        if p.is_dir():
            cand = p / "eval_log.jsonl"
            if cand.exists():
                out.append(cand)
        elif p.is_file():
            out.append(p)
    return out


def _fmt_extras(rec: dict) -> str:
    keep = []
    for k in ("will_prune_n", "pruned_n", "thresh_bin", "anchor_step", "offset",
              "wall_clock_sec"):
        if k in rec:
            v = rec[k]
            if isinstance(v, float):
                if k == "thresh_bin":
                    keep.append(f"tb={v:.4f}")
                elif k == "wall_clock_sec":
                    keep.append(f"{v:.1f}s")
                else:
                    keep.append(f"{k}={v:.3f}")
            else:
                short = {
                    "will_prune_n": "will",
                    "pruned_n": "n",
                    "anchor_step": "anchor",
                    "offset": "off",
                }.get(k, k)
                keep.append(f"{short}={v}")
    return " ".join(keep)


def summarize_one(jsonl_path: Path) -> None:
    title = jsonl_path.parent.name
    print(f"\n=== {title} ===")
    print(f"log: {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skip malformed line: {e}")
    print(f"entries: {len(records)}")
    if not records:
        return

    print(f"  {'step':>6} | {'phase':<22} | {'psnr':>6} | {'ssim':>6} | {'lpips':>5} | {'N':>8} | extra")
    for r in records:
        step = r.get("step", -1)
        phase = r.get("phase", "?")
        psnr_ = r.get("val/psnr", r.get("test/psnr", float("nan")))
        ssim_ = r.get("val/ssim", r.get("test/ssim", float("nan")))
        lpips_ = r.get("val/lpips", r.get("test/lpips", float("nan")))
        N = r.get("num_gaussians", 0)
        extra = _fmt_extras(r)
        try:
            print(f"  {step:>6} | {phase:<22} | "
                  f"{psnr_:>6.3f} | {ssim_:>6.4f} | {lpips_:>5.3f} | "
                  f"{N:>8} | {extra}")
        except (TypeError, ValueError):
            print(f"  {step:>6} | {phase:<22} | {psnr_} | {ssim_} | {lpips_} | {N} | {extra}")

    # Summary: best val/psnr (any phase except 'test'), and the final test.
    val_records = [r for r in records if r.get("phase") != "test"
                   and isinstance(r.get("val/psnr"), (int, float))]
    if val_records:
        best = max(val_records, key=lambda r: r["val/psnr"])
        print(f"  peak val/psnr = {best['val/psnr']:.3f} @ step {best['step']} "
              f"({best['phase']}, N={best['num_gaussians']})")

    test_recs = [r for r in records if r.get("phase") == "test"]
    if test_recs:
        t = test_recs[-1]
        print(f"  final test = psnr {t.get('test/psnr', float('nan')):.3f} | "
              f"ssim {t.get('test/ssim', float('nan')):.4f} | "
              f"lpips {t.get('test/lpips', float('nan')):.3f} | "
              f"N={t.get('num_gaussians', 0)}")

    # Pre/post prune diff table -- the headline number we actually care
    # about for evaluating the dual-prune strategy.
    pre = {r["step"]: r for r in records if r.get("phase") in ("pre_floater_prune", "pre_unseen_prune")}
    post = {r["step"]: r for r in records if r.get("phase") in ("post_floater_prune", "post_unseen_prune")}
    common = sorted(set(pre.keys()) & set(post.keys()))
    if common:
        print("  prune-diff (pre -> post):")
        for s in common:
            a = pre[s]; b = post[s]
            kind = a.get("phase").replace("pre_", "")
            dpsnr = b.get("val/psnr", float("nan")) - a.get("val/psnr", float("nan"))
            print(f"    step {s:>5} {kind:<14}  "
                  f"psnr {a.get('val/psnr', float('nan')):.3f} -> "
                  f"{b.get('val/psnr', float('nan')):.3f} (Δ={dpsnr:+.3f})  "
                  f"N {a.get('num_gaussians', 0)} -> {b.get('num_gaussians', 0)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="run dir(s), jsonl file(s), or glob(s)")
    args = ap.parse_args()

    any_found = False
    for arg in args.paths:
        files = _resolve(arg)
        if not files:
            print(f"[warn] no eval_log.jsonl matches: {arg}")
            continue
        for f in files:
            any_found = True
            summarize_one(f)
    return 0 if any_found else 1


if __name__ == "__main__":
    sys.exit(main())
