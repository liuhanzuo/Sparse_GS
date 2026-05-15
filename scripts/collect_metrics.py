"""Walk outputs/ and collect every run's metrics.json into one table.

Writes:
    outputs/_summary.md     (Markdown table, ready to paste)
    outputs/_summary.csv    (machine-readable)
    outputs/_summary.json   (raw payloads)

Usage:
    python scripts/collect_metrics.py
    python scripts/collect_metrics.py --root outputs/
"""
from __future__ import annotations

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Pick whichever metrics file has the largest test-view coverage.

    During the lifecycle of a run we may produce both:
      * ``metrics.json``      — written by the trainer at end of fit().
      * ``metrics_full.json`` — written by ``scripts/eval_ckpt.py`` whenever
        ``metrics.json`` already existed.
    The two may have been written with *different* ``--max-views``, so we
    can't blindly pick one. We pick the file with the largest
    ``num_test_views_used`` (ties: prefer ``metrics_full.json`` since that
    was written by the dedicated full-eval entry point).
    """
    candidates: List[Dict[str, Any]] = []
    for name in ("metrics.json", "metrics_full.json"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["_source"] = name
            payload["_dir"] = run_dir.name
            payload["_n_views_used"] = int(payload.get("num_test_views_used", 0) or 0)
            candidates.append(payload)
        except Exception as e:
            print(f"[warn] failed to parse {p}: {e}")
    if not candidates:
        return None
    # max coverage; ties broken by metrics_full.json
    candidates.sort(key=lambda p: (p["_n_views_used"], p["_source"] == "metrics_full.json"))
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs",
                    help="Root directory of runs.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    payloads: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _load(run_dir)
        if payload is None:
            continue
        payloads.append(payload)

    if not payloads:
        print(f"[collect_metrics] no metrics found under {root}")
        return

    # Sort by (scene, n_train_views, experiment name).
    def sort_key(p: Dict[str, Any]):
        return (
            str(p.get("scene", "")),
            int(p.get("n_train_views", 0) or 0),
            str(p.get("experiment", p.get("_dir", ""))),
        )
    payloads.sort(key=sort_key)

    # ---- Markdown ----
    rows = []
    for p in payloads:
        m = p.get("metrics", {})
        rows.append({
            "experiment": p.get("experiment") or p.get("_dir"),
            "scene": p.get("scene", "?"),
            "type": p.get("type", ""),
            "n": int(p.get("n_train_views", 0) or 0),
            "psnr": m.get("test/psnr"),
            "ssim": m.get("test/ssim"),
            "lpips": m.get("test/lpips"),
            "n_g": p.get("num_gaussians"),
            "n_test_used": p.get("num_test_views_used"),
            "n_test_total": p.get("num_test_views_total"),
            "src": p.get("_source"),
        })

    md_lines = [
        "# Run summary",
        "",
        "Auto-collected by `scripts/collect_metrics.py` from `metrics.json` / `metrics_full.json` per run.",
        "",
        "| experiment | scene | type | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #G | test-views | src |",
        "|---|---|---|:-:|---:|---:|---:|---:|:-:|:-:|",
    ]
    for r in rows:
        psnr_s = f"{r['psnr']:.4f}" if r['psnr'] is not None else "—"
        ssim_s = f"{r['ssim']:.4f}" if r['ssim'] is not None else "—"
        lpips_s = f"{r['lpips']:.4f}" if r['lpips'] is not None else "—"
        ng_s = f"{r['n_g']:,}" if r['n_g'] is not None else "—"
        tv = f"{r['n_test_used']}/{r['n_test_total']}" \
            if r['n_test_used'] is not None and r['n_test_total'] is not None else "—"
        md_lines.append(
            f"| `{r['experiment']}` | {r['scene']} | {r['type']} | {r['n']} | "
            f"{psnr_s} | {ssim_s} | {lpips_s} | {ng_s} | {tv} | {r['src']} |"
        )
    md_lines.append("")

    md_path = root / "_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[collect_metrics] wrote {md_path}")

    # ---- CSV ----
    csv_path = root / "_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[collect_metrics] wrote {csv_path}")

    # ---- JSON (raw) ----
    json_path = root / "_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2)
    print(f"[collect_metrics] wrote {json_path}")

    # ---- Console preview ----
    print()
    for line in md_lines[:60]:
        print(line)
    if len(md_lines) > 60:
        print(f"... (+{len(md_lines) - 60} more lines in _summary.md)")


if __name__ == "__main__":
    main()
