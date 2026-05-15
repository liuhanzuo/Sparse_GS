"""Aggregate Blender avg-8 results into a single markdown report.

Reads ``outputs/blender_<scene>_n8/metrics.json`` and
``outputs/blender_<scene>_n8_ssl_mv_dav2s/metrics.json`` for the 8
NeRF-Synthetic scenes (lego is sourced from outputs/lego_n8 and
outputs/lego_n8_ssl_mv_dav2s; the other 7 are produced by
run_blender_n8_pipeline.py).

Writes::

    outputs/blender_n8_table.md     # human-readable table
    outputs/blender_n8_table.json   # machine-readable

Usage::

    python scripts/dump_blender_n8_table.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"

_SCENES = ["chair", "drums", "ficus", "hotdog",
           "lego", "materials", "mic", "ship"]


def _load(scene: str, variant: str) -> Optional[dict]:
    if scene == "lego":
        # legacy run names
        if variant == "baseline":
            run = "lego_n8"
        else:
            run = "lego_n8_ssl_mv_dav2s"
    else:
        if variant == "baseline":
            run = f"blender_{scene}_n8"
        else:
            run = f"blender_{scene}_n8_ssl_mv_dav2s"
    p = _OUT / run / "metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "run": run,
        "psnr": d["metrics"].get("test/psnr"),
        "ssim": d["metrics"].get("test/ssim"),
        "lpips": d["metrics"].get("test/lpips"),
        "n_gauss": d.get("num_gaussians"),
        "wall": d.get("wall_seconds") or d.get("wall"),
        "n_test_views": d.get("num_test_views_used"),
    }


def _avg(vals: List[Optional[float]]) -> Optional[float]:
    vs = [v for v in vals if v is not None]
    return st.mean(vs) if vs else None


def main() -> int:
    rows: List[Tuple[str, Dict, Dict]] = []
    for s in _SCENES:
        b = _load(s, "baseline")
        d = _load(s, "dav2s")
        rows.append((s, b or {}, d or {}))

    # ---- markdown ----
    lines: List[str] = []
    lines.append("# Blender avg-8 @ n=8 — full sweep (literature protocol)")
    lines.append("")
    lines.append("All numbers from full 200-view test split eval"
                 " (`eval.num_test_renders=200`).")
    lines.append("")
    lines.append("| scene | base PSNR | dav2s PSNR | ΔPSNR | base SSIM | dav2s SSIM"
                 " | ΔSSIM | base LPIPS | dav2s LPIPS | ΔLPIPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base_psnr, dav_psnr = [], []
    base_ssim, dav_ssim = [], []
    base_lpips, dav_lpips = [], []
    for s, b, d in rows:
        bp, dp = b.get("psnr"), d.get("psnr")
        bs, ds = b.get("ssim"), d.get("ssim")
        bl, dl = b.get("lpips"), d.get("lpips")
        base_psnr.append(bp); dav_psnr.append(dp)
        base_ssim.append(bs); dav_ssim.append(ds)
        base_lpips.append(bl); dav_lpips.append(dl)

        def fmt(x, w=6):
            return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else "—".rjust(w)

        def dx(a, b, sign="+"):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return f"{b - a:+.3f}"
            return "—"

        lines.append(
            f"| {s} | {fmt(bp)} | {fmt(dp)} | {dx(bp, dp)} "
            f"| {fmt(bs)} | {fmt(ds)} | {dx(bs, ds)} "
            f"| {fmt(bl)} | {fmt(dl)} | {dx(bl, dl)} |"
        )
    abp = _avg(base_psnr); adp = _avg(dav_psnr)
    abs_ = _avg(base_ssim); ads = _avg(dav_ssim)
    abl = _avg(base_lpips); adl = _avg(dav_lpips)

    def safe(x, fmt="{:.3f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "—"

    def safe_dx(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return f"{b - a:+.3f}"
        return "—"

    lines.append(
        f"| **avg-8** | **{safe(abp)}** | **{safe(adp)}** | "
        f"**{safe_dx(abp, adp)}** "
        f"| **{safe(abs_)}** | **{safe(ads)}** | **{safe_dx(abs_, ads)}** "
        f"| **{safe(abl)}** | **{safe(adl)}** | **{safe_dx(abl, adl)}** |"
    )

    lines.append("")
    lines.append("## SOTA gap (Blender avg-8, n=8 PSNR)")
    lines.append("")
    lines.append("| method | PSNR | gap to ours (dav2s) |")
    lines.append("|---|---:|---:|")
    sota = [
        ("FSGS (ECCV'24)",     24.6),
        ("CoR-GS (ECCV'24)",   24.5),
        ("DNGaussian (CVPR'24)", 24.3),
        ("FreeNeRF (CVPR'23)", 24.3),
        ("RegNeRF (CVPR'22)",  23.9),
        ("DietNeRF (ICCV'21)", 23.6),
        ("SparseGS (3DV'24)",  22.8),
    ]
    for m, p in sota:
        gap = (adp - p) if isinstance(adp, (int, float)) else None
        lines.append(f"| {m} | {p:.1f} | "
                     f"{(f'{gap:+.2f} dB' if gap is not None else '—')} |")
    lines.append(f"| **Ours (avg-8, dav2s + mv)** | **{safe(adp)}** | — |")

    out_md = _OUT / "blender_n8_table.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")

    # ---- json ----
    out_json = _OUT / "blender_n8_table.json"
    out_json.write_text(json.dumps({
        "scenes": _SCENES,
        "rows": [
            {"scene": s, "baseline": b, "dav2s": d}
            for s, b, d in rows
        ],
        "avg": {
            "psnr": {"baseline": abp, "dav2s": adp},
            "ssim": {"baseline": abs_, "dav2s": ads},
            "lpips": {"baseline": abl, "dav2s": adl},
        },
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
