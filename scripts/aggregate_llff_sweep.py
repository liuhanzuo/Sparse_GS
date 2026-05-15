"""Aggregate LLFF sweep metrics and print a clean markdown table.

Reads outputs/llff_*_n3_{baseline,dav2s}/metrics.json and reports:
- per-scene PSNR / SSIM / LPIPS for baseline and dav2s
- delta (dav2s - baseline) per metric
- 8-scene averages
"""
from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SCENES = ["fern", "flower", "fortress", "horns",
          "leaves", "orchids", "room", "trex"]


def load(run: str) -> dict | None:
    p = OUT / run / "metrics.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    m = d.get("metrics", {})
    return {
        "psnr": m.get("test/psnr"),
        "ssim": m.get("test/ssim"),
        "lpips": m.get("test/lpips"),
        "n_test": d.get("num_test_views_used"),
        "sec": d.get("wall_clock_sec"),
        "ngs": d.get("num_gaussians"),
        "train_ids": d.get("train_view_ids"),
    }


def fmt(v, nd=3):
    if v is None:
        return "  --  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    rows = []
    for s in SCENES:
        b = load(f"llff_{s}_n3_baseline")
        d = load(f"llff_{s}_n3_dav2s")
        rows.append((s, b, d))

    header = (
        "| scene    | base PSNR | dav2s PSNR | ΔPSNR  "
        "| base SSIM | dav2s SSIM | ΔSSIM  "
        "| base LPIPS | dav2s LPIPS | ΔLPIPS |"
    )
    sep = (
        "|----------|----------:|-----------:|-------:"
        "|----------:|-----------:|-------:"
        "|-----------:|------------:|-------:|"
    )
    print(header)
    print(sep)

    # Averages over scenes that have both runs complete.
    psnr_b, psnr_d = [], []
    ssim_b, ssim_d = [], []
    lpips_b, lpips_d = [], []

    for s, b, d in rows:
        if b is None and d is None:
            print(f"| {s:<8} | (no runs) | | | | | | | | |")
            continue
        if b is None:
            b = {"psnr": None, "ssim": None, "lpips": None}
        if d is None:
            d = {"psnr": None, "ssim": None, "lpips": None}

        def delta(a, c, better="higher"):
            if a is None or c is None:
                return None
            return c - a  # for PSNR/SSIM, higher is better; for LPIPS lower is better

        dp = delta(b["psnr"], d["psnr"])
        ds = delta(b["ssim"], d["ssim"])
        dl = delta(b["lpips"], d["lpips"])

        if b["psnr"] is not None and d["psnr"] is not None:
            psnr_b.append(b["psnr"]); psnr_d.append(d["psnr"])
        if b["ssim"] is not None and d["ssim"] is not None:
            ssim_b.append(b["ssim"]); ssim_d.append(d["ssim"])
        if b["lpips"] is not None and d["lpips"] is not None:
            lpips_b.append(b["lpips"]); lpips_d.append(d["lpips"])

        print(
            f"| {s:<8} | "
            f"{fmt(b['psnr'], 2):>9} | {fmt(d['psnr'], 2):>10} | {fmt(dp, 2):>6} | "
            f"{fmt(b['ssim'], 4):>9} | {fmt(d['ssim'], 4):>10} | {fmt(ds, 4):>6} | "
            f"{fmt(b['lpips'], 4):>10} | {fmt(d['lpips'], 4):>11} | {fmt(dl, 4):>6} |"
        )

    # Averages row
    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    print("|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|")
    print(
        f"| **avg**  | "
        f"{fmt(avg(psnr_b), 2):>9} | {fmt(avg(psnr_d), 2):>10} | "
        f"{fmt((avg(psnr_d) - avg(psnr_b)) if psnr_b and psnr_d else None, 2):>6} | "
        f"{fmt(avg(ssim_b), 4):>9} | {fmt(avg(ssim_d), 4):>10} | "
        f"{fmt((avg(ssim_d) - avg(ssim_b)) if ssim_b and ssim_d else None, 4):>6} | "
        f"{fmt(avg(lpips_b), 4):>10} | {fmt(avg(lpips_d), 4):>11} | "
        f"{fmt((avg(lpips_d) - avg(lpips_b)) if lpips_b and lpips_d else None, 4):>6} |"
    )
    print(f"\n(averages computed over {len(psnr_b)}/{len(SCENES)} scenes with both runs complete)")


if __name__ == "__main__":
    main()
