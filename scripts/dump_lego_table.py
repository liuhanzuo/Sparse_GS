"""Dump a tidy lego PSNR/SSIM/LPIPS table from metrics_full.json files.

Prefers ``metrics_full.json`` (written by eval_ckpt on the full 200-view
split). Falls back to ``metrics.json`` if the full eval is missing so we
don't silently drop a row.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def row_for(d: Path):
    mfull = d / "metrics_full.json"
    m = d / "metrics.json"
    src = "full" if mfull.exists() else ("train" if m.exists() else None)
    if src is None:
        return None
    with open(mfull if src == "full" else m, encoding="utf-8") as f:
        j = json.load(f)
    mt = j.get("metrics", {})
    return dict(
        name=d.name,
        n=int(j.get("n_train_views", 0)),
        psnr=float(mt.get("test/psnr", float("nan"))),
        ssim=float(mt.get("test/ssim", float("nan"))),
        lpips=float(mt.get("test/lpips", float("nan"))),
        ng=int(j.get("num_gaussians", 0)),
        nv=int(j.get("num_test_views_used", 0)),
        src=src,
    )


def main() -> None:
    rows = []
    for d in sorted(OUT.glob("lego_n*")):
        if not d.is_dir():
            continue
        r = row_for(d)
        if r:
            rows.append(r)

    # group by n, sort by psnr descending within group
    rows.sort(key=lambda r: (r["n"], -r["psnr"]))

    print(f"{'experiment':<36} {'n':>3} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} "
          f"{'#G':>7} {'nV':>4} {'src':>5}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<36} {r['n']:>3} {r['psnr']:>8.4f} {r['ssim']:>7.4f} "
              f"{r['lpips']:>7.4f} {r['ng']:>7d} {r['nv']:>4d} {r['src']:>5}")

    # also write markdown
    md = ROOT / "outputs" / "lego_curve.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# lego — full 200-view eval\n\n")
        f.write("All numbers below are computed on the full 200-view test\n")
        f.write("split of the NeRF-Synthetic `lego` scene (re-eval via\n")
        f.write("`scripts/eval_ckpt_lego_all.py`). PSNR/SSIM/LPIPS use the\n")
        f.write("LPIPS-VGG metric as in the literature.\n\n")
        f.write("| experiment | n | PSNR | SSIM | LPIPS | #G | src |\n")
        f.write("|---|---:|---:|---:|---:|---:|:---:|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['n']} | {r['psnr']:.2f} | "
                    f"{r['ssim']:.3f} | {r['lpips']:.3f} | "
                    f"{r['ng']:,} | {r['src']} |\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
