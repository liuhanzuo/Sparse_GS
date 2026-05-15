"""Aggregate W3 dual-GS results vs W2-sotaview baseline.

Reads 16 metrics.json files (8 scenes × 2 setups: w2_sotaview / w3_dual_gs)
and writes a comparison table to ``outputs/w3_dual_gs_results.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def expname(scene: str, suffix: str) -> str:
    if scene == "lego":
        return f"lego_n8_{suffix}"
    return f"blender_{scene}_n8_{suffix}"


def load(scene: str, suffix: str):
    p = OUT / expname(scene, suffix) / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics") or {}
    return {
        "psnr": float(m.get("test/psnr", 0)),
        "ssim": float(m.get("test/ssim", 0)),
        "lpips": float(m.get("test/lpips", 0)),
        "n_gauss": int(d.get("num_gaussians", 0)),
    }


def main() -> int:
    rows = []
    for s in SCENES:
        b = load(s, "w2_sotaview")     # baseline (W2-sotaview)
        v = load(s, "w3_dual_gs")      # dual-GS
        rows.append((s, b, v))

    def avg(idx, key):
        vals = [r[idx][key] for r in rows if r[idx] is not None]
        return sum(vals) / len(vals) if vals else None

    avg_b_psnr  = avg(1, "psnr");  avg_v_psnr  = avg(2, "psnr")
    avg_b_ssim  = avg(1, "ssim");  avg_v_ssim  = avg(2, "ssim")
    avg_b_lpips = avg(1, "lpips"); avg_v_lpips = avg(2, "lpips")

    n_done = sum(1 for _, _, v in rows if v is not None)

    lines = []
    lines.append("# W3 dual-GS 全量对比（CoR-GS co-reg + co-prune）")
    lines.append("")
    lines.append(f"> **进度**：{n_done}/8 场景已完成。"
                 f"配置：所有场景在 W2-sotaview baseline 之上仅启用 `dual_gs.enabled=true`，超参一致 "
                 f"(seed_gs1=43, coprune_threshold=0.10, coprune_every=500, start_sample_pseudo=500)。")
    lines.append("")
    lines.append("## 8 场景对比")
    lines.append("")
    lines.append("| 场景 | W2-sotaview PSNR | **W3 dual** PSNR | Δ PSNR | W2-sota LPIPS | **W3 dual** LPIPS | Δ LPIPS |")
    lines.append("|---|---|---|---|---|---|---|")
    for s, b, v in rows:
        if b is None:
            lines.append(f"| {s} | – | – | – | – | – | – |"); continue
        if v is None:
            lines.append(f"| {s} | {b['psnr']:.3f} | _running_ | – | {b['lpips']:.4f} | – | – |"); continue
        d_psnr = v["psnr"] - b["psnr"]
        d_lpips = v["lpips"] - b["lpips"]
        d_str = f"**{d_psnr:+.3f}**" if abs(d_psnr) >= 0.5 else f"{d_psnr:+.3f}"
        dl_str = f"**{d_lpips:+.4f}**" if abs(d_lpips) >= 0.01 else f"{d_lpips:+.4f}"
        lines.append(
            f"| {s} | {b['psnr']:.3f} | **{v['psnr']:.3f}** | {d_str} | "
            f"{b['lpips']:.4f} | **{v['lpips']:.4f}** | {dl_str} |"
        )
    if avg_b_psnr is not None and avg_v_psnr is not None:
        d_psnr_avg = avg_v_psnr - avg_b_psnr
        d_lpips_avg = avg_v_lpips - avg_b_lpips
        lines.append(
            f"| **avg** | **{avg_b_psnr:.3f}** | **{avg_v_psnr:.3f}** | "
            f"**{d_psnr_avg:+.3f}** | {avg_b_lpips:.4f} | {avg_v_lpips:.4f} | "
            f"**{d_lpips_avg:+.4f}** |"
        )
    lines.append("")

    if n_done == 8:
        lines.append("## 总体观察")
        lines.append("")
        ups = [(s, v["psnr"] - b["psnr"]) for s, b, v in rows if b is not None and v is not None]
        ups_sorted = sorted(ups, key=lambda x: -x[1])
        lines.append("- **PSNR 增量排序**：" + ", ".join(f"{s}({d:+.2f})" for s, d in ups_sorted))
        lines.append(f"- **8-scene 平均 PSNR**：{avg_b_psnr:.3f} → {avg_v_psnr:.3f} (Δ={avg_v_psnr-avg_b_psnr:+.3f})")
        lines.append(f"- **8-scene 平均 LPIPS**：{avg_b_lpips:.4f} → {avg_v_lpips:.4f} (Δ={avg_v_lpips-avg_b_lpips:+.4f})")
        lines.append(f"- **8-scene 平均 SSIM**：{avg_b_ssim:.4f} → {avg_v_ssim:.4f}")
        lines.append("")

        lines.append("## 跟 SOTA 论文（NeRF-Synthetic n=8 avg, 粗略）")
        lines.append("")
        lines.append("| 阶段 | PSNR | SSIM | LPIPS |")
        lines.append("|---|---|---|---|")
        lines.append("| W1 baseline (uniform) | 19.626 | 0.8325 | 0.2099 |")
        lines.append("| W2 (uniform) | 19.679 | 0.8362 | 0.1968 |")
        lines.append("| W2-sotaview | 19.740 | 0.8303 | 0.1973 |")
        lines.append(f"| **W3 dual-GS** | **{avg_v_psnr:.3f}** | **{avg_v_ssim:.4f}** | **{avg_v_lpips:.4f}** |")
        lines.append("| DNGaussian 论文 (~粗略) | ~24 | ~0.88 | ~0.13 |")
        lines.append("| CoR-GS 论文 (~粗略) | ~26 | ~0.91 | ~0.10 |")

    out = OUT / "w3_dual_gs_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}  ({n_done}/8 done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
