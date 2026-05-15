"""
collect_x30k_sota.py — 扫描 outputs/blender_*_n8_w3_aggrprune_long_v6_pgan70_x30k/
                      下的 metrics.json，汇总成 SOTA 表格写到 outputs/_sota/。

用法（独立调用）:
    python scripts/collect_x30k_sota.py

输出:
    outputs/_sota/blender_n8_x30k_sota.json     # 全量结构化数据
    outputs/_sota/blender_n8_x30k_sota.md       # 人读 markdown 表
    outputs/_sota/blender_n8_x30k_sota.csv      # 给 Excel 的 csv

设计原则：
  - 容错：任何场景缺 metrics.json / 缺字段都用 '-' 占位，不抛错。
  - 增量：每跑完一个场景就可以单独调用一次刷新最新结果。
  - 同时收录 v6_pgan70 (15k baseline) 和 v6_pgan70_x30k (current SOTA)，
    方便横向对比 "x30k 相对 15k 的增益"。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs"
SOTA_DIR = OUT_BASE / "_sota"
SOTA_DIR.mkdir(parents=True, exist_ok=True)

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# 我们关心的两个 tag：
#   baseline (15k) 与 x30k 各做一栏，便于看 30k vs 15k 增益
TAGS = [
    ("v6_pgan70",      "15k"),    # baseline
    ("v6_pgan70_x30k", "30k"),    # candidate SOTA
]


def _safe_load_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metrics(meta: dict[str, Any] | None) -> dict[str, float | None]:
    """从一个 metrics.json 抽 PSNR/SSIM/LPIPS 与 best 的对应数值。"""
    if not meta:
        return {"psnr": None, "ssim": None, "lpips": None,
                "best_psnr": None, "best_step": None, "best_kind": None,
                "n_gaussians": None, "wall_clock_sec": None}

    m = meta.get("metrics") or {}
    bm = (meta.get("best_metrics") or {})
    return {
        "psnr":  m.get("test/psnr"),
        "ssim":  m.get("test/ssim"),
        "lpips": m.get("test/lpips"),
        "best_psnr": bm.get("test/psnr"),
        "best_step": meta.get("best_step"),
        "best_kind": (meta.get("pre_prune_test", {}) or {})
                       .get("best", {}).get("kind"),
        "n_gaussians": meta.get("num_gaussians"),
        "wall_clock_sec": meta.get("wall_clock_sec"),
    }


def _fmt(x: Any, prec: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def collect() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        for tag, label in TAGS:
            run_name = f"blender_{scene}_n8_w3_aggrprune_long_{tag}"
            meta_path = OUT_BASE / run_name / "metrics.json"
            meta = _safe_load_json(meta_path)
            metrics = _extract_metrics(meta)
            rows.append({
                "scene": scene,
                "tag": tag,
                "label": label,            # 15k / 30k
                "run_name": run_name,
                "exists": meta is not None,
                **metrics,
            })

    # 计算 avg-8（两个 tag 各一行），仅在该 tag 下 8 个场景全有结果时计算
    aggregates = {}
    for tag, label in TAGS:
        psnrs = [r["psnr"] for r in rows if r["tag"] == tag and r["psnr"] is not None]
        ssims = [r["ssim"] for r in rows if r["tag"] == tag and r["ssim"] is not None]
        lpips = [r["lpips"] for r in rows if r["tag"] == tag and r["lpips"] is not None]
        aggregates[tag] = {
            "label": label,
            "n_scenes": len(psnrs),
            "avg_psnr":  (sum(psnrs)/len(psnrs))   if len(psnrs) == len(SCENES) else None,
            "avg_ssim":  (sum(ssims)/len(ssims))   if len(ssims) == len(SCENES) else None,
            "avg_lpips": (sum(lpips)/len(lpips))   if len(lpips) == len(SCENES) else None,
            "psnr_when_partial":  (sum(psnrs)/len(psnrs))  if psnrs  else None,
            "ssim_when_partial":  (sum(ssims)/len(ssims))  if ssims  else None,
            "lpips_when_partial": (sum(lpips)/len(lpips))  if lpips  else None,
        }

    return {"rows": rows, "aggregates": aggregates}


def write_json(payload: dict[str, Any]) -> Path:
    p = SOTA_DIR / "blender_n8_x30k_sota.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_csv(payload: dict[str, Any]) -> Path:
    p = SOTA_DIR / "blender_n8_x30k_sota.csv"
    fieldnames = ["scene", "tag", "label", "exists", "psnr", "ssim", "lpips",
                  "best_psnr", "best_step", "best_kind",
                  "n_gaussians", "wall_clock_sec", "run_name"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in payload["rows"]:
            w.writerow({k: r.get(k) for k in fieldnames})
    return p


def write_md(payload: dict[str, Any]) -> Path:
    p = SOTA_DIR / "blender_n8_x30k_sota.md"
    rows = payload["rows"]
    aggs = payload["aggregates"]

    # 一行一个场景，列出 15k / 30k 两组指标 + delta
    by_scene: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in SCENES}
    for r in rows:
        by_scene[r["scene"]][r["label"]] = r

    lines: list[str] = []
    lines.append("# Blender n=8 — v6_pgan70 vs v6_pgan70_x30k SOTA tracker")
    lines.append("")
    lines.append("Auto-generated by `scripts/collect_x30k_sota.py`.")
    lines.append("Each metric row reads from `outputs/<run_name>/metrics.json`'s")
    lines.append("`metrics.test/*` (= final test). Missing scenes show `-`.")
    lines.append("")
    lines.append("## Per-scene (PSNR / SSIM / LPIPS)")
    lines.append("")
    lines.append("| scene | 15k PSNR | 30k PSNR | ΔPSNR | 15k SSIM | 30k SSIM | 15k LPIPS | 30k LPIPS | 30k #G | 30k time |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SCENES:
        a = by_scene[s].get("15k", {}) or {}
        b = by_scene[s].get("30k", {}) or {}
        ap, bp = a.get("psnr"), b.get("psnr")
        delta = (bp - ap) if (ap is not None and bp is not None) else None
        n_g = b.get("n_gaussians")
        wc  = b.get("wall_clock_sec")
        wc_str = f"{int(wc)//60}m" if isinstance(wc, (int, float)) else "-"
        lines.append("| {scene} | {ap} | {bp} | {dp} | {asx} | {bsx} | {al} | {bl} | {ng} | {wc} |".format(
            scene=s,
            ap=_fmt(ap, 3), bp=_fmt(bp, 3),
            dp=("+" + _fmt(delta, 3)) if (delta is not None and delta >= 0)
                else (_fmt(delta, 3) if delta is not None else "-"),
            asx=_fmt(a.get("ssim"), 4), bsx=_fmt(b.get("ssim"), 4),
            al=_fmt(a.get("lpips"), 4), bl=_fmt(b.get("lpips"), 4),
            ng=(f"{n_g:,}" if isinstance(n_g, int) else "-"),
            wc=wc_str,
        ))

    # avg-8 行
    g15 = aggs.get("v6_pgan70", {})
    g30 = aggs.get("v6_pgan70_x30k", {})
    a15p = g15.get("avg_psnr")  if g15.get("avg_psnr")  is not None else g15.get("psnr_when_partial")
    a30p = g30.get("avg_psnr")  if g30.get("avg_psnr")  is not None else g30.get("psnr_when_partial")
    a15s = g15.get("avg_ssim")  if g15.get("avg_ssim")  is not None else g15.get("ssim_when_partial")
    a30s = g30.get("avg_ssim")  if g30.get("avg_ssim")  is not None else g30.get("ssim_when_partial")
    a15l = g15.get("avg_lpips") if g15.get("avg_lpips") is not None else g15.get("lpips_when_partial")
    a30l = g30.get("avg_lpips") if g30.get("avg_lpips") is not None else g30.get("lpips_when_partial")

    delta_avg = (a30p - a15p) if (a15p is not None and a30p is not None) else None
    note15 = "" if g15.get("avg_psnr")  is not None else f" *(partial: {g15.get('n_scenes')}/8)*"
    note30 = "" if g30.get("avg_psnr")  is not None else f" *(partial: {g30.get('n_scenes')}/8)*"

    lines.append("| **avg-8** | **{ap}**{n15} | **{bp}**{n30} | **{dp}** | **{asx}** | **{bsx}** | **{al}** | **{bl}** |  |  |".format(
        ap=_fmt(a15p, 3), bp=_fmt(a30p, 3),
        dp=("+" + _fmt(delta_avg, 3)) if (delta_avg is not None and delta_avg >= 0)
            else (_fmt(delta_avg, 3) if delta_avg is not None else "-"),
        asx=_fmt(a15s, 4), bsx=_fmt(a30s, 4),
        al=_fmt(a15l, 4),  bl=_fmt(a30l, 4),
        n15=note15, n30=note30,
    ))
    lines.append("")
    lines.append("## SOTA gap to literature (Blender avg-8, n=8 PSNR)")
    lines.append("")
    lines.append("| method | PSNR | gap to ours (30k) |")
    lines.append("|---|---:|---:|")
    lit = [
        ("FSGS (ECCV'24)",       24.6),
        ("CoR-GS (ECCV'24)",     24.5),
        ("DNGaussian (CVPR'24)", 24.3),
        ("FreeNeRF (CVPR'23)",   24.3),
        ("RegNeRF (CVPR'22)",    23.9),
        ("DietNeRF (ICCV'21)",   23.6),
        ("SparseGS (3DV'24)",    22.8),
    ]
    full_30 = g30.get("avg_psnr") is not None
    for name, val in lit:
        if a30p is None:
            gap_str = "-"
        elif not full_30:
            gap_str = f"(partial {g30.get('n_scenes')}/8)"
        else:
            gap = a30p - val
            gap_str = f"{gap:+.3f} dB"
        lines.append(f"| {name} | {val} | {gap_str} |")
    ours_label = "Ours (avg-8, v6_pgan70_x30k)" if full_30 \
                 else f"Ours (partial {g30.get('n_scenes')}/8, v6_pgan70_x30k)"
    lines.append(f"| **{ours_label}** | **{_fmt(a30p, 3)}** | — |")
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    payload = collect()
    pj = write_json(payload)
    pc = write_csv(payload)
    pm = write_md(payload)
    print(f"[sota] wrote -> {pj}")
    print(f"[sota] wrote -> {pc}")
    print(f"[sota] wrote -> {pm}")

    # 简短摘要打到 stdout
    rows = payload["rows"]
    have30 = [r for r in rows if r["tag"] == "v6_pgan70_x30k" and r["psnr"] is not None]
    print(f"[sota] x30k done : {len(have30)}/{len(SCENES)} scenes")
    for r in have30:
        print(f"  {r['scene']:>10s}  PSNR={r['psnr']:.3f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}")


if __name__ == "__main__":
    main()
