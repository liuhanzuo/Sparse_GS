"""W1 Pearson Depth Loss + FSGS-aligned Schedule —— 实验聚合脚本.

读取已落盘的 outputs/blender_{scene}_n8_ssl_mv_dav2s_{variant}/metrics.json，
聚合三条曲线（depthv2 baseline / pearson_only / depthv2_pearson）的
avg PSNR / SSIM / LPIPS / num_gaussians，并以 depthv2 为基准计算 delta。

仅依赖 Python 标准库，禁止改动训练侧代码。
直接前台运行：python scripts/aggregate_w1_pearson.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- 常量 -----------------------------

SCENES: Tuple[str, ...] = (
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
)

# variant key -> (display label, default suffix, config file name)
DEFAULT_VARIANTS: Dict[str, Dict[str, str]] = {
    "depthv2": {
        "label": "depthv2 (baseline)",
        "suffix": "depthv2",
        "config": "blender_n8_depthv2.yaml",
    },
    "pearson": {
        "label": "pearson_only",
        "suffix": "pearson",
        "config": "blender_n8_pearson.yaml",
    },
    "depthv2_pearson": {
        "label": "depthv2_pearson",
        "suffix": "depthv2_pearson_idfix",
        "config": "blender_n8_depthv2_pearson.yaml",
    },
}

BASELINE_KEY = "depthv2"

# lego 的 depthv2 baseline 命名特殊（缺 blender_ 前缀），允许 fallback。
LEGO_DEPTHV2_FALLBACK = "lego_n8_ssl_mv_dav2s_depthv2"


# ----------------------------- 数据扫描 -----------------------------


def candidate_paths(outputs_root: Path, scene: str, suffix: str) -> List[Path]:
    """返回该 (scene, suffix) 组合下应当尝试的 metrics.json 候选路径列表."""
    primary = outputs_root / f"blender_{scene}_n8_ssl_mv_dav2s_{suffix}" / "metrics.json"
    candidates: List[Path] = [primary]
    if scene == "lego" and suffix == "depthv2":
        candidates.append(outputs_root / LEGO_DEPTHV2_FALLBACK / "metrics.json")
    return candidates


def scan_metrics_paths(
    outputs_root: Path, variants: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """枚举三条曲线 × 8 场景的 metrics.json 路径并标记三态."""
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for vk, vinfo in variants.items():
        suffix = vinfo["suffix"]
        table[vk] = {}
        for scene in SCENES:
            chosen: Optional[Path] = None
            attempted: List[str] = []
            for cand in candidate_paths(outputs_root, scene, suffix):
                attempted.append(str(cand))
                if cand.is_file():
                    chosen = cand
                    break
            if chosen is None:
                table[vk][scene] = {
                    "status": "MISSING",
                    "path": None,
                    "attempted": attempted,
                    "reason": "metrics.json 不存在",
                }
            else:
                table[vk][scene] = {
                    "status": "OK",
                    "path": chosen,
                    "attempted": attempted,
                    "reason": "",
                }
    return table


# ----------------------------- 数据解析 -----------------------------


def load_scene_metrics(path: Path) -> Dict[str, Any]:
    """从单个 metrics.json 抽取 PSNR/SSIM/LPIPS/num_gaussians.

    仅读取 metrics["test/psnr"]、metrics["test/ssim"]、metrics["test/lpips"]
    与顶层 num_gaussians；任何异常都归为 BROKEN。
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BROKEN",
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "num_gaussians": None,
            "reason": f"BROKEN: {type(exc).__name__}: {exc}",
        }

    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}

    def _f(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _i(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    psnr = _f(metrics.get("test/psnr"))
    ssim = _f(metrics.get("test/ssim"))
    lpips = _f(metrics.get("test/lpips"))
    n_gauss = _i(data.get("num_gaussians") if isinstance(data, dict) else None)

    missing_fields = [
        name
        for name, val in (
            ("test/psnr", psnr),
            ("test/ssim", ssim),
            ("test/lpips", lpips),
        )
        if val is None
    ]
    if missing_fields:
        return {
            "status": "BROKEN",
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "num_gaussians": n_gauss,
            "reason": f"BROKEN: 缺字段 {','.join(missing_fields)}",
        }

    return {
        "status": "OK",
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "num_gaussians": n_gauss,
        "reason": "",
    }


# ----------------------------- 聚合 -----------------------------


def aggregate_variant(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """对一条曲线的 8 个 scene 行做算术平均，剔除 N/A 而非补 0."""
    psnr_vals = [r["psnr"] for r in rows.values() if r.get("psnr") is not None]
    ssim_vals = [r["ssim"] for r in rows.values() if r.get("ssim") is not None]
    lpips_vals = [r["lpips"] for r in rows.values() if r.get("lpips") is not None]
    ng_vals = [r["num_gaussians"] for r in rows.values() if r.get("num_gaussians") is not None]
    available = [s for s, r in rows.items() if r.get("status") == "OK"]
    return {
        "psnr": mean(psnr_vals) if psnr_vals else None,
        "ssim": mean(ssim_vals) if ssim_vals else None,
        "lpips": mean(lpips_vals) if lpips_vals else None,
        "num_gaussians": int(round(mean(ng_vals))) if ng_vals else None,
        "n_available": len(available),
        "available_scenes": sorted(available),
    }


def aggregate_intersection(
    variant_rows: Dict[str, Dict[str, Dict[str, Any]]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """对“所有曲线都 OK”的全交集 scene 集合做 avg."""
    intersect: Optional[set] = None
    for rows in variant_rows.values():
        ok = {s for s, r in rows.items() if r.get("status") == "OK"}
        intersect = ok if intersect is None else (intersect & ok)
    if not intersect:
        return [], {}
    scenes_sorted = sorted(intersect)
    out: Dict[str, Dict[str, Any]] = {}
    for vk, rows in variant_rows.items():
        sub = {s: rows[s] for s in scenes_sorted}
        out[vk] = aggregate_variant(sub)
    return scenes_sorted, out


def compute_deltas(
    stats: Dict[str, Dict[str, Any]], baseline_key: str = BASELINE_KEY
) -> Dict[str, Dict[str, Optional[float]]]:
    """以 baseline 曲线为基准计算 delta."""
    base = stats.get(baseline_key, {})
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for vk, s in stats.items():
        if vk == baseline_key:
            deltas[vk] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
            continue
        d: Dict[str, Optional[float]] = {}
        for k in ("psnr", "ssim", "lpips"):
            if s.get(k) is None or base.get(k) is None:
                d[k] = None
            else:
                d[k] = s[k] - base[k]
        deltas[vk] = d
    return deltas


# ----------------------------- 渲染 -----------------------------


def fmt(v: Optional[float], digits: int) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def fmt_delta(v: Optional[float], digits: int) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{digits}f}"


def fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "N/A"
    return f"{int(v):,}"


def render_overview_table(
    variants: Dict[str, Dict[str, str]],
    stats: Dict[str, Dict[str, Any]],
    deltas: Dict[str, Dict[str, Optional[float]]],
) -> List[str]:
    lines = [
        "| Variant | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Avg N_Gaussians | 可用场景数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for vk in variants:
        s = stats.get(vk, {})
        d = deltas.get(vk, {})
        is_baseline = vk == BASELINE_KEY
        lines.append(
            "| {label} | {psnr} | {dpsnr} | {ssim} | {dssim} | {lpips} | {dlpips} | {ng} | {n} |".format(
                label=variants[vk]["label"],
                psnr=fmt(s.get("psnr"), 3),
                dpsnr="—" if is_baseline else fmt_delta(d.get("psnr"), 3),
                ssim=fmt(s.get("ssim"), 3),
                dssim="—" if is_baseline else fmt_delta(d.get("ssim"), 3),
                lpips=fmt(s.get("lpips"), 4),
                dlpips="—" if is_baseline else fmt_delta(d.get("lpips"), 4),
                ng=fmt_int(s.get("num_gaussians")),
                n=str(s.get("n_available", 0)),
            )
        )
    return lines


def render_per_scene_block(
    label: str,
    rows: Dict[str, Dict[str, Any]],
) -> List[str]:
    out = [f"#### {label}", "", "| scene | PSNR | SSIM | LPIPS | N_Gaussians | 备注 |", "| --- | --- | --- | --- | --- | --- |"]
    for scene in SCENES:
        r = rows.get(scene, {})
        status = r.get("status", "MISSING")
        if status == "OK":
            note = ""
            psnr = fmt(r.get("psnr"), 3)
            ssim = fmt(r.get("ssim"), 3)
            lpips = fmt(r.get("lpips"), 4)
            ng = fmt_int(r.get("num_gaussians"))
        else:
            note = r.get("reason") or status
            psnr = "N/A"
            ssim = "N/A"
            lpips = "N/A"
            ng = "N/A"
        out.append(f"| {scene} | {psnr} | {ssim} | {lpips} | {ng} | {note} |")
    out.append("")
    return out


def make_conclusions(
    stats_full: Dict[str, Dict[str, Any]],
    stats_inter: Dict[str, Dict[str, Any]],
    deltas_full: Dict[str, Dict[str, Optional[float]]],
    deltas_inter: Dict[str, Dict[str, Optional[float]]],
    schedule_observable: bool,
) -> List[str]:
    """基于聚合结果生成恰好 3 条结论."""
    # 选用全交集结果；若交集为空则退回 full
    use_inter = bool(stats_inter)
    deltas = deltas_inter if use_inter else deltas_full
    stats = stats_inter if use_inter else stats_full
    base_psnr = stats.get(BASELINE_KEY, {}).get("psnr")
    pear_d = deltas.get("pearson", {})
    both_d = deltas.get("depthv2_pearson", {})

    # (a) Pearson vs depthv2 baseline
    a_psnr = pear_d.get("psnr")
    a_ssim = pear_d.get("ssim")
    a_lpips = pear_d.get("lpips")
    if a_psnr is None:
        a_text = (
            "**(a) Pearson vs depthv2 baseline：** 无法判定，pearson_only 或 baseline 数据缺失。"
        )
    else:
        sign = "提升" if a_psnr > 0 else "下降"
        sig = "显著" if abs(a_psnr) >= 0.20 else ("不显著" if abs(a_psnr) < 0.05 else "轻微")
        a_text = (
            f"**(a) Pearson vs depthv2 baseline：** pearson_only 相对 baseline 在 PSNR 上"
            f"**{sig}{sign}** ΔPSNR={fmt_delta(a_psnr,3)}（"
            f"ΔSSIM={fmt_delta(a_ssim,3) if a_ssim is not None else 'N/A'}, "
            f"ΔLPIPS={fmt_delta(a_lpips,4) if a_lpips is not None else 'N/A'}）。"
        )
        if a_psnr <= 0:
            a_text += (
                " 可能原因：prior depth 来自 DAv2-small（disparity 语义），"
                "公式中已通过 -prior 与 1/(prior+200) 的 min 兼容符号歧义；"
                "本轮权重 0.05 + train-view-only 设定，Pearson 信号未压过 RGB+depthv2 主导项。"
            )

    # (b) Pearson 与 depthv2 是否互补
    b_psnr_both = both_d.get("psnr")
    b_psnr_pear = pear_d.get("psnr")
    if b_psnr_both is None or b_psnr_pear is None:
        b_text = "**(b) Pearson 与 depthv2 是否互补：** 无法判定，相关曲线数据缺失。"
    else:
        # both 相对 baseline 是否优于 pearson_only 相对 baseline；并且 both 相对 baseline 是否为正
        gain_both = b_psnr_both
        gain_pear = b_psnr_pear
        # 互补 = both 同时优于 pearson_only 且 ≥ baseline
        if gain_both > 0 and gain_both > gain_pear:
            b_text = (
                f"**(b) Pearson 与 depthv2 是否互补：** **互补**，depthv2_pearson 同时优于 baseline "
                f"({fmt_delta(gain_both,3)}) 与 pearson_only ({fmt_delta(gain_both - gain_pear,3)} vs pearson_only)。"
            )
        elif gain_both > 0 and gain_both <= gain_pear:
            b_text = (
                f"**(b) Pearson 与 depthv2 是否互补：** **弱互补**，depthv2_pearson 相对 baseline 提升 "
                f"{fmt_delta(gain_both,3)}，但与 pearson_only 几乎持平甚至更差，叠加未带来额外收益。"
            )
        else:
            b_text = (
                f"**(b) Pearson 与 depthv2 是否互补：** **未互补**，depthv2_pearson 相对 baseline "
                f"{fmt_delta(gain_both,3)}（depthv2_pearson vs pearson_only={fmt_delta(gain_both - gain_pear,3)}）。"
                f" 推测 depthv2 已经吃掉了大部分 depth 监督容量，Pearson 全局相关在低权重下被压制。"
            )

    # (c) FSGS-aligned warmup schedule
    if schedule_observable:
        c_text = (
            "**(c) FSGS-aligned warmup schedule：** 本轮配置启用了 pseudo-view 调度参数与 0.05 权重 warmup，"
            "但效果是否可观测仍需 W2 dual-GS / pseudo-view RGB 联合产出对照。"
        )
    else:
        c_text = (
            "**(c) FSGS-aligned warmup schedule：** 本轮为 **train-view-only** 设定，"
            "pseudo-view 渲染自约束未启用，schedule 仅作为权重 warmup 生效，"
            "无法在本表中独立验证其作用，留待 W2 接入 pseudo-view RGB/Depth 后单独消融。"
        )
    return [a_text, b_text, c_text]


def make_w2_decision(
    stats: Dict[str, Dict[str, Any]],
    deltas: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[str, List[str]]:
    """根据聚合结果给出 W2 决策建议."""
    pear_d = deltas.get("pearson", {}).get("psnr")
    both_d = deltas.get("depthv2_pearson", {}).get("psnr")
    reasons: List[str] = []

    pear_pos = pear_d is not None and pear_d > 0.05
    both_pos = both_d is not None and both_d > 0.05
    pear_neg = pear_d is not None and pear_d < -0.05
    both_neg = both_d is not None and both_d < -0.05

    if both_pos and (pear_pos or (pear_d is not None and pear_d > -0.05)):
        decision = "推荐进入 W2"
        reasons.append(
            f"depthv2_pearson 相对 baseline ΔPSNR={fmt_delta(both_d, 3)}，叠加在多场景平均上为正向。"
        )
        if pear_pos:
            reasons.append(
                f"pearson_only 单独也有 ΔPSNR={fmt_delta(pear_d, 3)}，说明 Pearson 监督本身有效，可在 W2 进一步引入 pseudo-view 渲染自约束放大收益。"
            )
        reasons.append(
            "W1 已锁定 train-view-only 场景的最佳组合，进入 W2 dual-GS / pseudo-view 实验风险可控。"
        )
    elif both_neg and pear_neg:
        decision = "不推荐进入 W2"
        reasons.append(
            f"两条 Pearson 相关曲线均劣于 baseline（pearson_only ΔPSNR={fmt_delta(pear_d, 3)}, depthv2_pearson ΔPSNR={fmt_delta(both_d, 3)}）。"
        )
        reasons.append(
            "应先排查 prior depth 语义/权重/warmup 区间，再考虑是否进入 W2 的 pseudo-view 与 dual-GS。"
        )
        reasons.append(
            "建议小范围调参实验（pearson_weight ∈ {0.02, 0.05, 0.1}、prior 取 -depth vs 1/(d+200)）后再回看 W1。"
        )
    else:
        decision = "有条件推荐"
        if both_d is not None:
            reasons.append(
                f"depthv2_pearson 相对 baseline ΔPSNR={fmt_delta(both_d, 3)}，方向不显著，但未明显恶化。"
            )
        if pear_d is not None:
            reasons.append(
                f"pearson_only ΔPSNR={fmt_delta(pear_d, 3)}，单独 Pearson 在低权重下信号偏弱。"
            )
        reasons.append(
            "条件：先在 1 个场景上做 pearson_weight + warmup 区间小型 sweep，确认 Pearson 的监督容量是否被 depthv2 吃掉，再决定是否进入 W2。"
        )
    return decision, reasons[:3]


# ----------------------------- 主流程 -----------------------------


def run(args: argparse.Namespace) -> int:
    outputs_root: Path = args.outputs_root
    report_path: Path = args.report

    variants: Dict[str, Dict[str, str]] = {}
    for vk, vinfo in DEFAULT_VARIANTS.items():
        variants[vk] = dict(vinfo)
    if args.depthv2_suffix:
        variants["depthv2"]["suffix"] = args.depthv2_suffix
    if args.pearson_suffix:
        variants["pearson"]["suffix"] = args.pearson_suffix
    if args.depthv2_pearson_suffix:
        variants["depthv2_pearson"]["suffix"] = args.depthv2_pearson_suffix

    # 1) 完整性预检
    print("=" * 72)
    print("[W1] Pearson Depth Loss + FSGS-aligned Schedule —— 实验聚合")
    print("=" * 72)
    print(f"outputs_root = {outputs_root}")
    print("variants:")
    for vk, vinfo in variants.items():
        print(
            f"  - {vk:<16s} suffix={vinfo['suffix']:<28s} config={vinfo['config']}"
        )
    print()

    path_table = scan_metrics_paths(outputs_root, variants)

    missing: List[Tuple[str, str, str]] = []
    for vk, rows in path_table.items():
        for scene, info in rows.items():
            if info["status"] != "OK":
                missing.append((vk, scene, info["reason"]))

    if missing:
        print(f"[WARN] 共发现 {len(missing)} 项 metrics.json 缺失/异常：")
        for vk, scene, reason in missing:
            print(f"  - {vk:<16s} {scene:<10s} {reason}")
    else:
        print("[OK] 24 个 metrics.json 全部就绪。")
    print()

    # lego depthv2 命名歧义提示
    lego_depthv2 = path_table.get("depthv2", {}).get("lego", {})
    if lego_depthv2.get("status") == "OK":
        chosen = lego_depthv2["path"]
        if LEGO_DEPTHV2_FALLBACK in str(chosen):
            print(
                f"[NOTE] lego depthv2 baseline 命中 fallback 命名：{chosen}"
                f"\n       （主路径 outputs/blender_lego_n8_ssl_mv_dav2s_depthv2/ 不存在，已自动采用上述目录。）"
            )
            print()

    # 2) 解析每个 metrics.json
    rows_by_variant: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for vk, rows in path_table.items():
        rows_by_variant[vk] = {}
        for scene in SCENES:
            info = rows[scene]
            if info["status"] != "OK":
                rows_by_variant[vk][scene] = {
                    "status": info["status"],
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "num_gaussians": None,
                    "reason": info.get("reason", info["status"]),
                    "path": None,
                }
                continue
            parsed = load_scene_metrics(info["path"])
            parsed["path"] = info["path"]
            rows_by_variant[vk][scene] = parsed

    # 3) 聚合：各自可用场景 avg + 全交集 avg
    stats_full: Dict[str, Dict[str, Any]] = {
        vk: aggregate_variant(rows) for vk, rows in rows_by_variant.items()
    }
    inter_scenes, stats_inter = aggregate_intersection(rows_by_variant)

    deltas_full = compute_deltas(stats_full)
    deltas_inter = compute_deltas(stats_inter) if stats_inter else {}

    # 4) stdout 总览表
    sets_per_variant = {vk: tuple(s.get("available_scenes", [])) for vk, s in stats_full.items()}
    same_set = len(set(sets_per_variant.values())) == 1

    print("[Overview] 各自可用场景 avg：")
    for line in render_overview_table(variants, stats_full, deltas_full):
        print(line)
    print()
    if not same_set and stats_inter:
        print(f"[Overview] 全交集 avg（{len(inter_scenes)} 个场景：{', '.join(inter_scenes)}）：")
        for line in render_overview_table(variants, stats_inter, deltas_inter):
            print(line)
        print()

    # 5+6) 渲染 markdown
    md_lines: List[str] = []
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    md_lines.append("# W1 Pearson Depth Loss + FSGS-aligned Schedule —— 实验聚合")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{now_iso}")
    md_lines.append(f"- 数据源根目录：`{outputs_root.as_posix()}`")
    md_lines.append("- 三条曲线：")
    for vk, vinfo in variants.items():
        md_lines.append(
            f"  - **{vinfo['label']}** —— `outputs/blender_{{scene}}_n8_ssl_mv_dav2s_{vinfo['suffix']}/metrics.json` "
            f"（config: `configs/{vinfo['config']}`）"
        )
    md_lines.append(
        f"- 聚合脚本：`scripts/aggregate_w1_pearson.py`（仅依赖标准库；本轮**不重训任何模型**）"
    )
    md_lines.append("")

    md_lines.append("## 1. 总览（avg over 各自可用场景）")
    md_lines.append("")
    md_lines.extend(render_overview_table(variants, stats_full, deltas_full))
    md_lines.append("")
    md_lines.append(
        "> ΔPSNR / ΔSSIM 越大越好（正值表示提升）；ΔLPIPS 越小越好（**负值表示提升**）。num_gaussians 取整。"
    )
    md_lines.append("")

    if not same_set and stats_inter:
        md_lines.append(f"### 1b. 全交集场景 avg（{len(inter_scenes)} 个场景）")
        md_lines.append("")
        md_lines.append(f"全交集场景集合：{', '.join(inter_scenes)}")
        md_lines.append("")
        md_lines.extend(render_overview_table(variants, stats_inter, deltas_inter))
        md_lines.append("")

    md_lines.append("## 2. Per-scene 明细")
    md_lines.append("")
    for vk in variants:
        md_lines.extend(render_per_scene_block(variants[vk]["label"], rows_by_variant[vk]))

    md_lines.append("## 3. 关键结论（3 条）")
    md_lines.append("")
    schedule_observable = False  # 本轮 W1 train-view-only，schedule 不能独立验证
    for c in make_conclusions(
        stats_full, stats_inter, deltas_full, deltas_inter, schedule_observable
    ):
        md_lines.append(f"- {c}")
    md_lines.append("")

    decision, reasons = make_w2_decision(
        stats_inter if stats_inter else stats_full,
        deltas_inter if deltas_inter else deltas_full,
    )
    md_lines.append("## 4. W2 决策建议")
    md_lines.append("")
    md_lines.append(f"- **结论：{decision}**")
    md_lines.append("- 理由：")
    for r in reasons:
        md_lines.append(f"  - {r}")
    md_lines.append("")
    md_lines.append(
        "> 本节为建议，**不自动启动 W2 训练或代码修改**；如继续 W2，请回到 plan 模式重新走需求与任务规划流程。"
    )
    md_lines.append("")

    # 6.5) 落盘
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] 报告已写入：{report_path}")
    print()

    # 7) 退出码：缺失 > 0 且未 --allow-partial 时返回非 0
    if missing and not args.allow_partial:
        print(
            "[EXIT 2] 存在 MISSING/BROKEN 项且未 --allow-partial。"
            " 请补齐缺失或加 --allow-partial 重新运行。"
        )
        return 2

    print("[EXIT 0] 聚合完成。")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Aggregate W1 Pearson ablation metrics into a markdown report.")
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=project_root / "outputs",
        help="outputs/ 根目录（默认 d:/SSL/sparse_gs/outputs）",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=project_root / "outputs" / "w1_pearson_ablation.md",
        help="报告输出路径（默认 outputs/w1_pearson_ablation.md）",
    )
    p.add_argument("--depthv2-suffix", type=str, default=None, help="覆盖 depthv2 baseline 的 variant 后缀")
    p.add_argument("--pearson-suffix", type=str, default=None, help="覆盖 pearson_only 的 variant 后缀")
    p.add_argument(
        "--depthv2-pearson-suffix",
        type=str,
        default=None,
        help="覆盖 depthv2_pearson 的 variant 后缀（默认 depthv2_pearson_idfix）",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许在存在 MISSING/BROKEN 项时继续聚合并以 0 退出",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
