"""W3 hotdog long-step ablation driver.

目的（用户提的假设验证）：
  原 7000 步 hotdog aggrprune 的 val_psnr 走的是
    step 2000 = 25.02 (peak, 一次 prune 前)
    step 4000 = 19.34 (第一次 floater-prune 后)
    step 6000 = 20.13 (恢复中)
    step 7000 = 23.50 (final)
  用户假设可能是 25 -> 19 -> 20 -> 24 -> 26... 的递进反弹被 7000 步早停截断。

本脚本串行跑 2 组对照（同一场景 hotdog，仅在剪枝强度上不同）：
  A) blender_hotdog_n8_w3_aggrprune_long  : 15000 步 + 激进 prune 安全阀
  B) blender_hotdog_n8_w3_dual_gs_long    : 15000 步 + 温和 prune 安全阀

判定：
  * A_final > 25.02  -> 用户假设成立，反弹真实存在
  * A_final - B_final > 1.0  -> prune 机制本身有效（不是单纯靠 step）
  * train_psnr 在每次 floater-prune 后呈"崩溃 -> 恢复 -> 新峰值"递进 -> 强证据

跑完自动提取每个 [eval @ N] val_psnr 与 [test] 最终值，打印对比。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# (tag, config path, output exp name)
EXPERIMENTS = [
    (
        "aggr",
        ROOT / "configs" / "_w3_aggrprune" / "blender_hotdog_n8_w3_aggrprune_long.yaml",
        "blender_hotdog_n8_w3_aggrprune_long",
    ),
    (
        "dual",
        ROOT / "configs" / "_w3_dual_gs" / "blender_hotdog_n8_w3_dual_gs_long.yaml",
        "blender_hotdog_n8_w3_dual_gs_long",
    ),
]


def already_done(exp_name: str) -> bool:
    p = ROOT / "outputs" / exp_name / "metrics.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return "test/psnr" in (d.get("metrics") or {})
    except Exception:
        return False


def run_one(tag: str, cfg: Path, exp_name: str) -> int:
    log = LOGS / f"w3hotdog_long_{tag}.log"
    print(f"\n[{tag}] >>> starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[{tag}]   cfg = {cfg}", flush=True)
    print(f"[{tag}]   log = {log}", flush=True)
    if not cfg.exists():
        print(f"[{tag}]   !! config missing, skipping", flush=True)
        return 1

    cmd = [sys.executable, "-m", "scripts.train", "--config", str(cfg)]
    t0 = time.time()
    last_print = t0
    last_progress_line = ""
    line_buf = ""

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, bufsize=0,
    )

    HEARTBEAT_S = 15.0
    with open(log, "w", encoding="utf-8") as f:
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            text = chunk.decode("utf-8", errors="replace")
            f.write(text)
            f.flush()
            line_buf += text

            parts = []
            while True:
                idxs = [i for i in (line_buf.find("\n"), line_buf.find("\r")) if i >= 0]
                if not idxs:
                    break
                cut = min(idxs)
                seg, line_buf = line_buf[:cut], line_buf[cut + 1:]
                if seg.strip():
                    parts.append(seg)

            for seg in parts:
                low = seg.lower()
                if any(k in low for k in (
                    "[w2-prune]", "[dual-prune]", "[dual]",
                    "[eval", "[test]", "traceback", "error", "warning",
                )):
                    print(f"[{tag}] {seg}", flush=True)
                    last_print = time.time()
                else:
                    last_progress_line = seg

            now = time.time()
            if now - last_print >= HEARTBEAT_S and last_progress_line:
                print(f"[{tag}] hb {last_progress_line[-160:]}", flush=True)
                last_print = now

    rc = proc.wait()
    dt = time.time() - t0
    print(f"[{tag}] <<< finished rc={rc} elapsed={dt:.1f}s at {time.strftime('%H:%M:%S')}", flush=True)
    return rc


def parse_eval_curve(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    pat = re.compile(r"\[eval @ (\d+)\] val/psnr=([\d.]+)")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [(int(m.group(1)), float(m.group(2))) for m in pat.finditer(text)]


def parse_prune_events(log_path: Path) -> list[tuple[int, str, int]]:
    """Return [(step, kind, pruned_N), ...] for floater/unseen prune events."""
    if not log_path.exists():
        return []
    out = []
    pat_floater = re.compile(r"\[w2-prune\] floater pruned N=(\d+) \(step=(\d+)")
    pat_unseen = re.compile(r"\[w2-prune\] unseen pruned N=(\d+) \(step=(\d+)")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in pat_floater.finditer(text):
        out.append((int(m.group(2)), "floater", int(m.group(1))))
    for m in pat_unseen.finditer(text):
        out.append((int(m.group(2)), "unseen", int(m.group(1))))
    out.sort()
    return out


def parse_test_metrics(exp_name: str) -> dict | None:
    p = ROOT / "outputs" / exp_name / "metrics.json"
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text(encoding="utf-8")).get("metrics") or {})
    except Exception:
        return None


def summarize() -> None:
    print("\n" + "=" * 80)
    print("HOTDOG LONG-STEP ABLATION  —  is the rebound real or step-bounded?")
    print("=" * 80)

    for tag, cfg, exp_name in EXPERIMENTS:
        log = LOGS / f"w3hotdog_long_{tag}.log"
        curve = parse_eval_curve(log)
        prunes = parse_prune_events(log)
        metrics = parse_test_metrics(exp_name)

        print(f"\n--- [{tag}]  cfg = {cfg.name} ---")
        if metrics is None:
            print(f"  test metrics: MISSING")
        else:
            print(f"  test/psnr  = {metrics.get('test/psnr', float('nan')):.3f}")
            print(f"  test/ssim  = {metrics.get('test/ssim', float('nan')):.3f}")
            print(f"  test/lpips = {metrics.get('test/lpips', float('nan')):.3f}")

        print(f"  prune events ({len(prunes)}):")
        for step, kind, n in prunes:
            print(f"    step={step:5d}  {kind:<7s}  pruned={n}")

        print(f"  val/psnr trajectory ({len(curve)} samples):")
        for step, psnr in curve:
            marker = ""
            for ps_step, _, _ in prunes:
                if abs(step - ps_step) <= 2:
                    marker = "   <- right after prune"
                elif 0 < step - ps_step <= 220:
                    marker = "   <- ~200 steps after prune"
                elif 0 < step - ps_step <= 520:
                    marker = "   <- ~500 steps after prune"
            print(f"    @ {step:5d}  val/psnr = {psnr:.3f}{marker}")

    # Cross-comparison
    a_metrics = parse_test_metrics(EXPERIMENTS[0][2])
    b_metrics = parse_test_metrics(EXPERIMENTS[1][2])
    print("\n" + "-" * 80)
    print("VERDICT")
    print("-" * 80)
    if a_metrics and b_metrics:
        a, b = a_metrics["test/psnr"], b_metrics["test/psnr"]
        baseline_peak = 25.02  # observed from prior run at step=2000
        baseline_final = 23.50
        print(f"  baseline (7K aggrprune)  : peak={baseline_peak:.2f}  final={baseline_final:.2f}")
        print(f"  aggrprune_long (15K)      : final={a:.3f}")
        print(f"  dual_gs_long   (15K)      : final={b:.3f}")
        print(f"  delta (aggr - dual)       : {a - b:+.3f}")
        print()
        if a > baseline_peak:
            print(f"  ✅ aggrprune_long > 7K peak ({baseline_peak:.2f}) — 用户假设成立，反弹真实存在")
        else:
            print(f"  ❌ aggrprune_long ({a:.2f}) <= 7K peak ({baseline_peak:.2f}) — 反弹未发生")
        if a - b > 1.0:
            print(f"  ✅ aggr - dual > 1.0 — 激进 prune 机制本身有效")
        elif a - b > 0:
            print(f"  ⚠️  aggr 仅微弱领先 dual ({a-b:+.2f}) — 可能只是 step 多了")
        else:
            print(f"  ❌ aggr 落后 dual — 激进 prune 在长 step 下也没用")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=None,
                    choices=[t for t, _, _ in EXPERIMENTS],
                    help="subset of tags to run")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if metrics.json exists")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip training, just print summary from existing logs/metrics")
    args = ap.parse_args()

    if args.summarize_only:
        summarize()
        return 0

    targets = args.tags if args.tags else [t for t, _, _ in EXPERIMENTS]
    for tag, cfg, exp_name in EXPERIMENTS:
        if tag not in targets:
            continue
        if not args.force and already_done(exp_name):
            print(f"[{tag}] already done -> skip", flush=True)
            continue
        rc = run_one(tag, cfg, exp_name)
        if rc != 0:
            print(f"[{tag}] failed with rc={rc}, aborting", flush=True)
            return rc

    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
