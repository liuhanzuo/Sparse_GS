"""W3-aggrprune-long_v3 driver — 4 scenes (hotdog/drums/materials/ship).

把 last floater prune 从 step=14000 提前到 11500，留 3500 步恢复期，
hotdog 上已验证 peak val_psnr=26.6（v2）。本脚本把同一调度套到
当前最差的 4 个场景，期望同样的 "晚 prune -> 充分恢复 -> 高 final"
现象在它们身上也成立。

使用：
    python scripts/run_w3_aggrprune_long_v3.py
    python scripts/run_w3_aggrprune_long_v3.py --scenes drums materials
    python scripts/run_w3_aggrprune_long_v3.py --force        # 重跑

每场景训练 ≈ 2300s（仅参考），4 场景串行 ≈ 2.5h。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# 4 scenes that had unsatisfying numbers under the original 7k schedule:
#   hotdog    23.50 / 23.97  (need to lift toward peak 26.6)
#   drums     15.40
#   materials 12.78  (worse than vanilla 14.79 — aggressively over-pruned)
#   ship      16.96
SCENES = ["hotdog", "drums", "materials", "ship"]


def cfg_for(scene: str) -> Path:
    return ROOT / "configs" / "_w3_aggrprune" / f"blender_{scene}_n8_w3_aggrprune_long_v3.yaml"


def expname_for(scene: str) -> str:
    return f"blender_{scene}_n8_w3_aggrprune_long_v3"


def already_done(scene: str) -> bool:
    metrics = ROOT / "outputs" / expname_for(scene) / "metrics.json"
    if not metrics.exists():
        return False
    try:
        d = json.loads(metrics.read_text(encoding="utf-8"))
        return "test/psnr" in (d.get("metrics") or {})
    except Exception:
        return False


def clear_eval_log(scene: str) -> None:
    """JSONL is append-mode — clear it before a fresh run."""
    p = ROOT / "outputs" / expname_for(scene) / "eval_log.jsonl"
    if p.exists():
        try:
            p.unlink()
            print(f"[{scene}]   cleared old eval_log.jsonl", flush=True)
        except Exception as e:
            print(f"[{scene}]   warn: failed to clear eval_log.jsonl: {e}", flush=True)


def run(scene: str) -> int:
    cfg = cfg_for(scene)
    log = LOGS / f"w3aggrlongv3_{scene}.log"
    print(f"\n[{scene}] >>> starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[{scene}]   cfg = {cfg}", flush=True)
    print(f"[{scene}]   log = {log}", flush=True)
    if not cfg.exists():
        print(f"[{scene}]   !! config missing, skipping", flush=True)
        return 1

    clear_eval_log(scene)

    cmd = [sys.executable, "-m", "scripts.train", "--config", str(cfg)]
    t0 = time.time()
    last_print = t0
    last_progress_line = ""
    line_buf = ""

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, bufsize=0,
    )

    HEARTBEAT_S = 10.0
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
                if any(k in low for k in ("[w2-prune]", "[dual-prune]", "[dual]",
                                          "[eval", "[test]", "traceback",
                                          "error", "warning",
                                          "pre_floater", "post_floater")):
                    print(f"[{scene}] {seg}", flush=True)
                    last_print = time.time()
                else:
                    last_progress_line = seg

            now = time.time()
            if now - last_print >= HEARTBEAT_S and last_progress_line:
                print(f"[{scene}] hb {last_progress_line[-160:]}", flush=True)
                last_print = now

    rc = proc.wait()
    dt = time.time() - t0
    print(f"[{scene}] <<< finished rc={rc} elapsed={dt:.1f}s at {time.strftime('%H:%M:%S')}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Subset of scenes to run (default: all 4).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if metrics.json already exists.")
    args = ap.parse_args()

    targets = args.scenes if args.scenes else SCENES
    bad = [s for s in targets if s not in SCENES]
    if bad:
        print(f"unknown scenes: {bad}", file=sys.stderr)
        return 2

    summary = []
    t_master = time.time()
    for s in targets:
        if not args.force and already_done(s):
            print(f"[{s}] already done -> skip", flush=True)
            summary.append((s, "skip", None))
            continue
        rc = run(s)
        summary.append((s, "ok" if rc == 0 else f"rc={rc}", rc))

    total = time.time() - t_master
    print("\n==================================================")
    print(f"W3-aggrprune-long_v3 4-scene driver summary  (total {total/60:.1f} min):")
    for s, status, rc in summary:
        print(f"  {s:10s} : {status}")
    print("==================================================")
    print("\n下一步：")
    print("  python scripts/log_summary.py outputs/blender_hotdog_n8_w3_aggrprune_long_v3 \\")
    print("                                outputs/blender_drums_n8_w3_aggrprune_long_v3 \\")
    print("                                outputs/blender_materials_n8_w3_aggrprune_long_v3 \\")
    print("                                outputs/blender_ship_n8_w3_aggrprune_long_v3")
    return 0 if all((rc in (None, 0)) for _, _, rc in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
