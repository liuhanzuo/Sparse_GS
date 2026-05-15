"""Watchdog: wait for aggr long metrics.json, then auto-launch dual long.

Detached daemon — survives the parent shell exiting. Run with
    python -u scripts/watchdog_dual_long.py &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGGR_METRICS = ROOT / "outputs" / "blender_hotdog_n8_w3_aggrprune_long" / "metrics.json"
DUAL_METRICS = ROOT / "outputs" / "blender_hotdog_n8_w3_dual_gs_long" / "metrics.json"
WATCH_LOG = ROOT / "outputs" / "logs" / "w3hotdog_long_watchdog.log"
DUAL_DRIVER_LOG = ROOT / "outputs" / "logs" / "w3hotdog_long_dual_driver.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def main() -> int:
    log(f"watchdog start (pid={os.getpid()})")
    log(f"  waiting for {AGGR_METRICS}")

    # Phase 1: wait for aggr completion
    while not AGGR_METRICS.exists():
        time.sleep(20)
    log("aggr metrics.json appeared -> launching dual long")

    if DUAL_METRICS.exists():
        log("dual metrics already exist -> skip launch")
        return 0

    # Phase 2: launch dual long
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "run_w3_hotdog_long.py"),
        "--tags", "dual",
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    log(f"  cmd = {' '.join(cmd)}")
    log(f"  log = {DUAL_DRIVER_LOG}")
    with open(DUAL_DRIVER_LOG, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env,
        )
        rc = proc.wait()
    log(f"dual driver finished rc={rc}")

    # Phase 3: write final summary
    if DUAL_METRICS.exists():
        log("dual metrics.json appeared -> running summarize-only")
        sum_log = ROOT / "outputs" / "logs" / "w3hotdog_long_summary.log"
        with open(sum_log, "w", encoding="utf-8") as f:
            subprocess.run(
                [sys.executable, "-u",
                 str(ROOT / "scripts" / "run_w3_hotdog_long.py"),
                 "--summarize-only"],
                cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env,
            )
        log(f"summary written to {sum_log}")
    else:
        log("WARNING: dual metrics still missing after driver returned")

    log("watchdog done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
