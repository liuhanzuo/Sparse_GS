"""W2-sotaview multi-scene serial driver.

Same logic as run_w2_all_scenes.py, but points at the W2-sotaview
configs (which only differ from W2 by switching the train-view sampler
to the DNG/CoR-GS fixed 8-view list).
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

# tqdm progress bars contain unicode block chars (▏▎▌...) that the default
# Windows console codepage (GBK) cannot encode. Force this driver's own
# stdout/stderr to UTF-8 with replace-on-error so the heartbeat print never
# crashes the whole run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover - older Pythons
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def cfg_for(scene: str) -> Path:
    if scene == "lego":
        return ROOT / "configs" / "_w2sota_view" / "lego_n8_w2_sotaview.yaml"
    return ROOT / "configs" / "_w2sota_view" / f"blender_{scene}_n8_w2_sotaview.yaml"


def expname_for(scene: str) -> str:
    if scene == "lego":
        return "lego_n8_w2_sotaview"
    return f"blender_{scene}_n8_w2_sotaview"


def already_done(scene: str) -> bool:
    metrics = ROOT / "outputs" / expname_for(scene) / "metrics.json"
    if not metrics.exists():
        return False
    try:
        d = json.loads(metrics.read_text(encoding="utf-8"))
        return "test/psnr" in (d.get("metrics") or {})
    except Exception:
        return False


def run(scene: str) -> int:
    cfg = cfg_for(scene)
    log = LOGS / f"w2sotaview_{scene}.log"
    print(f"\n[{scene}] >>> starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[{scene}]   cfg = {cfg}", flush=True)
    print(f"[{scene}]   log = {log}", flush=True)
    if not cfg.exists():
        print(f"[{scene}]   !! config missing, skipping", flush=True)
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
                if any(k in low for k in ("[w2-prune]", "[eval", "[test]", "traceback",
                                          "error", "warning")):
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
                    help="Subset of scenes to run (default: all 8).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if metrics.json already exists.")
    args = ap.parse_args()

    targets = args.scenes if args.scenes else SCENES
    bad = [s for s in targets if s not in SCENES]
    if bad:
        print(f"unknown scenes: {bad}", file=sys.stderr)
        return 2

    summary = []
    for s in targets:
        if not args.force and already_done(s):
            print(f"[{s}] already done -> skip", flush=True)
            summary.append((s, "skip", None))
            continue
        rc = run(s)
        summary.append((s, "ok" if rc == 0 else f"rc={rc}", rc))

    print("\n==================================================")
    print("W2-sotaview 8-scene driver summary:")
    for s, status, rc in summary:
        print(f"  {s:10s} : {status}")
    print("==================================================")
    return 0 if all((rc in (None, 0)) for _, _, rc in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
