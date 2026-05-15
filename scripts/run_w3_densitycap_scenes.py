"""W3-densitycap driver — start from drums single-scene validation.

针对 drums / materials / hotdog / ship 这 4 个 W3 dual-GS 中
val PSNR 严重欠拟合（drums/materials 跌到 12~15）的场景，
本次变体改的是 gsplat DefaultStrategy 的 densify 旋钮——而不是
事后剪枝（那是 aggrprune 的方向）。

第一阶段：先只跑 drums 验证 densitycap 是否能把 val PSNR 从
14.93 拉到 18+。如果有效，再扩到 materials / hotdog / ship。
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
else:  # pragma: no cover - older Pythons
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# 第一阶段只验证 drums。验证有效后再加入 ["materials", "hotdog", "ship"]。
SCENES = ["drums"]


def cfg_for(scene: str) -> Path:
    return ROOT / "configs" / "_w3_densitycap" / f"blender_{scene}_n8_w3_densitycap.yaml"


def expname_for(scene: str) -> str:
    return f"blender_{scene}_n8_w3_densitycap"


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
    log = LOGS / f"w3dcap_{scene}.log"
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
                if any(k in low for k in ("[w2-prune]", "[dual-prune]", "[dual]",
                                          "[eval", "[test]", "traceback",
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
                    help="Subset of scenes to run (default: drums only for phase-1).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if metrics.json already exists.")
    args = ap.parse_args()

    targets = args.scenes if args.scenes else SCENES
    # 不限制 scenes 必须在 SCENES 内——后续阶段直接 --scenes materials 即可。

    summary = []
    for s in targets:
        cfg = cfg_for(s)
        if not cfg.exists():
            print(f"[{s}] config not found at {cfg} — please create it first.", flush=True)
            summary.append((s, "no-cfg", 1))
            continue
        if not args.force and already_done(s):
            print(f"[{s}] already done -> skip", flush=True)
            summary.append((s, "skip", None))
            continue
        rc = run(s)
        summary.append((s, "ok" if rc == 0 else f"rc={rc}", rc))

    print("\n==================================================")
    print("W3-densitycap driver summary:")
    for s, status, rc in summary:
        print(f"  {s:10s} : {status}")
    print("==================================================")
    return 0 if all((rc in (None, 0)) for _, _, rc in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
