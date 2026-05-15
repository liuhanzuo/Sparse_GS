"""Detached launcher for run_blender_n8_pipeline.py.

Spawns the pipeline as a background process *without* the launcher
keeping it alive — closing the launcher does not kill the child,
mirroring scripts/launch_llff_sweep_detached.py.

Output is redirected to outputs/_blender_n8_pipeline.stdout.log
(the pipeline itself also writes its own log to
outputs/_blender_n8_pipeline.log; this just captures any uncaught
crash before the in-script logger initialises).

Usage::

    python scripts/launch_blender_n8_pipeline_detached.py
    python scripts/launch_blender_n8_pipeline_detached.py --scenes chair drums
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"
_PIPE_OUT = _OUT / "_blender_n8_pipeline.stdout.log"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="optional subset; default: all 7 missing scenes")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_ROOT / "scripts" / "run_blender_n8_pipeline.py")]
    if args.scenes:
        cmd += ["--scenes", *args.scenes]

    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # On Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP gives a true
    # background process not bound to this console.
    creationflags = 0
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        creationflags = (DETACHED_PROCESS
                         | CREATE_NEW_PROCESS_GROUP
                         | CREATE_NO_WINDOW)

    fout = open(_PIPE_OUT, "ab", buffering=0)
    p = subprocess.Popen(
        cmd, stdout=fout, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(_ROOT), env=env,
        creationflags=creationflags,
        close_fds=True,
    )
    # Give it a moment to be sure it started cleanly
    time.sleep(0.5)
    rc = p.poll()
    if rc is not None and rc != 0:
        print(f"pipeline child exited immediately rc={rc}; see {_PIPE_OUT}",
              file=sys.stderr)
        return rc or 1
    print(f"launched detached pipeline pid={p.pid}")
    print(f"  scene set: {args.scenes or '(all 7)'}")
    print(f"  child stdout/stderr -> {_PIPE_OUT}")
    print(f"  in-script log       -> {_OUT / '_blender_n8_pipeline.log'}")
    print(f"  per-scene state     -> {_OUT / '_blender_n8_state'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
