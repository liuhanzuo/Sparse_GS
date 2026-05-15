"""Spawn the LLFF sweep as a truly-detached Windows process (no console).

Key fix over earlier versions: we do NOT create a new process group or a
new console. The sweep master runs with DETACHED_PROCESS (no console at
all) and CREATE_BREAKAWAY_FROM_JOB (so it survives the IDE's cleanup of
its terminal job object). Child train.py workers are launched by the
sweep with CREATE_NO_WINDOW for the same reason.

Usage:
    python scripts/launch_llff_sweep_detached.py [--scenes ...]
Any args after the script name are forwarded to run_llff_sweep.py.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    python = sys.executable
    sweep = str(root / "scripts" / "run_llff_sweep.py")
    out_log = str(root / "outputs" / "_llff_sweep_master.log")
    err_log = str(root / "outputs" / "_llff_sweep_master.err")

    # Forward extra CLI args, or default to the four remaining scenes.
    extra = sys.argv[1:]
    if not extra:
        extra = ["--scenes", "leaves", "orchids", "room", "trex"]
    cmd = [python, sweep, *extra]

    creationflags = 0
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        # NOTE: intentionally NOT using CREATE_NEW_PROCESS_GROUP --- that
        # caused ctrl-close events from the IDE terminal to propagate.
        creationflags = DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB

    fout = open(out_log, "ab", buffering=0)
    ferr = open(err_log, "ab", buffering=0)
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=fout,
        stderr=ferr,
        creationflags=creationflags,
        close_fds=True,
        env=env,
    )
    print(f"launched PID={proc.pid}")
    print(f"master log: {out_log}")
    print(f"err log:    {err_log}")
    print(f"args:       {cmd}")


if __name__ == "__main__":
    main()
