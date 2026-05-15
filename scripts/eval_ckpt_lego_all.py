"""Re-evaluate every lego run on the FULL 200-view test split, one
subprocess per run.

Background
----------
``scripts/eval_ckpt.py --glob outputs/lego_n*`` is the right idea but in
practice the long-lived Python process trips ``cudaErrorUnknown`` after
~3 evals on Windows + 5090 + the gsplat CUDA extension (the same
context-rot we saw on the LLFF sweep). To work around that we just spawn
one subprocess per run; each one starts with a clean CUDA context.

We only re-eval if ``metrics_full.json`` is missing or older than
``ckpts/last.pt`` so reruns are cheap.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def needs_eval(run_dir: Path) -> bool:
    ck = run_dir / "ckpts" / "last.pt"
    if not ck.exists():
        return False
    mfull = run_dir / "metrics_full.json"
    if not mfull.exists():
        return True
    return mfull.stat().st_mtime < ck.stat().st_mtime


def main() -> None:
    runs = sorted([p for p in OUT.glob("lego_n*") if p.is_dir()
                   and (p / "config.yaml").exists()])
    todo = [r for r in runs if needs_eval(r)]
    print(f"[full_eval] {len(todo)}/{len(runs)} runs need re-eval")

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    creationflags = 0
    if os.name == "nt":
        creationflags = 0x08000000  # CREATE_NO_WINDOW

    log_dir = OUT / "_lego_full_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail, fails = 0, 0, []
    t0 = time.time()
    for r in todo:
        log_path = log_dir / f"{r.name}.log"
        cmd = [py, str(ROOT / "scripts" / "eval_ckpt.py"),
               "--run", str(r)]
        print(f"[run] {r.name}  > {log_path.name}", flush=True)
        with open(log_path, "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), env=env,
                                 creationflags=creationflags)
        if rc == 0 and (r / "metrics_full.json").exists():
            n_ok += 1
        else:
            n_fail += 1
            fails.append((r.name, rc))
            print(f"  [fail] rc={rc}  see {log_path}")
    dt = time.time() - t0
    print(f"\n[full_eval] {n_ok} ok, {n_fail} fail, {dt/60:.1f} min")
    for name, rc in fails:
        print(f"  {name}: rc={rc}")


if __name__ == "__main__":
    main()
