"""Train a sparse-view 3DGS model.

Usage (from ``d:/SSL/sparse_gs``):

    python -m scripts.train --config configs/sparse_view.yaml
"""

from __future__ import annotations

# Robust bootstrap: works for both ``python -m scripts.train`` and
# ``python d:/SSL/sparse_gs/scripts/train.py``.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap                                    # noqa: E402,F401

import argparse                                                   # noqa: E402
from pathlib import Path                                          # noqa: E402

from sparse_gs.trainer import Trainer, DualTrainer                # noqa: E402
from sparse_gs.utils.config import load_config, save_config       # noqa: E402

_ROOT = _bootstrap.PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--name", default=None, type=str,
                    help="override experiment.name (also output sub-dir)")
    ap.add_argument("--iterations", default=None, type=int,
                    help="override train.iterations")
    ap.add_argument("--n-views", default=None, type=int,
                    help="override data.n_train_views (sparse-view sweep)")
    ap.add_argument("--scene", default=None, type=str,
                    help="override data.scene")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name:
        cfg["experiment"]["name"] = args.name
    if args.iterations is not None:
        cfg["train"]["iterations"] = int(args.iterations)
    if args.n_views is not None:
        cfg["data"]["n_train_views"] = int(args.n_views)
        # auto-tag the output dir so sweeps don't overwrite each other.
        # Strip a trailing "_n<digits>" if it exists, then append the new one.
        # This way "lego_n6"        -> "lego_n3"
        #          "lego_n6_ssl_mv" -> "lego_n3_ssl_mv"  (suffix preserved)
        #          "lego"           -> "lego_n3"
        if args.name is None:
            import re
            cur = cfg["experiment"]["name"]
            m = re.match(r"^(.*?)_n\d+(.*)$", cur)
            if m:
                base, tail = m.group(1), m.group(2)
                cfg["experiment"]["name"] = f"{base}_n{args.n_views}{tail}"
            else:
                cfg["experiment"]["name"] = f"{cur}_n{args.n_views}"
    if args.scene is not None:
        cfg["data"]["scene"] = args.scene

    # Resolve relative paths against the project root, not the cwd.
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        cfg["data"]["root"] = str((_ROOT / data_root).resolve())

    out_root = Path(cfg["experiment"].get("output_dir", "outputs"))
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_dir = out_root / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    print(f"[run] output_dir = {out_dir}")

    # ``DualTrainer`` is a Trainer subclass that maintains a second,
    # independently-initialized Gaussian field and trains it jointly
    # with co-reg + co-prune (CoR-GS, ECCV'24). With ``cfg.dual_gs.enabled``
    # absent or false the subclass behaves identically to ``Trainer``,
    # but going through the subclass adds a small RNG snapshot cost we
    # don't want to pay on the baseline runs, so we branch.
    if bool((cfg.get("dual_gs", {}) or {}).get("enabled", False)):
        DualTrainer(cfg, output_dir=out_dir).fit()
    else:
        Trainer(cfg, output_dir=out_dir).fit()


if __name__ == "__main__":
    main()
