"""Predict the geometric coverage gain from switching to the SOTA 8-view list.

For each scene compare two train-view selections side by side:
  * "uniform": [0, 14, 28, 42, 57, 71, 85, 99]   (our current, np.linspace)
  * "SOTA":    [2, 16, 26, 55, 73, 76, 86, 93]   (DNGaussian + CoR-GS hard-coded)

For each strategy we report:
  * pairwise min angle (smaller = train views collide / waste budget)
  * pairwise median angle
  * test-view-to-nearest-train angle: median, p90, max
    (smaller = test set better covered)

This is geometry-only (no rendering), runs in <1 s and tells us upfront
whether the switch is expected to actually help.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nerf_synthetic"
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

UNIFORM = [0, 14, 28, 42, 57, 71, 85, 99]
SOTA = [2, 16, 26, 55, 73, 76, 86, 93]


def angle_deg(a, b):
    na, nb = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
    cos_t = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


def analyse(scene, ids):
    train_d = json.loads((DATA / scene / "transforms_train.json").read_text(encoding="utf-8"))
    test_d = json.loads((DATA / scene / "transforms_test.json").read_text(encoding="utf-8"))
    train_pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in train_d["frames"]])
    test_pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in test_d["frames"]])
    sel = train_pos[ids]
    pair = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            pair.append(angle_deg(sel[i], sel[j]))
    nearest = np.array([min(angle_deg(t, s) for s in sel) for t in test_pos])
    return {
        "pair_min": min(pair),
        "pair_median": float(np.median(pair)),
        "test_nearest_median": float(np.median(nearest)),
        "test_nearest_p90": float(np.percentile(nearest, 90)),
        "test_nearest_max": float(nearest.max()),
    }


def main():
    print(f"\n{'scene':10s} | {'strategy':8s} | pair_min/median  | test_nearest median/p90/max")
    print("-" * 84)
    for s in SCENES:
        for name, ids in [("uniform", UNIFORM), ("SOTA", SOTA)]:
            r = analyse(s, ids)
            print(
                f"{s:10s} | {name:8s} | {r['pair_min']:5.1f} / {r['pair_median']:5.1f}    | "
                f"{r['test_nearest_median']:5.1f} / {r['test_nearest_p90']:5.1f} / {r['test_nearest_max']:5.1f}"
            )
        print()


if __name__ == "__main__":
    main()
