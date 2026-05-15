"""Generate W3 dual-GS configs for all 8 NeRF-Synthetic scenes.

Each emitted config inherits the matching ``configs/_w2sota_view`` baseline
and only adds a ``dual_gs`` block. We don't sweep dual_gs hyperparameters
here — single point taken from chair validation.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "configs" / "_w3_dual_gs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def expname(scene: str) -> str:
    if scene == "lego":
        return "lego_n8_w3_dual_gs"
    return f"blender_{scene}_n8_w3_dual_gs"


def base_path(scene: str) -> str:
    if scene == "lego":
        return "../_w2sota_view/lego_n8_w2_sotaview.yaml"
    return f"../_w2sota_view/blender_{scene}_n8_w2_sotaview.yaml"


YAML_TEMPLATE = """_base_: {base}

experiment:
  name: {name}

# C-stage: CoR-GS dual-GS — second independently-initialized field +
# co-regularization on pseudo cams + co-pruning of inconsistent points.
# Hyperparameters validated on chair (single-scene smoke test); same
# values applied to all 8 scenes to keep the comparison fair.
dual_gs:
  enabled: true
  seed_gs1: 43
  coreg: true
  coreg_weight: 1.0
  start_sample_pseudo: 500
  end_sample_pseudo: 7000
  sample_pseudo_interval: 1
  pseudo_t_range: [0.2, 0.8]
  coprune: true
  coprune_threshold: 0.10
  coprune_start_iter: 1500
  coprune_stop_iter: 6500
  coprune_every_iter: 500
  safety_max_ratio: 0.10
"""


def main() -> None:
    for s in SCENES:
        cfg_path = OUT_DIR / f"{expname(s)}.yaml"
        cfg_path.write_text(
            YAML_TEMPLATE.format(base=base_path(s), name=expname(s)),
            encoding="utf-8",
        )
        print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
