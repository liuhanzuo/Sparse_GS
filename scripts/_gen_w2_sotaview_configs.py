"""Generate W2-sotaview configs for all 8 NeRF-Synthetic scenes.

Each config inherits the matching `_w2_prune` config and only overrides:
  * experiment.name -> "<orig_root>_w2_sotaview"
  * data.sparse_mode -> "sota_nerf_synthetic_n8"

This is the B-stage fix: switch the train-view sampler from naive
linspace -> the fixed 8-view list shared by DNGaussian and CoR-GS.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "_w2_prune"
DST = ROOT / "configs" / "_w2sota_view"
DST.mkdir(parents=True, exist_ok=True)

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def base_for(scene: str) -> Path:
    if scene == "lego":
        return SRC / "lego_n8_dav2s_depthv2_prune.yaml"
    return SRC / f"blender_{scene}_n8_dav2s_depthv2_prune.yaml"


def out_for(scene: str) -> Path:
    if scene == "lego":
        return DST / "lego_n8_w2_sotaview.yaml"
    return DST / f"blender_{scene}_n8_w2_sotaview.yaml"


def expname_for(scene: str) -> str:
    if scene == "lego":
        return "lego_n8_w2_sotaview"
    return f"blender_{scene}_n8_w2_sotaview"


CONTENT = """_base_: ../_w2_prune/{base_name}

experiment:
  name: {expname}

# B-stage fix: switch the train-view sampler from naive linspace
# (-> [0,14,28,42,57,71,85,99]) to the fixed 8-view list shared by
# DNGaussian (third_party/DNGaussian/scene/dataset_readers.py:331) and
# CoR-GS (third_party/CoR-GS/scene/dataset_readers.py:511):
#   [2, 16, 26, 55, 73, 76, 86, 93]
# Same setting both papers use for their NeRF-Synthetic n=8 numbers, so
# this lets us compare apples-to-apples against their reported results.
data:
  sparse_mode: sota_nerf_synthetic_n8
"""


def main() -> int:
    for s in SCENES:
        bf = base_for(s)
        if not bf.exists():
            raise FileNotFoundError(f"base config missing: {bf}")
        out = out_for(s)
        out.write_text(
            CONTENT.format(base_name=bf.name, expname=expname_for(s)),
            encoding="utf-8",
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
