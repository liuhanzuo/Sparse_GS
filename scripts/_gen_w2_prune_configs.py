"""One-shot generator for W2 per-scene configs. Throwaway script.

Run once after editing the chair template; produces 7 sibling YAMLs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "configs" / "_w2_prune"
ROOT.mkdir(parents=True, exist_ok=True)

TEMPLATE = """_base_: ../_depth_v2/{base_name}.yaml

experiment:
  name: {exp_name}

# W2: DNGaussian-style unseen prune + lightweight SparseGS-style floater prune.
# See blender_chair_n8_dav2s_depthv2_prune.yaml for full rationale.
strategy:
  prune:
    unseen:
      enabled: true
      start_iter: 2000
      every_iter: 2000
      stop_iter: 6500
    floater:
      enabled: true
      start_iter: 4000
      every_iter: 2000
      stop_iter: 6500
      alpha_thresh: 0.5
      thresh_bin: 0.05
      safety_max_ratio: 0.08
"""

SCENES = [
    # (base depthv2 file stem, experiment.name, output stem)
    ("blender_drums_n8_dav2s_depthv2",     "blender_drums_n8_ssl_mv_dav2s_depthv2_prune",     "blender_drums_n8_dav2s_depthv2_prune"),
    ("blender_ficus_n8_dav2s_depthv2",     "blender_ficus_n8_ssl_mv_dav2s_depthv2_prune",     "blender_ficus_n8_dav2s_depthv2_prune"),
    ("blender_hotdog_n8_dav2s_depthv2",    "blender_hotdog_n8_ssl_mv_dav2s_depthv2_prune",    "blender_hotdog_n8_dav2s_depthv2_prune"),
    ("lego_n8_dav2s_depthv2",              "lego_n8_ssl_mv_dav2s_depthv2_prune",              "lego_n8_dav2s_depthv2_prune"),
    ("blender_materials_n8_dav2s_depthv2", "blender_materials_n8_ssl_mv_dav2s_depthv2_prune", "blender_materials_n8_dav2s_depthv2_prune"),
    ("blender_mic_n8_dav2s_depthv2",       "blender_mic_n8_ssl_mv_dav2s_depthv2_prune",       "blender_mic_n8_dav2s_depthv2_prune"),
    ("blender_ship_n8_dav2s_depthv2",      "blender_ship_n8_ssl_mv_dav2s_depthv2_prune",      "blender_ship_n8_dav2s_depthv2_prune"),
]

count = 0
for base, exp, stem in SCENES:
    out_path = ROOT / f"{stem}.yaml"
    out_path.write_text(TEMPLATE.format(base_name=base, exp_name=exp), encoding="utf-8")
    print(f"  wrote {out_path}")
    count += 1
print(f"Done: {count} configs.")
