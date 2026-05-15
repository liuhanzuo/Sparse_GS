"""Merge diagnostic triptychs from drums/materials/chair into a single
labeled image so we can verify with one image read.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = Path(os.environ.get(
    "SPARSE_GS_DIAG_DIR",
    str(_REPO_ROOT / "outputs" / "_diag_w3_failed"),
))
SCENES = ["drums", "materials", "chair"]
VIEW = 50  # 取 v=50 一张代表
TILE_W = 480  # 每个 cell 的宽（pred|GT|diff 已经是 3 列）
ROW_H = 200

panels = []
for s in SCENES:
    p = DIAG / f"{s}_v{VIEW:03d}.png"
    im = Image.open(p).convert("RGB")
    # downscale，拼图小一点
    new_w = TILE_W
    new_h = int(im.height * (new_w / im.width))
    im = im.resize((new_w, new_h))
    # 在图顶部加一行黑底白字 label
    label_h = 26
    canvas = Image.new("RGB", (new_w, new_h + label_h), color=(0, 0, 0))
    canvas.paste(im, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{s}  v={VIEW}    [pred | GT(white-bg) | 5x|diff|]", fill=(255, 255, 255))
    panels.append(canvas)

# 垂直堆叠
W = panels[0].width
H = sum(p.height for p in panels)
out = Image.new("RGB", (W, H), color=(255, 255, 255))
y = 0
for p in panels:
    out.paste(p, (0, y))
    y += p.height
out_path = DIAG / "summary_v050.png"
out.save(out_path)
print("saved", out_path, out.size)
