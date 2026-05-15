"""Dump image_4 resolution for each LLFF scene."""
from pathlib import Path
from PIL import Image
root = Path(__file__).resolve().parent.parent / "data/nerf_llff_data"
for s in sorted(p.name for p in root.iterdir() if p.is_dir()):
    d = root / s / "images_4"
    if not d.is_dir():
        print(f"{s:10s} no images_4"); continue
    imgs = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.JPG"))
    if not imgs:
        print(f"{s:10s} empty"); continue
    w, h = Image.open(imgs[0]).size
    print(f"{s:10s}  N={len(imgs):3d}  res={w}x{h}  (pixels={w*h/1e3:.1f}k)")
