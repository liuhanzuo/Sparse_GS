"""Probe how to load DAv2-Small under transformers 5.2.0."""
import torch
from PIL import Image
import numpy as np

# attempt 1: AutoModelForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = r"D:\SSL\sparse_gs\outputs\_models\Depth-Anything-V2-Small-hf"

try:
    proc = AutoImageProcessor.from_pretrained(MODEL)
    print("processor OK", type(proc).__name__)
    mdl = AutoModelForDepthEstimation.from_pretrained(MODEL)
    print("model OK", type(mdl).__name__, "params=",
          sum(p.numel() for p in mdl.parameters()))
    # quick smoke
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    inp = proc(img, return_tensors="pt")
    print("input shape:", {k: v.shape for k, v in inp.items()})
    with torch.no_grad():
        out = mdl(**inp)
    print("output keys:", list(out.keys()) if hasattr(out, "keys") else "?")
    if hasattr(out, "predicted_depth"):
        print("predicted_depth shape:", out.predicted_depth.shape)
    print("OK")
except Exception as e:
    import traceback; traceback.print_exc()
