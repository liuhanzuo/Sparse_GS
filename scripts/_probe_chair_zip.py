"""Tiny probe: download just chair.zip, list its top-level entries,
then DELETE the zip if confirmed-correct so we don't waste disk
(the real pipeline will re-download via hf cache and that's instant
the second time)."""
from __future__ import annotations

from pathlib import Path
import time
from huggingface_hub import hf_hub_download
import zipfile

_OUT = Path(__file__).resolve().parents[1] / "outputs" / "_blender_n8_state" / "_probe_chair"
_OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
p = hf_hub_download(
    repo_id="nerfbaselines/nerfbaselines-data",
    repo_type="dataset",
    filename="blender/chair.zip",
    local_dir=str(_OUT),
    local_dir_use_symlinks=False,
)
dt = time.time() - t0
print(f"downloaded in {dt:.1f}s -> {p}")
print(f"size: {Path(p).stat().st_size/1024/1024:.1f} MB")

with zipfile.ZipFile(p, "r") as zf:
    names = zf.namelist()
    print(f"total entries: {len(names)}")
    print("first 15:")
    for n in names[:15]:
        print(" ", n)
    # detect whether top-level is "<scene>/" or stuff at root
    tops = sorted({n.split("/")[0] for n in names if "/" in n or n.endswith("/")})
    print(f"top-level dirs: {tops}")
