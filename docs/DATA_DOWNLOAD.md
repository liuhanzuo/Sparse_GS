# Data & Pretrained Weights — Download Guide

This repository's training scripts expect three external resources that
are **not committed** to git (sizes shown):

| Path                                       | Size    | Source |
|--------------------------------------------|---------|--------|
| `data/nerf_synthetic/`                     | 3.2 GB  | NeRF authors / Google Drive |
| `data/nerf_llff_data/` *(optional)*        | 2.8 GB  | LLFF authors / Google Drive |
| `models/depth_anything_v2_small/`          | 100 MB  | HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` |
| `models/depth_anything_v2_base/`  *(opt.)* | 390 MB  | HuggingFace `depth-anything/Depth-Anything-V2-Base-hf` |
| `models/depth_anything_v2_large/` *(opt.)* | 1.3 GB  | HuggingFace `depth-anything/Depth-Anything-V2-Large-hf` |

> ⚠️  These files **already exist** in this checkout (because they were
> migrated from the working `sparse_gs/` snapshot for convenience), but
> are listed in `.gitignore` so they will not be pushed. Treat the
> instructions below as the **canonical** way to recreate them on a
> fresh machine.

---

## 1. NeRF-Synthetic (Blender, 8 scenes)

Used by every `configs/_w*/blender_*.yaml`.

```powershell
# Option A — official Google Drive (mirrored by NeRF authors)
# https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view
# Download nerf_synthetic.zip (~700 MB) and unzip:
Expand-Archive nerf_synthetic.zip -DestinationPath data/

# Option B — HuggingFace mirror
huggingface-cli download bullshit/nerf_synthetic --repo-type dataset `
    --local-dir data/nerf_synthetic
```

Expected layout:
```
data/nerf_synthetic/
├── chair/   {train,val,test}/  transforms_*.json
├── drums/   …
├── ficus/   …
├── hotdog/  …
├── lego/    …
├── materials/  …
├── mic/     …
└── ship/    …
```

The 8-view sparse training subset is selected at runtime by
`SparseTrainSampler` (see `sparse_gs/datasets/sparse_sampler.py`) using a
deterministic seed → no extra step needed.

---

## 2. LLFF (forward-facing, 8 scenes) — optional

Used only by `configs/llff_*` and `configs/_llff_sweep/`.

```powershell
# Authors' Google Drive: https://drive.google.com/drive/folders/1cdW0YYUdwsMZTVvvIMdt8WhDH_FTbuVj
# Download nerf_llff_data.zip (~1.8 GB) and unzip:
Expand-Archive nerf_llff_data.zip -DestinationPath data/

# OR use the helper script (downloads + unzips into data/_cache then data/):
python scripts/download_llff.py
```

Expected layout:
```
data/nerf_llff_data/
├── fern/      images/  poses_bounds.npy
├── flower/    …
├── fortress/  …
├── horns/     …
├── leaves/    …
├── orchids/   …
├── room/      …
└── trex/      …
```

---

## 3. Depth-Anything-V2 weights

Used by `losses/depth_consist` and `pseudo_view`. The **Small** variant
is enough for the SOTA results (DAv2-Small + scale-shift-invariant L1).

```powershell
# HuggingFace snapshot (recommended — no auth required)
python -c @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='depth-anything/Depth-Anything-V2-Small-hf',
    local_dir='models/depth_anything_v2_small',
    local_dir_use_symlinks=False,
)
"@
```

Expected layout:
```
models/
├── depth_anything_v2_small/   ← required (~100 MB)
├── depth_anything_v2_base/    ← optional (~390 MB, slightly better SSI)
└── depth_anything_v2_large/   ← optional (~1.3 GB, marginally better SSI)
```

---

## 4. Per-scene depth caches *(generated, not downloaded)*

After the model + dataset are in place, run **once per scene** to
pre-compute the depth predictions used by `multiview_photo` and
`depth_consist`:

```powershell
python scripts/precompute_depth.py `
    --backend transformers `
    --model-id depth-anything/Depth-Anything-V2-Small-hf `
    --tag depth_anything_v2_small `
    --scenes hotdog drums chair ficus lego materials mic ship `
    --split train --n-views 8
```

This writes `data/nerf_synthetic/<scene>/_depth_cache/depth_anything_v2_small/*.npz`
(~30 MB per scene at full resolution; ~250 MB if you cache `train+val+test`).

---

## 5. Disk-space summary

A complete **reproduction-ready** working tree (data + models + this
repo's code + experiment archives) sums to roughly:

| Component                                           | Size    |
|-----------------------------------------------------|---------|
| `data/nerf_synthetic/` + per-scene `_depth_cache/`  | ~3.2 GB |
| `data/nerf_llff_data/` + per-scene `_depth_cache/`  | ~2.8 GB |
| `models/depth_anything_v2_*` (Small + Base + Large) | ~1.7 GB |
| `third_party/` (4 baseline repos)                   | ~165 MB |
| Source + configs + experiment archives              | <1 MB   |
| **Total**                                           | **~8 GB** |

During training, `outputs/<run_name>/` adds ~50–500 MB per run (mostly
checkpoints) plus optional render videos.
