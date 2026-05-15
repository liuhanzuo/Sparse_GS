# Third-Party Baselines

This project compares against four sparse-view 3DGS baselines. Their source code
is **not** vendored into this repository (each has its own upstream repo and
license).  If you want to reproduce side-by-side numbers, clone them manually
into the `third_party/` directory at the repo root.

## Layout

```
Sparse_GS_/
├── sparse_gs/            ← our method (versioned here)
├── third_party/          ← gitignored; clone baselines below into here
│   ├── CoR-GS/
│   ├── DNGaussian/
│   ├── FSGS/
│   └── SparseGS/
└── ...
```

## How to fetch the baselines

Run the following **inside the repo root** (`Sparse_GS_/`):

```bash
mkdir -p third_party && cd third_party

# 1) CoR-GS  — Co-Regularization for sparse-view 3DGS (CVPR'24)
git clone https://github.com/jiaw-z/CoR-GS.git

# 2) DNGaussian  — Depth-prior + Normalised 3DGS (CVPR'24)
git clone https://github.com/Fictionarry/DNGaussian.git

# 3) FSGS  — Few-Shot 3DGS via Gaussian Unpooling (ECCV'24)
git clone https://github.com/VITA-Group/FSGS.git

# 4) SparseGS  — Real-time 360° Sparse-View 3DGS (3DV'25)
git clone https://github.com/ForMyCat/SparseGS.git
```

Each baseline has its own `requirements.txt` / build steps; follow their
README.  We only consume their **printed metrics** (PSNR/SSIM/LPIPS) for our
SOTA comparison tables under `results/_sota/`.

## Why not include them as submodules?

* Their custom CUDA kernels (e.g. `diff-gaussian-rasterization-softmax`,
  `simple-knn`) ship with build artefacts that bloat repos by ~150 MB even on a
  fresh clone (we observed this firsthand).
* Their licenses differ from ours; mixing them into one repo would be murky.
* Reproducibility is preserved by pinning the **upstream commits we tested
  against** in the table below.

## Pinned commits

| Baseline    | Upstream commit (date)      | Notes |
|-------------|-----------------------------|-------|
| CoR-GS      | _record after first clone_  | LLFF + Blender |
| DNGaussian  | _record after first clone_  | LLFF + Blender |
| FSGS        | _record after first clone_  | LLFF only      |
| SparseGS    | _record after first clone_  | Blender only   |

> Update this table by running `git -C third_party/<baseline> rev-parse --short HEAD`
> after a successful build.
