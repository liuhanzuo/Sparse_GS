# sparse_gs — Sparse-View 3D Gaussian Splatting (SSL Project)

A minimal, modular **sparse-view 3DGS** baseline built on top of
[`gsplat`](https://github.com/nerfstudio-project/gsplat). This repo is the
starting point of our SSL course project, on which we will incrementally add
self-supervised regularizers (multi-view photometric consistency, depth /
geometry consistency, pseudo-view consistency, feature-level consistency,
teacher–student EMA, optionally a PatchGAN adversarial loss, …).

The current commit contains **stage A (env)** + **stage B (baseline)** +
**stage C (SSL hooks)** + **stage D — first working SSL loss**
(`multiview_photo` + occlusion-aware variant, **+0.79 / +2.43 / +2.06 dB**
on lego at n_views = 3 / 6 / 9 with the recommended `_mv_v3` config;
LPIPS 0.448 / 0.271 / 0.122 respectively).

## 1. Status

| Stage | Description | Status |
|------|-------------|--------|
| A | Repo skeleton, deps, configs, IO | ✅ |
| B | Baseline: dense-init Gaussians, gsplat rasterization, L1 + DSSIM, `DefaultStrategy` densify/prune, eval (PSNR/SSIM) | ✅ |
| C | Hook points for SSL losses & teacher EMA (registered, no-op placeholders) | ✅ |
| D | Concrete SSL losses: **multiview_photo (working)**, pseudo_view + ema_teacher (modular, neutral on vanilla 3DGS) | ✅ first loss |
| E | External-prior SSL (monocular depth, DINOv2 features), occlusion-aware reproj, mesh extraction, paper experiments | ⏳ TODO |

End-to-end has been verified on the dev machine (RTX 5090 Laptop, Win 11,
Py 3.14, torch 2.10+cu128, gsplat 1.5.3) using a procedurally-generated
mini scene (`scripts/make_mini_dataset.py` + `configs/sparse_view_mini.yaml`):

```
[data] NerfSyntheticDataset(scene=_mini, train=4/8, val=4, test=4, scene_scale=3.395)
[model] initial #gaussians = 20000
... 1500 iters in 16.2 s, 94 it/s ...
[eval @ 1500] val/psnr=16.47 | val/ssim=0.83
[test] test/psnr=27.42 | test/ssim=0.90  (#G=36629)
```

The numbers themselves are not meaningful (the mini scene is a coloured
cube), but they confirm that the rasterizer, photometric loss, densify /
prune, optimizer state mutation, evaluator, checkpoint save/load and
render-from-ckpt all work.

## 2. Tested environment

The code was developed against:

- Windows 11, PowerShell
- Python 3.14.3
- PyTorch 2.10.0 + cu128
- gsplat 1.5.3 (pure-Python wheel; CUDA kernels are JIT-compiled by `ninja`
  on first call to `gsplat.rasterization` — expect a 1-3 min stall the very
  first time you launch training)
- NVIDIA RTX 5090 Laptop, Driver 591.91, CUDA 13.1 toolkit available
- 24 GB VRAM is more than enough for NeRF-Synthetic at 800×800

It should also run on Linux without changes.

## 3. Install

```powershell
# from inside d:/SSL
pip install gsplat --no-build-isolation
pip install pyyaml tqdm imageio "imageio[ffmpeg]" pillow tensorboard
```

PyTorch must be installed first and must match your CUDA driver. We rely on
the user's existing torch install (the project does *not* pin a torch
version because the correct wheel depends on the GPU).

## 4. Data

We default to **NeRF-Synthetic** (`lego`, `chair`, …). It is small (~700 MB
for all 8 scenes), self-contained (no COLMAP), and the de-facto sparse-view
benchmark.

Put it under `d:/SSL/sparse_gs/data/nerf_synthetic/<scene>/` so that the
following files exist:

```
data/nerf_synthetic/lego/
    transforms_train.json
    transforms_val.json
    transforms_test.json
    train/r_0.png   r_1.png   ...
    val/...
    test/...
```

Download links (manual — we do not auto-fetch):

- Mirror 1: <https://huggingface.co/datasets/nerf-synthetic-dataset/nerf-synthetic-dataset>
- Mirror 2: official Google Drive link from the NeRF paper

## 5. Run the baseline

```powershell
cd d:/SSL/sparse_gs

# (A) End-to-end smoke test on a generated cube scene (~20 s, no download).
python scripts/make_mini_dataset.py
python scripts/train.py  --config configs/sparse_view_mini.yaml
python scripts/render.py --config configs/sparse_view_mini.yaml ^
                         --ckpt outputs/mini/ckpts/last.pt

# (B) Real sparse-view experiment on NeRF-Synthetic lego (after download).
python scripts/train.py  --config configs/sparse_view.yaml
python scripts/render.py --config configs/sparse_view.yaml ^
                         --ckpt outputs/lego_n6/ckpts/last.pt

# (C) SSL: cross-view photometric consistency. +3.13 dB at n=6 vs (B).
python scripts/train.py --config configs/sparse_view_ssl_mv.yaml
# Sweep over view counts (suffix preserved → no overwrite):
python scripts/train.py --config configs/sparse_view_ssl_mv.yaml --n-views 3
python scripts/train.py --config configs/sparse_view_ssl_mv.yaml --n-views 9

# (C2) SSL mv + strict occlusion check (ablation: over-rejects at n≤3).
python scripts/train.py --config configs/sparse_view_ssl_mv_v2.yaml --n-views 6

# (C3) **Recommended SSL setting.** mv + sparse-view-friendly occlusion
# check (τ=0.10, occlusion_start_iter=3000, alpha_thresh=0.3).
# +2.43 dB vs baseline at n=6, tied with v1 at n=3 with better LPIPS.
python scripts/train.py --config configs/sparse_view_ssl_mv_v3.yaml --n-views 3
python scripts/train.py --config configs/sparse_view_ssl_mv_v3.yaml --n-views 6
```

A standalone CUDA smoke test that doesn't even need a dataset:

```powershell
python scripts/smoke_test.py
```

Outputs land in `outputs/<run_name>/`:

```
outputs/lego_n6/
    ckpts/        # *.pt
    renders/      # PNGs of held-out test views
    tb/           # TensorBoard logs
    config.yaml   # frozen copy of the config used
```

## 6. Repo layout

```
sparse_gs/
├── configs/
│   ├── base.yaml                # shared defaults
│   ├── sparse_view.yaml         # baseline: NeRF-Synthetic lego, n_views=6
│   ├── sparse_view_ssl.yaml     # SSL: pseudo_view + ema_teacher (modular, neutral on vanilla 3DGS)
│   ├── sparse_view_ssl_mv.yaml  # SSL: multiview_photo v1 (no occlusion check — ablation)
│   ├── sparse_view_ssl_mv_v2.yaml # SSL: v1 + strict occlusion check (ablation, over-rejects @ n=3)
│   ├── sparse_view_ssl_mv_v3.yaml # SSL: v1 + sparse-view-friendly occlusion check (**recommended**)
│   └── sparse_view_mini.yaml    # CPU-friendly procedural cube smoke test
├── sparse_gs/
│   ├── datasets/
│   │   ├── nerf_synthetic.py    # transforms_*.json loader, OpenCV viewmats
│   │   ├── sparse_sampler.py    # deterministic train-view subsampling
│   │   └── pseudo_pose.py       # SLERP-based novel-pose sampling for SSL
│   ├── models/
│   │   ├── gaussians.py         # GaussianModel (means/scales/quats/opacities/sh0/shN)
│   │   └── ema.py               # snapshot/EMA teacher (densify-aware)
│   ├── rendering/
│   │   └── gsplat_renderer.py   # thin gsplat.rasterization wrapper (with detach_geometry path)
│   ├── strategies/
│   │   └── densify.py           # gsplat.DefaultStrategy + per-param optimizers
│   ├── losses/
│   │   ├── photometric.py       # L1 + DSSIM
│   │   └── ssl.py               # SSL bank: multiview_photo (working), pseudo_view, ema_teacher, ...
│   ├── trainer/
│   │   └── trainer.py           # training/eval loop, teacher build & update
│   └── utils/
│       ├── config.py
│       ├── metrics.py           # PSNR, SSIM
│       └── io.py
├── scripts/
│   ├── train.py                 # `--n-views N` for sparse-view sweeps
│   ├── render.py
│   ├── smoke_test.py            # standalone CUDA smoke test
│   └── make_mini_dataset.py
├── outputs/
│   ├── baseline_lego.md         # baseline numbers (n=3/6/9/12)
│   └── ssl_lego.md              # SSL ablations + final headline result
└── README.md
```

## 7. Where the SSL hooks live

All self-supervised losses are added in **one** place:
`sparse_gs/losses/ssl.py::SSLLossBank`.

The trainer calls

```python
ssl_loss, ssl_logs = self.ssl_bank(
    step=step,
    rendered=rendered,            # dict: rgb, depth, alpha, info
    gt_rgb=gt_rgb,
    camera=camera,                # current train view
    pose_pool=self.train_cameras, # for sampling pseudo-views / neighbors
    teacher=self.teacher,         # optional EMA copy, lazily built
    gaussians=self.gaussians,
    renderer=self.renderer,
    background=self.background,
)
loss = base_loss + ssl_loss
```

To add a new SSL term, register it in `SSLLossBank.LOSSES` and toggle it
from the YAML config, e.g.:

```yaml
ssl:
  multiview_photo: { enabled: true,  weight: 0.10, n_neighbors: 1 }
  pseudo_view:     { enabled: false, weight: 0.05, detach_geometry: true }
  ema_teacher:     { enabled: false, weight: 0.02, snapshot_every: 2000 }
  depth_consist:   { enabled: false }
  feature:         { enabled: false }
```

### 7.1 Critical engineering note: gsplat densify & SSL gradients

`gsplat.DefaultStrategy.step_post_backward` reads `info["means2d"].grad`
of the **main forward** to drive split / duplicate. **Any** SSL loss
whose gradient flows back through that *same* `info` will inflate
`grad2d` and cause the Gaussian count to explode. We therefore offer
two safe patterns and use both:

1. **Separate forward + ignored info** (used by `multiview_photo`).
   The SSL loss does its own `renderer.render(...)`; the resulting
   `info` is never handed to the strategy. Geometry parameters still
   receive the SSL gradient via standard autograd, but the strategy's
   density-control state is unaffected.
2. **`detach_geometry=True`** (used by `pseudo_view` and
   `ema_teacher`). The renderer detaches `means / scales / quats`
   before rasterization, so the SSL branch only updates colors and
   opacities. Densify is unaffected; geometry is unaffected; only
   appearance is regularized.

See `outputs/ssl_lego.md` for the ablation that motivated this.

## 8. What is **not** done yet

- ❌ **External-prior SSL.** `multiview_photo` already gives +0.9 / +3.1 / +2.6 dB
  at n=3/6/9 without any external model. The next big lever is monocular
  depth pseudo-labels (Marigold / Depth-Anything-V2) and DINOv2 feature
  consistency, both already wired as `depth_consist` / `feature` stubs.
- ✅ **Occlusion-aware reprojection.** `multiview_photo` supports a
  forward-backward depth consistency check (`occlusion_check: true`).
  Student depth is rendered from the neighbor view (no-grad,
  detach_geometry=True) and compared against the warped z; pixels with
  `|z_b − D_b(uv)| / z_b > τ` or low cam_b alpha are dropped.
  Two configs are shipped: `_mv_v2.yaml` (strict τ=0.05) as an ablation
  and `_mv_v3.yaml` (loose τ=0.10 + `occlusion_start_iter=3000` +
  `occlusion_alpha_thresh=0.3`) as the recommended setting. See
  `PROJECT_STATUS.md §3.1` for the full comparison table.
- ❌ PatchGAN discriminator + adversarial loss (out of scope for sparse-view).
- ❌ COLMAP / MipNeRF360 / DTU loaders (only NeRF-Synthetic so far).
- ❌ Mesh extraction / 2DGS / TSDF (proposal mentioned SparseSurf —
  out of scope for the photometric baseline; volumetric 3DGS only).
- ✅ **LPIPS metric.** Reported automatically alongside PSNR/SSIM when
  the `lpips` PyPI package is installed (VGG backbone). Toggle with
  `eval.lpips: false` in the config. Silently skipped if the package
  (or its cached weights) is not available, so existing runs do not
  break.

## 9. Known issues / assumptions

1. **First-call JIT compile.** `gsplat.rasterization` triggers ninja to
   build CUDA kernels the first time it is invoked. On Windows + RTX 5090
   (sm_120) this takes ~2 min and needs a working MSVC + CUDA 12.8/13.x
   toolkit. ``scripts/_bootstrap.py`` activates the latest installed VS
   automatically (vcvars64), prepends user-site ``Scripts/`` (where
   ``ninja.exe`` lives) to ``PATH``, sets
   ``NVCC_PREPEND_FLAGS=-allow-unsupported-compiler -Xcompiler /Zc:preprocessor``
   and applies a one-line patch to PyTorch's
   ``c10/cuda/CUDACachingAllocator.h`` (Windows ``rpcndr.h`` redefines
   ``small`` as ``char``, which collides with a parameter name). All of
   this is no-op on Linux. Set
   ``SPARSE_GS_SKIP_BOOTSTRAP=1`` if you've already activated a Developer
   Command Prompt yourself.
2. **Camera convention.** We use **OpenCV** (X right, Y down, Z forward)
   throughout. NeRF-Synthetic stores OpenGL poses (Y up, Z back), so the
   loader applies a fixed `diag(1, -1, -1)` flip on the camera-to-world
   rotation. See `datasets/nerf_synthetic.py`.
3. **White background.** NeRF-Synthetic images are RGBA on a transparent
   background. We composite onto white at load time (the standard
   convention) and pass `backgrounds=white` to `rasterization`.
4. **Per-parameter optimizers.** `DefaultStrategy.step_post_backward`
   mutates optimizer state in-place during densify/prune, so each Gaussian
   parameter (means, scales, quats, opacities, sh0, shN) has its **own**
   `Adam`. This is the pattern used by gsplat's own examples and is
   required — sharing one optimizer across all params will silently break
   densify.
5. **Sparse-view determinism.** Train-view subsampling is fully determined
   by `data.train_view_ids` (explicit list) or `data.n_train_views` + a
   fixed RNG seed, so different runs see exactly the same views.
6. **`packed=False` by default.** gsplat 1.5.3's packed path has a shape
   mismatch when `render_mode='RGB+ED'` and ``backgrounds`` is non-None;
   the assert ``backgrounds.shape == image_dims + (channels,)`` fails
   because ``image_dims`` collapses to ``()`` in packed mode while the
   input still carries a camera batch axis. We default to ``packed:false``
   to dodge this; for our sparse-view (≤16 cameras) workload it is
   indistinguishable in throughput.
```
