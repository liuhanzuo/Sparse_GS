# Sparse_GS — Sparse-View 3D Gaussian Splatting (n=8 NeRF-Synthetic)

A self-contained, modular **sparse-view 3DGS** pipeline built on
[`gsplat`](https://github.com/nerfstudio-project/gsplat). The current
SOTA chain is **`v6_pgan70_x30k`** (= Dual-GS + aggressive prune + wide
pseudo-view sampling + DepthAnything-V2 depth prior + 70×70 PatchGAN
with DiffAug + 30k iterations w/ a 6k-step finetune tail).

> This repo is a **clean snapshot** of the working `sparse_gs` project,
> trimmed to the SOTA chain only. Heavy artefacts (datasets, checkpoints,
> per-experiment renders, stderr logs) are intentionally **not** included.

---

## 1. Current SOTA (NeRF-Synthetic, n=8 train views, 200-view test split)

Single-scene reference (full numbers in
[`results/_sota/blender_n8_x30k_sota.md`](results/_sota/blender_n8_x30k_sota.md)):

| scene  | 15k PSNR | **30k PSNR** | ΔPSNR | 30k SSIM | 30k LPIPS | #G      | wall  |
|--------|---------:|-------------:|------:|---------:|----------:|--------:|------:|
| hotdog | 24.747   | **25.759**   | +1.01 | 0.9378   | 0.1026    | 457,643 | 61 min|
| drums  | 15.707   | _running_    | —     | 0.8256   | 0.2048    | —       | —     |
| chair / ficus / lego / materials / mic / ship | _queued_ | — | — | — | — | — | — |

**vs literature (Blender avg-8 n=8 PSNR):**
FSGS 24.6 · CoR-GS 24.5 · DNGaussian 24.3 · FreeNeRF 24.3 · RegNeRF 23.9
· DietNeRF 23.6 · SparseGS 22.8. **Ours (hotdog only): 25.76.**

The remaining 7 scenes are wired up (configs + run script ready) but
have not yet been trained in this snapshot — see §5.

---

## 2. Method overview (what makes `v6_pgan70_x30k` work)

```
        NeRF-Synthetic (8 train views)
                   │
          ┌────────┴────────┐ Dual GS field (seed 42 / 43)
          ▼                 ▼
       gs0 (seed 42)     gs1 (seed 43)            ── coreg + coprune
          │                                         (every 500 iters)
          ▼
   gsplat.rasterization (RGB + Expected-Depth)
          │
          ├── L1 + 0.2·DSSIM on 8 GT views
          ├── multiview_photo (cross-view reproj, occlusion-gated)
          ├── depth_consist  (DepthAnything-V2-Small, SSI L1)
          ├── pseudo_view    (wide sampler: 40% interp / 30% extrap / 30% sphere)
          └── PVD: 70×70 PatchGAN + DiffAug, hinge G/D loss
                   │ floater_guidance: D's fake-logit gates the prune
                   ▼
   DefaultStrategy   +   AggressivePruneStrategy
   (densify/split)       (unseen + alpha-floater w/ decaying threshold)
                   │
                   ▼
   30k iters · LR exp-decay · pre-prune & periodic ckpts (1500-step in tail)
                   │
                   ▼
   final test on 200 views — best ckpt auto-selected from 13 snapshots
```

Key engineering rules (must keep):

1. **OpenCV cameras everywhere** (`diag(1,-1,-1)` flip on NeRF-Synthetic load).
2. **Per-parameter Adams** — densify mutates optimizer state in-place.
3. **SSL gradient ≠ densify state.** Two safe patterns enforced by
   the renderer API: `detach_geometry=True` for color/opacity-only
   gradients, or a **separate** forward whose `info` is never handed
   to the strategy. Violating this 2× the Gaussian count and dropped
   PSNR from 15.7 → 10.1 in early ablations.
4. **PVD never returns gradients to the strategy.** It uses pattern (1)
   + the AggressivePruneStrategy reads D's logits as a *score*, not as
   a loss term contributing to `means2d.grad`.

---

## 3. Repo layout

```
Sparse_GS_/
├── README.md                           ← you are here
├── .gitignore
├── configs/                           (146 yaml files across 11 ablation lines)
│   ├── base.yaml                       ← shared defaults
│   ├── _blender_n8/                    ← w0 — vanilla 3DGS @ 8 views
│   ├── _depth_v2/                      ← w1 — depth-anything-v2 prior
│   ├── _gan/                           ← early GAN ablation (small / full)
│   ├── _llff_sweep/                    ← LLFF baselines + DAv2 (forward-facing)
│   ├── _w2sota_view/                   ← w2 — sota-view sampling baseline
│   ├── _w2_prune/                      ← w2 — unseen-view floater pruning
│   ├── _w3_dual_gs/                    ← w3 — Dual-GS only (no aggrprune)
│   ├── _w3_aggrprune/  ★               ← w3 — Dual-GS + AggressivePrune (SOTA chain)
│   │   ├── *_v3.yaml                   ← long training (15k)
│   │   ├── *_v4_pvd.yaml               ← + PVD as floater detector (G adv=0)
│   │   ├── *_v5_widepvd.yaml           ← + wide pseudo sampler
│   │   ├── *_v6_pgan70.yaml            ← + 70×70 PatchGAN + DiffAug
│   │   └── *_v6_pgan70_x30k.yaml       ← + 30k iter (24k + 6k tail)  ★ current SOTA
│   ├── _w3_densitycap/                 ← w3 — Gaussian-count ceiling
│   ├── _w3_lpips/                      ← w3 — perceptual aux loss
│   └── _w3_nofloater/                  ← w3 — ablation w/o floater prune
├── sparse_gs/                          ← core library (29 .py files)
│   ├── datasets/                       ← nerf_synthetic, llff, sparse_sampler, pseudo_pose
│   ├── losses/                         ← photo, ssl, pvd, gan, diffaug, perceptual
│   ├── models/                         ← gaussians, ema
│   ├── rendering/                      ← gsplat_renderer (with detach_geometry)
│   ├── strategies/                     ← densify (DefaultStrategy), post_prune (Aggressive)
│   ├── trainer/                        ← trainer.py (62k LoC), dual_gs.py
│   └── utils/                          ← config, depth_prior, io, metrics
├── scripts/                           (64 files — train + analysis + ablation drivers)
│   ├── train.py / render.py / eval_ckpt.py / smoke_test.py
│   ├── precompute_depth.py             ← DAv2 depth cache builder
│   ├── make_mini_dataset.py            ← procedural cube smoke scene
│   ├── download_llff.py                ← LLFF dataset fetcher
│   ├── run_x30k_all_scenes.ps1         ← SOTA driver (all 8 scenes)
│   ├── run_w*_*_scenes.py              ← per-ablation drivers (w2 / w3 / lpips / nofloater / …)
│   ├── aggregate_w*_*.py               ← per-ablation result aggregators
│   ├── collect_x30k_sota.py            ← metrics.json → SOTA tables
│   ├── collect_metrics.py / dump_*_table.py  ← legacy aggregators
│   ├── _smoke_pseudo_wide.py / _smoke_v6_pgan70.py  ← unit tests
│   ├── _peek_eval.py / _b_*.py / b_*.py    ← diagnostic / plotting helpers
│   ├── _gen_*_configs.py               ← per-ablation config generators
│   ├── watchdog_dual_long.py / retry_failed_runs.py / log_summary.py
│   └── (~30 more probe / dump / sweep helpers)
├── results/                           (committed — small, version-controlled)
│   ├── _sota/                          ← auto-generated SOTA tables (md / json / csv)
│   ├── hotdog_x30k/                    ← reference run: metrics + eval_log + config
│   └── experiments/                    ← 137 experiments × {metrics, eval_log, config}, 0.43 MB
├── data/                              (not committed — see docs/DATA_DOWNLOAD.md)
│   ├── nerf_synthetic/<scene>/         ← 3.2 GB (8 scenes + per-scene _depth_cache)
│   ├── nerf_llff_data/<scene>/         ← 2.8 GB (8 scenes + per-scene _depth_cache)
│   └── _cache/nerf_llff_data.zip       ← LLFF download cache
├── models/                            (not committed — see docs/DATA_DOWNLOAD.md)
│   ├── depth_anything_v2_small/        ← ~100 MB (used by SOTA)
│   ├── depth_anything_v2_base/         ← ~390 MB (optional)
│   └── depth_anything_v2_large/        ← ~1.3 GB (optional)
├── third_party/                       (165 MB — comparison baselines, not committed)
│   ├── CoR-GS/  DNGaussian/  FSGS/  SparseGS/
├── docs/
│   ├── PROJECT_STATUS.md               ← single source of truth (history + open issues)
│   ├── DATA_DOWNLOAD.md                ← dataset & weight fetching guide
│   ├── README_legacy.md                ← original sparse_gs README (kept for reference)
│   └── plans/                          ← w1 / w2 design docs (requirements + tasks)
├── tests/                              ← pytest unit tests
├── README.md                           ← (this file)
└── .gitignore
```

---

## 4. Reproduce the hotdog SOTA

### 4.1 Environment

- Windows 11 + PowerShell *or* Linux
- Python 3.14, PyTorch 2.10 + cu128, gsplat 1.5.3
- 24 GB VRAM is enough for NeRF-Synthetic 800×800

```powershell
# from inside Sparse_GS_/

# 1) Install PyTorch first, matching your CUDA driver. We do NOT pin a
#    torch version because the right wheel depends on the GPU. Example:
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Install the rest of the runtime deps (gsplat, lpips, transformers,
#    huggingface_hub, imageio, pyyaml, tqdm, pillow, ...):
pip install -r requirements.txt

# 3) (Optional) extras only needed for dataset downloaders / diagnostic plots:
pip install -r requirements-dev.txt
```

`gsplat` ships a pure-Python wheel; its CUDA kernels are JIT-compiled by
`ninja` on the first call to `gsplat.rasterization` (1–3 min stall the very
first time). If JIT fails on your box, fall back to
`pip install gsplat --no-build-isolation`.

### 4.2 Data & weights

Follow [`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md) to fetch:

1. `data/nerf_synthetic/` (NeRF-Synthetic, 3.2 GB)
2. `models/depth_anything_v2_small/` (DAv2-Small weights, 100 MB)
3. *(optional)* `data/nerf_llff_data/`, larger DAv2 variants

Then precompute per-scene depth caches **once**:

```powershell
python scripts/precompute_depth.py --backend transformers `
    --model-id depth-anything/Depth-Anything-V2-Small-hf `
    --tag depth_anything_v2_small `
    --scenes hotdog drums chair ficus lego materials mic ship `
    --split train --n-views 8
```

### 4.3 Train (single scene)

```powershell
conda activate gaussian_splatting
$env:PYTHONIOENCODING = "utf-8"
python -u scripts/train.py `
    --config configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml `
    | Tee-Object -FilePath outputs\logs\x30k_hotdog.log
```

Wall-time on RTX 5090 Laptop: **~62 min** (training 30k + final test on
200 views + 13 pre-prune checkpoint tests). Output goes to
`outputs/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k/`.

### 4.4 Train all 8 scenes serially

```powershell
.\scripts\run_x30k_all_scenes.ps1
# refresh the SOTA table afterwards:
python scripts/collect_x30k_sota.py
```

Total budget ≈ 8 × 62 min ≈ 8.3 h.

---

## 5. Status and what's missing

| Item | State |
|---|---|
| `v6_pgan70_x30k` config — all 8 scenes wired | ✅ |
| `v6_pgan70_x30k` trained — hotdog | ✅ (PSNR 25.76) |
| `v6_pgan70_x30k` trained — drums | 🟡 (15k version 15.71; 30k queued) |
| `v6_pgan70_x30k` trained — chair / ficus / lego / materials / mic / ship | ⏳ queued |
| Auto SOTA collector | ✅ |
| 70×70 PatchGAN unit test | ✅ |
| Wide pseudo sampler unit test | ✅ |

For the deeper history (every ablation, why each piece exists, what
failed) read [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

---

## 6. License & attribution

Built on top of `gsplat` (Apache-2.0). Depth prior uses
`DepthAnything-V2-Small` weights (open).
LPIPS metric uses the official `lpips` PyPI package.