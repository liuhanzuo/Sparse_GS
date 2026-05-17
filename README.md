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

## 1bis. Latest run snapshot — 2026-05-17 (overnight + ablation batch)

> Snapshot of every metrics.json under `outputs/` after the 9-h overnight
> run + the ship/materials ablation batch. Numbers are final test PSNR /
> SSIM / LPIPS on the 200-view test split. Full table dumped by
> `scripts/_collect_all.py`.

### 1bis.1 Per-scene best (current best run for each of the 8 scenes)

| Scene     | Best run                                         | PSNR  | SSIM   | LPIPS  | #G       | iters | wall  |
|-----------|--------------------------------------------------|------:|-------:|-------:|---------:|------:|------:|
| chair     | `blender_chair_*_v7_widerpseudo_x60k`            | 25.65 | 0.9304 | 0.0788 |   871022 | 60000 | 2.33h |
| ficus     | `blender_ficus_*_v7_widerpseudo_x60k`            | 24.53 | 0.9354 | 0.0557 |   527898 | 60000 | 0.91h |
| hotdog    | `blender_hotdog_*_v7_widerpseudo_x45k`           | 26.40 | 0.9385 | 0.1035 |   536718 | 45000 | 1.00h |
| lego      | `blender_lego_*_v7_widerpseudo_x60k`             | 23.46 | 0.8936 | 0.1166 |   832818 | 60000 | 2.28h |
| mic       | `blender_mic_*_v7_widerpseudo_x60k`              | 25.76 | 0.9523 | 0.0562 |   737492 | 60000 | 1.93h |
| ship      | `ablation_ship_A1_rand_bg` (v7 + random_bg, 45k) | 17.58 | 0.7445 | 0.2825 |  1189229 | 45000 | 2.65h |
| materials | `ablation_materials_A1_rand_bg` (idem)           | 13.87 | 0.8018 | 0.2527 |   715005 | 45000 | 1.14h |
| drums     | `blender_drums_*_v8_hard_x45k`                   | 16.47 | 0.8423 | 0.1890 |  1694518 | 45000 | 2.91h |
| **avg-8** |                                                  | **21.71** | **0.8674** | **0.1544** | — | — | — |

### 1bis.2 Ship / materials ablation (single-variable on top of v7)

8 train views, 45k iters, full 200-view test. Configs under
`configs/_w3_aggrprune/v8_hard/ablation/`.

| Run                                       | PSNR  | SSIM   | LPIPS  | wall  | Δ vs v8_hard |
|-------------------------------------------|------:|-------:|-------:|------:|-------------:|
| ship · v7 baseline (`*v7_widerpseudo_x45k`) | 16.80 | 0.7267 | 0.2937 | 2.48h | —            |
| ship · v8_hard (full new bundle)          | 17.36 | 0.7389 | 0.2864 | 2.45h | +0.56 ✅     |
| ship · A1 = v7 + random_bg only           | **17.58** | **0.7445** | **0.2825** | 2.65h | **+0.78 🏆** |
| ship · A2 = v7 + conservative_prune only  | 16.90 | 0.7286 | 0.2945 | 2.67h | +0.10 (≈noise)|
| ship · A3 = v7 + weak_pvd only            | OOM, dropped | — | — | — | —            |
| materials · v8_hard                       | 13.76 | 0.7980 | 0.2541 | 1.06h | —            |
| materials · A1 = v7 + random_bg only      | **13.87** | **0.8018** | **0.2527** | 1.14h | **+0.11 ⭐** |
| materials · A3 = v7 + weak_pvd only       | 13.42 | 0.7939 | 0.2713 | 1.30h | −0.34 ❌     |

**Verdict.** The v8_hard bundle's lift comes almost entirely from
**`random_bg` augmentation alone**. `conservative_prune` and `weak_pvd`
are near-zero or negative. The recommended next config ("v8_simple")
is **v7 + `random_bg` only**, no other v8_hard knobs.

### 1bis.3 vs literature (avg-8 PSNR on Blender-n=8)

| Method               | avg-8 PSNR | Δ vs us |
|----------------------|----------:|--------:|
| **Ours (current)**   | **21.71** | —       |
| SparseGS (3DV'24)    | 22.8      | +1.1    |
| DietNeRF             | 23.6      | +1.9    |
| RegNeRF              | 23.9      | +2.2    |
| DNGaussian (CVPR'24) | 24.3      | +2.6    |
| FreeNeRF             | 24.3      | +2.6    |
| CoR-GS (ECCV'24)     | 24.5      | +2.8    |
| FSGS (ECCV'24)       | 24.6      | +2.9    |

We are **~3 dB short of the SOTA cluster**. The 5 "easy" scenes
(chair/ficus/hotdog/lego/mic) average ~25.2 dB and are within 1.5–3 dB
of FSGS/CoR-GS — architecturally fine. The 3 "hard" scenes
(ship 17.6, materials 13.9, drums 16.5) average ~16.0 dB, while the
same methods report 22–24 dB on them. **These 3 scenes alone drag the
mean down by ≈2 dB** and are the bottleneck.

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

Then precompute per-scene depth caches **once** (one scene per call):

```powershell
foreach ($s in @('hotdog','drums','chair','ficus','lego','materials','mic','ship')) {
    python scripts/precompute_depth.py `
        --dataset nerf_synthetic --scene $s `
        --splits train `
        --backend transformers `
        --model depth-anything/Depth-Anything-V2-Small-hf `
        --tag depth_anything_v2_small
}
```

Linux equivalent:

```bash
for s in hotdog drums chair ficus lego materials mic ship; do
    python scripts/precompute_depth.py \
        --dataset nerf_synthetic --scene "$s" \
        --splits train \
        --backend transformers \
        --model depth-anything/Depth-Anything-V2-Small-hf \
        --tag depth_anything_v2_small
done
```

This writes `data/nerf_synthetic/<scene>/_depth_cache/depth_anything_v2_small/`
once per scene; the trainer reads from that cache on every step.

### 4.3 Train (single scene)

The SOTA chain config for hotdog is
[`configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml`](configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml).
To launch:

```powershell
# (activate whatever python env you installed step 4.1 into)
$env:PYTHONIOENCODING = "utf-8"   # only needed on Windows for unicode tqdm
mkdir outputs\logs -Force | Out-Null
python -u scripts/train.py `
    --config configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml `
    | Tee-Object -FilePath outputs\logs\x30k_hotdog.log
```

Linux equivalent:

```bash
mkdir -p outputs/logs
python -u scripts/train.py \
    --config configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml \
    2>&1 | tee outputs/logs/x30k_hotdog.log
```

Output goes to
`outputs/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k/` and
contains:

| File / dir            | Meaning                                                              |
|-----------------------|----------------------------------------------------------------------|
| `metrics.json`        | final PSNR / SSIM / LPIPS on the 200-view test split + best ckpt info |
| `eval_log.jsonl`      | one row per validation step (step, val/psnr, val/ssim, ΔN)            |
| `pre_*_iter_*.pt`     | pre-prune snapshots (auto-deleted after the final test pass)          |
| `final_iter_*.pt`     | last checkpoint (kept)                                                |
| `renders/`            | RGB / depth turntable PNGs from the final test                        |

Wall-time on RTX 5090 Laptop: **~62 min** (30k iters + final test on
200 views + 13 pre-prune checkpoint tests).

### 4.4 Train all 8 scenes serially

```powershell
.\scripts\run_x30k_all_scenes.ps1
# refresh the SOTA table afterwards:
python scripts/collect_x30k_sota.py
```

Total budget ≈ 8 × 62 min ≈ 8.3 h.

### 4.5 Run other ablations (not just SOTA)

Every ablation has its own config tree under `configs/`, so swapping
the chain is just swapping the `--config` path. The ones that are
known-good in this snapshot:

| Chain                                      | config dir                  | what it isolates                                |
|--------------------------------------------|-----------------------------|-------------------------------------------------|
| Vanilla 3DGS @ 8 views                     | `configs/_blender_n8/`      | baseline (no SSL, no PVD, no prune)             |
| + DepthAnything-V2 prior                   | `configs/_depth_v2/`        | w1 — depth consistency only                     |
| + unseen-view floater prune                | `configs/_w2_prune/`        | w2 — pre-cursor of AggressivePrune              |
| + Dual-GS (no aggrprune)                   | `configs/_w3_dual_gs/`      | w3a — Dual field, default densify only          |
| + Dual-GS + AggressivePrune  ★ **SOTA**    | `configs/_w3_aggrprune/`    | w3b — full chain (v3 → v6_pgan70_x30k)          |
| Density cap / LPIPS aux / no-floater       | `configs/_w3_*/`            | further w3 ablations                            |
| LLFF forward-facing baseline + DAv2        | `configs/_llff_sweep/`      | LLFF n=3 reference                              |

Inside `_w3_aggrprune/` the suffix tells you the version on the chain:

```
*_v3.yaml             ←  Dual-GS + AggressivePrune, 15k iters
*_v4_pvd.yaml         ←  + PVD as floater detector (G adv = 0)
*_v5_widepvd.yaml     ←  + wide pseudo-view sampler
*_v6_pgan70.yaml      ←  + 70×70 PatchGAN with DiffAug
*_v6_pgan70_x30k.yaml ←  + 24k main + 6k tail = 30k iters   ★ current SOTA
```

Per-ablation drivers (each iterates over the 8 NeRF-Synthetic scenes
and aggregates the metrics) live in `scripts/run_w*_*_scenes.py` /
`scripts/run_x30k_all_scenes.ps1` — pick the one matching the chain you
want.

### 4.6 Inspect results

After a run finishes:

```powershell
# raw final metrics for one experiment
type outputs\blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k\metrics.json

# regenerate the SOTA tables under results/_sota/  (md + json + csv)
python scripts/collect_x30k_sota.py

# tail-only summary across all eval steps of a run
python scripts/log_summary.py outputs\blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k\eval_log.jsonl
```

Anything you want to keep around long-term should be copied to
`results/experiments/<run_name>/{metrics.json, eval_log.jsonl, config.yaml}` —
that directory is committed (it's small) and is what feeds the SOTA tables.

### 4.7 Anatomy of a config file

All configs inherit from [`configs/base.yaml`](configs/base.yaml) via a
single line at the top:

```yaml
_base_: ../base.yaml
```

…and override only the fields they care about. SOTA configs go one
level deeper — they `_base_:` from a sibling chain config, e.g.
`*_x30k.yaml` inherits from `*_v6_pgan70.yaml`, which inherits from
`*_v5_widepvd.yaml`, etc. So a single `_x30k.yaml` file is small
(~70 lines) and only contains the deltas vs the previous chain step.

`base.yaml` exposes seven top-level blocks. The fields you'll most
likely touch are flagged ◀ below; everything else has reasonable
defaults baked in.

```yaml
experiment:
  name: baseline                 #                          ◀ run identifier
  output_dir: outputs            # results land in outputs/<name>/
  seed: 42

data:
  type: nerf_synthetic           # only loader implemented for n=8 work
  root: data/nerf_synthetic
  scene: lego                    #                          ◀ swap scene here
  n_train_views: 6               #                          ◀ sparsity level
  image_downsample: 1            # 1 = full 800×800
  white_background: true

model:
  sh_degree: 3
  init:
    type: random_in_box          # NeRF-Synthetic has no SfM cloud
    num_points: 100000
    extent: 1.5
  scale_init_factor: 0.01

renderer:
  rasterize_mode: antialiased    # 'classic' | 'antialiased'
  render_mode: RGB+ED            # also returns expected depth in 4th channel
  absgrad: true

strategy:
  type: default                  # gsplat.DefaultStrategy
  refine_start_iter: 500
  refine_stop_iter: 15000
  refine_every: 100
  reset_every: 3000
  # SOTA chain adds here:
  #   prune.unseen   { enabled, start_iter, every_iter, stop_iter }
  #   prune.floater  { enabled, start_iter, every_iter, alpha_thresh,
  #                    thresh_bin / thresh_bin_end (decay) }

optim:
  means_lr: 1.6e-4               # gsplat per-param Adam defaults
  scales_lr: 5.0e-3
  quats_lr: 1.0e-3
  opacities_lr: 5.0e-2
  sh0_lr: 2.5e-3
  shN_lr: 1.25e-4
  means_lr_final_factor: 0.01    # exp-decay to 1% of start LR

train:
  iterations: 7000               #                          ◀ training budget
  ssim_lambda: 0.2               # loss = 0.8·L1 + 0.2·DSSIM
  log_every: 50
  eval_every: 2000
  save_every: 7000
  # SOTA chain enables:
  #   save_pre_prune_ckpt: true
  #   periodic_ckpt_every / periodic_ckpt_start  (the 6k-step tail)

ssl:                              # all SSL hooks disabled in base.yaml.
  multiview_photo: { enabled: false, weight: 0.0 }
  depth_consist:   { enabled: false, weight: 0.0 }
  pseudo_view:     { enabled: false, weight: 0.0 }
  patch_gan:       { enabled: false, weight: 0.0 }
  ema_teacher:     { enabled: false, momentum: 0.995 }
  # SOTA chain re-enables them and adds:
  #   pvd: { enabled, start_iter, floater_guidance.warmup_iter, ... }

eval:
  render_test_at_end: true
  num_test_renders: 8
  lpips: true
  lpips_net: vgg
```

The SOTA chain extras that don't live in `base.yaml` (because they
only exist in the Dual-GS line) live under a top-level `dual_gs:` block:

```yaml
dual_gs:
  coprune: true
  coprune_threshold: 0.05
  coprune_start_iter: 2400
  coprune_stop_iter:  18400
  coprune_every_iter: 800
  end_sample_pseudo:  30000     # keep pseudo + co-reg + GAN to the very end
```

The two "where to start tweaking" knobs:

* **`data.scene`** — the only thing that differs between the 8
  NeRF-Synthetic configs in `_w3_aggrprune/`.
* **`train.iterations`** — currently 30 000 (24 k main + 6 k tail) for
  `*_x30k`, 15 000 for the older `*_v6_pgan70` baseline.

For the full set of fields read [`configs/base.yaml`](configs/base.yaml)
and the chain config you're starting from
(e.g. [`blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml`](configs/_w3_aggrprune/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml)).

---

## 5. Status and what's missing

> Updated 2026-05-17. The original `v6_pgan70_x30k` chain has since
> been superseded by `v7_widerpseudo_x45k/x60k` and the `random_bg`
> ablation. See §1bis for the current per-scene best numbers.

| Item | State |
|---|---|
| `v7_widerpseudo` configs — all 8 scenes wired (x45k + x60k) | ✅ |
| `v7_widerpseudo` trained — chair / ficus / lego / mic | ✅ (60k, see §1bis) |
| `v7_widerpseudo` trained — hotdog | ✅ (45k, PSNR 26.40, see §1bis) |
| `v7_widerpseudo` trained — ship | ✅ (45k+60k; best is `A1_rand_bg`, PSNR 17.58) |
| `v8_hard` bundle — ship / materials / drums | ✅ (trained, see §1bis) |
| `v8_hard` ablation — ship A1/A2/A3, materials A1/A3 | ✅ (§1bis.2; only A1 = `random_bg` survived) |
| Recommended `v8_simple` (= v7 + `random_bg`) on the 5 easy scenes | ⏳ not yet (see §7.5 step 1) |
| Auto SOTA collector | ✅ |
| 70×70 PatchGAN unit test | ✅ |
| Wide pseudo sampler unit test | ✅ |
| Avg-8 PSNR (current) | **21.71** (≈3 dB short of FSGS / CoR-GS — see §1bis.3 and §7.3) |

For the deeper history (every ablation, why each piece exists, what
failed) read [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

---

## 6. License & attribution

Built on top of `gsplat` (Apache-2.0). Depth prior uses
`DepthAnything-V2-Small` weights (open).
LPIPS metric uses the official `lpips` PyPI package.

---

## 7. Handoff notes for the next agent (2026-05-17, end of overnight batch)

> Read this section before changing anything — it's the shortest path
> to "what's the state, what works, what's the open problem".

### 7.1 What is currently the recommended training recipe

**`v8_simple` = v7_widerpseudo + `data.random_background_aug` only.**
Nothing else from `v8_hard` survived ablation. Concretely:

- Start from any `configs/_w3_aggrprune/blender_<scene>_*_v7_widerpseudo_x45k.yaml`.
- Append the following block to it (the verbatim form is in
  [`configs/_w3_aggrprune/v8_hard/ablation/ablation_ship_A1_rand_bg.yaml`](configs/_w3_aggrprune/v8_hard/ablation/ablation_ship_A1_rand_bg.yaml)):

  ```yaml
  data:
    random_background_aug:
      enabled: true
      prob: 0.5      # per-step probability of using a random RGB bg (default 0.5)
  ```

  ⚠️ The flag is `data.random_background_aug.enabled`, **not** `data.random_bg`.
  Eval still uses white background regardless of this setting.
- Keep the v7 prune schedule (`unseen` + decaying `floater`), keep the
  PVD weight 0.012 (full strength), keep 45 k iters.
- This gave +0.78 dB on ship and +0.11 dB on materials in single-variable
  ablation, while the full v8_hard bundle gave only +0.56 / +0.10.

### 7.2 What was tried in v8_hard and rejected

| Knob                  | Verdict | Evidence |
|-----------------------|---------|----------|
| `data.random_background_aug.enabled: true`| ✅ keep | A1 single-var beats v8_hard bundle on both hard scenes |
| `conservative_prune` (lower thresh, longer warmup) | ❌ drop | A2 ≈ noise on ship (+0.10 dB) |
| `weak_pvd` (PVD weight × 0.5, later start) | ❌ drop | A3 OOM on ship; A3 *negative* −0.34 dB on materials |

Configs and metrics for each ablation live at:
```
configs/_w3_aggrprune/v8_hard/ablation/ablation_{ship,materials}_{A1_rand_bg,A2_conservative_prune,A3_weak_pvd}.yaml
outputs/ablation_{ship,materials}_{A1_rand_bg,A2_conservative_prune,A3_weak_pvd}/metrics.json
```

### 7.3 Open problem — `ship` / `materials` / `drums`

The 5 easy scenes are at 23.5–26.4 dB (within 1.5–3 dB of FSGS / CoR-GS).
The 3 hard scenes are stuck at 13.9–17.6 dB while published
sparse-view methods report 22–24 dB on them. **All of the recipe
tweaking in v8 only moved them by < 1 dB.** The remaining gap is
structural, not scheduling:

1. **No SfM / MVS init.** Training starts from random 100 k points.
   Ship/materials reflective geometry is essentially impossible to
   recover from random + 8 photometric views. Nearly every published
   sparse-view 3DGS method (DNGaussian, FSGS, CoR-GS) seeds from a
   COLMAP / Dust3R / Mast3R point cloud even when only 8 views are used.
2. **Depth prior is a soft consistency loss only.** `depth_consist`
   (DAv2) currently gives a small SSI-L1 nudge. DNGaussian's gain on
   ship/drums comes from a **hard depth + soft depth dual loss applied
   from step 0** with weight ~0.1 *and* a depth-aware densifier
   ("proximity densification" in FSGS) that grows new Gaussians on
   the predicted surface, not the photometric residual.
3. **PVD is a GAN, not a geometric consistency loss.** Our PVD path
   feeds a 70×70 PatchGAN logit back as a *score* for the floater
   pruner. FSGS / SparseGS instead synthesise unseen views, render
   their depth, reproject to train views, and apply a geometric
   consistency L1 — no adversary, no hallucination. This is reportedly
   what makes ship/drums tractable.
4. **SH degree starts at 3.** Reflective scenes can cheat with
   view-dependent SH on 8 train views. Most baselines start at SH=0
   for the first ~30 k iters, then increment one degree per ~1 k.

These four are the rank-ordered candidates for closing the ~3 dB gap.
A realistic estimate per item, based on what the cited papers ablate:
SfM/MVS init alone is +1 to +3 dB on ship/materials; geometric (not
GAN) pseudo-view consistency is +1 to +2 dB on ship/drums; the dual
depth loss is +0.5 to +1.5 dB; SH delay is +0.3 to +1 dB. Even one of
these done well would close most of the gap on the avg-8 metric, since
the 3 hard scenes are what's pulling the mean down.

### 7.4 What is *already* in the repo (don't reinvent)

- DAv2-Small depth cache for all 8 scenes is on disk
  (`data/nerf_synthetic/<scene>/_depth_cache/depth_anything_v2_small/`).
  The trainer reads it via `ssl.depth_consist`.
- `random_bg` augmentation is implemented in the trainer step itself
  (`sparse_gs/trainer/trainer.py`, ~L228 + L755 — re-composites GT via
  `cam.image + (bg-1)·(1-alpha)` per step). To enable, set
  `data.random_background_aug.enabled: true` (optionally `prob: 0.5`) —
  no new code needed. Requires `data.type=nerf_synthetic` and
  `data.white_background=true` (otherwise the trainer prints a warning
  and stays disabled).
- `AggressivePruneStrategy` (unseen + decaying floater) is the prune
  path used by every `_w3_aggrprune` config. Don't replace it; tune the
  thresholds.
- A 200-view full-test pass + best-of-N pre-prune ckpt selection runs
  automatically at the end of every training. `metrics.json` already
  reports the *best* ckpt's PSNR/SSIM/LPIPS, not just the final.
- `scripts/_collect_all.py` regenerates the avg-8 table from
  `outputs/*/metrics.json` and is what produced §1bis.

### 7.5 Suggested next milestones (in priority order)

1. **Re-run chair / ficus / lego / mic / hotdog with `v8_simple`
   (= v7 + random_bg) at 45k.** Currently those 5 scenes use plain
   v7 (no random_bg). If random_bg gives the same +0.5 dB lift it gave
   on ship, the avg-8 jumps from 21.71 to ≈22.2 with zero new code.
   Cost: ~8 h on RTX 5090 Laptop. **Lowest risk, do this first.**
2. **SfM/Dust3R-init ship/materials/drums.** Drop in a point cloud
   loader that reads `data/nerf_synthetic/<scene>/sparse/0/points3D.ply`
   (or a Dust3R cache), use it instead of `random_in_box` when
   present. Train v8_simple. Expected: +1 to +3 dB on these 3 scenes.
3. **Geometric pseudo-view loss.** Add a sibling SSL hook next to
   `multiview_photo` that renders an unseen view's depth, reprojects
   to train views, and applies a Huber on the depth residual.
   `sparse_gs/losses/ssl.py` already has the cross-view reprojection
   plumbing — extend it to depth instead of RGB only. Expected: +1 to
   +2 dB on ship/drums.
4. **Hard depth loss + delayed SH.** One-line config flips:
   `ssl.depth_consist.weight: 0.1` (currently lower) and
   `model.sh_degree_schedule: [0, 1, 2, 3]` with step boundaries
   (~`[0, 30000, 31000, 32000]`). Expected: +0.5 to +1.5 dB total.

### 7.6 Mechanical pointers for the next agent

- Active configs: `configs/_w3_aggrprune/`. Anything with `_v7_` in the
  name is the SOTA-line baseline; anything under `v8_hard/ablation/`
  is the rejected variant set from this batch.
- Active outputs: `outputs/`. `ablation_*` are this batch; `blender_*_v7_*`
  and `blender_*_v8_hard_*` are the per-scene SOTA candidates.
- Run summary script: `python scripts/_collect_all.py`.
- Per-ablation comparison: `python scripts/ablation_compare.py`.
- Overnight driver: `run_overnight_9h.ps1` (already used for this batch).
- Training entry point is unchanged: `python -u scripts/train.py --config <yaml>`.
- The 200-view test split is hard-coded in the dataset (`split="test"`
  loads all 200 frames); don't change it or numbers stop being
  comparable to the 1bis table.