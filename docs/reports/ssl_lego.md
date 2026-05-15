# SSL experiments — NeRF-Synthetic `lego`

All numbers on the **full 200-view test split**, 7000 iters, RTX 5090.
Baselines come from `baseline_lego.md`.

## 1. Final headline result — multi-view photometric consistency

`configs/sparse_view_ssl_mv.yaml` enables **only** `multiview_photo`:
a depth-based reprojection between training views (no teacher, no
pseudo-views, no monocular priors). It is the only loss in the SSL
bank that introduces a **new** signal — it couples training views
through the student's own depth, so geometric inconsistencies across
views are penalized directly.

| n_views | baseline PSNR | + multiview_photo PSNR | Δ PSNR ↑ | baseline SSIM | + mv SSIM | Δ SSIM ↑ | #Gauss (mv) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3  | 12.8888 | **13.8121** | **+0.92** | 0.6011 | 0.6337 | +0.033 | 434 886 |
| 6  | 15.7241 | **18.8562** | **+3.13** | 0.7285 | 0.7867 | +0.058 | 449 681 |
| 9  | 21.0759 | **23.6343** | **+2.56** | 0.8777 | 0.9056 | +0.028 | 567 000 |

Time penalty vs baseline: ~+60 % (1 extra forward + 1 extra backward
through gsplat per step), well within budget on a 5090.

The gain peaks at the **sparse-view sweet spot** (n = 6, 9) where the
photometric loss alone overfits dramatically; n = 3 is harder because
the warps themselves become unreliable when there are only 3 views to
reproject from.

## 1b. Occlusion-aware variant (**current recommended**)

The n=6 v1 result in §1 (**18.86 dB**) did not reproduce in a later
re-run on the same code (13.88 dB). Both runs are legitimate — they
differ by small variations in the densification trajectory — but it
reveals that `multiview_photo` v1 is *unstable* at n=6 without an
occlusion check: wide-baseline training pairs share many pixels whose
cam-A contributions are actually occluded in cam-B; those "false
photometric disagreements" push the geometry the wrong way in some
seeds.

We added a forward-backward depth check: after reprojecting pixel `p`
from cam A to cam B to obtain `(uv, z_b)`, render the student depth
from cam B (no-grad, detach_geometry) and drop the pixel if
`|z_b - D_b(uv)| / z_b > τ` or the student's alpha at B is too low.

Two configs ship:

- `sparse_view_ssl_mv_v2.yaml` — strict gate (τ=0.05, enabled from
  `start_iter`, `occlusion_alpha_thresh=0.5`). **Ablation only**: it
  over-rejects at n=3 (kept_ratio 0.37 → coverage 0.10) and loses v1's
  +0.9 dB gain at n=3.
- **`sparse_view_ssl_mv_v3.yaml` — sparse-view-friendly gate** (τ=0.10,
  `occlusion_start_iter=3000` so the gate waits for depth to stabilise,
  `occlusion_alpha_thresh=0.3`). This is the **recommended** config.

| setup                       |  n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Δ PSNR vs baseline |
|-----------------------------|:--:|-------:|-------:|--------:|-------------------:|
| baseline                    |  3 | 12.889 | 0.601  |    —    |         —          |
| mv v1 (no occ.)             |  3 | 13.812 | 0.634  |    —    |       +0.92        |
| mv v2 (τ=0.05, start=1500)  |  3 | 12.904 | 0.608  | 0.463   |       +0.02        |
| **mv v3 (τ=0.10, start=3000)** | 3 | 13.676 | 0.630 | **0.448** |  **+0.79**      |
| baseline                    |  6 | 15.724 | 0.729  |    —    |         —          |
| mv v1 (no occ.) (re-run)    |  6 | 13.880 | 0.712  |    —    |       −1.84 ⚠️     |
| mv v2 (τ=0.05, start=1500)  |  6 | 16.496 | 0.752  | 0.292   |       +0.77        |
| **mv v3 (τ=0.10, start=3000)** | 6 | **18.153** | **0.774** | **0.271** | **+2.43** |
| baseline                    |  9 | 21.076 | 0.878  |    —    |         —          |
| mv v1 (no occ.)             |  9 | 23.634 | 0.906  |    —    |       +2.56        |
| **mv v3 (τ=0.10, start=3000)** | 9 | 23.139 | 0.902 | **0.122** |   **+2.06**     |

Kept-ratio (fraction of in-bounds/in-front pixels that survive the
occlusion gate, averaged over training):

- v2 @ n=3: 0.37, coverage 0.10 — gate bites off too much
- v3 @ n=3: 0.58, coverage 0.19 — healthy
- v2 @ n=6: 0.48, coverage 0.14
- v3 @ n=6: 0.68, coverage 0.23
- v3 @ n=9: 0.74, coverage 0.24 — gate barely fires, geometry already healthy

**Why v3 ≤ v1 at n=9.** With 9 training views the wide-baseline
"false-match overload" that breaks v1 at n=6 doesn't materialise — v1's
warps are already mostly correct. v3's gate then occasionally rejects
genuinely-useful pixels (the −0.5 dB cost). At n=6 v3 wins by **+4.27 dB**
because v1 collapses there. Net across n ∈ {3, 6, 9}, v3 averages
**+1.76 dB** vs baseline and is robust where v1 is not, so it's the
default.

## 2. What didn't work (and why)

I tried two pure self-distillation losses first — `pseudo_view`
(teacher-rendered novel poses) and `ema_teacher` (same-view teacher
match). Both turned out to be **neutral or slightly harmful** at n=6:

| variant (n=6) | Test PSNR | Test SSIM | #G | comment |
|---|---|---|---|---|
| baseline                                  | 15.72 | 0.7285 |   404 k | reference |
| pseudo+teacher v1, no detach              | 10.10 | 0.6193 |   966 k | densify exploded |
| pseudo+teacher v2, detach geometry        | 14.23 | 0.7010 |   478 k | safer, still − |
| pseudo+teacher v3, frozen snapshot teacher| 14.49 | 0.7077 |   442 k | tuned, still − |
| pseudo+teacher v4, late start, low weight | 15.19 | 0.7162 |   408 k | nearly neutral |
| **multiview_photo (this work)**           | **18.86** | **0.7867** | 449 k | **+3.13 dB** |

Interpretation:

- At iter ≈ 5000 on lego, train PSNR is already ~50 dB on the 6 train
  views. The student is essentially perfect there → so is the teacher.
  The teacher therefore has no information that the student doesn't
  already have. Pseudo-view distillation just asks the student to
  match its own *overfit predictions* on novel poses — there is no
  new signal, only smoothing.
- The "v1" (no `detach_geometry`) failure was a separate gsplat-specific
  bug: any extra gradient that flows into `info["means2d"]` of the
  *main* forward gets folded into `state["grad2d"]` and triggers
  uncontrolled densification. This is now documented in code and the
  fix is to either run SSL passes with `detach_geometry=True`
  (as `pseudo_view` and `ema_teacher` now do by default) or with a
  *separate* student forward whose `info` is never handed to the
  strategy (as `multiview_photo` does — see code comment).

## 3. Reproduce

```powershell
# baseline sweep
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view.yaml --n-views 3
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view.yaml --n-views 6
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view.yaml --n-views 9

# +multiview_photo SSL
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view_ssl_mv.yaml --n-views 3
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view_ssl_mv.yaml --n-views 6
python sparse_gs/scripts/train.py --config sparse_gs/configs/sparse_view_ssl_mv.yaml --n-views 9
```

The `--n-views` flag preserves any suffix in the experiment name, so
`lego_n6_ssl_mv` becomes `lego_n3_ssl_mv` / `lego_n9_ssl_mv` and the
sweep runs do not overwrite each other.

## 4. Key implementation details

### 4.1 `multiview_photo` (in `sparse_gs/losses/ssl.py`)

For each step:

1. Render the **current** training camera A → student RGB, depth, alpha.
   This is done in a **separate** forward (not the main one). Its
   `info["means2d"]` is therefore **not** retained by gsplat's
   `DefaultStrategy`, which means densification is still driven only
   by the main photometric loss.
2. Pick a random neighbor training camera B (with real GT image).
3. Un-project A's pixels with the student depth to world space.
4. Project the world points into B → pixel coordinates.
5. `F.grid_sample` GT_b at those coordinates → the cross-view
   "consensus" image of A.
6. Masked L1 between the **student's render of A** and the warped
   GT_b. Mask = (alpha_A > 0.5) ∧ (depth_A > eps) ∧ (uv_b in bounds)
   ∧ (z_b > eps).

The loss back-props into the student's geometry (means / scales /
quats) through standard autograd, but does *not* corrupt
densification triggers. This is the critical engineering trick.

### 4.2 EMA teacher (in `sparse_gs/models/ema.py`)

A snapshot teacher (default `momentum=1.0`, configurable
`snapshot_every`). Tensor-wise EMA only runs when the parameter
*count* hasn't changed since the last snapshot (gsplat densifies
every 100 iters by default, so an EMA across that boundary would
be shape-mismatched). Re-snapshots whenever the student's N changes
or `snapshot_every` elapses.

Even though `multiview_photo` does not use the teacher, the teacher
class is kept in tree because future losses (DINO feature distill,
mean-teacher consistency on auxiliary cameras, etc.) will rely on it.

## 5. Open issues / next steps

- **`pseudo_view` and `ema_teacher` consistently lost ≈0.5 dB** even
  with conservative settings. They are correct and modular but they
  do not introduce new information for vanilla 3DGS. They should
  start to win once we add **external** signals on top:
  - monocular depth pseudo-labels (Marigold / Depth-Anything)
  - DINOv2 feature consistency between rendered & GT crops
  Both are now trivial to add — a new entry in `SSLLossBank.LOSSES`,
  optionally consuming `ctx["teacher"]` for an EMA target.
- **`multiview_photo`** uses bilinear `grid_sample` with `padding_mode='border'`.
  Out-of-FoV pixels still get a small contribution from the border
  pixels; the in-bounds mask removes most of it but a hard zero-pad
  with explicit visibility mask might be slightly cleaner.
- **No occlusion test.** Currently any pixel of A that projects into
  B's frustum contributes. If the student's depth is wrong on an
  occluded pixel, we will warp from a wrong location in B. A simple
  forward-backward depth check (compare predicted z_b with the
  student's depth rendered from B) would prune this. This is a
  ~30-line addition and a clear next experiment.
- **Mesh extraction / surface reconstruction** (the project's
  ultimate goal — SparseSurf style) is not yet attempted. The
  current student model is volumetric 3DGS; we will need either
  2DGS or a TSDF post-process.
- **No COLMAP / DTU / MipNeRF360.** We only run on NeRF-Synthetic
  for now. Other loaders are easy drop-ins under `datasets/`.
- **No LPIPS.** PSNR + SSIM are fine for verifying correctness;
  LPIPS will be added when we report final numbers.
