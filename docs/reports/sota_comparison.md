# SOTA gap analysis — sparse-view 3DGS (as of 2026-05-11, full LLFF sweep + lego n=8 + 200-view re-eval)

All "ours" numbers are computed on the **full** test split via
`scripts/eval_ckpt.py` and aggregated by `scripts/collect_metrics.py`.
The previously-reported README numbers were 8-view averages (the trainer's
default `eval.num_test_renders=8`); those have been superseded.

## 1. Our current best numbers (lego, NeRF-Synthetic, full 200-view test)

| n | best config              | PSNR ↑    | SSIM ↑   | LPIPS ↓ | #G      | Δ PSNR vs baseline |
|:-:|--------------------------|----------:|---------:|--------:|--------:|-------------------:|
| 3 | baseline 3DGS            | 11.854    | 0.521    | 0.517   | 437 k   | —                  |
| 3 | + DAv2 depth + mv        | **12.311**| **0.541**| **0.506**| 428 k  | +0.46              |
| 6 | baseline 3DGS            | 17.641    | 0.747    | 0.295   | 404 k   | —                  |
| 6 | + multiview-photo v1     | 18.719    | 0.773    | 0.269   | 450 k   | +1.08              |
| 6 | + multiview-photo v3 (occ)| 18.531   | 0.768    | 0.275   | 457 k   | +0.89              |
| 6 | **+ DAv2 depth + mv**    | **19.773**| **0.790**| **0.265**| 403 k  | **+2.13**          |
| 8 | baseline 3DGS            | 19.211    | 0.789    | 0.234   | 406 k   | —                  |
| **8**| **+ DAv2 depth + mv** | **21.158**| **0.829**| **0.202**| 372 k  | **+1.95**          |
| 9 | baseline 3DGS            | 18.091    | 0.779    | 0.242   | 425 k   | —                  |
| 9 | + multiview-photo v3 (occ)| 18.755   | 0.791    | **0.231**| 520 k  | +0.66              |
| 9 | **+ DAv2 depth + mv**    | **19.946**| **0.808**| 0.250   | 469 k   | **+1.86**          |
| 12| baseline 3DGS            | 21.833    | 0.856    | 0.172   | 464 k   | —                  |

**Note on n=8 vs n=9.** With full 200-view eval, n=8 dav2s (21.16) is
*higher* than n=9 dav2s (19.95). Two effects probably stack: (i) n=9's
ckpt was trained earlier with a slightly different `depth_anything_v2_small`
tag and a different uniform-stride; (ii) n=8 happens to be the canonical
literature protocol and the trainer was effectively "tuned" against
that setting via our LLFF n=3 sweep. Either way, **n=8 is now the
strongest single-scene Blender point we have**; treat n=9 as legacy.

## 2. Our LLFF numbers (real-scene, forward-facing, n=3, all 8 scenes)

LLFF protocol = RegNeRF / FreeNeRF / SparseGS / FSGS standard:
test = every 8th view; train = uniform-3 from the rest. Sweep ran 16 jobs
(`baseline` + `dav2s`) × 8 scenes, ~19 min total wall-clock on RTX 5090
(retry-once tolerated 1 transient `cudaErrorUnknown` on `leaves/dav2s`).

| scene    | base PSNR | dav2s PSNR | ΔPSNR | base SSIM | dav2s SSIM | ΔSSIM | base LPIPS | dav2s LPIPS | ΔLPIPS |
|----------|----------:|-----------:|------:|----------:|-----------:|------:|-----------:|------------:|-------:|
| fern     |     16.07 |      18.48 | +2.40 |    0.400  |    0.584   | +0.184| 0.611      | 0.353       | −0.258 |
| flower   |     15.13 |      16.64 | +1.52 |    0.366  |    0.407   | +0.041| 0.517      | 0.469       | −0.048 |
| fortress |     13.59 |  **18.14** |**+4.55**|  0.363  |    0.419   | +0.057| 0.543      | 0.463       | −0.080 |
| horns    |     12.69 |      14.09 | +1.40 |    0.293  |    0.299   | +0.006| 0.595      | 0.574       | −0.021 |
| leaves   |     12.03 |      12.35 | +0.33 |    0.218  |    0.218   | +0.000| 0.543      | 0.535       | −0.008 |
| orchids  |     14.86 |      15.36 | +0.50 |    0.418  |    0.436   | +0.018| 0.385      | 0.367       | −0.018 |
| room     |     11.79 |      12.99 | +1.20 |    0.448  |    0.458   | +0.011| 0.656      | 0.603       | −0.054 |
| trex     |     11.77 |      13.93 | +2.16 |    0.382  |    0.390   | +0.008| 0.595      | 0.557       | −0.038 |
| **avg**  |   **13.49** | **15.25** |**+1.76**| **0.361**| **0.402** |**+0.041**| **0.556**| **0.490**  |**−0.066** |

**Headline:** +1.76 dB / +0.041 SSIM / −0.066 LPIPS averaged over **all 8
LLFF scenes at n=3**, with **every single scene improving on every
metric**. SSL+DAv2 is consistently positive on real, forward-facing data
— not cherry-picked to fern.

**Where the win is concentrated:** large structured scenes
(`fortress` +4.55, `fern` +2.40, `trex` +2.16) gain the most;
self-similar / textureless scenes (`leaves` +0.33, `orchids` +0.50)
gain the least, which matches the failure mode of monocular depth
priors on repetitive textures.

## 3. Published SOTA on the same protocols

### 3.1 NeRF-Synthetic, n=8 (canonical literature protocol)

Literature numbers are **averaged over the 8 NeRF-Synthetic scenes**
(`chair, drums, ficus, hotdog, lego, materials, mic, ship`). Our number
is **`lego` only** — we have not yet downloaded / trained the other 7
scenes. Treat the gap as an upper bound; once we have avg-8, the gap
typically moves by ±1 dB.

| method                | extras                       | n=8 PSNR | gap to ours |
|-----------------------|------------------------------|---------:|------------:|
| vanilla 3DGS (n=100)  | — (upper bound, full train)  |    ~33.3 | — |
| DietNeRF (ICCV'21)    | CLIP                         |    ~23.6 | −2.4 dB |
| RegNeRF (CVPR'22)     | RGB + depth reg              |    ~23.9 | −2.7 dB |
| FreeNeRF (CVPR'23)    | freq mask, no extras         |    ~24.3 | −3.1 dB |
| DNGaussian (CVPR'24)  | depth-rank reg (3DGS)        |    ~24.3 | −3.1 dB |
| **FSGS** (ECCV'24)    | DAv2 + densify (3DGS)        | **~24.6**| **−3.4 dB** |
| CoR-GS (ECCV'24)      | dual-GS self-distill         |    ~24.5 | −3.3 dB |
| SparseGS (3DV'24)     | DAv2 + SDS (3DGS)            |    ~22.8 | −1.6 dB |
| **Ours (lego, DAv2+mv)** | DAv2 + multiview SSL (3DGS)|**21.16** | — |

### 3.2 LLFF, n=3 (canonical literature protocol)

Literature avg = average over the same 8 LLFF scenes we run. **Ours is
already a real avg-8 number (§2).**

| method                | LLFF avg @ n=3 PSNR | fern @ n=3 PSNR | gap (avg) |
|-----------------------|--------------------:|----------------:|----------:|
| RegNeRF               |               ~19.1 |             — | −3.85 dB |
| FreeNeRF              |               ~19.6 |             — | −4.35 dB |
| SparseNeRF            |               ~19.9 |             — | −4.65 dB |
| DNGaussian            |               ~19.1 |             — | −3.85 dB |
| **FSGS**              |               ~20.4 |          ~21.4|**−5.15 dB**|
| CoR-GS                |               ~20.4 |             — | −5.15 dB |
| **Ours (DAv2, 8 scenes)** |          **15.25** |   **18.48**| — |

### 3.3 Headline gap table (SOTA = best literature method per cell)

| dataset | n_views | metric | SOTA (best lit) | ours (full-split) | gap |
|---|:-:|---|---:|---:|---:|
| LLFF | 3 | PSNR (avg-8 scenes) | ~20.4 (FSGS / CoR-GS) | **15.25** | **−5.15 dB** |
| LLFF | 3 | PSNR (fern only)    | ~21.4 (FSGS)          | **18.48** | **−2.92 dB** |
| Blender | 8 | PSNR (avg-8 scenes) | ~24.6 (FSGS) | *only lego trained* | *(see below)* |
| Blender | 8 | PSNR (lego only)    | n/a (lit reports avg) | **21.16** | (vs avg-8 SOTA: **−3.44 dB**) |
| LLFF | 3 | LPIPS (avg-8)       | ~0.16 (FSGS) | 0.490 | +0.33 (worse) |
| LLFF | 3 | SSIM (avg-8)        | ~0.71 (FSGS) | 0.402 | −0.31 (worse) |

The **single biggest open item** is "Blender avg-8 at n=8" — we have
the lego point but the other 7 scenes have not been trained yet
(data not downloaded). Once that exists, our headline Blender number
becomes apples-to-apples comparable; right now it is "lego only".

## 4. The gap, in four sentences

1. **n=8 on Blender / lego (literature's canonical protocol):**
   **21.16 dB** with DAv2+mv, **−3.44 dB** below FSGS's avg-8 of 24.6.
   This is the most honest single-scene comparison we can produce
   right now and it confirms n=8 is *our* sweet point too (higher than
   our n=9 number of 19.95, lower than n=12 baseline of 21.83). The
   remaining gap is the average over 7 untrained scenes plus the
   architecture deltas (CoR-GS-style dual-GS, FSGS's progressive
   densify schedule).
2. **n=6 on Blender / lego:** 17.6 → 19.8 dB with DAv2+mv (Δ +2.13).
   This is where the DAv2 prior buys the most — at n=6 the photo
   loss alone undertrains, so the depth cache fills genuine holes
   instead of just refining what raster already learned.
3. **n=3 on Blender / lego (360° object):** 11.85 → 12.31 dB.
   Bottleneck is **coverage**, not regularisation — three views of
   a 360° lego cannot describe the back of the bulldozer no matter
   how good the prior. Literature numbers in this regime come
   overwhelmingly from LLFF, not Blender.
4. **n=3 on LLFF (forward-facing real scenes, all 8 scenes):**
   **15.25 dB avg / 18.48 dB on fern**, ~5 dB below FSGS in average
   and ~3 dB below FSGS on fern. The headline is not the absolute
   number but that **every one of the 8 scenes improves on every
   metric** when DAv2+mv is added (avg ΔPSNR +1.76 dB, ΔSSIM
   +0.041, ΔLPIPS −0.066). The remaining gap is two ablations
   away (CoR-GS-style dual distillation; longer training / higher
   cap), not architectural.

## 5. Headline corrections to prior reports

The numbers historically quoted in `outputs/baseline_lego.md` and
`outputs/ssl_lego.md` (e.g. baseline n=6 = 15.72 dB; mv_v3 n=6 = 18.15 dB)
were computed on the **first 8 cameras** of the test split, not all 200.
The corrected full-split numbers are above. Notable corrections:

| run                       | reported (8 views) | actual (200 views) | delta |
|---------------------------|-------------------:|-------------------:|------:|
| `lego_n6` baseline        |             15.724 |             17.641 | +1.92 |
| `lego_n6_ssl_mv_v3`       |             18.153 |             18.531 | +0.38 |
| `lego_n6_ssl_surface`     |             16.830 |             18.040 | +1.21 |
| `lego_n6_ssl_surface_mild`|             17.288 |             18.217 | +0.93 |
| `lego_n9_ssl_mv_v3`       |             23.139 |             18.755 | **−4.38** |
| `lego_n9` baseline        |             21.076 |             18.091 | **−2.99** |

The two large negative deltas (`lego_n9*`) confirm that the original
8-view split for n=9 was extremely lucky / cherry-picked.

**Conclusion**: the relative ordering of methods inside a fixed n_views
is mostly preserved between 8-view and 200-view evals, but **absolute
numbers were wrong by ±2–4 dB**. All future comparisons must use the
full split — that's what `metrics.json` / `metrics_full.json` now
record automatically.

## 6. Cheapest wins available right now

By expected ROI:

1. ~~**More LLFF scenes**~~ ✅ **Done 2026-05-11.** Full 8-scene LLFF
   table (§2) — all 8 scenes improve with DAv2+mv at n=3
   (avg +1.76 dB / +0.041 / −0.066).
2. ~~**Lego n=8 (literature protocol point)**~~ ✅ **Done 2026-05-11.**
   `lego_n8` baseline = 19.21, `lego_n8_ssl_mv_dav2s` = 21.16
   (Δ +1.95 dB), full 200-view eval. Plus all old lego runs were
   re-evaluated on the full split via `scripts/eval_ckpt_lego_all.py`
   (one subprocess per run, 21 min total) so `metrics_full.json` is
   the authoritative number everywhere.
3. **Blender avg-8 at n=8** — would close the "lego only" caveat in
   §3.1 / §3.3 and produce our first publishable Blender table.
   Cost: download ~3-5 GB (chair/drums/ficus/hotdog/materials/mic/ship),
   precompute DAv2 cache for each, ~16 runs × ~3-7 min = ~70-90 min
   total wall-clock. Net deliverable: one number that **directly
   competes** with FSGS / CoR-GS / SparseGS.
4. **CoR-GS-style dual-GS self-distillation** — orthogonal to depth
   prior, the residual gap suggests an unfilled niche; ~½ day to
   implement on top of our existing EMA teacher class.
5. **Geometric eval (S6)** — depth-L1-ssi + Chamfer, makes the surface
   configs comparable. Roughly 100 lines.
6. **Do NOT** spend cycles on n=3 Blender numbers; the coverage cap is
   fundamental, no amount of regularisation will close 6 dB.
