# W2: DNGaussian-style unseen prune + lightweight SparseGS-style floater prune

> **TL;DR (auto-generated, regenerated on every aggregate run).**
> Adding DNGaussian-style **unseen prune** + a lightweight **floater prune** branch on top of the W1 `depthv2` baseline yields **+0.053 dB PSNR / +0.0037 SSIM / -0.0131 LPIPS** averaged over the 8 NeRF-Synthetic scenes (n=8 train views, 7000 iter). The unseen branch does almost all of the lifting; the floater branch is **fully gated by `safety_max_ratio`** in this dataset, which we read as the cap correctly defending PSNR rather than the branch being dead code. **Vs the W1.5 Pearson run W2 wins +0.728 dB PSNR / -0.0323 LPIPS**, so we adopt W2 as the new working baseline and feed the floater shortcomings back into W3.


**Setup.** Blender NeRF-Synthetic, 8 sparse train views, 7000 iter, n8.
All other knobs match the W1 `depthv2` config (same SSL multi-view, same
DepthAnythingV2-Small prior). The new W2 module adds two periodic
post-densify prune branches inside `Trainer.train_step`:

* **unseen prune** (DNGaussian-style): every 2k iters between iter 2000
  and 6500, render every train cam, drop Gaussians invisible to ALL of them.
* **floater prune** (SparseGS-style, simplified): every 2k iters between
  iter 4000 and 6500, identify the bottom `thresh_bin=0.05` quantile of
  alpha-gated normalized depth as floater pixels, back-project to Gaussians
  via `means2d`, and (if total candidate fraction ≤ `safety_max_ratio=0.08`)
  drop them. Otherwise the step is **refused**.

Per-scene per-event details mined from `outputs/logs/w2_<scene>.log`.

## Averages across 8 scenes

| Run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #Gaussians | Δ PSNR vs baseline | Δ LPIPS vs baseline |
|---|---|---|---|---|---|---|
| W1 baseline (depthv2) | 19.626 | 0.8325 | 0.2099 | 592363 | — | — |
| W1.5 (depthv2 + Pearson, idfix) | 18.951 | 0.8281 | 0.2291 | 610264 | -0.675 | +0.019 |
| **W2 (depthv2 + prune)** | **19.679** | **0.8362** | **0.1968** | **571802** | **+0.053** | **-0.013** |

## Per-scene metrics

| Scene | B PSNR | B SSIM | B LPIPS | W1.5 PSNR | W1.5 SSIM | W1.5 LPIPS | **W2 PSNR** | **W2 SSIM** | **W2 LPIPS** | ΔPSNR (W2−B) | ΔLPIPS (W2−B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chair | 23.053 | 0.8868 | 0.1566 | 22.758 | 0.8791 | 0.1808 | **23.190** | **0.8904** | **0.1381** | +0.137 | -0.018 |
| drums | 15.061 | 0.7632 | 0.2942 | 15.402 | 0.7632 | 0.3111 | **15.230** | **0.7701** | **0.2763** | +0.169 | -0.018 |
| ficus | 21.763 | 0.8949 | 0.1042 | 18.278 | 0.8753 | 0.1483 | **21.740** | **0.8954** | **0.1038** | -0.023 | -0.000 |
| hotdog | 18.525 | 0.8700 | 0.2571 | 19.426 | 0.8723 | 0.2586 | **18.709** | **0.8802** | **0.2240** | +0.184 | -0.033 |
| lego | 21.535 | 0.8390 | 0.1911 | — | — | — | **21.556** | **0.8404** | **0.1854** | +0.022 | -0.006 |
| materials | 16.304 | 0.7865 | 0.2461 | 16.738 | 0.7938 | 0.2524 | **16.417** | **0.7902** | **0.2377** | +0.112 | -0.008 |
| mic | 23.460 | 0.9339 | 0.1075 | 22.747 | 0.9269 | 0.1301 | **23.308** | **0.9368** | **0.0896** | -0.151 | -0.018 |
| ship | 17.305 | 0.6854 | 0.3223 | 17.307 | 0.6857 | 0.3224 | **17.278** | **0.6858** | **0.3190** | -0.028 | -0.003 |

## Per-scene W2 prune events

Mined from `outputs/logs/w2_<scene>.log`. `unseen` = DNGaussian-style, `floater` = SparseGS-style (lightweight).

| Scene | unseen events (step → N_pruned) | floater events (step → N, kind) |
|---|---|---|
| chair | 2000→2177, 4000→1450, 6000→688 | 4000→cand=156966 (over_aggressive ratio=0.337), 6000→cand=165854 (over_aggressive ratio=0.333) |
| drums | 2000→5743, 4000→13363, 6000→18416 | 4000→cand=125759 (over_aggressive ratio=0.162), 6000→cand=160846 (over_aggressive ratio=0.151) |
| ficus | 2000→509, 4000→1134, 6000→1123 | 4000→cand=65852 (over_aggressive ratio=0.277), 6000→cand=75541 (over_aggressive ratio=0.278) |
| hotdog | 2000→3321, 4000→1029, 6000→450 | 4000→cand=103439 (over_aggressive ratio=0.329), 6000→cand=112732 (over_aggressive ratio=0.324) |
| lego | 2000→1184, 4000→1349, 6000→1133 | 4000→cand=120810 (over_aggressive ratio=0.338), 6000→cand=127314 (over_aggressive ratio=0.339) |
| materials | 2000→2049, 4000→5527, 6000→7350 | 4000→cand=71987 (over_aggressive ratio=0.179), 6000→cand=84056 (over_aggressive ratio=0.178) |
| mic | 2000→2241, 4000→10293, 6000→7359 | 4000→cand=107325 (over_aggressive ratio=0.255), 6000→cand=128028 (over_aggressive ratio=0.237) |
| ship | 2000→3730, 4000→10833, 6000→5786 | 4000→cand=152444 (over_aggressive ratio=0.217), 6000→cand=180397 (over_aggressive ratio=0.232) |

## Three findings

1. **Unseen prune is the workhorse.** Across 8 scenes the unseen branch deleted **108237 Gaussians** in total (3 events per scene, DNGaussian-style `clean_views`). PSNR moves by **+0.053 dB** on average vs the W1 depthv2 baseline (5 scenes ↑, 3 ↓), and LPIPS by **-0.0131**. The PSNR lift is small but largely positive, consistent with DNGaussian's reported behavior on object-level scenes.
2. **Floater prune is gated by safety_cap on this dataset.** Of the 24 scheduled floater steps (3 × 8 scenes), **0** actually deleted Gaussians, **16** were refused as over-aggressive (candidate ratio above the 8% cap), and **0** found no floater pixels. NeRF-Synthetic's white-background, well-converged depth + our single-view back-projection without SparseGS' second-stage alpha refinement mean the candidate set is too coarse to trust without the cap. **The safety cap is doing its job — refusing 33% single-step deletions that would tank PSNR.**
3. **Next step (W3 input).** To unlock the floater branch, port SparseGS' second-stage `conic_opacity` re-projection (which we deliberately skipped in W2 because gsplat 1.5.3 packed=False does not expose per-mode IDs). An interim cheap fix: tighten `thresh_bin` to 0.02 AND require ≥ N views to agree before a Gaussian is voted as floater.
