# Depth SSL v2 — Global-Local + gradient depth consistency

Implemented DNGaussian-style depth SSL on top of current `DAv2+mv` recipe:

- global SSI depth/disparity consistency (existing term)
- local patch-wise SSI consistency (`local_weight=0.5`, `local_patch_size=96`, `local_n_patches=8`)
- normalized depth-gradient consistency (`grad_weight=0.2`)

Full Blender avg-8 @ n=8, full 200-view test split.

| scene | DAv2+mv PSNR | depthv2 PSNR | ΔPSNR | DAv2+mv SSIM | depthv2 SSIM | ΔSSIM | DAv2+mv LPIPS | depthv2 LPIPS | ΔLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chair | 22.195 | 23.053 | +0.858 | 0.877 | 0.887 | +0.010 | 0.164 | 0.157 | -0.007 |
| drums | 15.023 | 15.061 | +0.038 | 0.753 | 0.763 | +0.010 | 0.305 | 0.294 | -0.010 |
| ficus | 21.844 | 21.763 | -0.081 | 0.896 | 0.895 | -0.001 | 0.101 | 0.104 | +0.004 |
| hotdog | 18.392 | 18.525 | +0.134 | 0.865 | 0.870 | +0.005 | 0.273 | 0.257 | -0.016 |
| lego | 21.158 | 21.535 | +0.377 | 0.829 | 0.839 | +0.010 | 0.202 | 0.191 | -0.011 |
| materials | 16.259 | 16.304 | +0.045 | 0.784 | 0.787 | +0.002 | 0.246 | 0.246 | +0.001 |
| mic | 23.135 | 23.460 | +0.325 | 0.930 | 0.934 | +0.004 | 0.112 | 0.107 | -0.004 |
| ship | 16.943 | 17.305 | +0.362 | 0.681 | 0.685 | +0.005 | 0.322 | 0.322 | +0.001 |
| **avg-8** | **19.369** | **19.626** | **+0.257** | **0.827** | **0.832** | **+0.006** | **0.215** | **0.210** | **-0.006** |

## Takeaway

Depth SSL v2 is a real improvement over current `DAv2+mv`:

- PSNR improves on 7/8 scenes.
- SSIM improves on 7/8 scenes.
- LPIPS improves on 5/8 scenes and avg improves.
- The only clear regression is `ficus`, where monocular depth/local-depth priors appear to over-regularize an already easy scene.

This is much stronger than both PatchGAN trials. The evidence supports continuing along geometry-aware SSL rather than adversarial appearance regularization.

## Next tuning direction

The current weights are aggressive (`local_weight=0.5`, `grad_weight=0.2`). Since `ficus` regresses, next try a milder version:

- `local_weight=0.25`
- `grad_weight=0.1`
- optionally start local/grad later (`start_iter=1500`) while keeping global depth at `500`

This may keep gains on chair/lego/ship while reducing over-regularization on ficus/materials.
