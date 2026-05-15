# PatchGAN trial — DAv2+mv + appearance-only GAN

Setup: conservative PatchGAN, `weight=0.002`, `start_iter=4000`, `every=4`, `detach_geometry=true`, spectral norm, hinge loss. Full 200-view test eval.

| scene | method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #G |
|---|---|---:|---:|---:|---:|
| chair | DAv2+mv | 22.1948 | 0.8768 | **0.1639** | 485001 |
| chair | DAv2+mv+GAN | 22.1692 | **0.8773** | 0.1648 | 491040 |
| chair | Δ GAN | -0.0256 | +0.0004 | +0.0008 | +6039 |
| drums | DAv2+mv | 15.0235 | **0.7531** | **0.3046** | 1231610 |
| drums | DAv2+mv+GAN | 14.9587 | 0.7525 | 0.3084 | 1192830 |
| drums | Δ GAN | -0.0648 | -0.0006 | +0.0038 | -38780 |

## Conclusion

This conservative PatchGAN setting does **not** improve the blur problem on the two tested scenes.

- `chair`: PSNR slightly drops, LPIPS slightly worsens; SSIM is basically unchanged.
- `drums`: all three metrics move in the wrong direction, especially LPIPS.

Interpretation: the visible blur/artifacts are likely dominated by geometry / view-consistency errors rather than missing 2D patch realism. A small appearance-only GAN is safe to run, but it is not a useful next mainline direction in this configuration.

Recommended next steps:

1. Do not expand GAN to avg-8 unless we want a negative-result ablation for the report.
2. Put effort into densify/training schedule and confidence/occlusion handling instead.
3. If we still want a GAN variant later, try it only as a stronger perceptual/LPIPS ablation and expect PSNR risk.
