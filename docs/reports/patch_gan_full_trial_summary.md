# Full PatchGAN trial — chair n=8

Setup: DAv2+mv baseline plus stronger/full PatchGAN: `weight=0.005`, `start_iter=3000`, `every=1`, `detach_geometry=false`, spectral norm, hinge loss. Full 200-view test eval.

| method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #G | wall sec |
|---|---:|---:|---:|---:|---:|
| DAv2+mv | **22.1948** | 0.8768 | 0.1639 | 485001 | 305.5 |
| DAv2+mv + GAN small (appearance-only) | 22.1692 | 0.8773 | 0.1648 | 491040 | 290.3 |
| DAv2+mv + GAN full (geometry-active) | 22.1692 | **0.8774** | **0.1633** | 485521 | 381.8 |

## Delta vs DAv2+mv

| method | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|
| GAN small | -0.0256 | +0.0004 | +0.0008 |
| GAN full | -0.0256 | +0.0006 | -0.0007 |

## Visual check

See `outputs/_visual_checks/chair_full_gan_visual_check.jpg`.

Per-view, full GAN gives tiny mixed changes:

- Best PSNR views vs DAv2+mv: view 196 (+0.37 dB), 195 (+0.30 dB), 55 (+0.24 dB).
- Worst PSNR views: view 5 (-0.54 dB), 8 (-0.39 dB), 11 (-0.36 dB).

The contact sheet does not show a clear deblurring effect. The method difference is concentrated around edges/noise/floater regions, not coherent texture sharpening.

## Conclusion

Full GAN is **not a clear win**. It slightly improves LPIPS and SSIM on `chair`, but PSNR drops by the same amount as the appearance-only GAN, runtime increases, and visual sharpening is not obvious. This supports the hypothesis that the remaining blur is mostly geometry / view-consistency / floater ambiguity, not a missing 2D patch realism signal.

Recommendation: keep GAN as a negative/auxiliary ablation, but do not expand it to avg-8 unless needed for the proposal narrative. The next practical direction should be densify/training schedule and geometry confidence rather than GAN.
