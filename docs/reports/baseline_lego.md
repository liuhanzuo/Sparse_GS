# Vanilla 3DGS baseline — NeRF-Synthetic `lego`

All numbers produced by `configs/sparse_view.yaml` (L1 + 0.2·DSSIM, no SSL),
evaluated on the full 200-view test split.

| n_views | train ids (deterministic uniform) | Test PSNR ↑ | Test SSIM ↑ | #Gaussians | wall-clock |
|:---:|:---|:---:|:---:|:---:|:---:|
| 3  | `[0, 50, 99]`                                     | 12.8888 | 0.6011 | 343256 |  87 s |
| 6  | `[0, 20, 40, 59, 79, 99]`                         | 15.7241 | 0.7285 | 404153 |  99 s |
| 9  | `[0, 12, 25, 37, 50, 62, 74, 87, 99]`             | 21.0759 | 0.8777 | 425081 | 103 s |
| 12 | `[0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99]`  | 21.1470 | 0.8857 | 463575 |  98 s |

Hardware: RTX 5090, CUDA 13.1, gsplat 1.5.3 (JIT), 7000 iters per run.

## Observations

1. Severe overfitting — train PSNR saturates at ~50 dB while test sits at 12–21 dB.
2. `n=9` and `n=12` are nearly tied → the overfitting regime is `n ≤ 9`.
   This is the sweet spot where SSL / adversarial regularization should help.
3. Densify produces 340k–460k Gaussians regardless of view count; sparse-view
   caps (e.g. max-N or opacity-aware prune) are a reasonable next lever.

## How to reproduce

```powershell
python d:\SSL\sparse_gs\scripts\train.py --config d:\SSL\sparse_gs\configs\sparse_view.yaml --n-views 3
python d:\SSL\sparse_gs\scripts\train.py --config d:\SSL\sparse_gs\configs\sparse_view.yaml --n-views 6
python d:\SSL\sparse_gs\scripts\train.py --config d:\SSL\sparse_gs\configs\sparse_view.yaml --n-views 9
python d:\SSL\sparse_gs\scripts\train.py --config d:\SSL\sparse_gs\configs\sparse_view.yaml --n-views 12
```
