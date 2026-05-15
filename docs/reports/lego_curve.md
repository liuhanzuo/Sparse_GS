# lego — full 200-view eval

All numbers below are computed on the full 200-view test
split of the NeRF-Synthetic `lego` scene (re-eval via
`scripts/eval_ckpt_lego_all.py`). PSNR/SSIM/LPIPS use the
LPIPS-VGG metric as in the literature.

| experiment | n | PSNR | SSIM | LPIPS | #G | src |
|---|---:|---:|---:|---:|---:|:---:|
| lego_n3_ssl_mv_v3 | 3 | 13.68 | 0.630 | 0.448 | 446,703 | full |
| lego_n3_ssl_mv_dav2s | 3 | 12.31 | 0.541 | 0.506 | 427,953 | full |
| lego_n3_ssl_mv | 3 | 11.89 | 0.521 | 0.515 | 434,886 | full |
| lego_n3 | 3 | 11.85 | 0.521 | 0.517 | 437,356 | full |
| lego_n3_ssl_mv_depth | 3 | 11.60 | 0.511 | 0.537 | 1,011,690 | full |
| lego_n3_ssl_surface | 3 | 11.46 | 0.527 | 0.508 | 604,821 | full |
| lego_n3_ssl_mv_v2 | 3 | 11.36 | 0.508 | 0.529 | 485,390 | full |
| lego_n6_ssl_mv_dav2s | 6 | 19.77 | 0.790 | 0.265 | 402,599 | full |
| lego_n6_ssl_mv | 6 | 18.72 | 0.773 | 0.269 | 449,681 | full |
| lego_n6_ssl_mv_v3 | 6 | 18.53 | 0.768 | 0.275 | 456,981 | full |
| lego_n6_ssl_surface_mild | 6 | 18.22 | 0.768 | 0.271 | 472,769 | full |
| lego_n6_ssl_mv_v2 | 6 | 18.07 | 0.760 | 0.284 | 465,937 | full |
| lego_n6_ssl_surface | 6 | 18.04 | 0.768 | 0.271 | 538,277 | full |
| lego_n6 | 6 | 17.64 | 0.747 | 0.295 | 404,153 | full |
| lego_n6_ssl_v4 | 6 | 17.34 | 0.733 | 0.301 | 408,307 | full |
| lego_n6_ssl | 6 | 16.99 | 0.724 | 0.306 | 442,164 | full |
| lego_n6_ssl_mv_depth | 6 | 16.95 | 0.723 | 0.353 | 700,779 | full |
| lego_n8_ssl_mv_dav2s | 8 | 21.16 | 0.829 | 0.202 | 372,305 | full |
| lego_n8 | 8 | 19.21 | 0.789 | 0.234 | 405,948 | full |
| lego_n9_ssl_mv_dav2s | 9 | 19.95 | 0.808 | 0.250 | 468,841 | full |
| lego_n9_ssl_mv_depth_alpha | 9 | 18.84 | 0.782 | 0.251 | 677,864 | full |
| lego_n9_ssl_mv | 9 | 18.77 | 0.790 | 0.231 | 567,000 | full |
| lego_n9_ssl_mv_v3 | 9 | 18.75 | 0.791 | 0.231 | 520,284 | full |
| lego_n9 | 9 | 18.09 | 0.779 | 0.242 | 425,081 | full |
| lego_n12 | 12 | 21.83 | 0.856 | 0.172 | 463,575 | full |
