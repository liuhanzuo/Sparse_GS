# W1 Pearson Depth Loss + FSGS-aligned Schedule —— 实验聚合

- 生成时间：2026-05-12T13:56:39+08:00
- 数据源根目录：`D:/SSL/sparse_gs/outputs`
- 三条曲线：
  - **depthv2 (baseline)** —— `outputs/blender_{scene}_n8_ssl_mv_dav2s_depthv2/metrics.json` （config: `configs/blender_n8_depthv2.yaml`）
  - **pearson_only** —— `outputs/blender_{scene}_n8_ssl_mv_dav2s_pearson/metrics.json` （config: `configs/blender_n8_pearson.yaml`）
  - **depthv2_pearson** —— `outputs/blender_{scene}_n8_ssl_mv_dav2s_depthv2_pearson_idfix/metrics.json` （config: `configs/blender_n8_depthv2_pearson.yaml`）
- 聚合脚本：`scripts/aggregate_w1_pearson.py`（仅依赖标准库；本轮**不重训任何模型**）

## 1. 总览（avg over 各自可用场景）

| Variant | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Avg N_Gaussians | 可用场景数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depthv2 (baseline) | 19.626 | — | 0.832 | — | 0.2099 | — | 592,363 | 8 |
| pearson_only | 14.796 | -4.830 | 0.745 | -0.087 | 0.3183 | +0.1084 | 955,694 | 8 |
| depthv2_pearson | 19.186 | -0.440 | 0.828 | -0.004 | 0.2281 | +0.0182 | 586,480 | 8 |

> ΔPSNR / ΔSSIM 越大越好（正值表示提升）；ΔLPIPS 越小越好（**负值表示提升**）。num_gaussians 取整。

## 2. Per-scene 明细

#### depthv2 (baseline)

| scene | PSNR | SSIM | LPIPS | N_Gaussians | 备注 |
| --- | --- | --- | --- | --- | --- |
| chair | 23.053 | 0.887 | 0.1566 | 507,842 |  |
| drums | 15.061 | 0.763 | 0.2942 | 1,274,680 |  |
| ficus | 21.763 | 0.895 | 0.1042 | 286,514 |  |
| hotdog | 18.525 | 0.870 | 0.2571 | 358,847 |  |
| lego | 21.535 | 0.839 | 0.1911 | 380,886 |  |
| materials | 16.304 | 0.787 | 0.2461 | 517,877 |  |
| mic | 23.460 | 0.934 | 0.1075 | 591,550 |  |
| ship | 17.305 | 0.685 | 0.3223 | 820,705 |  |

#### pearson_only

| scene | PSNR | SSIM | LPIPS | N_Gaussians | 备注 |
| --- | --- | --- | --- | --- | --- |
| chair | 17.220 | 0.793 | 0.2676 | 953,739 |  |
| drums | 13.450 | 0.702 | 0.3640 | 1,540,973 |  |
| ficus | 19.322 | 0.859 | 0.1917 | 404,589 |  |
| hotdog | 13.044 | 0.724 | 0.4427 | 661,458 |  |
| lego | 14.836 | 0.726 | 0.3182 | 815,870 |  |
| materials | 12.834 | 0.719 | 0.3211 | 757,874 |  |
| mic | 17.946 | 0.880 | 0.1841 | 751,987 |  |
| ship | 9.718 | 0.559 | 0.4570 | 1,759,065 |  |

#### depthv2_pearson

| scene | PSNR | SSIM | LPIPS | N_Gaussians | 备注 |
| --- | --- | --- | --- | --- | --- |
| chair | 22.758 | 0.879 | 0.1808 | 544,896 |  |
| drums | 15.402 | 0.763 | 0.3111 | 1,082,334 |  |
| ficus | 18.278 | 0.875 | 0.1483 | 373,514 |  |
| hotdog | 19.426 | 0.872 | 0.2586 | 385,778 |  |
| lego | 20.827 | 0.827 | 0.2213 | 419,995 |  |
| materials | 16.738 | 0.794 | 0.2524 | 403,214 |  |
| mic | 22.747 | 0.927 | 0.1301 | 658,700 |  |
| ship | 17.307 | 0.686 | 0.3224 | 823,412 |  |

## 3. 关键结论（3 条）

- **(a) Pearson vs depthv2 baseline：** pearson_only 相对 baseline 在 PSNR 上**显著下降** ΔPSNR=-4.830（ΔSSIM=-0.087, ΔLPIPS=+0.1084）。 可能原因：prior depth 来自 DAv2-small（disparity 语义），公式中已通过 -prior 与 1/(prior+200) 的 min 兼容符号歧义；本轮权重 0.05 + train-view-only 设定，Pearson 信号未压过 RGB+depthv2 主导项。
- **(b) Pearson 与 depthv2 是否互补：** **未互补**，depthv2_pearson 相对 baseline -0.440（depthv2_pearson vs pearson_only=+4.389）。 推测 depthv2 已经吃掉了大部分 depth 监督容量，Pearson 全局相关在低权重下被压制。
- **(c) FSGS-aligned warmup schedule：** 本轮为 **train-view-only** 设定，pseudo-view 渲染自约束未启用，schedule 仅作为权重 warmup 生效，无法在本表中独立验证其作用，留待 W2 接入 pseudo-view RGB/Depth 后单独消融。

## 4. W2 决策建议

- **结论：不推荐进入 W2**
- 理由：
  - 两条 Pearson 相关曲线均劣于 baseline（pearson_only ΔPSNR=-4.830, depthv2_pearson ΔPSNR=-0.440）。
  - 应先排查 prior depth 语义/权重/warmup 区间，再考虑是否进入 W2 的 pseudo-view 与 dual-GS。
  - 建议小范围调参实验（pearson_weight ∈ {0.02, 0.05, 0.1}、prior 取 -depth vs 1/(d+200)）后再回看 W1。

> 本节为建议，**不自动启动 W2 训练或代码修改**；如继续 W2，请回到 plan 模式重新走需求与任务规划流程。
