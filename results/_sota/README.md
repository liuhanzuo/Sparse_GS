# SOTA Records — Blender n=8 (sparse-view)

本目录记录 sparse_gs 在 NeRF-Synthetic n=8 sparse-view 设置下，当前
**v6_pgan70 (15k baseline)** 与 **v6_pgan70_x30k (current SOTA candidate)**
两组配置下的 full 200-view test 结果。

## 文件清单（自动生成，请不要手改）

| 文件 | 用途 |
|---|---|
| `blender_n8_x30k_sota.json` | 全量结构化数据（每个 scene × tag 一条记录，含 PSNR/SSIM/LPIPS、N_gaussians、wall-clock 等） |
| `blender_n8_x30k_sota.csv`  | 同上数据的 CSV 视图，方便丢进 Excel/Sheets 做图 |
| `blender_n8_x30k_sota.md`   | 同上数据的 Markdown 视图，给 paper / iwiki 用 |

## 当前 protocol

- 数据集：NeRF-Synthetic, 8 train views（`seed=42` 确定性挑选）。
- Eval：full 200-view test split, antialiased rasterization, white background。
- 15k baseline：`configs/_w3_aggrprune/blender_<scene>_n8_w3_aggrprune_long_v6_pgan70.yaml`
- 30k SOTA   ：`configs/_w3_aggrprune/blender_<scene>_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml`
  - 30k = v6_pgan70 的 15000 步等比拉伸到 24000 + 6000 步尾段精修。
  - 尾段每 1500 步存一个 periodic snapshot；训练结束后对所有 snapshot 跑全量 test，
    选 PSNR 最高者写到 `metrics.json:best_*`，然后**删除所有 snapshot 文件**（磁盘零残留）。

## 数据来源

每个数据点都来自 `outputs/<run_name>/metrics.json` 中的 `metrics.test/*`
（即 `trainer._final_test()` 的输出，与 `eval_log.jsonl` 的 `final_test` 行完全一致）。

## 更新流程

1. 训练任一场景：
   ```powershell
   python -u scripts/train.py --config configs/_w3_aggrprune/blender_<scene>_n8_w3_aggrprune_long_v6_pgan70_x30k.yaml
   ```
2. 训练结束后会自动生成 `outputs/<run_name>/metrics.json`。
3. 刷新本目录的 SOTA 表格：
   ```powershell
   python -u scripts/collect_x30k_sota.py
   ```
4. 一次性串行跑所有 7 个剩余场景（hotdog 已完成会被默认跳过）：
   ```powershell
   .\scripts\run_x30k_all_scenes.ps1
   ```
   该脚本会在每个场景跑完后自动调用 `collect_x30k_sota.py` 刷新这里的表格。

## 已知 SOTA（截至当前）

详见 `blender_n8_x30k_sota.md`。最近一次手工锚点：

| scene  | tag | PSNR  | SSIM   | LPIPS  | source |
|---|---|---:|---:|---:|---|
| hotdog | v6_pgan70_x30k | **25.759** | 0.9378 | 0.1026 | `outputs/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70_x30k/` |
| hotdog | v6_pgan70 (15k) | 24.747 | 0.9388 | 0.1033 | `outputs/blender_hotdog_n8_w3_aggrprune_long_v6_pgan70/` |
