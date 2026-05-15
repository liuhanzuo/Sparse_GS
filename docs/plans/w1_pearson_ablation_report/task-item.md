# 实施计划

- [ ] 1. 完整性预检：枚举三条曲线 × 8 场景的 metrics.json 路径
   - 编写 `scripts/aggregate_w1_pearson.py` 骨架，定义 SCENES 和 VARIANTS 常量（默认 `depthv2`、`pearson`、`depthv2_pearson_idfix`），并支持通过 CLI 参数覆盖 variant 后缀
   - 实现 `scan_metrics_paths(outputs_root)` 函数，枚举 `outputs/blender_{scene}_n8_ssl_mv_dav2s_{variant}/metrics.json` 共 24 项，区分 `OK / MISSING / BROKEN` 三态，返回结构化字典
   - 在 stdout 打印缺失清单（如有），并在缺失数 > 0 时通过 `ask_user_input` 让用户决定是否继续部分聚合（命中默认 idfix 命名歧义时同样走 ask_user_input）
   - _需求：1.1、1.2、1.3、5.2_

- [ ] 2. 解析 metrics.json，抽取 PSNR/SSIM/LPIPS/num_gaussians
   - 实现 `load_scene_metrics(path)`，仅读取 `metrics["test/psnr"]`、`metrics["test/ssim"]`、`metrics["test/lpips"]` 与顶层 `num_gaussians`，对 `null/缺失` 返回 `None` 并标记 `N/A`
   - 显式禁止读取 `train/` 前缀字段；对解析异常归类为 `BROKEN` 并保留原因字符串
   - _需求：2.1、2.2_

- [ ] 3. 计算每条曲线的 avg 指标和 Δ 相对 depthv2 baseline
   - 实现 `aggregate_variant(rows)`，对每条曲线分别按算术平均计算 PSNR/SSIM/LPIPS/num_gaussians（剔除 N/A 场景，不补 0）
   - 实现 `compute_deltas(stats, baseline_key="depthv2")`，PSNR/SSIM 为 `value - baseline`，LPIPS 同样为差值（负值表示提升），保留 PSNR/SSIM 3 位、LPIPS 4 位、num_gaussians 取整
   - 同时计算“**全交集场景** avg”和“**各自可用场景** avg”两组数据，以应对场景集合不一致的情况
   - _需求：2.3、2.4、2.5_

- [ ] 4. 生成 stdout 总览表 + 退出码逻辑
   - 在 stdout 打印与 markdown 一致的总览表（Variant / PSNR / ΔPSNR / SSIM / ΔSSIM / LPIPS / ΔLPIPS / Avg N_Gaussians / 可用场景数）
   - 当存在 MISSING/BROKEN 且用户未授权部分聚合时返回非 0 退出码；正常完成返回 0
   - _需求：4.4、5.1_

- [ ] 5. 渲染 markdown 报告主体（标题、数据源、总览表、per-scene 表）
   - 实现 `render_markdown(stats, per_scene_rows, sources)`：写入 W1 标题、ISO8601 生成时间、三条曲线的 outputs 目录前缀和对应 config 文件名（`blender_n8_depthv2.yaml`、`blender_n8_pearson.yaml`、`blender_n8_depthv2_pearson.yaml`）
   - 输出总览表（3 行：`depthv2 (baseline)`、`pearson_only`、`depthv2_pearson`）
   - 对每条曲线输出 8 行 per-scene 表（chair → ship），缺失项标 `N/A` 并在“备注”列注明原因（`metrics.json 不存在` / `BROKEN: …` / `OOM` / `训练崩溃`）
   - 若三条曲线场景集合不一致，同时附上“全交集 avg”小表
   - _需求：3.1、3.2、3.3、2.5_

- [ ] 6. 渲染“关键结论 3 条 + W2 决策建议”小节
   - 基于聚合结果按规则文本化生成恰好 3 条结论：(a) Pearson vs depthv2 baseline；(b) Pearson 与 depthv2 是否互补（`mode=both` 是否优于各自单独）；(c) FSGS-aligned warmup schedule 在 train-view-only 设定下是否产生可观测影响（如未启用 pseudo-view 则照实写明“train-view-only，schedule 未生效”）
   - 当结果负向或不显著时，结论中显式列出可能原因（prior depth 语义、权重 0.05、warmup 区间、train-view-only 限制）
   - 末尾输出 “W2 决策建议” 小节，给出 `推荐进入 W2 / 不推荐进入 W2 / 有条件推荐` 三选一 + 至多 3 条理由，明确不自动启动 W2
   - _需求：3.4、3.5、3.6、5.3_

- [ ] 7. 在当前可见命令行前台运行脚本并落盘报告
   - 直接前台运行 `python scripts/aggregate_w1_pearson.py`（禁止 Start-Process / Start-Job / 隐藏窗口启动 [[memory:ykqwnv9c]]）
   - 验证退出码、stdout 总览表，并确认 `outputs/w1_pearson_ablation.md` 已写入
   - 检查脚本仅依赖标准库，未引入 pandas/matplotlib/远程下载；未触动 `sparse_gs/losses/ssl.py`、`configs/blender_n8_*.yaml`、`scripts/train.py`、`third_party/`
   - _需求：4.1、4.2、4.3、5.1_

- [ ] 8. 终态汇报
   - 用 `open_result_view` 打开 `outputs/w1_pearson_ablation.md`
   - 在最终消息中明确给出：三条曲线 avg PSNR delta、是否值得进入 W2、有无异常需用户介入；不主动启动 W2 训练或代码修改
   - _需求：3.6、5.3、5.4_
