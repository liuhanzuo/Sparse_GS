# 实施计划

- [ ] 1. 定位现有 depth loss、配置与 pseudo-view 接入点
   - 核对 `outputs/sota_source_audit.md`、`PROJECT_STATUS.md`、`sparse_gs/losses/ssl.py`、FSGS `loss_utils.py` 与 `train.py` 中的 W1 事实源。
   - 在现有训练入口、loss 聚合逻辑、配置加载逻辑中定位最小改动点，确认 `depthv2`、prior depth 供给语义和 pseudo-view 机制是否已存在。
   - 若 pseudo-view 机制不存在，仅记录为 W1 不扩展项，后续只实现 train-view Pearson。
   - _需求：1.1、1.2、1.3、1.4、1.5、1.6、1.7、3.8、4.1、4.9、9.2、9.5_

- [ ] 2. 在 `sparse_gs/losses/ssl.py` 实现 FSGS 全局 Pearson depth loss
   - 新增 `pearson_depth_loss(rend_depth, prior_depth)`，支持 `[H,W]` 与 `[B,H,W]` 输入并返回标量 Tensor。
   - 按 FSGS 公式实现两路 prior 变换并取 `min`，保留 `+200` magic number，使用去均值和标准差归一化计算 Pearson correlation。
   - 不修改 `depthv2` 公式，不在函数内部添加 mask 或 detach prior。
   - _需求：2.1、2.2、2.3、2.4、2.5、2.6、2.7、2.8、2.10、9.4_

- [ ] 3. 为 Pearson depth loss 添加单元测试
   - 覆盖相同 depth loss 接近 0、随机 depth 与反向 depth loss 接近 2、`.backward()` 可用且产生梯度。
   - 覆盖 `[B,H,W]` batch 输入，确保 batch-friendly 行为稳定。
   - 将测试接入项目现有测试方式，避免引入无关测试框架或重写现有测试结构。
   - _需求：6.1、6.2、6.3、6.4、2.2、2.3_

- [ ] 4. 接入 `ssl.depth.mode`、`pearson_weight` 与 `depthv2_weight` 配置控制
   - 在现有 loss registry 或训练 loss 聚合逻辑中支持 `depthv2`、`pearson`、`both` 三种模式。
   - 保持缺省行为兼容现有 baseline；`depthv2` 模式只走现有分支，`pearson` 模式只走 Pearson，`both` 模式同时叠加。
   - 对 prior depth 是 metric depth 还是 disparity 的语义做接入层处理，并在必要位置留下简明说明。
   - _需求：3.1、3.2、3.3、3.4、3.5、3.7、3.8、9.1、9.3、9.4_

- [ ] 5. 补充分项日志或 metrics，验证 `both` 模式叠加正确
   - 在训练日志或 metrics 输出中分别记录 `depthv2` loss、Pearson loss 与 total loss 中对应权重后的贡献。
   - 添加轻量测试或 smoke 日志检查，确认 `mode="both"` 时两个 loss 均被计算并加到 total。
   - 避免改变现有 baseline 的日志语义或评测脚本。
   - _需求：3.3、3.6、6.5、7.1、9.3_

- [ ] 6. 对齐已有 pseudo-view schedule 与 warmup 权重
   - 若项目已有 pseudo-view 触发逻辑，接入 `start_sample_pseudo`、`end_sample_pseudo`、`sample_pseudo_interval` 配置字段。
   - 将触发条件调整为 `start_sample_pseudo < iter < end_sample_pseudo` 且满足采样间隔。
   - 实现 `w_pseudo = min(max(iter - start_sample_pseudo, 0) / 500, 1.0)`，并保证 pseudo-view depth 约束不使用 GT depth。
   - 若项目无 pseudo-view 机制，不新增完整 pseudo-view 系统，只保持 W1 train-view Pearson 路径可运行。
   - _需求：4.1、4.2、4.3、4.4、4.5、4.8、4.9、4.10、9.2、9.5_

- [ ] 7. 添加 pseudo warmup 与 schedule 行为测试
   - 测试 `iter == start_sample_pseudo` 时 `w_pseudo == 0`。
   - 测试 `iter == start_sample_pseudo + 500` 时 `w_pseudo == 1.0`。
   - 测试 start/end 边界外不触发 pseudo-view sampling，边界内且满足 interval 才触发。
   - _需求：4.3、4.4、4.5、4.6、4.7、6.6_

- [ ] 8. 新建两份 Blender n8 Pearson 消融配置
   - 基于实际存在的 Blender n8 `depthv2` baseline 配置复制或继承，创建 `configs/blender_n8_pearson.yaml` 与 `configs/blender_n8_depthv2_pearson.yaml`。
   - `blender_n8_pearson.yaml` 设置 `ssl.depth.mode: pearson` 与 `pearson_weight: 0.05`。
   - `blender_n8_depthv2_pearson.yaml` 设置 `ssl.depth.mode: both`，保留现有 `depthv2` 权重并加入 `pearson_weight: 0.05`。
   - 在配置中显式写入 pseudo schedule 参数，避免训练代码硬编码 Blender n8 总迭代设置。
   - _需求：5.1、5.2、5.3、5.4、5.5、5.6、5.7、4.8_

- [ ] 9. 运行代码测试与 1 scene / 2k iter smoke test
   - 运行 Pearson loss、`both` 模式和 pseudo warmup 相关单元测试。
   - 使用一个 Blender n8 scene 运行 2k iter smoke test，检查无 NaN、loss 曲线正常、日志中 depthv2/Pearson 分项和 pseudo warmup 输出正确。
   - 若 smoke test 失败，修复可定位问题后重跑；若无法定位，记录阻塞原因等待用户决策。
   - _需求：6.1、6.2、6.3、6.4、6.5、6.6、6.7、6.8、3.6_

- [ ] 10. 运行三条 Blender n8 曲线并汇总评测指标
   - 使用相同 seed、相同迭代预算和项目现有训练/评测脚本运行或复用 `depthv2` baseline、Pearson-only、depthv2+Pearson 三条曲线。
   - 覆盖 `chair`、`drums`、`ficus`、`hotdog`、`lego`、`materials`、`mic`、`ship` 八个 scene。
   - 对每条曲线汇总 avg PSNR、SSIM、LPIPS；任何 OOM、崩溃或评测失败都在结果中显式标注。
   - _需求：7.1、7.2、7.3、7.4、7.5、7.6、7.7、7.8、7.9_

- [ ] 11. 生成 `outputs/w1_pearson_ablation.md` 消融报告
   - 写入三条曲线的 avg 指标表，并增加相对 `depthv2` baseline 的 delta 列。
   - 写入 8 个 scene 的 per-scene 指标表，失败 scene 标注失败原因。
   - 给出不超过 3 条关键结论，覆盖 Pearson 是否提升、是否与 `depthv2` 互补、warmup schedule 是否起作用；负向结果照实分析。
   - 只汇报 W1 结果、PSNR delta、是否建议进入 W2 和是否需要用户介入，不主动进入 W2。
   - _需求：8.1、8.2、8.3、8.4、8.5、8.6、8.7、9.6_
