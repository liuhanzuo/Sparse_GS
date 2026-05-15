# 需求文档

## 引言

本功能面向 `d:/SSL/sparse_gs` 项目的 W1 消融实验：在现有 `depthv2` 基线之上，以最小改动引入 FSGS 风格的全局 Pearson depth loss，并将已有 pseudo-view sampling schedule 对齐 FSGS 原文。实现范围严格限制在 W1，不引入 dual GS、co-prune、unpooling、unseen prune 或 SparseGS local patch Pearson 等后续工作。

需求重点包括：新增 train-view Pearson depth loss、通过配置开关与 `depthv2` 并行控制、按 FSGS 规则调整 pseudo-view schedule、补充单元测试、创建两份 Blender n8 配置、运行可对比消融实验，并输出 `outputs/w1_pearson_ablation.md` 结果文档。

## 需求

### 需求 1：基于本地事实源完成 W1 实现前核对

**用户故事：** 作为一名研究工程师，我希望在修改代码前读取本地审计文档和 SOTA 仓库源码，以便 Pearson loss 公式、schedule 和实验边界严格符合已核对事实。

#### 验收标准

1. WHEN 开始 W1 工作 THEN 系统 SHALL 先读取 `d:/SSL/sparse_gs/outputs/sota_source_audit.md` 全文，并以该文档作为事实优先级最高的依据。
2. WHEN 需要了解项目状态和 baseline 指标 THEN 系统 SHALL 读取 `d:/SSL/sparse_gs/PROJECT_STATUS.md`。
3. WHEN 需要理解现有 SSL depth loss THEN 系统 SHALL 读取 `d:/SSL/sparse_gs/sparse_gs/losses/ssl.py`，且 SHALL 不修改现有 `depthv2` 实现语义。
4. WHEN 核对 Pearson depth loss 公式 THEN 系统 SHALL 读取 `d:/SSL/sparse_gs/third_party/FSGS/utils/loss_utils.py` 中 `pearson_depth_loss` 与 `local_pearson_loss` 相关实现。
5. WHEN 核对 pseudo-view schedule THEN 系统 SHALL 读取 `d:/SSL/sparse_gs/third_party/FSGS/train.py` 中 `start_sample_pseudo`、`end_sample_pseudo`、采样间隔和 warmup 相关段落。
6. IF 公式、超参或 schedule 存在歧义 THEN 系统 SHALL 优先回到 `third_party/FSGS/` 本地源码核对，而 SHALL NOT 联网搜索或凭记忆实现。
7. IF FSGS 源码与早期整合方案不一致 THEN 系统 SHALL 以 `outputs/sota_source_audit.md` 和 FSGS 本地源码为准。

### 需求 2：新增 FSGS 风格全局 Pearson depth loss

**用户故事：** 作为一名研究工程师，我希望在 `sparse_gs/losses/ssl.py` 中新增 FSGS 风格的全局 Pearson depth loss，以便对 train view 的渲染深度和先验深度施加尺度/符号鲁棒的相关性约束。

#### 验收标准

1. WHEN 实现 Pearson loss THEN 系统 SHALL 在 `d:/SSL/sparse_gs/sparse_gs/losses/ssl.py` 中新增 `pearson_depth_loss(rend_depth: Tensor, prior_depth: Tensor) -> Tensor`。
2. WHEN `rend_depth` 和 `prior_depth` 输入形状为 `[H,W]` THEN 系统 SHALL 正确展平并计算全局 Pearson loss。
3. WHEN `rend_depth` 和 `prior_depth` 输入形状为 `[B,H,W]` THEN 系统 SHALL batch-friendly 地计算 loss，并返回可参与训练的标量 Tensor。
4. WHEN 计算 Pearson loss THEN 系统 SHALL 使用 FSGS 原版公式：`min(1 - pearson(-prior_depth.flatten(), rend_depth.flatten()), 1 - pearson(1.0 / (prior_depth.flatten() + 200), rend_depth.flatten()))`。
5. WHEN 使用 `1.0 / (prior_depth + 200)` 分支 THEN 系统 SHALL 保留 `+200` magic number，不得改成其他常数。
6. WHEN 计算 correlation THEN 系统 SHALL 对输入去均值并按标准差归一化，避免仅计算未中心化 cosine similarity。
7. WHEN 实现函数 THEN 系统 SHALL NOT detach `prior_depth`，调用方负责确保先验本身不需要梯度。
8. WHEN 输入包含 NaN、0 或无效区域 THEN 系统 SHALL 假设调用方已经过滤或清理，`pearson_depth_loss` 内部不额外实现 mask 逻辑。
9. IF pseudo-view 分支已有可用的双渲染深度组织 THEN 系统 MAY 新增 `pearson_depth_loss_pseudo(rend_depth_a, rend_depth_b)`；否则系统 SHALL 只完成 train-view Pearson，不因 pseudo helper 阻塞 W1。
10. IF 现有 `depthv2` 是 L1 版 patch-norm depth THEN 系统 SHALL 保持该实现不变，并将 Pearson 作为并行新分支接入。

### 需求 3：通过配置开关接入 depth loss registry 或训练逻辑

**用户故事：** 作为一名研究工程师，我希望通过配置选择 `depthv2`、`pearson` 或二者叠加，以便公平运行 W1 消融实验并保留现有 baseline。

#### 验收标准

1. WHEN 配置 `ssl.depth.mode` 为 `depthv2` THEN 系统 SHALL 只启用现有 `depthv2` loss，并保持 baseline 行为不变。
2. WHEN 配置 `ssl.depth.mode` 为 `pearson` THEN 系统 SHALL 只启用新增 Pearson depth loss，且 SHALL 关闭 `depthv2` 分支。
3. WHEN 配置 `ssl.depth.mode` 为 `both` THEN 系统 SHALL 同时计算 `depthv2` 和 Pearson depth loss，并将二者按各自权重加到 total loss。
4. WHEN 配置包含 `ssl.depth.pearson_weight` THEN 系统 SHALL 使用该权重缩放 Pearson loss，默认建议值为 `0.05`。
5. WHEN 配置包含 `ssl.depth.depthv2_weight` THEN 系统 SHALL 使用该权重缩放 `depthv2` loss，并保持现有 baseline 权重语义。
6. WHEN `mode="both"` 运行训练 THEN 系统 SHALL 在日志或 metrics 中分别记录 `depthv2` loss 与 Pearson loss，便于确认二者都被叠加。
7. IF 配置中缺少新增字段 THEN 系统 SHALL 提供与现有 baseline 兼容的默认行为，不应破坏原有 `blender_n8_depthv2` 配置。
8. IF prior depth 的语义被确认是 Blender metric depth 而非 MiDaS/DPT disparity THEN 系统 SHALL 在接入层处理符号或 disparity 转换问题，并 SHALL 在实现或文档中说明选择依据。

### 需求 4：对齐 FSGS pseudo-view sampling schedule

**用户故事：** 作为一名研究工程师，我希望现有 pseudo-view sampling schedule 与 FSGS 原文一致，以便 Pearson 消融实验中的时间安排可解释、可复现。

#### 验收标准

1. WHEN 项目中已有 pseudo-view 机制 THEN 系统 SHALL 定位现有触发逻辑，并只修改 schedule 参数和 warmup 权重，不重写采样策略。
2. WHEN 新增或读取 pseudo-view schedule 参数 THEN 系统 SHALL 支持 `start_sample_pseudo`、`end_sample_pseudo` 和 `sample_pseudo_interval`。
3. WHEN 当前迭代 `iter` 满足 `start_sample_pseudo < iter < end_sample_pseudo` 且满足采样间隔 THEN 系统 SHALL 触发 pseudo-view sampling。
4. WHEN 当前迭代不满足 `start_sample_pseudo < iter < end_sample_pseudo` THEN 系统 SHALL 不触发 pseudo-view sampling。
5. WHEN 计算 pseudo loss warmup THEN 系统 SHALL 使用 `w_pseudo = min(max(iter - start_sample_pseudo, 0) / 500, 1.0)`。
6. WHEN `iter == start_sample_pseudo` THEN 系统 SHALL 得到 `w_pseudo == 0`。
7. WHEN `iter == start_sample_pseudo + 500` THEN 系统 SHALL 得到 `w_pseudo == 1.0`。
8. WHEN Blender n8 总迭代数不同于 FSGS 原始 10k THEN 系统 SHALL 通过配置写入缩放后的 start/end，而 SHALL NOT 在训练代码中 hardcode 特定总迭代数。
9. IF 项目尚未实现 pseudo-view 机制 THEN 系统 SHALL 不在 W1 重写或新增完整 pseudo-view 系统，并 SHALL 只完成 train-view Pearson 与相关实验。
10. WHEN pseudo-view 使用 depth 约束 THEN 系统 SHALL 区分 train view 与 pseudo view：train view 使用 prior depth Pearson，pseudo view 只能使用渲染 depth 自身 Pearson 约束，不使用 GT depth。

### 需求 5：新增 Blender n8 消融配置且不覆盖 baseline

**用户故事：** 作为一名研究工程师，我希望新增独立配置文件运行 Pearson 消融，以便保留现有 baseline 并确保三条曲线可对比。

#### 验收标准

1. WHEN 创建 Pearson-only 配置 THEN 系统 SHALL 新建 `d:/SSL/sparse_gs/configs/blender_n8_pearson.yaml`。
2. WHEN 创建 depthv2+Pearson 配置 THEN 系统 SHALL 新建 `d:/SSL/sparse_gs/configs/blender_n8_depthv2_pearson.yaml`。
3. WHEN 新建配置 THEN 系统 SHALL NOT 覆盖或破坏现有 `blender_n8_depthv2.yaml` 或同等 baseline 配置。
4. WHEN 配置 `blender_n8_pearson.yaml` THEN 系统 SHALL 设置 depth mode 为 `pearson`，并使用 `pearson_weight: 0.05`。
5. WHEN 配置 `blender_n8_depthv2_pearson.yaml` THEN 系统 SHALL 设置 depth mode 为 `both`，并同时保留现有 `depthv2` 权重和 `pearson_weight: 0.05`。
6. WHEN 配置 pseudo schedule THEN 系统 SHALL 在配置文件中显式写入 start/end/interval 或引用项目支持的等效字段，避免训练代码硬编码。
7. IF baseline 配置实际文件名与 `blender_n8_depthv2.yaml` 不完全一致 THEN 系统 SHALL 根据 `configs/` 中实际文件命名复制或继承正确 baseline 配置。

### 需求 6：补充代码层单元测试与 smoke test

**用户故事：** 作为一名研究工程师，我希望 Pearson loss 和 pseudo warmup 有明确测试，以便在长训练前发现公式、梯度或 schedule 错误。

#### 验收标准

1. WHEN 测试两张相同 depth 图 THEN `pearson_depth_loss` SHALL 返回接近 0 的 loss，目标阈值为 `< 1e-3`。
2. WHEN 测试随机 depth 与反向 depth THEN `pearson_depth_loss` SHALL 返回接近 2 的 loss，允许合理数值误差。
3. WHEN 对 Pearson loss 调用 `.backward()` THEN 系统 SHALL 不报错，且渲染深度输入 SHALL 获得有效梯度。
4. WHEN 测试 `[B,H,W]` 输入 THEN 系统 SHALL 覆盖 batch-friendly 行为。
5. WHEN `mode="both"` THEN 测试或日志验证 SHALL 证明 `depthv2` 与 Pearson 都被计算并叠加到 total loss。
6. WHEN 测试 pseudo warmup THEN 系统 SHALL 验证 `iter=start` 时权重为 0、`iter=start+500` 时权重为 1.0。
7. BEFORE 运行完整 8 scene 长训练 THEN 系统 SHALL 先运行 1 个 scene + 2k iter 的 smoke test，确认无 NaN、loss 曲线正常且资源使用合理。
8. IF smoke test 失败 THEN 系统 SHALL 修复可定位的问题后再运行全量实验，或在无法定位时记录阻塞原因并等待用户决策。

### 需求 7：运行三条 Blender n8 消融曲线

**用户故事：** 作为一名研究工程师，我希望用相同 seed 和相同迭代预算运行三条 Blender n8 曲线，以便评估 Pearson 是否优于或互补于 `depthv2`。

#### 验收标准

1. WHEN 运行实验 THEN 系统 SHALL 使用项目现有训练和评测脚本，不自行重写 metric 计算。
2. WHEN 运行 baseline 曲线 THEN 系统 SHALL 使用现有 `depthv2` 配置；IF 已有可信 baseline 结果 THEN 系统 MAY 复用并说明来源。
3. WHEN 运行 Pearson-only 曲线 THEN 系统 SHALL 使用 `configs/blender_n8_pearson.yaml`。
4. WHEN 运行 depthv2+Pearson 曲线 THEN 系统 SHALL 使用 `configs/blender_n8_depthv2_pearson.yaml`。
5. WHEN 运行三条曲线 THEN 系统 SHALL 使用相同 seed、相同训练迭代预算和相同评测流程。
6. WHEN 运行 Blender n8 THEN 系统 SHALL 覆盖 8 个 scene：`chair`、`drums`、`ficus`、`hotdog`、`lego`、`materials`、`mic`、`ship`。
7. WHEN 单个 scene 出现 OOM、崩溃或评测失败 THEN 系统 SHALL 在结果表中标注失败状态，而 SHALL NOT 静默跳过。
8. WHEN 在 Windows PowerShell 环境运行长训练 THEN 系统 SHOULD 使用不阻塞主进程的方式或独立日志，并将日志写入 `d:/SSL/sparse_gs/outputs/logs/` 下的 W1 相关文件。
9. WHEN 完成每条曲线评测 THEN 系统 SHALL 汇总 avg PSNR、SSIM、LPIPS。

### 需求 8：输出 W1 Pearson 消融报告

**用户故事：** 作为一名研究工程师，我希望训练完成后生成结构化消融报告，以便快速判断 Pearson 是否值得进入后续 W2 工作。

#### 验收标准

1. WHEN 三条曲线完成或有明确失败状态 THEN 系统 SHALL 新建 `d:/SSL/sparse_gs/outputs/w1_pearson_ablation.md`。
2. WHEN 编写报告 THEN 系统 SHALL 包含三条曲线的 avg PSNR、SSIM、LPIPS 表格。
3. WHEN 编写 avg 表格 THEN 系统 SHALL 包含相对 `depthv2` baseline 的 delta 列。
4. WHEN 编写报告 THEN 系统 SHALL 包含 8 个 scene 的 per-scene 指标表，并标注失败 scene 的失败原因。
5. WHEN 编写结论 THEN 系统 SHALL 提供不超过 3 条关键结论，覆盖 Pearson 是否提升、是否与 `depthv2` 互补、warmup schedule 是否起作用。
6. IF Pearson 或叠加结果为负向 THEN 系统 SHALL 照实记录，不 cherry-pick，并分析可能原因，例如 prior depth 质量、权重尺度或 schedule 设置。
7. WHEN W1 完成 THEN 系统 SHALL 不主动进入 W2，仅报告是否值得进入 W2 以及是否需要用户介入。

### 需求 9：严格控制实现边界和改动范围

**用户故事：** 作为一名研究工程师，我希望 W1 改动尽可能小且不越界，以便实验结果能归因于 Pearson loss 和 FSGS schedule，而不是其他重构或算法变化。

#### 验收标准

1. WHEN 修改代码 THEN 系统 SHALL 优先编辑现有文件，仅在测试、配置或报告确实需要时新增文件。
2. WHEN 实现 W1 THEN 系统 SHALL NOT 引入 dual GS、co-prune、unpooling、unseen prune、SparseGS local patch Pearson 或其他 W2/W3/W4 内容。
3. WHEN 修改现有训练逻辑 THEN 系统 SHALL 保持改动最小，不进行无关重构。
4. WHEN 修改 `depthv2` 相关代码 THEN 系统 SHALL 只在接入层控制启停或权重，不改变 `depthv2` 的 loss 公式。
5. IF 发现项目无 pseudo-view 机制 THEN 系统 SHALL 不因实现 pseudo-view 而扩大 W1 范围。
6. WHEN 完成 W1 THEN 系统 SHALL 只汇报 W1 结果、PSNR delta、是否建议进入 W2 和异常情况。
