# 需求文档

## 引言

本功能用于完成 **W1 阶段**（Pearson Depth Loss + FSGS-aligned Schedule）的**实验聚合与决策产出**，即把已经训练完成的三条曲线 × Blender n8 八个场景（chair / drums / ficus / hotdog / lego / materials / mic / ship）的结果统一聚合成一份对比表，并基于该表给出是否进入 W2 的建议。

本功能**不重训任何模型**，也**不修改任何训练/损失/调度相关代码**。本功能的边界严格限定为：

- 读取已经落盘的 `outputs/blender_{scene}_n8_ssl_mv_dav2s_{variant}/metrics.json`（`variant` ∈ {`depthv2`、`pearson`、`depthv2_pearson_idfix`}）；
- 计算每条曲线的 avg PSNR / SSIM / LPIPS，并相对 `depthv2` 基线给出 delta；
- 把上述结果写入 `outputs/w1_pearson_ablation.md`；
- 基于结果给出**3 条**关键结论与“是否进入 W2”的建议。

本功能符合 W1 Agent Prompt 的“完成信号”要求（`outputs/w1_pearson_ablation.md`），并继承用户运行偏好（如需补跑任何缺失场景，只能在当前可见命令行直接前台运行）。

## 需求

### 需求 1

**用户故事：** 作为研究工程师，我希望有一个**完整性预检**步骤，确认三条曲线 × 8 场景共 24 个 `metrics.json` 是否全部存在，以便在聚合前发现缺失或异常，避免静默跳过。

#### 验收标准

1. WHEN 进入聚合流程 THEN 系统 SHALL 枚举 `outputs/blender_{scene}_n8_ssl_mv_dav2s_{variant}/metrics.json`，其中 `scene ∈ {chair, drums, ficus, hotdog, lego, materials, mic, ship}`、`variant ∈ {depthv2, pearson, depthv2_pearson_idfix}`，共 24 项。
2. IF 任一 `metrics.json` 不存在或无法解析 THEN 系统 SHALL 在控制台和报告中以醒目的“MISSING / BROKEN”标记列出该 (scene, variant)，并继续聚合其余项，不得静默跳过、不得 cherry-pick。
3. IF 缺失项数 > 0 AND 用户尚未确认“先出部分报告” THEN 系统 SHALL 在 `ask_user_input` 中明确列出缺失清单，但本轮**不主动重训**（重训属于 W1 训练阶段，已结束）。
4. WHEN 完整性预检通过或用户确认部分聚合 THEN 系统 SHALL 进入需求 2。

### 需求 2

**用户故事：** 作为研究工程师，我希望从每个 `metrics.json` 中抽取 **PSNR / SSIM / LPIPS / num_gaussians**，并按曲线聚合 avg，以便横向比较 depthv2 baseline、only Pearson 和 depthv2 + Pearson 三种配置。

#### 验收标准

1. WHEN 解析 `metrics.json` THEN 系统 SHALL 读取 `metrics["test/psnr"]`、`metrics["test/ssim"]`、`metrics["test/lpips"]` 和顶层 `num_gaussians`，禁止使用 `train/` 前缀的指标。
2. WHEN 任一字段为 `null` 或缺失 THEN 系统 SHALL 把该 scene 在该曲线上的 metric 标记为 `N/A`，并在 avg 时**剔除该 scene**而非补 0；剔除情况必须在报告“说明”一栏注明。
3. WHEN 计算 avg THEN 系统 SHALL 对**完整可用场景集合**做算术平均，PSNR / SSIM 保留 3 位小数、LPIPS 保留 4 位小数、num_gaussians 取整。
4. WHEN avg 计算完成 THEN 系统 SHALL 以 `depthv2` 曲线为基准计算 ΔPSNR / ΔSSIM / ΔLPIPS，正负号方向：PSNR、SSIM 越大越好（正值表示提升），LPIPS 越小越好（负值表示提升）。
5. IF 三条曲线的可用场景集合不一致 THEN 系统 SHALL 同时输出“**全交集场景** avg”和“**各自 8 场景**（含 N/A 剔除）avg”两组数字，避免不公平比较。

### 需求 3

**用户故事：** 作为研究工程师，我希望产出一份结构清晰的 markdown 报告 `outputs/w1_pearson_ablation.md`，以便我和后续 reviewer 能在一页之内完成 W1 决策。

#### 验收标准

1. WHEN 生成报告 THEN 系统 SHALL 在文件首部写入：W1 标题、生成时间（ISO8601）、数据源描述（三条曲线对应的 outputs 目录前缀和 config 文件名）。
2. WHEN 写入“总览表” THEN 系统 SHALL 输出一个 markdown 表格，列依次为 `Variant | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Avg N_Gaussians | 可用场景数`，行依次为 `depthv2 (baseline)`、`pearson_only`、`depthv2_pearson`。
3. WHEN 写入“per-scene 表” THEN 系统 SHALL 对每条曲线输出一段独立的 8 行表格（chair → ship），列为 `scene | PSNR | SSIM | LPIPS | N_Gaussians | 备注`，缺失项用 `N/A` 并在“备注”里写明原因（e.g. `metrics.json 不存在` / `OOM` / `训练崩溃`）。
4. WHEN 写入“关键结论” THEN 系统 SHALL **恰好输出 3 条**结论，且必须分别覆盖：(a) Pearson 是否对 depthv2 baseline 形成提升；(b) Pearson 与 depthv2 是否互补（即叠加是否优于各自单独）；(c) FSGS-aligned warmup schedule 在本轮 train-view-only 设定下是否产生可观测影响（如不可观测，照实写）。
5. WHEN 实验结果为负向或互不显著 THEN 系统 SHALL **照实写**，禁止挑选子集场景包装成正面结论；并在结论里给出可能原因（prior depth 语义、权重 0.05、warmup 区间、train-view-only 限制等）。
6. WHEN 报告末尾 THEN 系统 SHALL 输出“W2 决策建议”小节，明确给出 `推荐进入 W2 / 不推荐进入 W2 / 有条件推荐` 中的一项，并列出至多 3 条理由；此节内容仅作建议，**不得**自动启动 W2。

### 需求 4

**用户故事：** 作为研究工程师，我希望聚合脚本的实现尽量轻量，不引入新依赖、不修改训练侧代码，以便保持 W1 “最小改动” 的原则与 audit 结果对齐。

#### 验收标准

1. WHEN 实现聚合 THEN 系统 SHALL 仅新增 1 个脚本文件（建议路径：`scripts/aggregate_w1_pearson.py`）和 1 个报告文件（`outputs/w1_pearson_ablation.md`）；禁止改动 `sparse_gs/losses/ssl.py`、`configs/blender_n8_*.yaml`、`scripts/train.py` 以及任何 `third_party/`。
2. WHEN 脚本运行 THEN 系统 SHALL 仅依赖 Python 标准库 + 项目已存在的依赖（`json`、`pathlib`、`statistics` 等），禁止引入 pandas / matplotlib / 远程下载。
3. WHEN 脚本对外暴露入口 THEN 系统 SHALL 支持 `python scripts/aggregate_w1_pearson.py` 直接运行，并允许通过可选参数覆盖三条曲线的 variant 后缀（默认值为 `depthv2`、`pearson`、`depthv2_pearson_idfix`），便于后续替换 `idfix` 为后续命名而无需改源码。
4. WHEN 脚本运行结束 THEN 系统 SHALL 在 stdout 打印总览表（与 markdown 报告一致），并以非 0 退出码反映“缺失项 > 0 但用户未授权部分聚合”的异常情形。

### 需求 5

**用户故事：** 作为研究工程师，我希望整个聚合过程严格遵守用户的运行偏好和工作流约束，以便结果可信且可复核。

#### 验收标准

1. WHEN 运行聚合脚本 THEN 系统 SHALL 在当前可见命令行直接前台运行，禁止使用 `Start-Process`、`Start-Job`、隐藏窗口或新开 cmd 终端启动 [[memory:ykqwnv9c]]。
2. WHEN 任何环节出现歧义（如某 scene 出现多个 `_idfix` / `_idfix2` 同名后缀目录）THEN 系统 SHALL **不臆测**，先用 `ask_user_input` 确认采用哪一个目录，再继续；默认应采用 W1 Agent Prompt 中定义的 `depthv2_pearson_idfix` 命名。
3. WHEN 报告写完 THEN 系统 SHALL 把 `outputs/w1_pearson_ablation.md` 作为最终交付，并在汇报中明确给出：三条曲线 avg PSNR delta、是否值得进入 W2、有无异常需用户介入；**不得**主动启动 W2 的训练或代码修改。
4. IF 用户在 review 后提出新需求（如增加 per-scene 排序、绘制曲线、扩展到 LLFF 等）THEN 系统 SHALL 回到 plan 模式重新走需求与任务规划流程，而不是在本次实施中顺手扩展范围。
