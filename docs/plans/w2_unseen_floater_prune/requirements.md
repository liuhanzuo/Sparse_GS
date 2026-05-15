# 需求文档

## 引言

本功能在 W1 已交付的 `depthv2` baseline 基础上，引入 **W2：unseen-prune + 轻量 floater-prune**，目标是直接打掉少视角 3DGS 中两类几何噪声：

1. **从未被任何训练相机可见的高斯**（"unseen 高斯"）—— DNGaussian `train_blender.py` 中 `clean_views` / `unvisible_pnts` 路径的对齐实现。
2. **位于深度直方图最近 K% 处、明显穿出场景表面的"漂浮高斯"**（"floater 高斯"）—— SparseGS `prune_utils_torch.identify_floaters` 的轻量版（仅依赖 alpha-depth，不依赖项目当前 rasterizer 不暴露的 mode/conic_opacity）。

本功能**严格不动**：
- `sparse_gs/losses/ssl.py`（depthv2 / pearson 实现）
- `sparse_gs/models/gaussians.py`（参数布局）
- `sparse_gs/rendering/gsplat_renderer.py`（rasterizer 包装）
- `sparse_gs/strategies/densify.py`（gsplat `DefaultStrategy` 配置）
- 任何 `third_party/`
- 任何已有 config 文件

唯一改动点：
- `sparse_gs/trainer/trainer.py` 中加 1 个方法 `_periodic_unseen_floater_prune(step)`，在已有 `_enforce_cap_max()` 之后调用；
- 新建 1 个工具模块 `sparse_gs/strategies/post_prune.py` 提供独立、可单测的 prune 函数；
- 新增 2 个 config（chair 单场景 smoke test + 8-scene 全量），不修改既有 config。

W2 完成的判据是产出 `outputs/w2_unseen_floater_ablation.md`，含 8 场景 vs `depthv2` baseline 的 ΔPSNR/ΔSSIM/ΔLPIPS/ΔN_Gaussians，并由聚合脚本 `scripts/aggregate_w2_prune.py`（复用 W1 聚合脚本设计）一键生成。

## 需求

### 需求 1

**用户故事：** 作为研究工程师，我希望以最小改动新增 `unseen_prune` 与 `floater_prune` 两个 *后置剪枝* 模块，挂在 `gsplat.DefaultStrategy.step_post_backward` 之后、`_enforce_cap_max` 之后，以便在不破坏 gsplat 内部 densify 状态的前提下，按调度周期性清理几何噪声。

#### 验收标准

1. WHEN 训练循环每步执行 THEN 系统 SHALL 在已有 `_enforce_cap_max()` 之后**立即**调用 `_periodic_unseen_floater_prune(step)`，且该方法只在调度命中时才触发实际 prune，否则零成本提前 return。
2. WHEN 实际 prune 触发 THEN 系统 SHALL 仅通过 `gsplat.strategy.ops.remove(params, optimizers, state, mask)` 修改高斯集合（与项目现有 `_enforce_cap_max` 一致），禁止直接 `del`、禁止 `prune_points` 自定义实现，禁止改写 `gaussians.params` 张量。
3. WHEN prune 触发 THEN 系统 SHALL 用 `torch.no_grad()` 包裹整个流程；禁止在前向 / 反向计算图中执行 prune。
4. IF 某步 `info["radii"]` 与 `info["gaussian_ids"]` 同时为 None（异常 packed 模式或空场景）THEN 系统 SHALL 跳过该步并打印 `[w2-prune] skip step={step}: no visibility info`，**不得 raise**。
5. WHEN 一次 prune 后 N_gaussians 减少了 ≥1 THEN 系统 SHALL 通过 `tb.add_scalar("stats/prune_unseen_n", n)` 与 `tb.add_scalar("stats/prune_floater_n", n)` 上报；并在 stdout 周期性 log（默认每次 prune 命中都 log 一行）。

### 需求 2

**用户故事：** 作为研究工程师，我希望 `unseen_prune` 严格对齐 DNGaussian 的 `clean_views` 语义：累加所有训练相机的 `visibility`，把"从未被任何训练相机看到"的高斯一次性清掉。

#### 验收标准

1. WHEN `unseen_prune` 命中 THEN 系统 SHALL 对当前所有 `self._train_cams_gpu`（已经在 GPU 上的全部训练相机）逐个调用 `self.renderer.render(...)`，复用现有 GSplatRenderer，**禁止新建 rasterizer**。
2. WHEN 累加可见性 THEN 系统 SHALL 兼容两种 packed 模式：
   - `packed=True`：使用 `info["gaussian_ids"]`（局部索引），`scatter_add_` 到长度 N 的全局 bool 张量
   - `packed=False`：使用 `info["radii"] > 0`（已经是长度 N 的张量），直接 `|=` 累加
3. WHEN 一个高斯在所有训练相机渲染中均未可见 THEN 系统 SHALL 将其标记为 `unseen`；`mask = ~visible_pnts` 即为 prune mask。
4. WHEN 某次 unseen-prune 命中但 `mask.sum() == 0` THEN 系统 SHALL 跳过 `remove` 调用，仅 log "no unseen gaussian"。
5. WHEN `cfg["strategy"]["prune"]["unseen"]["enabled"]: false` THEN 系统 SHALL 完全不渲染、不计算可见性。
6. WHEN `unseen_prune` 默认调度 THEN 系统 SHALL 触发于 `start_iter`、之后每隔 `every_iter` 一次、直到 `stop_iter`；默认 `start=2000, every=2000, stop=iterations-500`，与 gsplat `refine_stop_iter=15000` 兼容（即 unseen-prune 主要作用于 densify 末期与收敛期）。

### 需求 3

**用户故事：** 作为研究工程师，我希望 `floater_prune` 对齐 SparseGS `identify_floaters` 的轻量语义（基于 alpha-depth 直方图前 K% 阈值识别近距离漂浮像素，再回投到对应高斯），但**只**使用项目当前 rasterizer 暴露的字段（`alpha`、`depth`、`info["radii"]` / `info["gaussian_ids"]`），不引入 `mode_id` / `conic_opacity` 等任何 SparseGS 自定义字段。

#### 验收标准

1. WHEN `floater_prune` 命中 THEN 系统 SHALL 对每个训练相机渲染得到 `depth (H,W,1)` 与 `alpha (H,W,1)`，归一化深度（`(d - d.min()) / (d.max() - d.min() + eps)`），仅保留 `alpha > alpha_thresh`（默认 0.5）的像素参与直方图。
2. WHEN 阈值确定 THEN 系统 SHALL 取 `floater_thresh_bin`（默认 0.13）所对应的归一化深度分位作为 cut-off，把归一化深度小于 cut-off 的像素标为 floater 像素（**这是 SparseGS `identify_floaters` 的核心简化**：不做 GHT、不做 KL 匹配，单纯取 quantile 阈值）。
3. WHEN floater 像素 → floater 高斯 的回投 THEN 系统 SHALL 使用 `info["gaussian_ids"]`（packed=True）或对应 view 的可见性掩码（packed=False）取并集，跨相机汇总。**禁止**像 SparseGS 一样基于 `means2D` + `conic_opacity` 计算 alpha 二次回投，因为项目 rasterizer 不暴露这些字段。
4. IF packed=False 且无 `gaussian_ids` THEN 系统 SHALL 退化为"该 view 所有可见且其 mean2D 落在 floater 像素的高斯"；mean2D 由 `info["means2d"]` 提供（gsplat 1.5 暴露），floater 像素掩码再 round 到整像素索引检索。
5. WHEN floater_prune 命中但累计 mask 为空或 mask 比例 > `safety_max_ratio`（默认 0.05，即一次最多 prune 5%）THEN 系统 SHALL **拒绝执行此次 prune**，log "floater-prune over-aggressive" 并跳过；安全阈值用于防止误删大量好高斯。
6. WHEN `cfg["strategy"]["prune"]["floater"]["enabled"]: false` THEN 系统 SHALL 完全不计算 floater，不影响 unseen 分支。
7. WHEN `floater_prune` 默认调度 THEN 系统 SHALL 触发于 `start_iter=4000`、`every_iter=2000`、`stop_iter=iterations-500`（**比 unseen 更晚**：因为 floater 检测依赖 depth 已经收敛到合理量级，太早会误删）。

### 需求 4

**用户故事：** 作为研究工程师，我希望 W2 配置层完全沿用项目现有 `_base_` + override 的层级结构，不引入新的 cfg 加载机制。

#### 验收标准

1. WHEN 新增 W2 config THEN 系统 SHALL 在 `configs/` 下新建 2 个文件：
   - `configs/blender_n8_depthv2_prune.yaml`：`_base_: _depth_v2/blender_chair_n8_dav2s_depthv2.yaml` 仅作 schema 模板（实际 8 场景跑用 `--scene` 覆盖），开 `unseen + floater` prune
   - `configs/_w2_prune/blender_{scene}_n8_dav2s_depthv2_prune.yaml`：8 个 per-scene 文件（chair/drums/ficus/hotdog/lego/materials/mic/ship），与 `_depth_v2/` 目录平行
2. WHEN config 中加 prune schema THEN 系统 SHALL 在 `strategy:` 下添加：
   ```yaml
   strategy:
     prune:
       unseen:
         enabled: true
         start_iter: 2000
         every_iter: 2000
         stop_iter: 6500
       floater:
         enabled: true
         start_iter: 4000
         every_iter: 2000
         stop_iter: 6500
         alpha_thresh: 0.5
         thresh_bin: 0.13
         safety_max_ratio: 0.05
   ```
3. WHEN 旧 config（不含 `strategy.prune`）加载 THEN 系统 SHALL 视为 `unseen.enabled=false, floater.enabled=false`（向后兼容），既有 `depthv2` baseline 行为完全不变。
4. WHEN 写代码读取 cfg THEN 系统 SHALL 仅从 `cfg["strategy"]["prune"]` 取值，禁止下钻 `gsplat.DefaultStrategy.__init__` 的私有字段。

### 需求 5

**用户故事：** 作为研究工程师，我希望 W2 实施有最低限度的单元测试与 smoke test，避免 8 场景全量训练后才发现 bug。

#### 验收标准

1. WHEN 写完 `post_prune.py` THEN 系统 SHALL 提供 `pytest` 可运行的小型单元测试 `tests/test_post_prune.py`，至少覆盖：
   - `compute_unseen_mask` 在"全部高斯都被某 view 看到"的情形下返回全 False
   - `compute_unseen_mask` 在"前 K 个高斯从未可见"的情形下返回前 K 个为 True 的 mask
   - `compute_floater_mask` 在 `safety_max_ratio` 触发时返回空 mask
2. WHEN 单元测试运行 THEN 系统 SHALL 仅依赖 `torch` + `pytest`，不依赖 GPU，不依赖 gsplat 真实 rasterizer（用 mock info dict）。
3. WHEN smoke test THEN 系统 SHALL 先在 `chair` 单场景跑完整 `iterations=7000` 的训练（不裁短），并比对：
   - PSNR 不低于 baseline `depthv2 chair` 的 23.053 - 0.5（即 ≥ 22.55）
   - N_Gaussians **降低** 至少 5%（相对 baseline 的 507842）
   - 训练日志中能看到至少 1 次 `[w2-prune] unseen pruned N=...` 与 ≥1 次 `[w2-prune] floater pruned N=...`
4. IF smoke test 不达标 THEN 系统 SHALL 在进入 8 场景前先调参（提高 `safety_max_ratio` / 推迟 `start_iter`），**禁止**直接进入 8 场景全量。

### 需求 6

**用户故事：** 作为研究工程师，我希望 W2 跑完后产出与 W1 同构的 markdown ablation 报告，能直接给 reviewer 看清"加 prune 后是否优于 depthv2 baseline"。

#### 验收标准

1. WHEN W2 8 场景全部训练完毕 THEN 系统 SHALL 通过新增脚本 `scripts/aggregate_w2_prune.py`（**复用 W1 聚合脚本结构**：standard library only、无 pandas）输出 `outputs/w2_unseen_floater_ablation.md`。
2. WHEN 报告写入 THEN 系统 SHALL 包含：
   - 总览表：`depthv2 (baseline)` vs `depthv2 + unseen_prune` vs `depthv2 + unseen+floater_prune` 三行（如时间允许跑分组消融；若仅跑全量组合，则两行）
   - per-scene 表：8 行，含 PSNR / SSIM / LPIPS / N_Gaussians + 备注
   - 关键结论 3 条：(a) 几何噪声是否真为瓶颈；(b) unseen_prune 是否独立有效；(c) floater_prune 是否带来额外收益
   - W3 决策建议：`推荐 / 不推荐 / 有条件推荐` + 至多 3 条理由
3. WHEN 报告写完 THEN 系统 SHALL 不主动启动 W3，等用户 review 后再走 plan 流程。

### 需求 7

**用户故事：** 作为研究工程师，我希望整个 W2 工作流严守用户运行偏好与项目工作流约束。

#### 验收标准

1. WHEN 任何训练运行 THEN 系统 SHALL 在当前可见命令行直接前台运行 `python scripts/train.py --config ... --scene ... --name ...`，禁止 `Start-Process` / `Start-Job` / 隐藏窗口 / 后台进程 [[memory:ykqwnv9c]]。
2. WHEN 出现命名歧义（如 lego baseline 路径已知 fallback）THEN 系统 SHALL 直接复用 W1 聚合脚本的 `LEGO_DEPTHV2_FALLBACK` 处理逻辑，不重新发明。
3. WHEN 任意环节 OOM / 训练崩溃 THEN 系统 SHALL 在 per-scene 表"备注"列照实写明，禁止静默跳过、禁止 cherry-pick。
4. WHEN W2 实施过程中需要修改任何**未在需求 1–6 列出的文件** THEN 系统 SHALL 先停止、用 `ask_user_input`（或在汇报中明确告知）确认，禁止"顺手优化"。
5. WHEN 出现 plan 流程未涵盖的新需求（如改 prior depth 模型、扩 LLFF、加 dual-GS）THEN 系统 SHALL 退回 plan 模式重新走需求-任务规划流程，**不得**在本次实施中扩张范围。
