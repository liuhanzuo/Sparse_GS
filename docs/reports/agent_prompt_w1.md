# Agent Prompt — W1: Pearson Depth Loss + FSGS-aligned Schedule

> 把下面整段直接粘给新 agent 开工。路径都是绝对路径，Windows / PowerShell。

---

## 你是谁

你是一名研究工程师，接手项目 `d:/SSL/sparse_gs`（3DGS + SSL，稀疏视角重建）。
前一位 agent 已经完成 **SOTA 仓库源码核对**，结论写在：

- `d:/SSL/sparse_gs/outputs/sota_source_audit.md`（**事实凭证，先读这个**）
- `d:/SSL/sparse_gs/outputs/sota_tricks_and_ssl_integration.md`（更早的整合方案，部分条目已被 audit 修订，以 audit 为准）
- `d:/SSL/sparse_gs/PROJECT_STATUS.md`（项目当前状态、已有 baseline 数字）

4 个官方仓库已 clone 到本地，**禁止联网重新下载**：

- `d:/SSL/sparse_gs/third_party/FSGS/`
- `d:/SSL/sparse_gs/third_party/CoR-GS/`
- `d:/SSL/sparse_gs/third_party/DNGaussian/`
- `d:/SSL/sparse_gs/third_party/SparseGS/`

需要核对公式/超参时**必须回到 third_party 读源码**，不要凭记忆或上网搜。

---

## 本轮目标（W1，只做这一件事）

在现有 `depthv2` 基线之上，引入 **FSGS 式 Pearson depth loss**，并把 **pseudo-view sampling schedule** 对齐 FSGS 原文，产出一次可对比的消融实验。

**不要越界做 W2/W3/W4 的东西**（dual GS、co-prune、unpooling、unseen prune 都先不碰）。

---

## 必须遵守的事实（来自 audit，别搞错）

1. **Pearson depth loss 的正确公式**（FSGS 原版，`third_party/FSGS/utils/loss_utils.py`）：

   ```python
   # rend_depth: [H,W] 渲染深度（disparity-like，越近值越大）
   # prior_depth: [H,W] 单目先验（MiDaS/DPT 类，disparity）
   loss = min(
       1 - pearson(-prior_depth.flatten(),          rend_depth.flatten()),
       1 - pearson(1.0/(prior_depth.flatten()+200), rend_depth.flatten()),
   )
   ```

   `+200` 是论文里的 magic number，不要改。两种形式取 min 是为了兼容单目先验的符号/尺度歧义。

2. **FSGS pseudo-view schedule**（`third_party/FSGS/train.py`）：
   - `start_sample_pseudo` 默认 2000，`end_sample_pseudo` 默认 9500
   - 权重 warmup：`w = min((iter - start_sample_pseudo) / 500, 1.0)`
   - 只在 `start < iter < end` 且满足采样间隔时触发
   - pseudo view 用 **渲染 depth 自身 Pearson 约束**（没有 GT depth），train view 才有 prior depth Pearson

3. **Pearson 不是 FSGS 独有**，SparseGS 也有，而且 SparseGS 额外有 **local patch Pearson**。**本轮只做 global Pearson（FSGS 版），不做 SparseGS 的 local patch 版**。

4. 现有 `depthv2`（在 `sparse_gs/losses/ssl.py`）是 **L1 版 patch-norm depth**（DNGaussian 的自扩展实现，原文是 MSE）。**本轮不要动 depthv2 的实现**，只把 Pearson 作为**并行新分支**加进去，通过 config 开关控制。

---

## 代码改动范围（越小越好）

### 1. `sparse_gs/losses/ssl.py`

新增函数：

```python
def pearson_depth_loss(rend_depth: Tensor, prior_depth: Tensor) -> Tensor:
    """FSGS-style global Pearson depth loss, train-view only.
    rend_depth / prior_depth: [H,W] or [B,H,W], same shape, no mask here
    (caller is responsible for masking invalid pixels before calling).
    """
    # 展平 + 去均值 + 归一化 std → pearson corr
    # 两种先验形式取 min，见上面公式
```

注意：
- 实现要 **batch 友好**（支持 `[B,H,W]`），但先以 `[H,W]` 单图为主，内部 `reshape(-1)`。
- **不要 detach prior**（FSGS 原实现没 detach，prior 本来就 requires_grad=False）。
- 如果 `prior_depth` 有无效区（NaN/0），调用方负责过滤，这个函数假设输入已 clean。

可选 helper：`pearson_depth_loss_pseudo(rend_depth_a, rend_depth_b)`，用于 pseudo view 渲染自约束（看你们 pseudo 分支怎么组织，如果暂时不方便就**本轮只做 train view Pearson**）。

### 2. `sparse_gs/losses/` 或 `configs/` 里的 loss registry

加开关，例如 config 里：

```yaml
ssl:
  depth:
    mode: "pearson"        # 新增，可选: "depthv2" | "pearson" | "both"
    pearson_weight: 0.05   # FSGS 默认，先别动
    depthv2_weight: 1.0    # 保持你们现在的权重
```

`mode="both"` 允许同时挂两个 loss，用于消融。

### 3. pseudo-view schedule

找到你们现有 pseudo-view 触发逻辑（应该在 `sparse_gs/trainer/` 或 `train.py` 附近），**参考 FSGS**：

- 加参数：`start_sample_pseudo`, `end_sample_pseudo`, `sample_pseudo_interval`
- 权重 warmup：`w_pseudo = min(max(iter - start, 0) / 500, 1.0)`
- 你们 Blender n8 现在总 iter 多少？如果是 30k，**按比例缩放 start/end**（原 FSGS 是 10k 训练；建议 start=2000、end=iter_total*0.95，保持相对位置一致），这个缩放比例写死在 config 里，不要 hardcode。

**如果你们已有 pseudo view 机制**，只改 schedule 参数和 warmup 权重，不要重写采样策略。

### 4. Config 文件

在 `configs/` 下**新建**（不要覆盖现有 config）：

- `configs/blender_n8_pearson.yaml`：只开 Pearson，关 depthv2
- `configs/blender_n8_depthv2_pearson.yaml`：depthv2 + Pearson 叠加

现有 baseline `configs/blender_n8_depthv2.yaml`（或类似名，看实际）保持不动，作为对照组。

---

## 验收标准

### 代码层面

- [ ] `pearson_depth_loss` 有单元测试：
  - 两张相同 depth → loss ≈ 0（可能不严格 0 因为 `1/(x+200)` 形式，但应 < 1e-3）
  - 随机 depth vs 反向 depth → loss ≈ 2（两个 pearson 都接近 -1）
  - 支持 autograd，`.backward()` 不报错
- [ ] `mode="both"` 时两个 loss 都被正确叠加到 total，log 里能分别看到
- [ ] pseudo warmup 的 `w` 在 `iter=start` 时是 0、`iter=start+500` 时是 1.0，可打 log 验证

### 实验层面

跑 **3 条曲线**（Blender n8，同 seed，同 iter 预算）：

1. `blender_n8_depthv2.yaml`（现有 baseline，可能已有结果就复用）
2. `blender_n8_pearson.yaml`（只 Pearson）
3. `blender_n8_depthv2_pearson.yaml`（depthv2 + Pearson）

每条跑 **8 scene 全量**（chair/drums/ficus/hotdog/lego/materials/mic/ship），报 **avg PSNR / SSIM / LPIPS**。

单 scene 若 OOM / 跑崩，在结果表里标注，不要静默跳过。

### 产出文档

训练完成后在 `d:/SSL/sparse_gs/outputs/` 下新建 `w1_pearson_ablation.md`：

- 3 条曲线的 avg 指标表（带 delta 列，相对 depthv2 baseline）
- 每个 scene 的 per-scene 指标（小字号表格即可）
- **关键结论 3 条**（不超过 3 条）：Pearson 是否提升、是否和 depthv2 互补、warmup schedule 是否起作用
- 如果结果负向，**照实写，不要 cherry-pick**；分析原因（prior depth 质量？权重？schedule？）

---

## 工作流程要求

1. **先读**（不写代码）：
   - `outputs/sota_source_audit.md` 全文
   - `PROJECT_STATUS.md`
   - `sparse_gs/losses/ssl.py` 现有实现
   - `third_party/FSGS/utils/loss_utils.py` 的 `pearson_depth_loss` 和 `local_pearson_loss`
   - `third_party/FSGS/train.py` 里 `start_sample_pseudo` 相关段落

2. **用 `todo_write` 建 todo list**，推荐分解：
   - [ ] 读源码 + 定位项目 pseudo-view 现有代码
   - [ ] 写 `pearson_depth_loss` + 单元测试
   - [ ] 接入 loss registry + config 开关
   - [ ] 调 pseudo-view schedule
   - [ ] 写两个新 config
   - [ ] 跑 3 条曲线
   - [ ] 产出 `w1_pearson_ablation.md`

3. **每完成一个 todo 立即 mark completed**，不要攒。

4. **遇到歧义**：**不要臆测**，去 `third_party/FSGS` 读源码。如果源码也说不清，在 todo 里记一条"待用户决策"，先用最合理默认值推进。

5. **运行长训练前**：先跑 **1 个 scene + 2k iter 的 smoke test**，确认 loss 曲线正常、没 NaN、GPU 利用率合理，再铺全量。

6. **不要重构**现有代码，不要"顺手"改你觉得不优雅的地方。**最小改动**。

---

## 踩坑预警

1. **Pearson loss 的梯度尺度**和 L1 很不一样（corr 是 -1~1 区间的），weight 不能直接照搬 depthv2 的。FSGS 默认 0.05，先用它；如果 loss 曲线压不过 RGB loss 再调。

2. **prior depth 的符号**：MiDaS/DPT 输出是 **disparity**（近处大），Blender GT depth 是 **metric depth**（近处小）。用哪个作为 prior？
   - 如果你们 `depthv2` 现在用的是 Blender GT depth → **Pearson 公式里 prior 要取负号** 或 **用 `1/(depth+eps)` 转成 disparity**。
   - 这块**务必读你们现有 depth 供给代码**确认语义，不要盲抄 FSGS（FSGS 用的是 MiDaS）。

3. **pseudo-view 如果还没实现**：本轮 **只做 train-view Pearson**，pseudo-view 留给 W2。不要因为 pseudo 做不动就卡住整个 W1。

4. **Windows PowerShell 跑训练**：长任务建议用 `Start-Job` 或直接在独立窗口跑，不要阻塞 agent 主进程。日志写到 `outputs/logs/w1_*.log`。

5. **评测脚本**：用项目现有的 eval 脚本，**不要自己重写 metric**。如果找不到，在 `scripts/` 下找 `eval_*.py` 或 `test.py`。

---

## 完成信号

全部做完后，你的最后一个工具调用应该是 `open_result_view`，target 是 `d:/SSL/sparse_gs/outputs/w1_pearson_ablation.md`。

然后简要汇报：
- 3 条曲线的 avg PSNR delta
- 是否值得进入 W2
- 有无异常需要用户介入

**不要主动进 W2**，等用户 review W1 结果后再说。
