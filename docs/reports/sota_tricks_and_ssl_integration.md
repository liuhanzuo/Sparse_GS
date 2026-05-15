# SOTA tricks × 我们的 SSL —— 差距诊断与融合方案

> 配套文档：`outputs/sota_comparison.md`（差距定量化）、
> `outputs/depthv2_global_local_summary.md`（DNGaussian-style depth v2 结果）、
> `outputs/patch_gan_full_trial_summary.md`（GAN 负结果）、
> `outputs/blender_n8_table.md`（Blender avg-8 实测）。
> 本文件只谈 **tricks 对齐 + 结合方案**，不重复差距数字。
> 更新于 2026-05-11。

---

## 0. TL;DR

我们目前的 `DAv2 + multiview-photo (+ depth v2)` 把 Blender avg-8 n=8 推到 **19.37 dB**，
距 FSGS / CoR-GS（~24.5）仍有 **~5 dB**。`depth-v2` 把 DNGaussian 的 global-local +
梯度一致性吃进来，只拿到 +0.26 dB —— 说明"depth 类 trick 的红利基本吃完了"。
**剩下的 5 dB 不在 loss，在 densify / 初始化 / pseudo-view / dual model 这四条线**。
SOTA 的 main contributions 中，**有 3 个是我们还没碰过的增量正交方向**：

| SOTA contribution | 本质 | 是否 depth 以外的新信息 | 与我们 SSL 的关系 |
|---|---|:-:|---|
| FSGS **Gaussian Unpooling** | 在邻近点之间主动插点，把"空心洞"填掉 | ✅ | **正交**，直接替换/增强 DefaultStrategy |
| CoR-GS **双 GS + co-pruning / co-reg** | 用两个独立初始化 GS 互为教师，剔除只有一个模型拟合的 floater | ✅ | **正交**，天然契合我们已有的 `EMATeacher` 架子 |
| SparseGS **unseen-view floater pruning** | 在 pseudo view 上检测"从未见过但透明度高"的 Gaussian，直接删 | ✅ | **正交**，而且能救活我们现在 "pseudo_view = 中性" 的尴尬 |
| DNGaussian **global-local + grad depth** | 尺度-偏移不变的多尺度 depth reg | ⚠️ 已吃 | 我们 `depth v2` 已实现，收益已榨干 |
| SparseGS **SDS / diffusion prior** | 2D diffusion 当外部先验 | ⚠️ 可选 | 工程重、吃显存，性价比最低 |
| FSGS/SparseGS **pseudo-view 采样** | 在训练位姿之间插值采样虚拟相机 | ✅ | 我们有框架但"没外部先验 → 中性"，要靠上面三项激活 |

**最关键的洞察**：我们 `PROJECT_STATUS.md §2` 里有一条负结果 —— 
*"pseudo_view / ema_teacher 在 vanilla 3DGS 上是中性/负向的，因为 student 在
train view 上已经完美，teacher 不带新信息"*。**这三条 SOTA trick 的共同点，
就是它们都在给 pseudo view / teacher 注入"新信息"**：

- FSGS Unpooling → 让几何本身变得不完美（有洞 → 填洞），teacher 就有事做了；
- CoR-GS dual → 用一个**独立初始化**的第二模型当 teacher，天然避开 student ≈ teacher；
- SparseGS unseen pruning → 显式利用 "novel pose 上 student 自己都不自信" 的信号。

所以我们现有的 SSL 基础设施（EMA teacher、pseudo_view、detach_geometry、
occlusion check、depth prior cache）**不是白建的**：它们是激活这些 SOTA
tricks 的前置，现在缺的是把上面三个"信息源"接上去的那一小块胶水。

---

## 1. 各 SOTA 方法的 tricks 清单（拆到能落地）

对每个方法，列出 (a) main contribution、(b) 具体 tricks、(c) 解决的什么问题、
(d) 与我们当前方法的关系。

### 1.1 FSGS (ECCV'24, 24.6 dB avg-8 @ n=8)

- **Main contribution**：*Gaussian Unpooling* —— 在邻近点中点主动插入新 Gaussian。
- **Tricks**：
  1. **Proximity-guided densification**：不依赖 `grad2d` 阈值，而是基于 kNN
     距离大的点插值（"稀疏处补人"），与 3DGS 原生 split/clone **并行使用**。
  2. **Depth prior + Pearson correlation loss**：不是 L1 / SSI-L1，而是 Pearson
     相关系数（对尺度-偏移完全不变，比 median/MAD 更鲁棒）。
  3. **Pseudo-view 采样**：在训练相机之间做球面插值，渲染后用 depth prior
     做正则。
  4. **Warmup schedule**：先 warmup 几何（~500 iter）再开 depth / pseudo。
- **解决的问题**：
  - 我们 `PROJECT_STATUS §4 O5` 提到 "n=3 下覆盖是硬上限"—— Unpooling 正面
    攻击这个问题：训练视图稀疏时，`grad2d` 本身就稀疏，3DGS 的 densify
    触发次数不够；Unpooling 用几何 proximity 显式补点。
  - Pearson loss 解决 DAv2 depth 在室内/forward-facing 尺度偏移不稳定时
    SSI-L1 会把错误尺度也拟进去的问题。
- **与我们的关系**：
  - Pearson 替换：**10 行代码**，直接在 `losses/ssl.py::_depth_consist`
    加一个 `mode: pearson` 选项。预计 +0.1~0.3 dB，便宜。
  - Unpooling 替换：**不是 10 行**，要改 `strategies/densify.py`，但
    gsplat 的 `DefaultStrategy` 暴露了 `refine()` 钩子，可以做成一个
    `UnpoolingStrategy` wrap / 并行。这是**性价比最高**的一条路。

### 1.2 CoR-GS (ECCV'24, ~24.5 dB avg-8 @ n=8)

- **Main contribution**：训练**两个独立初始化的 3DGS 模型**，用它们在训练视图的
  *点云差异* 和 novel view 的 *渲染差异* 做互相监督。
- **Tricks**：
  1. **Co-pruning**：同一 Gaussian 位置附近没有 "对家" 的点 → 判定为 floater，
     剪掉。本质上就是"两次独立随机初始化后都稳定存活的点才保留"。
  2. **Pseudo-view co-regularization**：novel pose 上，两个模型应该渲染出
     一致的 RGB；不一致的 pixel 就是两个模型各自过拟合到训练视图的 floater。
  3. **Uncertainty weighting**：一致度高的像素权重大，低的权重小（避免两个
     都错的时候相互拉向错误答案）。
  4. 两模型**不共享** densify state / 优化器。
- **解决的问题**：
  - 我们 `PROJECT_STATUS §2 contribution 4` 的负结果 *"pseudo_view 中性/-0.5 dB"* —— 
    根因是 EMA teacher 其实就是 student 自己（snapshot mode），没有新信息。
    CoR-GS 的 dual 模型架构本质上**彻底消除**这个问题：teacher 是独立初始化的
    第二个模型，它在哪里过拟合、在哪里稳定，统计上独立。
  - 也解决 `sota_comparison §4(4)` 说的 "floater ambiguity" ——
    PatchGAN 试图用 2D 判别器解决，失败了；CoR-GS 的 floater 检测是
    **几何/view-consistency** 层面的，正好对上我们的诊断。
- **与我们的关系**：
  - 我们已经有 `sparse_gs/models/ema.py::EMATeacher`，里面已经处理了 densify
    boundary、snapshot 接口。**把 EMATeacher 替换为 "第二个独立初始化的
    GaussianModel + 自己的 Strategy + 自己的 optimizer"，就等价于 CoR-GS 的
    dual model**。`pseudo_view` 的 context 改用第二个模型渲染，
    `_ema_teacher_consist` 自然升级为 `_dual_gs_consist`。
  - 训练时间和显存大约 ×1.5~1.7（不是 ×2，因为 batch 共享、Adam state
    是独立但总量加一倍）。RTX 5090 可接受。

### 1.3 SparseGS (3DV'24, 22.8 dB avg-8 @ n=8)

- **Main contribution**：把 pseudo view 当一等公民，引入三个专门为它设计的 loss。
- **Tricks**：
  1. **Softmax-weighted depth correctness loss**：不是直接 L1(depth, prior)，
     而是沿射线做 softmax 后的 "期望深度 vs prior 深度" 的差，对 multi-modal
     射线（有多个 surface）更鲁棒。
  2. **Unseen-view floater pruning**：在 pseudo view 上渲染，把 *只在 pseudo
     view 上被看到、但训练视图从未见过* 的高不透明 Gaussian 直接裁掉 ——
     这些就是 floater 的几何定义。
  3. **SDS / diffusion prior**：用 Stable Diffusion 给 pseudo view 打分；工程最重。
  4. **Patch-based depth ranking loss**：在小 patch 内只惩罚 depth 顺序错误，
     忽略绝对值。
- **解决的问题**：
  - 解决"novel view 上长出 floater"这个 3DGS 通病。我们的 mv-photo v3 的
    occlusion gate 是"被遮挡就不监督"，是**被动防御**；SparseGS 的 unseen
    pruning 是**主动进攻**，二者互补。
  - Depth ranking 对 LLFF / 室内场景尤其友好，因为绝对尺度对 monocular
    depth 本来就不可靠。
- **与我们的关系**：
  - Unseen-view pruning：可以直接加到 `_multiview_photo` 的 occlusion check
    旁边 —— 我们已经在渲染 neighbor view 了，多一步 "在 pseudo pose 渲染
    一次，统计每个 Gaussian 在 pseudo view 的可见度" 就能得到裁剪信号。
    落点是 `strategies/densify.py` 的 `refine()` 钩子。
  - Depth ranking loss：在 `_depth_consist` 里加 `mode: ranking`，
    以 patch 为单位算 pairwise sign 一致性，**纯加项**，不影响现有管线。
  - SDS：不建议碰，投入产出比最差，`sota_comparison §6` 已经把它排除。

### 1.4 DNGaussian (CVPR'24, 24.3 dB avg-8 @ n=8)

- **Main contribution**：*Hard/Soft depth reg + Global-Local normalization*。
- **Tricks**：
  1. **Hard depth**：直接对中心（center）深度做 L1。
  2. **Soft depth**：对 alpha-blended expected depth 做 L1。
  3. **Global-local normalization**：全图 SSI + 多个 local patch 各自 SSI，
     防止远景小物体被近景大物体主导 normalize。
- **解决的问题**：DAv2 在一张图内尺度不均匀（近+远同时存在）时的归一化偏差。
- **与我们的关系**：**我们已经实现并验证**（`outputs/depthv2_global_local_summary.md`），
  avg-8 +0.26 dB，7/8 scene 改善。**这条路的红利已经榨干**，不要再加 depth 项。

---

## 2. 我们现有方法的真实痛点（从 `PROJECT_STATUS` + 实测推出）

按"还有多少 dB 可捞"排序：

| # | 痛点 | 证据 | 可解释的 SOTA trick |
|:-:|---|---|---|
| **P1** | **densify 在稀疏视图下触发次数不足**，"空心区域"永远补不上点 | `#G` 在 baseline 和 SSL 间几乎不变（400~450k）；n=3 几何极残缺 | FSGS Unpooling |
| **P2** | **novel view 上有大量 floater**，student 自己都不自信 | mv-photo v3 的 occlusion gate `kept_ratio` 在 n=6 = 0.68 → 32% pixel 被认定不一致 | SparseGS unseen-pruning、CoR-GS co-pruning |
| **P3** | **pseudo_view / ema_teacher 是"零信息"** | `PROJECT_STATUS §2` 写得很清楚：student ≈ teacher | CoR-GS dual model（唯一能真正破坏这个对称） |
| **P4** | **n=3 Blender 硬上限** = 360° 覆盖问题 | `sota_comparison §4(3)` | 没有任何 2D 正则能解决；只能靠 pseudo view + 外部先验（A1/B2） |
| **P5** | **forward-facing 场景 SSIM/LPIPS 离 SOTA 最远**（LLFF SSIM 0.40 vs 0.71） | §3.2 | 架构问题 + 训练长度，非 loss 问题 |
| **P6** | depth prior 尺度不变归一化还能再稳一点 | ficus 在 depth v2 下退化 | FSGS Pearson loss |

**重要**：P4 / P5 用 loss 改不动，只能靠架构/数据；P1 / P2 / P3 是
**能用几百行代码撬动 dB** 的地方，也是下一步的主战场。

---

## 3. 融合方案：SSL ⊕ SOTA tricks

我们不要"抄某个 SOTA"，而是把 **我们 SSL 框架当作底座**，选择性地嫁接
SOTA tricks。原则：

1. **保留 SSL 的 contribution 叙事**：multiview-photo (无先验 cross-view)、
   `detach_geometry` / occlusion gate、EMA teacher、depth-SSI+local+grad。
2. **嫁接正交信息源**：FSGS Unpooling（几何补点）、
   CoR-GS Dual GS（独立 teacher）、SparseGS Unseen-pruning（floater 清除）。
3. **每嫁接一项都做 A/B**，保持 `PROJECT_STATUS §3` 的"每条贡献都有消融支撑"
   的学术规范。

### 3.1 短期（≤1 周，期望 +1~2 dB，不改架构）

| # | 工作 | 落点 | 预期 | 工作量 |
|:-:|---|---|:-:|:-:|
| **F1** | **Pearson depth loss**（FSGS 风格） | `losses/ssl.py::_depth_consist`: 增加 `mode: pearson`，(x - x̄)(y - ȳ) / σ_xσ_y | +0.1~0.3 dB，修 ficus 退化 | 半天 |
| **F2** | **Depth ranking loss**（SparseGS 风格） | 同文件，新 helper `_depth_rank_patch`，在 96×96 patch 内采样点对算 sign 一致性 | LLFF 类场景 +0.2~0.5 dB | 1 天 |
| **F3** | **Sparse-tuned densify**（`PROJECT_STATUS TODO B3`） | `configs/base.yaml` 把 `refine_stop_iter` 往后推，`grow_grad2d` 降低，或换 `MCMCStrategy` | +0.3~1 dB "免费" | 半天 |
| **F4** | **Pseudo-view warmup schedule**（SparseGS/FSGS 风格） | `trainer.py`：pseudo view 从 iter 3000 才开，初期只监督 train views + mv | 修 `pseudo_view = -0.5 dB` | 1 天 |

F1-F4 加起来**不新增任何可训练参数，不动 EMA / strategy 代码**，是纯
"按钮"级改动，适合当下一周的 sanity-check 和论文消融表填格子。

### 3.2 中期（1~2 周，期望 +2~3 dB，动一次架构）

**F5. Dual Gaussian Radiance Field（CoR-GS 风格，但嫁接到我们 SSL 框架上）**

这是最有价值的一条路。现有脚手架：

- `sparse_gs/models/ema.py::EMATeacher` 已经处理了 densify boundary ✅
- `sparse_gs/losses/ssl.py::_pseudo_view / _ema_teacher_consist` 已经有
  render-with-alternative-model 的调用路径 ✅
- `detach_geometry` / strategy-info-isolation 已经保证 SSL 不污染 densify ✅

缺的就是：

- 把 `EMATeacher` 改造 / 复制为一个 `IndependentTwinGS`：**自己的**
  `GaussianModel`、**自己的**独立 init（不同随机 seed 的 random-in-box）、
  **自己的**optimizer + strategy。和 student 完全对称、交替训练或同步训练。
- 训练循环：每步 forward 两次（student / twin），两者分别用 photo loss，
  然后加 "co-reg" 项（pseudo view 渲染一致）和 "co-prune" 触发器（twin 附近
  没有对应点就剪）。
- pseudo_view / ema_teacher loss 改为用 twin 渲染做 target，学生消费。
  这样 `PROJECT_STATUS §2` 的"pseudo_view 中性"负结果直接被激活为正向。

预期：Blender avg-8 从 19.4 → 21~22 dB（CoR-GS 的 +2 dB 增量近乎可复现）。
**同时让我们已有的 EMA / pseudo_view 基础设施第一次真正产生收益** —— 这是
很干净的叙事：*"SSL 框架是正确的，之前只是缺一个独立教师"*。

工作量：代码 ~400~600 行（主要是 trainer 里的双模型循环 + 两套 optimizer），
训练成本 ×1.5~1.7。

### 3.3 中期 · Plan B（和 F5 正交，可并行）

**F6. Gaussian Unpooling（FSGS 风格）**

- 落点：`sparse_gs/strategies/densify.py`，新增 `UnpoolingStrategy`（或在
  `DefaultStrategy` 前后插入一个 pre-refine 钩子）。
- 逻辑：每 `unpool_every` 步，对每个 Gaussian 的 kNN（k=3~5）距离做统计，
  距离 top-K% 的点在中点 clone 一个新点，继承 SH / opacity。
- 配合 F3 的 densify 调参，把"n=3 下 Gaussian 总数偏少"直接打穿。
- 这条路和 F5 完全正交 —— F5 改 loss / teacher，F6 改 densify。

预期：n=3 / 6 上 +0.5~1.5 dB（FSGS 的 unpool 独立消融在论文里约 +0.8 dB）。
工作量 ~200~300 行。

### 3.4 中期 · Plan C（SparseGS floater pruning）

**F7. Unseen-view floater pruning**

- 落点：`strategies/densify.py` 的 `refine()` 后钩子 / 或单独一个
  `prune_unseen_in_pseudo` hook。
- 逻辑：每 N 步在若干 pseudo pose 上渲染一次，统计每个 Gaussian 的可见度
  `vis = Σ_{pseudo pose} α_contribution`；若 `vis_train < ε` 且 `vis_pseudo > τ`
  且 `opacity > 0.5` → 判定为 unseen-floater，剪。
- 与我们 mv-photo v3 的 occlusion gate 思路一致（都利用"被遮挡/看不到的
  Gaussian 是可疑的"），但一个是 loss 端，一个是 prune 端。
- 成本：每 N 步一次无梯度 render，和 occlusion check 在同一个数量级。

预期：LPIPS 收益大于 PSNR，解决 `patch_gan_full_trial_summary` 结尾
"剩余模糊主要是 floater" 的诊断。工作量 ~150~250 行。

### 3.5 长期（2~4 周，期望 +1 dB 但故事价值最大）

**F8. DINOv2 feature consistency on novel views**（`TODO B1`）

- 在 F5 / F6 / F7 打完之后才做，因为现在 pseudo view 上 student ≈ teacher，
  DINOv2 cos 一致性几乎一定是 0，加了也没用。
- 一旦 F5 就位，pseudo view 上 student 和 twin 渲染会有语义偏差，DINOv2
  cosine 就能把这偏差压到 "真实语义分布"上，而不是简单"两个模型像"。
- 正是 `PROJECT_STATUS §2 B1` 已经在路线图上的条目，用 F5 激活它。

**F9. Surface extraction & Chamfer eval**（`TODO A2 / S6`）

- 等 F6 (Unpooling) 收紧几何后再做，否则 mesh 会是 floater 主导。
- 能让 `sparse_view_ssl_surface*` 这一系列配置第一次有**该被评估的指标**，
  解除 `TODO S5` 的"PSNR 看不到 surface 收益"的尴尬。

---

## 4. 推荐下一步执行顺序

按 ROI 倒序：

```
Week 1:  F1 (Pearson)  + F3 (densify schedule)  + F4 (pseudo warmup)
         ↓  一次 avg-8 n=8 跑完：期望 19.4 → 20.5 dB 左右
Week 2:  F5 (Dual GS)  ←  最大的一条，应该用整周打磨
         ↓  一次 avg-8 n=8 跑完：期望 20.5 → 22.0 dB 左右
Week 3:  F6 (Unpooling)  + F7 (unseen pruning)
         ↓  一次 avg-8 n=8：期望 22.0 → 23.0 dB；LPIPS / SSIM 显著改善
Week 4:  F8 (DINOv2) + F2 (depth ranking) + 论文消融 / 图表
         ↓  冲 23.5~24 dB，正式 vs FSGS / CoR-GS 对齐（gap < 1 dB）
```

**每一条都是独立消融**，即使中途某条 F 失败，其他条的收益仍保留，
叙事不会塌。这也避免 "SDS / diffusion" 那种"要么都成要么全砸"的大赌。

---

## 5. 论文叙事角度（顺便）

把 F1~F7 装进我们已有的故事里，只需要改一句：

> 旧版：*"我们提出一套对 densify 安全的 SSL 框架，以 multiview-photo
> 作为主要 contribution；pseudo_view / EMA teacher 是负结果，证明
> 纯 self-distillation 无效。"*
>
> 新版：*"我们提出一套对 densify 安全的 SSL 框架，其价值在于**为几何
> 层面的外部信息源提供安全的注入接口**：我们在同一框架下集成
> (i) multiview-photo（无先验 cross-view）、(ii) DAv2 + DNGaussian-style
> global-local depth、(iii) CoR-GS-style dual-GS consistency、
> (iv) SparseGS-style unseen-view pruning 与 FSGS-style proximity
> unpooling。`pseudo_view / EMA teacher` 在 vanilla 3DGS 上是负结果
> 这一事实，恰恰证明 (iii)(iv) 的必要性 —— 我们的框架是第一个在同一
> SSL bank 内把它们组合起来并给出受控消融的工作。"*

这样我们的 "negative result" 反而变成了 "为什么 dual / unseen-pruning
非做不可" 的动机章节，而不是一段尴尬的 ablation。

---

## 6. 附：每条 SOTA trick → 代码落点快查表

| SOTA trick | 新增/修改文件 | 函数/类 | 是否需要动 strategy | 是否需要动 trainer |
|---|---|---|:-:|:-:|
| FSGS Pearson depth (F1) | `losses/ssl.py` | `_depth_consist` 新增 `mode: pearson` | ✗ | ✗ |
| SparseGS depth ranking (F2) | `losses/ssl.py` | 新 `_depth_rank_patch` helper | ✗ | ✗ |
| Sparse-tuned densify (F3) | `configs/base.yaml` | 调 `grow_grad2d / refine_stop_iter` | ✗ | ✗ |
| Pseudo warmup (F4) | `losses/ssl.py` / `trainer.py` | `start_iter` 字段全局化 | ✗ | 轻微 |
| **CoR-GS Dual GS (F5)** | `models/ema.py` → `twin.py`, `trainer/trainer.py` | 新 `IndependentTwinGS`，trainer 双循环 | 新 strategy 实例 | 大改 |
| **FSGS Unpooling (F6)** | `strategies/densify.py` | 新 `UnpoolingStrategy` 或 pre-refine hook | ✓ | ✗ |
| **SparseGS unseen pruning (F7)** | `strategies/densify.py` | `prune_unseen_hook` | ✓ | 轻微 |
| DINOv2 feature (F8) | `losses/ssl.py` | `_feature_consist` 去 stub 化 | ✗ | ✗ |
| Mesh extraction (F9) | `scripts/extract_mesh.py` (new) | — | ✗ | ✗ |

---

## 7. 总结一句话

**我们 SSL 框架的价值不是"现在就赢 SOTA"，而是它为 SOTA 的三条正交
contribution（Unpooling、Dual GS、Unseen Pruning）提供了一个干净的
注入口子。** 把这三条接上、每条都做独立消融，就能把 avg-8 Blender n=8
从 19.4 dB 推到 23~24 dB，同时让 `PROJECT_STATUS §2` 里那条"pseudo_view
负结果"从负债变资产。
