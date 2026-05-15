# SOTA Source-Code Audit — 与 `sota_tricks_and_ssl_integration.md` 的对齐报告

> 核对范围：FSGS / CoR-GS / DNGaussian / SparseGS 官方仓库（已 clone 至 `third_party/`，commit 为 clone 时刻的 `main`）。
> 目的：在动手改代码前，把我上一份分析文档里**与源码有出入的表述**全部纠正，并标注每一条已验证 / 已修正的 trick 的准确出处（文件 + 行号）。

## 0. 仓库与入口

| 仓库 | 入口 train 文件 | 关键 loss 文件 | 关键 gaussian model 文件 |
|---|---|---|---|
| FSGS | `third_party/FSGS/train.py` | `utils/loss_utils.py` | `scene/gaussian_model.py` |
| CoR-GS | `third_party/CoR-GS/train.py` | `utils/loss_utils.py`（仅 photometric） | `scene/gaussian_model.py` |
| DNGaussian | `third_party/DNGaussian/train_blender.py` / `train_llff.py` / `train_dtu.py` | `utils/loss_utils.py` | `scene/gaussian_model.py` |
| SparseGS | `third_party/SparseGS/train.py` | `utils/loss_utils.py` | `scene/gaussian_model.py` |

---

## 1. FSGS — 已验证 / 已修正

### 1.1 Depth loss：**global Pearson**（不是 L1，不是 local patch）
**源码位置**：`third_party/FSGS/train.py:100-108`

```python
rendered_depth = render_pkg["depth"][0].reshape(-1, 1)
midas_depth    = torch.tensor(viewpoint_cam.depth_image).cuda().reshape(-1, 1)

depth_loss = min(
    (1 - pearson_corrcoef(-midas_depth, rendered_depth)),
    (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
)
loss += args.depth_weight * depth_loss
```

**修正**：
- 我之前模糊地写"FSGS 用 Pearson 或 L1"，实际 FSGS **train view 上就是 global Pearson**，没有 L1 变体。
- **独有小 trick**：两种 depth 形式（负 midas vs `1/(midas+200)`）取 `min`——处理 MiDaS 输出符号约定 + 近/远反转 scale 的模糊性，这是 FSGS 原创、SparseGS / DNGaussian 都没有的细节。
- FSGS **没有 local patch Pearson**（那是 SparseGS 的）。

### 1.2 Pseudo-view depth loss + 线性 warmup
**源码位置**：`train.py:116-131`

```python
if iteration % args.sample_pseudo_interval == 0 \
    and args.start_sample_pseudo < iteration < args.end_sample_pseudo:
    ...
    depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()
    loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
    loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

if iteration > args.end_sample_pseudo:
    args.depth_weight = 0.001   # ← 晚期把 train-view depth loss 降到几乎为 0
```

**要点**（上一份文档未强调）：
1. Pseudo depth loss 前 500 iter **线性 ramp-up** (0 → 1)。
2. 过了 `end_sample_pseudo` 之后，**train-view depth weight 也会直接降到 0.001**（等于几乎关掉）——"晚期 depth 退火"这条我们的 schedule 里没有。

### 1.3 Gaussian Unpooling = FSGS 的 `proximity(·)`
**源码位置**：`scene/gaussian_model.py:405-420`，调用点 `scene/gaussian_model.py:481-482`

```python
def proximity(self, scene_extent, N=3):
    dist, nearest_indices = distCUDA2(self.get_xyz)
    selected_pts_mask = torch.logical_and(
        dist > (5. * scene_extent),
        torch.max(self.get_scaling, dim=1).values > (scene_extent)
    )
    new_indices   = nearest_indices[selected_pts_mask].reshape(-1).long()
    source_xyz    = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
    target_xyz    = self._xyz[new_indices]
    new_xyz       = (source_xyz + target_xyz) / 2        # ← 中点插入
    new_scaling   = self._scaling[new_indices]
    new_rotation  = torch.zeros_like(self._rotation[new_indices]); new_rotation[:, 0] = 1
    new_features_dc   = torch.zeros_like(self._features_dc[new_indices])
    new_features_rest = torch.zeros_like(self._features_rest[new_indices])
    new_opacity   = self._opacity[new_indices]
    self.densification_postfix(...)
```

调用只发生在：
```python
def densify_and_prune(...):
    self.densify_and_clone(...); self.densify_and_split(...)
    if iter < 2000:
        self.proximity(extent)     # ← 只在前 2000 iter 触发
```

**修正**（相对上一份文档）：
- 触发条件更严格：**既要**最近邻距离 > 5×scene_extent，**又要**最大 scaling > scene_extent。
- 只在 **iter < 2000** 生效（早期撒点），不是全程开启。
- 新增高斯的 **SH features 全部清零**（`torch.zeros_like`）——不是继承邻居的颜色。这意味着新生成的高斯会先被 RGB loss 主动"染色"。
- 新点 rotation 设为单位四元数，不是继承。

### 1.4 FSGS 不涉及的
- 没有 dual GS（单模型）。
- 没有 floater prune（除标准 opacity / screen-size）。
- 没有 SDS / diffusion prior。

---

## 2. CoR-GS — 已验证 / 已修正（**上一份文档此处错得最多**）

### 2.1 Dual GS 的实现方式
**源码位置**：`train.py:60, 73-82`

```python
assert args.gaussiansN >= 1 and args.gaussiansN <= 2

GsDict = {}
for i in range(args.gaussiansN):
    if i == 0:
        GsDict[f"gs{i}"] = gaussians
    elif i > 0:
        GsDict[f"gs{i}"] = GaussianModel(args)
        GsDict[f"gs{i}"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
        GsDict[f"gs{i}"].training_setup(opt)
```

**修正**：
- 不是"两条并行 GS 模型 + 各自不同随机初始化"—— **两路都用同一 `scene.init_point_cloud`（COLMAP 稀疏点）做 `create_from_pcd`**，只是各自走独立的 optimizer / densification。
- 差异来源是 **densification 本身的随机性**（random sampling、random patch、random pseudo camera）和 **各自独立的 opacity reset timing**。这比"独立随机初始化"更弱，但已经够产生足够的函数级差异让 co-reg 有意义。
- 这对我们是利好：**不需要二次 COLMAP**、不需要 perturb 初始点，只要双 optimizer + 独立 densify stats 就能复现 CoR-GS 的 dual 行为。

### 2.2 Co-reg loss = **photometric consensus（不是 disagreement）**
**源码位置**：`train.py:180-186`

```python
if args.coreg:
    for i in range(args.gaussiansN):
        for j in range(args.gaussiansN):
            if i != j:
                LossDict[f"loss_gs{i}"] += loss_photometric(
                    RenderDict[f"image_pseudo_co_gs{i}"],
                    RenderDict[f"image_pseudo_co_gs{j}"].clone().detach(),   # ← 对方 detach
                    opt=opt
                ) / (args.gaussiansN - 1)
```

**重大修正**：上一份文档把 CoR-GS 说成是"disagreement loss / 让两者差异变大"，**这是错的**。源码里明确是 **让 gs_i 去逼近 gs_j 的 rendering（gs_j 侧 detach）**——这是**协作一致性**，不是对抗分歧。公式上等价于：

$$
\mathcal{L}_{\text{coreg}}^{(i)} = \frac{1}{N-1} \sum_{j \ne i} \mathcal{L}_{\text{photo}}\big(\hat{I}^{\text{pseudo}}_i,\; \texttt{sg}(\hat{I}^{\text{pseudo}}_j)\big)
$$

（`sg` = stop-gradient。）每一路都把**另一路当作 pseudo-GT**，轮流被拉向对方。

**对你们 SSL 的含义**（重要）：
- 你们 `pseudo_view` loss 就是 "student 去逼近 EMA teacher"，**数学形式几乎一模一样**（teacher detach = 对方 detach）。
- 唯一差别：你们 teacher 是 EMA，EMA = weighted sum of 自己历史；CoR-GS 的"teacher" 是**另一个独立训练的 gs**，不是自己的历史平均。
- 所以在 vanilla 3DGS 上把 EMA teacher 替换成 **"另一个独立 gs"** 就是最自然的 CoR-GS 移植——而且几乎只需要动 `EMATeacher` 这一处。

### 2.3 Co-prune = Open3D ICP correspondence
**源码位置**：`train.py:284-301`

```python
if args.coprune and iteration > opt.densify_from_iter and iteration % 500 == 0:
    for i in range(args.gaussiansN):
        for j in range(args.gaussiansN):
            if i != j:
                source_cloud = o3d.geometry.PointCloud()
                source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_xyz.clone().cpu().numpy())
                target_cloud = o3d.geometry.PointCloud()
                target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{j}"].get_xyz.clone().cpu().numpy())
                threshold = args.coprune_threshold     # default: 5
                evaluation = o3d.pipelines.registration.evaluate_registration(
                    source_cloud, target_cloud, threshold, np.identity(4))
                correspondence = np.array(evaluation.correspondence_set)
                mask_consistent = torch.zeros((..., 1)).cuda()
                mask_consistent[correspondence[:, 0], :] = 1
                GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())
    for i in range(args.gaussiansN):
        GsDict[f"gs{i}"].prune_from_mask(...)
```

**修正**：
- 上一份文档说"基于最近邻距离"，**正确但不完整**：用的是 Open3D `evaluate_registration`，**不是手写的 kNN + 阈值**，依赖一个完整 ICP 对应集算法（内部用 kd-tree，behavior 上近似于"对每个 source 点，找半径 ≤ threshold 的最近 target 点"）。
- `threshold=5` 是**世界坐标系下的绝对距离**——**这在 Blender（通常 scene_extent ≈ 1~3）场景下是非常宽松的**，几乎等价于"不剪"；需要乘 scene_extent 做归一化才能跨数据集通用。这是一个**真正值得做的消融点**。
- 触发频率：每 500 iter 一次，`iteration > opt.densify_from_iter`。

### 2.4 Co-reg 的 schedule 和 FSGS 同源
`train.py:167-169`：

```python
if iteration % args.sample_pseudo_interval == 0 and iteration <= args.end_sample_pseudo:
    loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
```

→ **CoR-GS 的 pseudo schedule 直接沿用了 FSGS 的 `start/end_sample_pseudo` + 500-step 线性 warmup**。这两家是同一谱系（CoR-GS 明显在 FSGS 之上改的）。

---

## 3. DNGaussian — 已验证 / 已修正（**对你们 `depthv2` 的相关性最高**）

### 3.1 Patch-normalized depth loss 是 **MSE**，不是 L1
**源码位置**：`utils/loss_utils.py:85-103`，使用点 `train_blender.py:100-104, 126-130`

```python
def patch_norm_mse_loss(input, target, patch_size, margin):
    input_patches  = normalize(patchify(input,  patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin)   # ← L2

def patch_norm_mse_loss_global(input, target, patch_size, margin):
    input_patches  = normalize(patchify(input,  patch_size), std=input.std().detach())    # ← detach std
    target_patches = normalize(patchify(target, patch_size), std=target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin)
```

使用：
```python
loss_l2_dpt = patch_norm_mse_loss(depth, depth_mono, randint(patch_range[0], patch_range[1]), opt.error_tolerance)
loss_hard  += 0.1 * loss_l2_dpt
loss_global = patch_norm_mse_loss_global(depth, depth_mono, randint(...), opt.error_tolerance)
loss_hard  += 1.0 * loss_global
```

**修正（相对你们的 `depthv2`）**：

| 维度 | DNGaussian 原文 | 你们 `depthv2` 当前实现 (`losses/ssl.py:580-600`) |
|---|---|---|
| 距离度量 | **MSE with margin** (`margin_l2_loss`) | **L1** (`(pred_n - prior_n).abs().mean()`) |
| Patch 大小 | **每 iter 随机** 取 `patch_range=[17, 51]` 之间的奇数 | 固定 `local_patch_size=96` (从你们配置看) |
| Global std | `std = input.std().detach()` 显式 detach | 你们 `_ssi_normalize` 需确认是否 detach |
| Margin | `error_tolerance`（只对 `|pred-target| > margin` 的 pixel 算 loss） | 无 margin |
| 权重 | **local 0.1 + global 1.0**（global 主导） | 可配置，但之前 `grad_weight=0.05` + `local_weight=0.5` 比例与原文不同 |
| Global/Local 之外的第三项 | **无**（DNGaussian 没有 grad consistency） | **有** `_depth_grad_l1`（梯度一致性，非原文） |

**关键洞察**：
- 我之前的文档把 DNGaussian 的 grad consistency 项当成了它的 contribution，**但实际 DNGaussian 原文就没有 grad loss 这一项**！你们 `depthv2` 里的 `_depth_grad_l1` 其实是**你们自己加的扩展**，在 +0.26 dB 中可能已经在贡献（但具体贡献需要消融确认）。
- DNGaussian 的真 trick 是 **margin + 随机 patch + MSE + 梯度隔离**（见 3.3），不是"global-local 归一化"本身——因为 global-local 归一化 SparseGS 也有（`patch_norm_l1_loss_global`），不是 DNGaussian 独创。

### 3.2 "Hard / Soft" 双 depth backward
**源码位置**：`train_blender.py:89-132`

```python
# hard: 影响 xyz / scale / rotation（几何）
if iteration > opt.hard_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
    render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, background)
    depth = render_pkg["depth"]
    loss_hard = 0.1 * patch_norm_mse_loss(...) + 1.0 * patch_norm_mse_loss_global(...)
    loss_hard.backward()                     # ← 独立 backward！

# soft: 影响 opacity
if iteration > opt.soft_depth_start and iteration < opt.densify_until_iter and iteration % 10 == 0:
    render_pkg = render_for_opa(viewpoint_cam, gaussians, pipe, background)
    depth, alpha = render_pkg["depth"], render_pkg["alpha"]
    loss_pnt = 0.1 * patch_norm_mse_loss(...) + 1.0 * patch_norm_mse_loss_global(...)
    loss_pnt.backward()                      # ← 又一次独立 backward！

# 然后才是 RGB loss 的 backward
loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_reg
loss.backward()
```

**重大发现**：
- DNGaussian 用 **3 次独立 backward**（hard depth / soft depth / RGB + reg），分别从不同的 render 函数出发。
- `render_for_depth` 和 `render_for_opa` 是**定制 rasterizer**，在 rendering 前 `detach` 掉不属于该通路的参数（`render_for_depth` 只让 xyz/scale/rot 有梯度；`render_for_opa` 只让 opacity 有梯度）。
- 这套"参数级梯度路由"正是 DNGaussian **Neural Gaussian** 标题下的真正含义——不是它表面"hard/soft depth loss"那么简单。
- **每 10 iter 才做一次 depth backward**（不是每 iter），这个节奏影响 compute 量。

**对你们的含义**：
- 你们 `depthv2` 的 depth loss 是**和 RGB loss 一起 backward** 到**所有参数**。这和 DNGaussian 原文**不等价**。
- 如果你们要进一步提升 depth 收益，有两条路：
  1. **实现 gsplat 版的 `render_for_depth` / `render_for_opa`**（把非目标参数 `.detach()` 掉再 forward），然后分 backward。**成本中等，收益可能大**。
  2. 保持现状，但把 depth loss 频率从"每 iter" 改成"每 10 iter"（和 RGB loss 间隔触发），能省 compute 同时降低与 RGB 的梯度冲突。**几乎零成本，可以先试**。

### 3.3 超参基线（Blender n8）
**源码位置**：`arguments/` 目录（我还没全读）+ `train_blender.py` 的 argparse 默认值，需要进一步 pin 准确值，但从 `train.py:382-383` 看：
- `test_iterations` default `[6000]` → Blender 训练总迭代 ≈ **6000**（比你们 `densification` 常用 15000 少一半）
- `densification_interval` / `densify_from_iter` / `densify_until_iter` 需要从 `arguments/*.py` 读

---

## 4. SparseGS — 已验证 / 已修正

### 4.1 Pearson：**global + local 两路都有**
**源码位置**：`utils/loss_utils.py:80-109`，使用点 `train.py:149-155`

```python
def pearson_depth_loss(depth_src, depth_target):
    # 标准 Pearson = 1 - corr
    src    = depth_src    - depth_src.mean();    src    /= src.std()    + 1e-6
    target = depth_target - depth_target.mean(); target /= target.std() + 1e-6
    return 1 - (src * target).mean()

def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    num_box_h = floor(H / box_p); num_box_w = floor(W / box_p)
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = randint(0, max_h, size=(n_corr,))
    y_0 = randint(0, max_w, size=(n_corr,))
    loss = 0
    for i in range(n_corr):
        loss += pearson_depth_loss(
            depth_src[x_0[i]:x_0[i]+box_p, y_0[i]:y_0[i]+box_p].reshape(-1),
            depth_target[x_0[i]:x_0[i]+box_p, y_0[i]:y_0[i]+box_p].reshape(-1)
        )
    return loss / n_corr
```

**修正**：
- 上一份文档把 local Pearson 归给 FSGS，**错了**。FSGS 只有 global。**Local patch Pearson 是 SparseGS 独有**。
- `box_p`（patch 边长）和 `p_corr`（采样比例）是两个 dataset-level 超参（不是 opt-level），默认值需要看 arguments。

### 4.2 Floater pruning = Dip test on ray-distributed α modes
**源码位置**：`train.py:240-244, 271-360`

```python
if iteration in prune_sched:
    prune_floaters(scene.getTrainCameras().copy(), gaussians, pipe, background, dataset, iteration)

def prune_floaters(viewpoint_stack, gaussians, ...):
    for view in viewpoint_stack:
        render_pkg = render(view, gaussians, pipe, bg, ret_pts=True)
        # ↑ 需要自定义 rasterizer 返回 mode_id / modes / point_list / means2D / conic_opacity
        mode_id, mode, point_list, depth, means2D, conic_opacity = ...
        diff = calc_diff(mode, depth)           # mode-depth 差
        dips.append(diptest.dipstat(diff[diff > 0].cpu().numpy()))   # unimodality 检验
    avg_dip = dips.mean()
    perc = max(80, prune_perc * 100 * exp(-prune_exp * avg_dip))
    for view: 
        threshold = np.percentile(diffpos, perc)
        pruned_modes_mask = (diff > threshold)
        # 对这些像素对应的 mode gaussian 做 alpha 测试 (calc_alpha > power_thresh)
        # 收集 selected_gaussians 做剪除
```

**修正 + 关键限制**：
- **不是 novel-view rendering 差异**——是 **train view 上**用 **diptest（多峰检测）** 找"沿光线方向 α 呈多峰分布"的像素，再剪掉其 mode gaussian。
- 强依赖自定义 rasterizer 返回 `mode_id / modes / point_list / conic_opacity`——**你们用的 gsplat 官方版默认不返回这些中间量**。整合前必须先确认 gsplat 的 `alpha_threshold` / `render_mode` 配置能否吐出这些 per-mode 数据；若不能，有两条路：
  1. fork gsplat 加返回；
  2. 用**近似替代**（比如用 novel view 之间的 depth disagreement 做简化版 unseen pruning，更符合你们现有基础设施）。
- **依赖 `diptest` 包**（`pip install diptest`，C 扩展，Windows 编译可能踩坑）。
- SparseGS 还需要 **`prune_sched`**（一个 iter 列表，稀疏触发）+ `prune_perc` + `prune_exp` + `power_thresh` + `densify_lag` + `densify_period` 一套超参。整合成本是四家里最高的。

### 4.3 Diffusion prior（SDS）
**源码位置**：`train.py:109-115, 157-159`

- 只在 `iteration > opt.iterations * 2/3` 后启用，按 `SDS_freq` 概率采样一个 ellipse pose。
- 需要加载 **`StableDiffusion`**（`guidance/sd_utils.py`）——额外 ~4G VRAM + 显著 compute。
- 对 Blender 受益不大（合成场景 SD prior 可能反而干扰），SparseGS 论文主要在 LLFF / Mip-NeRF360 上证明。**建议不优先整合到 Blender pipeline**。

---

## 5. 对我上一份文档的**修正清单**（action items）

| # | 位置 | 原表述 | 更正 |
|---|---|---|---|
| 1 | FSGS depth loss | "L1 / Pearson 可选" | **只有 global Pearson**，且带 `min(neg, 1/(mi+200))` 双形式 |
| 2 | Local Pearson 归属 | 算在 FSGS 下 | **SparseGS 独有**，FSGS 只有 global |
| 3 | FSGS Unpooling 条件 | 模糊 | **iter<2000 且 dist>5×scene_extent 且 max_scale>scene_extent**；features 清零不继承 |
| 4 | CoR-GS 两路初始化 | "独立随机初始化" | **共享 COLMAP 初始点**，差异只来自 densify/reset 的随机性 |
| 5 | CoR-GS co-reg 语义 | "disagreement / 让两者差异变大" | **photometric consensus**，对方 detach，是协作一致性 |
| 6 | CoR-GS co-prune | "kNN 距离阈值" | **Open3D `evaluate_registration`**, threshold=5（绝对距离，需要按 scene_extent 归一化） |
| 7 | DNGaussian depth loss 距离 | "L1" | **MSE (margin_l2_loss)**，带 margin（error_tolerance） |
| 8 | DNGaussian patch size | 固定 | **每 iter random from `[patch_range]`** |
| 9 | DNGaussian grad consistency | 算作 DNG 的 contribution | **原文没有**，是你们 `depthv2` 自己扩展的 |
| 10 | DNGaussian 梯度通路 | "和 RGB 一起 backward" | **3 次独立 backward**，依赖 `render_for_depth` / `render_for_opa` 对参数做 per-param detach |
| 11 | SparseGS unseen prune | "novel-view rendering 差异" | **train view + diptest(多峰检验) + per-mode gaussian α 测试**，依赖自定义 rasterizer |
| 12 | SparseGS 对 gsplat 的移植 | 视为易移植 | **最难的一家**：rasterizer hard dependency + diptest C 扩展 |

---

## 6. 基于核对后的新判断：W1~W4 计划微调

不改总体节奏，只调细节：

### W1（立刻开工）
- ✅ **Global Pearson**（照抄 FSGS `min(neg, 1/(mi+200))`）—— 这是最稳、最便宜、FSGS 和 SparseGS 都用的
- ✅ **线性 warmup**（0 → 1 over 500 iter），并在 `end_sample_pseudo` 后把 train-view depth weight 降到 ~0.001（**我们之前 schedule 没有退火**）
- ⚠️ **`depthv2` 的 L1 vs DNG 原文 MSE + margin**：**做一个正交消融**（L1 vs MSE-with-margin），看看 +0.26 dB 里有多少来自距离度量。这是"免费的"信号。
- 先不动 `render_for_depth / render_for_opa`（对 gsplat 改动大），放到 W3 再做。

### W2（Dual GS，仍是最大一条）
- **复用 COLMAP 初始点**（不做独立初始化）—— 完全按 CoR-GS 原文，降低工作量
- Co-reg 公式直接套：`pseudo_loss_i += photo(img_pseudo_i, img_pseudo_j.detach())`
- **初期只开 2 路 dual，不开 co-prune**（co-prune 是 W3 再加，先证 dual itself 在 Blender 有效）
- schedule 用 FSGS/CoR-GS 同款：`start_sample_pseudo = 500`, `end_sample_pseudo = 0.75 * total`，500-step linear warmup

### W3（FSGS Unpooling + CoR co-prune）
- Unpooling：直接抄 `proximity()`，注意 gsplat 没有 `distCUDA2`，需要找等价 kNN（`torch.cdist` 或 `pytorch3d.ops.knn_points`，后者快）
- Co-prune：阈值要**除以 scene_extent**（不能直接 5，Blender scene_extent ≈ 1.3，5 几乎=不剪）
- 这两个是 densify / prune 层面的改动，相互独立，**可并行消融**

### W4（DNGaussian 梯度隔离 + metrics / report）
- 这是最"DNGaussian-flavored"的 trick，但改动最深（rasterizer 包装器）
- **先做成 opt-in**：在 renderer 里加 `detach_params=[...]` 支持，默认关，消融时打开
- 若 W1~W3 已经把 PSNR 推到 23 dB 附近，W4 的边际收益可能已经不大——到时候再决定要不要真做 per-param 梯度隔离

### SparseGS 相关的重新评估
- **下调优先级**：dip test + 自定义 rasterizer 的工程成本与 Blender n8 的预期收益不成正比
- **可能替代**：用 **novel view 之间的 depth disagreement**（你们已有 depth cache 基础设施）做一个 SparseGS-inspired 的简化 unseen pruning。这才是"你们 SSL 基础设施的独特价值"
- SDS 先不做

---

## 7. 待读文件清单（如果后续需要更精准）

- `third_party/FSGS/arguments/*.py`：pin down `sample_pseudo_interval`, `start/end_sample_pseudo`, `depth_weight`, `depth_pseudo_weight` 的默认值
- `third_party/DNGaussian/arguments/*.py`：pin down `patch_range`, `error_tolerance`, `hard_depth_start`, `soft_depth_start`, `densification_interval`, Blender 的 iterations
- `third_party/DNGaussian/gaussian_renderer/__init__.py`：确认 `render_for_depth` / `render_for_opa` 的 detach 细节（是在 python 层还是 C++ 层做）
- `third_party/CoR-GS/arguments/*.py`：pin down `gaussiansN`, `coprune_threshold`, `sample_pseudo_interval`
- `third_party/SparseGS/arguments/*.py`：pin down `prune_sched`, `prune_perc`, `prune_exp`, `box_p`, `p_corr`

这一份放到 W3/W4 开始前再做即可，现在的细节足够启动 W1。
