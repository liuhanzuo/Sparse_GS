# 实施计划

- [ ] 1. 创建 `sparse_gs/strategies/post_prune.py`：定义 `compute_unseen_mask` 与 `compute_floater_mask` 纯函数
   - 函数 1：`compute_unseen_mask(visibility_count: Tensor[N]) -> Tensor[N, bool]`，输入累加可见性张量、返回 `count == 0` 的 bool mask
   - 函数 2：`compute_floater_mask(depth, alpha, gaussian_ids_or_radii, *, alpha_thresh, thresh_bin, packed) -> Tensor[N, bool]`，按"alpha 过滤 → 归一化 depth → 按 thresh_bin quantile 截断 → 回投到高斯"的顺序实现
   - 函数 3：`apply_safety_cap(mask, max_ratio) -> Tuple[Tensor[N, bool], bool]`，若 mask 比例超过 max_ratio 则返回空 mask + over_aggressive=True
   - 仅依赖 `torch` + 标准库，可在 CPU 上运行；签名与文档与 requirements 2/3 完全对齐
   - _需求：1.2、2.1-2.3、3.1-3.5_

- [ ] 2. 在 `sparse_gs/trainer/trainer.py` 新增 `_periodic_unseen_floater_prune(step)`，挂在 `_enforce_cap_max()` 之后
   - 读取 `cfg["strategy"]["prune"]`（缺失视为全关 → 向后兼容）
   - 调度逻辑：仅在 `step >= start_iter and step <= stop_iter and (step - start_iter) % every_iter == 0` 时触发；否则零成本提前 return
   - 触发时按"unseen 先 → floater 后"的顺序：unseen 完成后用最新 N 重新算 floater 的 mask 长度
   - 全程 `torch.no_grad()`；调用 `gsplat.strategy.ops.remove(params, optimizers, state, mask)` 与 `_enforce_cap_max` 一致
   - 异常 `info` 跳过并 print `[w2-prune] skip step={step}: no visibility info`，**不 raise**
   - 命中 prune 后用 `tb.add_scalar("stats/prune_unseen_n", n)` / `stats/prune_floater_n` 上报，同时 stdout 打印
   - _需求：1.1-1.5、2.4-2.6、3.5-3.7、4.3、4.4_

- [ ] 3. 实现 `_collect_visibility_for_unseen()` 与 `_collect_floater_pixels_for_floater()` 两个 trainer 私有方法
   - `_collect_visibility`：对 `self._train_cams_gpu` 逐个 `renderer.render(...)`，按 packed 模式把可见性 `scatter_add_` / `|=` 到长度 N 的 bool 张量；最后用 `compute_unseen_mask` 得 unseen mask
   - `_collect_floater_pixels`：同上渲染，每 view 取 `depth/alpha/info`，调用 `compute_floater_mask` 得 per-view floater 高斯 mask；跨 view 取 `|`
   - 两者**复用现有 `self.renderer`**，禁止新建 rasterizer
   - _需求：2.1-2.3、3.1-3.4_

- [ ] 4. 在两个 base config 中加 `strategy.prune` schema（默认全关），并新建 W2 专用 config
   - 修改 `configs/base.yaml`（如存在 strategy 段；若无则跳过此项，由各 W2 config 自行声明完整段）补默认 `strategy.prune.{unseen,floater}.enabled=false`，**不改变既有行为**
   - 新建 `configs/blender_n8_depthv2_prune.yaml`（chair smoke 模板 + 完整 prune schema，按需求 4.2 给出默认值）
   - 新建 `configs/_w2_prune/blender_{scene}_n8_dav2s_depthv2_prune.yaml`（8 个文件，与 `_depth_v2/` 平行）
   - 仅 `_base_` 引用既有 depthv2 配置 + `strategy.prune` 覆盖，**禁止改写既有 depthv2 / pearson config**
   - _需求：4.1-4.4_

- [ ] 5. 写单元测试 `tests/test_post_prune.py`
   - 测试 1：构造 `visibility_count = torch.zeros(N)` 全零 + 末位 `+1`，验证 `compute_unseen_mask` 返回除末位外都为 True
   - 测试 2：mock 一个 64×64 假 depth/alpha + 假 `gaussian_ids`，验证 `compute_floater_mask` 在 `thresh_bin=0.13` 下选出的 mask 大小 ≈ 13% 像素覆盖范围
   - 测试 3：构造 mask 比例 60%，验证 `apply_safety_cap(mask, max_ratio=0.05)` 返回空 mask 且 `over_aggressive=True`
   - 仅 `torch` + `pytest`，无 GPU 依赖
   - _需求：5.1、5.2_

- [ ] 6. chair 单场景 smoke test（必须前台运行）
   - 在当前可见命令行直接运行：`python scripts/train.py --config configs/_w2_prune/blender_chair_n8_dav2s_depthv2_prune.yaml --scene chair --name blender_chair_n8_ssl_mv_dav2s_depthv2_prune_smoke`
   - 训练完成后立刻读 `outputs/.../metrics.json`，校验：PSNR ≥ 22.55、N_Gaussians ≤ 482450（即 ≥5% 削减）、stdout 中含 `[w2-prune] unseen pruned N=...` 与 `[w2-prune] floater pruned N=...` 各 ≥1
   - 如不达标 → 调参（先放宽 `safety_max_ratio` 或推迟 `start_iter`），不直接进入 8 场景
   - _需求：5.3、5.4、7.1_

- [ ] 7. 8 场景全量训练（顺序前台运行，每场景一个 todo 子项推进）
   - 顺序跑 chair → drums → ficus → hotdog → lego → materials → mic → ship 共 8 个；每个跑完立即读 metrics.json 校验产物
   - 训练命令统一：`python scripts/train.py --config configs/_w2_prune/blender_{scene}_n8_dav2s_depthv2_prune.yaml --scene {scene} --name blender_{scene}_n8_ssl_mv_dav2s_depthv2_prune`
   - 严禁 Start-Process / Start-Job / 隐藏进程 [[memory:ykqwnv9c]]；OOM/崩溃如实记录，不静默跳过
   - _需求：5.3、7.1、7.3_

- [ ] 8. 创建 `scripts/aggregate_w2_prune.py`（复用 W1 聚合脚本结构）并产出 `outputs/w2_unseen_floater_ablation.md`
   - 直接 import 或复制 `scripts/aggregate_w1_pearson.py` 中的 `load_scene_metrics / aggregate_variant / compute_deltas / render_overview_table / render_per_scene_block`，仅改 VARIANTS 与报告标题
   - VARIANTS：`depthv2 (baseline)` / `depthv2_prune` 两条曲线（首版仅跑全量组合）
   - 报告含：总览表（含 ΔPSNR/ΔSSIM/ΔLPIPS/ΔN_Gaussians）、per-scene 表、关键结论 3 条（覆盖 6.2 a/b/c）、W3 决策建议（推荐/不推荐/有条件 + 至多 3 条理由）
   - 在当前可见命令行直接前台运行 `python scripts/aggregate_w2_prune.py`，落盘 `outputs/w2_unseen_floater_ablation.md`
   - _需求：6.1-6.3、7.1_

- [ ] 9. 终态汇报
   - 直接给出报告路径 `outputs/w2_unseen_floater_ablation.md`（项目环境无 `open_result_view`，沿用 W1 处理方式）
   - 在最终消息中明确给出：两条曲线 avg PSNR/N_Gaussians delta、是否值得进入 W3、有无异常需用户介入
   - **不主动启动 W3**；如用户希望继续，回 plan 模式重新走需求-任务规划流程
   - _需求：6.3、7.4、7.5_
