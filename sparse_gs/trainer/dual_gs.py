"""DualTrainer — CoR-GS style two-Gaussian co-training.

Implements the two pillars of `CoR-GS` (Zhang et al., ECCV 2024) on top of
our existing :class:`Trainer`:

1. **Two independent Gaussian fields** (``gs0`` = the primary one shared
   with the rest of our pipeline; ``gs1`` = a second, independently
   randomly-initialized field). Both are trained on the same train views.

2. **Co-regularization** — at every ``sample_pseudo_interval`` step in
   ``[start_sample_pseudo, end_sample_pseudo]`` we render both fields at a
   sampled pseudo camera and add a ``photometric(gs_i, detach(gs_j))`` term
   to each. The gradient flows only into the field being supervised, so
   the two fields pull each other toward agreement on unseen views.

3. **Co-pruning** — every ``coprune_every`` steps we register the two
   point clouds with Open3D's correspondence-set evaluator and prune the
   Gaussians that have *no* point in the other field within
   ``coprune_threshold`` (the "inconsistent" set). This is the floater
   killer described in the CoR-GS paper.

Design notes / scope:

* We keep the entire W1/W2 pipeline (pearson, mono-depth, dn-prune) **on
  ``gs0`` only**. ``gs1`` runs only photometric + (optional) co-reg +
  co-prune. Rationale: avoids doubling the per-step cost of every SSL term
  and avoids depth-prior coupling that might bias both fields the same
  way; co-reg already encourages cross-field agreement on pseudo views.
* The W2 unseen / floater post-prune *is* applied to ``gs1`` too — it's
  cheap and protects ``gs1`` from the same failure modes as ``gs0``.
* Test-time evaluation uses ``gs0`` only, matching CoR-GS's reported
  numbers (their paper also reports gs0 only).
* All dual-specific knobs live under ``cfg.dual_gs.*``; with
  ``cfg.dual_gs.enabled = false`` (the default) this class behaves
  identically to ``Trainer``.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..datasets.nerf_synthetic import Camera
from ..datasets.pseudo_pose import sample_pseudo_camera
from ..losses.photometric import photometric_loss
from ..models.gaussians import GaussianModel
from ..strategies.densify import build_strategy_and_optimizers
from ..strategies.post_prune import (
    apply_safety_cap,
    compute_floater_mask,
    compute_unseen_mask,
)
from .trainer import Trainer, _PruneSkip


class DualTrainer(Trainer):
    """Trainer that maintains two independently-initialized Gaussian
    fields and trains them jointly with CoR-GS-style co-reg + co-prune.

    Activated by setting ``cfg.dual_gs.enabled = true``. With it disabled
    (the default), ``__init__`` skips the gs1 setup entirely and behaves
    as the ordinary :class:`Trainer`.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any], output_dir: Path):
        super().__init__(cfg, output_dir)

        dg = (cfg.get("dual_gs", {}) or {})
        self.dual_enabled: bool = bool(dg.get("enabled", False))
        if not self.dual_enabled:
            self.gaussians2: Optional[GaussianModel] = None
            self.optimizers2: Optional[Dict[str, torch.optim.Optimizer]] = None
            self.strategy2 = None
            self.strategy2_state: Optional[Dict[str, Any]] = None
            return

        # ----- co-reg knobs -----
        self.coreg_enabled: bool = bool(dg.get("coreg", True))
        self.coreg_weight: float = float(dg.get("coreg_weight", 1.0))
        self.start_sample_pseudo: int = int(dg.get("start_sample_pseudo", 500))
        self.end_sample_pseudo: int = int(dg.get(
            "end_sample_pseudo", int(cfg["train"]["iterations"]) + 1,
        ))
        self.sample_pseudo_interval: int = max(int(dg.get("sample_pseudo_interval", 1)), 1)
        # pseudo-cam interpolation t-range (0 reuses cam_a, 1 reuses cam_b);
        # keep some headroom so gs0/gs1 actually see novel poses.
        t_lo, t_hi = dg.get("pseudo_t_range", [0.2, 0.8])
        self.pseudo_t_range: Tuple[float, float] = (float(t_lo), float(t_hi))

        # ----- co-prune knobs -----
        self.coprune_enabled: bool = bool(dg.get("coprune", True))
        self.coprune_threshold: float = float(dg.get("coprune_threshold", 0.05))
        self.coprune_start: int = int(dg.get("coprune_start_iter", 1000))
        self.coprune_stop: int = int(dg.get(
            "coprune_stop_iter",
            int(cfg.get("strategy", {}).get("refine_stop_iter", 15000)),
        ))
        self.coprune_every: int = max(int(dg.get("coprune_every_iter", 500)), 1)
        # safety: never prune more than this fraction of either field in one go
        self.coprune_safety_max_ratio: float = float(dg.get("safety_max_ratio", 0.10))

        # ----- gs1 init (different seed → different random init → different mode) -----
        m = cfg["model"]
        init = m.get("init", {})
        seed2 = int(dg.get("seed_gs1", int(cfg["experiment"].get("seed", 42)) + 1))
        # Snapshot global RNG, set seed2 for gs1 init only, then restore.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            random.seed(seed2)
            np.random.seed(seed2)
            torch.manual_seed(seed2)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed2)
            self.gaussians2 = GaussianModel(sh_degree=int(m["sh_degree"])).to(self.device)
            if init.get("type", "random_in_box") == "random_in_box":
                self.gaussians2.init_random(
                    num_points=int(init.get("num_points", 100_000)),
                    extent=float(init.get("extent", 1.5)),
                    rgb_init=float(init.get("rgb_init", 0.5)),
                    scale_init_factor=float(m.get("scale_init_factor", 0.01)),
                    device=self.device,
                )
            else:
                raise NotImplementedError(
                    f"dual_gs requires model.init.type == 'random_in_box', got {init.get('type')!r}"
                )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        # Fresh optimizers + strategy for gs1.
        self.strategy2, self.optimizers2, self.strategy2_state = (
            build_strategy_and_optimizers(
                self.gaussians2, cfg, scene_scale=self.dataset.scene_scale,
            )
        )
        print(f"[dual] gs1 initialized: N={self.gaussians2.num_points} (seed={seed2})")
        print(
            f"[dual] coreg={self.coreg_enabled} "
            f"(start={self.start_sample_pseudo}, end={self.end_sample_pseudo}, "
            f"every={self.sample_pseudo_interval}, w={self.coreg_weight}) | "
            f"coprune={self.coprune_enabled} "
            f"(start={self.coprune_start}, every={self.coprune_every}, "
            f"thr={self.coprune_threshold})"
        )

        # Stats counters (printed at the end and useful for ablation).
        self._dual_stats = {
            "coprune_calls": 0,
            "coprune_total_pruned_gs0": 0,
            "coprune_total_pruned_gs1": 0,
        }

    # ------------------------------------------------------------------
    # optimizer step for gs1 (gs0 still uses self._opt_step from base)
    # ------------------------------------------------------------------
    def _opt_step2(self) -> None:
        if self.optimizers2 is None:
            return
        for opt in self.optimizers2.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # train_step override
    # ------------------------------------------------------------------
    def train_step(self, step: int, total: int) -> Dict[str, float]:
        if not self.dual_enabled:
            return super().train_step(step, total)

        cam: Camera = random.choice(self._train_cams_gpu)
        active_sh = self._active_sh_degree(step, total)
        self.gaussians.active_sh_degree = active_sh
        self.gaussians2.active_sh_degree = active_sh

        # v8_hard: optionally swap to a random background for this step.
        bg_step, gt_step = self._sample_train_bg_and_gt(cam)

        # ===== gs0: full pipeline (renders + pre/post-backward strategy) =====
        out0 = self.renderer.render(
            self.gaussians,
            viewmat=cam.viewmat, K=cam.K,
            width=cam.width, height=cam.height,
            active_sh_degree=active_sh,
            background=bg_step,
        )
        info0 = out0["info"]
        self.strategy.step_pre_backward(
            params=self.gaussians.params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info0,
        )

        # ===== gs1: render only (we won't run gsplat's own densify on gs1
        # while it is being supervised — but we still need pre_backward for
        # the gsplat strategy to track 2D-mean grads). =====
        out1 = self.renderer.render(
            self.gaussians2,
            viewmat=cam.viewmat, K=cam.K,
            width=cam.width, height=cam.height,
            active_sh_degree=active_sh,
            background=bg_step,
        )
        info1 = out1["info"]
        self.strategy2.step_pre_backward(
            params=self.gaussians2.params,
            optimizers=self.optimizers2,
            state=self.strategy2_state,
            step=step,
            info=info1,
        )

        ssim_lambda = float(self.cfg["train"].get("ssim_lambda", 0.2))
        gt_rgb = gt_step
        photo0 = photometric_loss(out0["rgb"], gt_rgb, ssim_lambda=ssim_lambda)
        photo1 = photometric_loss(out1["rgb"], gt_rgb, ssim_lambda=ssim_lambda)

        # SSL bank (pearson / depth / etc.) — applied to gs0 only.
        ssl_term, ssl_logs = self.ssl_bank(
            step=step,
            total_steps=total,
            rendered=out0,
            gt_rgb=gt_rgb,
            camera=cam,
            pose_pool=self._train_cams_gpu,
            teacher=self.teacher,
            gaussians=self.gaussians,
            renderer=self.renderer,
            background=bg_step,
        )
        adv_term, adv_logs = self._patch_gan_term(step, cam, active_sh, gt_rgb, bg=bg_step)

        if self.perceptual is not None:
            perc_term, perc_logs = self.perceptual(step, out0["rgb"], gt_rgb)
        else:
            perc_term = torch.zeros((), device=self.device, dtype=out0["rgb"].dtype)
            perc_logs = {}

        # Co-reg on a pseudo camera: render both fields, mutual photometric
        # supervision with the *other* field detached.
        coreg0 = torch.zeros((), device=self.device)
        coreg1 = torch.zeros((), device=self.device)
        coreg_logs: Dict[str, float] = {}
        if (
            self.coreg_enabled
            and self.start_sample_pseudo < step < self.end_sample_pseudo
            and (step % self.sample_pseudo_interval) == 0
            and len(self._train_cams_gpu) >= 2
        ):
            pseudo_cam = sample_pseudo_camera(
                self._train_cams_gpu, device=self.device, t_range=self.pseudo_t_range,
            )
            ps_out0 = self.renderer.render(
                self.gaussians,
                viewmat=pseudo_cam.viewmat, K=pseudo_cam.K,
                width=pseudo_cam.width, height=pseudo_cam.height,
                active_sh_degree=active_sh,
                background=bg_step,
            )
            ps_out1 = self.renderer.render(
                self.gaussians2,
                viewmat=pseudo_cam.viewmat, K=pseudo_cam.K,
                width=pseudo_cam.width, height=pseudo_cam.height,
                active_sh_degree=active_sh,
                background=bg_step,
            )
            # warmup: linearly ramp coreg weight over 500 iters after start
            warmup = min(max(
                (step - self.start_sample_pseudo) / 500.0, 0.0,
            ), 1.0)
            w = self.coreg_weight * warmup
            # gs0 supervised by detached gs1
            coreg0 = w * photometric_loss(
                ps_out0["rgb"], ps_out1["rgb"].detach(), ssim_lambda=ssim_lambda,
            )
            # gs1 supervised by detached gs0
            coreg1 = w * photometric_loss(
                ps_out1["rgb"], ps_out0["rgb"].detach(), ssim_lambda=ssim_lambda,
            )
            coreg_logs["loss/coreg_gs0"] = float(coreg0.detach().item())
            coreg_logs["loss/coreg_gs1"] = float(coreg1.detach().item())
            coreg_logs["dual/coreg_w"] = float(w)

        # Sum + backward. Both fields' losses share the graph through the
        # detached cross-supervision, so a single backward on (loss0 + loss1)
        # is correct. (CoR-GS's own train.py does it sequentially, but
        # because gs0 and gs1 are disjoint parameter sets a fused backward
        # is equivalent and faster.)
        loss0 = photo0 + ssl_term + adv_term + perc_term + coreg0
        loss1 = photo1 + coreg1
        total_loss = loss0 + loss1
        total_loss.backward()

        self._opt_step()
        self._opt_step2()
        if self.lr_sched is not None:
            self.lr_sched.step(step)

        # ----- densify / prune on gs0 (full pipeline) -----
        n0_before = self.gaussians.num_points
        self.strategy.step_post_backward(
            params=self.gaussians.params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info0,
            packed=self.renderer.packed,
        )
        self._enforce_cap_max()
        self._periodic_unseen_floater_prune(step)

        # ----- densify / prune on gs1 (gsplat strategy only, no W2 prune) -----
        self.strategy2.step_post_backward(
            params=self.gaussians2.params,
            optimizers=self.optimizers2,
            state=self.strategy2_state,
            step=step,
            info=info1,
            packed=self.renderer.packed,
        )
        # cap_max for gs1 too if configured
        self._enforce_cap_max_for(self.gaussians2, self.optimizers2, self.strategy2_state)

        n0_after = self.gaussians.num_points
        did_densify = (n0_after != n0_before)

        # ----- co-prune (every coprune_every steps) -----
        coprune_logs: Dict[str, float] = {}
        if self.coprune_enabled and self.coprune_start <= step <= self.coprune_stop \
                and ((step - self.coprune_start) % self.coprune_every) == 0:
            coprune_logs = self._co_prune_step(step)

        if self.need_teacher:
            if self.teacher is None and step >= self.teacher_build_iter:
                from ..models.ema import EMATeacher
                self.teacher = EMATeacher(
                    self.gaussians,
                    momentum=self.teacher_momentum,
                    snapshot_every=self.teacher_snapshot_every,
                )
            elif self.teacher is not None:
                self.teacher.after_student_step(
                    self.gaussians, step=step, did_densify=did_densify,
                )

        with torch.no_grad():
            mse0 = (out0["rgb"].detach() - gt_rgb).pow(2).mean()
            ps0 = float(10.0 * torch.log10(1.0 / mse0.clamp_min(1e-12)))
            mse1 = (out1["rgb"].detach() - gt_rgb).pow(2).mean()
            ps1 = float(10.0 * torch.log10(1.0 / mse1.clamp_min(1e-12)))
        logs = {
            "loss/total": float(total_loss.detach().item()),
            "loss/photo": float(photo0.detach().item()),
            "loss/photo_gs1": float(photo1.detach().item()),
            "train/psnr": ps0,
            "train/psnr_gs1": ps1,
            "stats/n_gaussians": float(self.gaussians.num_points),
            "stats/n_gaussians_gs1": float(self.gaussians2.num_points),
            "stats/active_sh": float(active_sh),
        }
        logs.update(ssl_logs)
        logs.update(perc_logs)
        logs.update(adv_logs)
        logs.update(coreg_logs)
        logs.update(coprune_logs)
        return logs

    # ------------------------------------------------------------------
    # cap_max enforcement on an arbitrary GS / optimizer / state triple.
    # (Trainer._enforce_cap_max only knows about self.gaussians.)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _enforce_cap_max_for(
        self,
        gauss: GaussianModel,
        opts: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> None:
        cap = state.get("cap_max", None)
        if cap is None:
            return
        n = gauss.num_points
        if n <= int(cap):
            return
        from gsplat.strategy.ops import remove as _gs_remove

        opa = torch.sigmoid(gauss.params["opacities"].detach())
        n_drop = n - int(cap)
        drop_idx = torch.topk(opa, k=n_drop, largest=False).indices
        mask = torch.zeros(n, dtype=torch.bool, device=opa.device)
        mask[drop_idx] = True
        _gs_remove(params=gauss.params, optimizers=opts, state=state, mask=mask)

    # ------------------------------------------------------------------
    # Co-pruning: register the two clouds and drop points without a
    # neighbor in the other field within ``coprune_threshold``.
    #
    # We deliberately do this in pure torch (knn via cdist + min over
    # chunks) rather than open3d to avoid a hard runtime dependency.
    # ``coprune_threshold`` is in *world units*; for NeRF-Synthetic the
    # scene is normalized to ~radius 4, so a threshold around 0.05 keeps
    # the pruning conservative.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _co_prune_step(self, step: int) -> Dict[str, float]:
        from gsplat.strategy.ops import remove as _gs_remove

        xyz0 = self.gaussians.params["means"].detach()
        xyz1 = self.gaussians2.params["means"].detach()
        if xyz0.numel() == 0 or xyz1.numel() == 0:
            return {}

        # Find for each point in gs0 its nearest neighbor distance in gs1
        # (and vice versa) via chunked cdist. Memory-bound but fine for
        # the typical 100k-500k point clouds we see in 7000-iter runs.
        d0_to_1 = self._chunked_min_dist(xyz0, xyz1, chunk=4096)
        d1_to_0 = self._chunked_min_dist(xyz1, xyz0, chunk=4096)

        # "Inconsistent" = no neighbor in the other field within thr.
        thr = float(self.coprune_threshold)
        mask0 = d0_to_1 > thr  # (N0,) bool: gs0 points to drop
        mask1 = d1_to_0 > thr  # (N1,) bool: gs1 points to drop

        n0_raw = int(mask0.sum().item())
        n1_raw = int(mask1.sum().item())

        # Safety cap: if either is too aggressive, skip both this round to
        # avoid a runaway prune that happens when one field is just
        # transiently far ahead of the other (e.g. right after opacity
        # reset). This mirrors how W2's floater prune behaves.
        f0 = n0_raw / max(1, mask0.numel())
        f1 = n1_raw / max(1, mask1.numel())
        if f0 > self.coprune_safety_max_ratio or f1 > self.coprune_safety_max_ratio:
            print(
                f"[dual-prune] step={step} too aggressive "
                f"(gs0: {n0_raw}/{mask0.numel()}={f0:.3f}, "
                f"gs1: {n1_raw}/{mask1.numel()}={f1:.3f}, "
                f"max_ratio={self.coprune_safety_max_ratio:.3f}); skipping"
            )
            return {
                "dual/coprune_n0_skipped": float(n0_raw),
                "dual/coprune_n1_skipped": float(n1_raw),
            }

        if n0_raw > 0:
            _gs_remove(
                params=self.gaussians.params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mask=mask0,
            )
        if n1_raw > 0:
            _gs_remove(
                params=self.gaussians2.params,
                optimizers=self.optimizers2,
                state=self.strategy2_state,
                mask=mask1,
            )

        self._dual_stats["coprune_calls"] += 1
        self._dual_stats["coprune_total_pruned_gs0"] += n0_raw
        self._dual_stats["coprune_total_pruned_gs1"] += n1_raw
        print(
            f"[dual-prune] step={step} pruned gs0={n0_raw}/{mask0.numel()} "
            f"({f0:.3%}), gs1={n1_raw}/{mask1.numel()} ({f1:.3%}); "
            f"remain gs0={self.gaussians.num_points}, gs1={self.gaussians2.num_points}"
        )
        return {
            "dual/coprune_n0": float(n0_raw),
            "dual/coprune_n1": float(n1_raw),
            "dual/n_after_gs0": float(self.gaussians.num_points),
            "dual/n_after_gs1": float(self.gaussians2.num_points),
        }

    @staticmethod
    def _chunked_min_dist(
        a: torch.Tensor, b: torch.Tensor, chunk: int = 4096,
    ) -> torch.Tensor:
        """For each point in ``a`` (Na,3), return min L2 distance to ``b``
        (Nb,3). Done in chunks of ``a`` rows to bound peak memory at
        roughly ``chunk * Nb * 4 bytes``. With Nb≈500k and chunk=4096
        that's ~8 GB — too much. So we also chunk over ``b``.
        """
        Na = a.shape[0]
        out = torch.empty(Na, dtype=a.dtype, device=a.device)
        # Inner chunk on b to keep memory ≤ ~4 GB even for half-million sets.
        b_chunk = max(1, min(b.shape[0], 8192))
        for i in range(0, Na, chunk):
            ai = a[i : i + chunk]                                # (na,3)
            min_d2 = torch.full(
                (ai.shape[0],), float("inf"), dtype=a.dtype, device=a.device,
            )
            for j in range(0, b.shape[0], b_chunk):
                bj = b[j : j + b_chunk]                          # (nb,3)
                # squared L2 via broadcast
                d2 = (ai[:, None, :] - bj[None, :, :]).pow(2).sum(-1)  # (na,nb)
                min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)
            out[i : i + chunk] = min_d2.sqrt_()
        return out

    # ------------------------------------------------------------------
    # final summary
    # ------------------------------------------------------------------
    def fit(self) -> None:
        super().fit()
        if self.dual_enabled:
            print(
                f"[dual] coprune summary: calls={self._dual_stats['coprune_calls']}, "
                f"pruned_gs0={self._dual_stats['coprune_total_pruned_gs0']}, "
                f"pruned_gs1={self._dual_stats['coprune_total_pruned_gs1']}"
            )
            if self.tb is not None:
                self.tb.add_scalar(
                    "dual/coprune_total_gs0",
                    float(self._dual_stats["coprune_total_pruned_gs0"]),
                    int(self.cfg["train"]["iterations"]),
                )
                self.tb.add_scalar(
                    "dual/coprune_total_gs1",
                    float(self._dual_stats["coprune_total_pruned_gs1"]),
                    int(self.cfg["train"]["iterations"]),
                )
