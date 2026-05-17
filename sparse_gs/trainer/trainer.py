"""End-to-end training/eval loop for sparse-view 3DGS.

Order of operations per step (matches gsplat's official ``simple_trainer``):

    1. pick a train camera (uniform random)
    2. render -> info, rgb, depth
    3. strategy.step_pre_backward(info)         # accumulates 2D grads
    4. compute photometric + (optional) SSL losses
    5. backward()
    6. for each per-param optimizer: step() then zero_grad()
    7. strategy.step_post_backward()            # densify / prune in-place
    8. (optional) update EMA teacher           [stub]

After training we render the held-out test views and report PSNR / SSIM.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..datasets.nerf_synthetic import Camera, NerfSyntheticDataset
from ..datasets.llff import LLFFDataset
from ..datasets.pseudo_pose import sample_pseudo_camera
from ..losses.gan import PatchGANController
from ..losses.perceptual import PerceptualLoss
from ..losses.photometric import photometric_loss
from ..losses.pvd import PseudoViewDiscriminator
from ..losses.ssl import SSLLossBank
from ..models.ema import EMATeacher
from ..models.gaussians import GaussianModel
from ..rendering.gsplat_renderer import GSplatRenderer
from ..strategies.densify import build_strategy_and_optimizers
from ..strategies.post_prune import (
    apply_safety_cap,
    compute_floater_mask,
    compute_unseen_mask,
)
from ..utils.io import load_checkpoint, save_checkpoint, save_image
from ..utils.metrics import LPIPSMetric, psnr, ssim, hwc_to_bchw


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MeansLRScheduler:
    """Exponential decay on the means LR from initial -> initial*final_factor."""

    def __init__(self, opt: torch.optim.Optimizer, total_steps: int, final_factor: float):
        self.opt = opt
        self.total = max(int(total_steps), 1)
        self.final_factor = float(final_factor)
        self.lr0 = [g["lr"] for g in opt.param_groups]

    def step(self, it: int) -> None:
        t = min(max(it / self.total, 0.0), 1.0)
        scale = math.exp(t * math.log(max(self.final_factor, 1e-12)))
        for g, lr0 in zip(self.opt.param_groups, self.lr0):
            g["lr"] = lr0 * scale


class Trainer:
    def __init__(self, cfg: Dict[str, Any], output_dir: Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = torch.device(cfg["experiment"].get("device", "cuda"))
        _set_seed(int(cfg["experiment"].get("seed", 42)))

        # ---- data ----
        d = cfg["data"]
        dtype = d.get("type", "nerf_synthetic")
        if dtype == "nerf_synthetic":
            self.dataset = NerfSyntheticDataset(
                root=d["root"],
                scene=d["scene"],
                n_train_views=int(d.get("n_train_views", 6)),
                train_view_ids=d.get("train_view_ids"),
                image_downsample=int(d.get("image_downsample", 1)),
                white_background=bool(d.get("white_background", True)),
                seed=int(cfg["experiment"].get("seed", 42)),
                sparse_mode=str(d.get("sparse_mode", "uniform")),
                depth_prior=d.get("depth_prior"),
            )
        elif dtype == "llff":
            self.dataset = LLFFDataset(
                root=d["root"],
                scene=d["scene"],
                n_train_views=int(d.get("n_train_views", 3)),
                train_view_ids=d.get("train_view_ids"),
                image_downsample=int(d.get("image_downsample", 4)),
                white_background=bool(d.get("white_background", False)),
                recenter=bool(d.get("recenter", True)),
                rescale=bool(d.get("rescale", True)),
                seed=int(cfg["experiment"].get("seed", 42)),
                sparse_mode=str(d.get("sparse_mode", "uniform")),
                depth_prior=d.get("depth_prior"),
            )
        else:
            raise NotImplementedError(f"data.type={dtype!r} not implemented yet")
        print("[data]", self.dataset)

        # Pre-move all training cameras to GPU **once**. Sparse-view setups have
        # very few train views (typically 3-9), so the memory cost is trivial,
        # but doing this saves a per-step H2D copy of (image, viewmat, K, alpha,
        # mono_depth). On Windows + recent CUDA drivers the per-step
        # ``.to(device, non_blocking=True)`` of unpinned CPU tensors has been
        # observed to occasionally trigger transient ``unknown error`` failures
        # in downstream gsplat kernels. Caching avoids this entirely.
        self._train_cams_gpu = [c.to(self.device) for c in self.dataset.train]

        # ---- pseudo-view wide-sampling switch (Option B1) ----
        # Reads cfg.pseudo.wide_sampling (optional). When enabled, the
        # global ``sample_pseudo_camera`` will sample from a mixture of
        # interp / extrap / sphere-uniform distributions instead of pure
        # train-pair interpolation. This is a no-op when not configured.
        from ..datasets.pseudo_pose import set_wide_sampling
        set_wide_sampling(cfg.get("pseudo", {}).get("wide_sampling"))

        # ---- model ----
        m = cfg["model"]
        self.gaussians = GaussianModel(sh_degree=int(m["sh_degree"])).to(self.device)
        init = m.get("init", {})
        if init.get("type", "random_in_box") == "random_in_box":
            self.gaussians.init_random(
                num_points=int(init.get("num_points", 100_000)),
                extent=float(init.get("extent", 1.5)),
                rgb_init=float(init.get("rgb_init", 0.5)),
                scale_init_factor=float(m.get("scale_init_factor", 0.01)),
                device=self.device,
            )
        else:
            raise NotImplementedError(f"model.init.type={init['type']!r}")
        print(f"[model] initial #gaussians = {self.gaussians.num_points}")

        # ---- renderer ----
        r = cfg["renderer"]
        self.renderer = GSplatRenderer(
            sh_degree=int(m["sh_degree"]),
            near_plane=float(r.get("near_plane", 0.01)),
            far_plane=float(r.get("far_plane", 1.0e10)),
            rasterize_mode=str(r.get("rasterize_mode", "antialiased")),
            packed=bool(r.get("packed", True)),
            absgrad=bool(r.get("absgrad", True)),
            render_mode=str(r.get("render_mode", "RGB+ED")),
        )

        # ---- strategy + optimizers ----
        self.strategy, self.optimizers, self.strategy_state = build_strategy_and_optimizers(
            self.gaussians, cfg, scene_scale=self.dataset.scene_scale,
        )

        # Optional means-LR exponential schedule.
        self.lr_sched: Optional[_MeansLRScheduler] = None
        means_final = float(cfg["optim"].get("means_lr_final_factor", 0.0) or 0.0)
        if 0.0 < means_final < 1.0:
            self.lr_sched = _MeansLRScheduler(
                self.optimizers["means"],
                int(cfg["train"]["iterations"]),
                means_final,
            )

        # ---- losses ----
        self.ssl_bank = SSLLossBank(cfg.get("ssl", {}))
        print("[ssl]", self.ssl_bank)

        # Optional appearance-only PatchGAN regularizer. It is kept outside
        # SSLLossBank because it owns a discriminator and optimizer.
        pg_cfg = (cfg.get("ssl", {}).get("patch_gan", {}) or {})
        self.patch_gan: Optional[PatchGANController] = None
        if bool(pg_cfg.get("enabled", False)):
            self.patch_gan = PatchGANController(pg_cfg, self.device)
            print(
                "[gan] PatchGAN enabled "
                f"(weight={self.patch_gan.weight}, start={self.patch_gan.start_iter}, "
                f"every={self.patch_gan.every}, detach_geometry={self.patch_gan.detach_geometry})"
            )

        # Pseudo-View Discriminator (PVD): a discriminator whose negative
        # distribution is *pseudo-view renders* (where floaters surface),
        # not training-view renders (which already match GT very well).
        # Two consumers:
        #   1) a small G-side adv term (optional, weight may be 0)
        #   2) a per-pixel "fakeness" map used to gate floater-prune
        # See ``losses/pvd.py`` for the full design write-up.
        pvd_cfg = (cfg.get("ssl", {}).get("pvd", {}) or {})
        self.pvd: Optional[PseudoViewDiscriminator] = None
        if bool(pvd_cfg.get("enabled", False)):
            self.pvd = PseudoViewDiscriminator(pvd_cfg, self.device)
            guid = (pvd_cfg.get("floater_guidance", {}) or {})
            print(
                "[pvd] PseudoViewDiscriminator enabled "
                f"(start={self.pvd.start_iter}, every={self.pvd.every}, "
                f"weight={self.pvd.weight}, patch={self.pvd.patch_size}, "
                f"arch={self.pvd.arch}, "
                f"diffaug={'on' if self.pvd.diffaug_cfg is not None else 'off'}, "
                f"guidance={'on' if bool(guid.get('enabled', False)) else 'off'})"
            )

        # Optional differentiable perceptual loss (LPIPS-VGG used as a train
        # regularizer, not just an eval metric). Lives under cfg.ssl.perceptual
        # to keep all add-on losses grouped together.
        perc_cfg = (cfg.get("ssl", {}).get("perceptual", {}) or {})
        self.perceptual: Optional[PerceptualLoss] = None
        if bool(perc_cfg.get("enabled", False)):
            self.perceptual = PerceptualLoss(perc_cfg, self.device)
            print(f"[ssl] {self.perceptual!r}")

        # ---- background (white) ----
        if bool(d.get("white_background", True)):
            self.background = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        else:
            self.background = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # ---- random-background train-time augmentation (v8_hard) ----
        # cfg.data.random_background_aug:
        #   enabled: bool                 (default false)
        #   prob:    float in [0,1]       (default 0.5; per-step probability of using random bg)
        # Only meaningful for nerf_synthetic (we need per-pixel alpha to recompose GT).
        # White-bg eval is *always* preserved -- this only fires inside train_step.
        rb_cfg = d.get("random_background_aug") or {}
        self._rand_bg_enabled = bool(rb_cfg.get("enabled", False))
        self._rand_bg_prob = float(rb_cfg.get("prob", 0.5))
        self._rand_bg_active = self._rand_bg_enabled and (dtype == "nerf_synthetic") and bool(d.get("white_background", True))
        if self._rand_bg_enabled and not self._rand_bg_active:
            print(
                f"[rand_bg] requested but disabled: dtype={dtype} white_background={bool(d.get('white_background', True))} "
                "(random_background_aug requires nerf_synthetic + white_background=true)"
            )
        elif self._rand_bg_active:
            print(f"[rand_bg] enabled (prob={self._rand_bg_prob:.2f}); eval still uses white bg")

        # ---- TB ----
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        except Exception as e:  # noqa: BLE001
            print(f"[tb] disabled ({e})")
            self.tb = None

        # ---- post-prune eval scheduler ----
        # When a floater-prune *actually removes points* (not just runs and
        # clears the buffer), we schedule extra eval probes at
        #   step+1, step+200, step+500
        # so the dip-then-recover trajectory is sampled densely enough to
        # tell whether prune is doing structural good or only transient harm.
        # Each entry is a tuple (step_to_run, anchor_prune_step, offset).
        self._post_prune_eval_pending: list = []
        self._post_prune_eval_offsets = tuple(
            int(x) for x in (
                self.cfg.get("train", {}).get("post_prune_eval_offsets", (1, 200, 500))
            )
        )

        # ---- pre-prune snapshot bookkeeping ----
        # When ``train.save_pre_prune_ckpt`` is True, every prune (unseen /
        # floater) writes the pre-cut GS params to disk. After training is
        # done, ``_test_pre_prune_ckpts`` reloads each one, runs the full
        # test pass (LPIPS included), records the metrics in ``eval_log
        # .jsonl`` under ``phase=pre_prune_ckpt_test``, then deletes the
        # ckpt file. Picks the best PSNR (vs. the final-train ckpt) into
        # ``metrics.json:best_*`` so post-hoc analysis is one-stop.
        self._pre_prune_ckpts: list = []  # list of dicts: {path, step, kind, n_pruned, thresh_bin}

        # ---- EMA teacher (built lazily on first need) ----
        self.teacher: Optional[EMATeacher] = None
        ema_cfg = cfg.get("ssl", {}).get("ema_teacher", {}) or {}
        self.teacher_momentum = float(ema_cfg.get("momentum", 1.0))
        self.teacher_snapshot_every = int(ema_cfg.get("snapshot_every", 0))
        self.teacher_build_iter = int(ema_cfg.get("build_iter", 0))
        self.need_teacher = self.ssl_bank.requires_teacher()
        if self.need_teacher:
            print(f"[ssl] EMA teacher will be built at iter {self.teacher_build_iter} "
                  f"(momentum={self.teacher_momentum}, "
                  f"snapshot_every={self.teacher_snapshot_every})")

        # ---- LPIPS (optional, lazy-constructed perceptual metric) ----
        # Controlled by cfg.eval.lpips (default True). If the package is not
        # installed we silently fall back to PSNR/SSIM only.
        self.lpips_metric: Optional[LPIPSMetric] = None
        if bool(cfg.get("eval", {}).get("lpips", True)):
            net = str(cfg.get("eval", {}).get("lpips_net", "vgg"))
            self.lpips_metric = LPIPSMetric(net=net, device=self.device)
            print(f"[eval] {self.lpips_metric}")

        # ---- structured eval log (jsonl) -------------------------------
        # Every eval (regular eval_every, scheduled post-prune offsets,
        # AND the new pre/post-floater-prune probes) is appended here as
        # one JSON object per line. This is the source of truth for
        # post-hoc trajectory analysis -- stdout / Tee-Object can be
        # noisy or even silently drop stderr on Windows PowerShell.
        self._eval_log_path = self.output_dir / "eval_log.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use line buffering so a kill -9 still leaves a complete file.
        self._eval_log_fp = open(self._eval_log_path, "a", encoding="utf-8", buffering=1)
        print(f"[eval] jsonl log -> {self._eval_log_path}")

    # ------------------------------------------------------------------
    # structured eval logger
    # ------------------------------------------------------------------
    def _log_eval(self, step: int, phase: str, metrics: Dict[str, float],
                  **extras: Any) -> None:
        """Append one eval record to ``eval_log.jsonl``.

        ``phase`` is one of:
          - ``regular``                  : scheduled by ``train.eval_every``
          - ``pre_floater_prune``        : just before a floater prune that
                                            actually removes points
          - ``post_floater_prune``       : right after that same prune
          - ``pre_unseen_prune`` /
            ``post_unseen_prune``        : same idea for unseen prune
          - ``post_floater_offset``      : the +1/+200/+500 dip-recovery probes
          - ``test``                     : final held-out test eval
        Extras (e.g. ``pruned_n``, ``thresh_bin``) are merged verbatim.
        """
        if self._eval_log_fp is None:
            return
        import json
        record = {
            "step": int(step),
            "phase": str(phase),
            "num_gaussians": int(self.gaussians.num_points),
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()},
            **extras,
        }
        try:
            self._eval_log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # pragma: no cover
            print(f"[eval-log] write failed: {e}")

    @torch.no_grad()
    def _eval_val(self) -> Dict[str, float]:
        """Convenience: run val eval with a *small* sampler.

        训练过程中的 val 只是观察曲线趋势，不需要跑完整 100 张：
        我们读 ``eval.num_val_renders``（默认 8），并独立用
        ``eval.lpips_in_train``（默认 False）控制是否算 LPIPS。
        最终 test eval (``render_test_at_end``) 仍按 ``num_test_renders``
        全量评估 + 算 LPIPS，不受这里影响。
        """
        eval_cfg = self.cfg.get("eval", {}) or {}
        n_val = int(eval_cfg.get("num_val_renders", 8) or 0) or None
        return self.evaluate(
            self.dataset.val, tag="val",
            max_views=n_val,
            use_lpips=bool(eval_cfg.get("lpips_in_train", False)),
        )

    # ------------------------------------------------------------------
    # one optimizer step over all per-param Adams
    # ------------------------------------------------------------------
    def _opt_step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Optional hard cap on N to keep single-render kernel time bounded.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _enforce_cap_max(self) -> None:
        """If ``strategy.cap_max`` is set and N > cap, prune the lowest-opacity
        Gaussians down to the cap (in-place). gsplat's ``DefaultStrategy``
        does not support this natively (only ``MCMCStrategy`` does); on
        Windows + recent CUDA drivers an unbounded N is the most common
        cause of TDR resets in long single rasterization kernels.
        """
        cap = self.strategy_state.get("cap_max", None)
        if cap is None:
            return
        n = self.gaussians.num_points
        if n <= int(cap):
            return
        from gsplat.strategy.ops import remove as _gs_remove

        # logits -> probabilities; lower opacity = first to go.
        opa = torch.sigmoid(self.gaussians.params["opacities"].detach())
        n_drop = n - int(cap)
        # ``topk`` with ``largest=False`` returns the n_drop smallest indices.
        drop_idx = torch.topk(opa, k=n_drop, largest=False).indices
        mask = torch.zeros(n, dtype=torch.bool, device=opa.device)
        mask[drop_idx] = True
        _gs_remove(
            params=self.gaussians.params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # W2: periodic post-densify prune (unseen + floater)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _periodic_unseen_floater_prune(self, step: int) -> None:
        """DNGaussian-style ``clean_views`` + lightweight SparseGS
        ``identify_floaters``. Both branches are independently scheduled and
        independently disable-able through ``cfg.strategy.prune``.

        Order: unseen first (cheap, robust), then floater (depends on a
        relatively well-formed depth, so we run it later in training).
        """
        prune_cfg = (self.cfg.get("strategy", {}) or {}).get("prune", {}) or {}
        unseen_cfg = (prune_cfg.get("unseen", {}) or {})
        floater_cfg = (prune_cfg.get("floater", {}) or {})

        do_unseen = bool(unseen_cfg.get("enabled", False)) and self._prune_due(unseen_cfg, step)
        do_floater = bool(floater_cfg.get("enabled", False)) and self._prune_due(floater_cfg, step)
        if not (do_unseen or do_floater):
            return

        from gsplat.strategy.ops import remove as _gs_remove

        if do_unseen:
            try:
                visibility_count = self._collect_visibility_for_unseen()
            except _PruneSkip as e:
                print(f"[w2-prune] skip step={step}: {e}")
                visibility_count = None
            if visibility_count is not None:
                mask = compute_unseen_mask(visibility_count)
                n_pruned = int(mask.sum().item())
                if n_pruned > 0:
                    # PRE-prune snapshot: persist the *exact* steady-state
                    # parameters just before we cut, so we can later test
                    # each candidate and pick the actual best (val/test gap
                    # at the local peak is otherwise invisible). No eval is
                    # done here -- training stays as fast as possible.
                    self._save_pre_prune_ckpt(step, kind="unseen", n_pruned=n_pruned)
                    self._log_eval(step, "pre_unseen_prune", {},
                                   will_prune_n=n_pruned)
                    _gs_remove(
                        params=self.gaussians.params,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        mask=mask,
                    )
                    print(f"[w2-prune] unseen pruned N={n_pruned} (step={step}, "
                          f"remain={self.gaussians.num_points})")
                    if self.tb is not None:
                        self.tb.add_scalar("stats/prune_unseen_n", float(n_pruned), step)
                else:
                    print(f"[w2-prune] unseen: no unseen gaussian (step={step})")

        if do_floater:
            safety_max_ratio = float(floater_cfg.get("safety_max_ratio", 0.05))
            # ---- thresh_bin schedule ----
            # If `thresh_bin_end` is provided, linearly anneal thresh_bin from
            # `thresh_bin` (= start value) at `thresh_bin_decay_start` (defaults
            # to floater.start_iter) down to `thresh_bin_end` at
            # `thresh_bin_decay_end` (defaults to floater.stop_iter).
            # This implements "early coarse / late fine" pruning: at the
            # beginning we let SparseGS-style identify_floaters hit the lower
            # 5% depth quantile (it is OK to be aggressive while densify is
            # still re-growing things), and at the end we shrink to e.g. 1%
            # so the residual cut-off only removes the most-egregious
            # near-camera floaters and the recovery cost stays bounded.
            tb_start = float(floater_cfg.get("thresh_bin", 0.13))
            tb_end = floater_cfg.get("thresh_bin_end", None)
            if tb_end is None:
                thresh_bin_eff = tb_start
            else:
                tb_end = float(tb_end)
                d0 = int(floater_cfg.get("thresh_bin_decay_start",
                                         floater_cfg.get("start_iter", 0)))
                d1 = int(floater_cfg.get("thresh_bin_decay_end",
                                         floater_cfg.get("stop_iter", step)))
                if d1 <= d0:
                    thresh_bin_eff = tb_end
                else:
                    frac = (step - d0) / float(d1 - d0)
                    frac = max(0.0, min(1.0, frac))
                    thresh_bin_eff = tb_start + (tb_end - tb_start) * frac
            try:
                floater_mask = self._collect_floater_pixels_for_floater(
                    alpha_thresh=float(floater_cfg.get("alpha_thresh", 0.5)),
                    thresh_bin=thresh_bin_eff,
                    step=step,
                )
            except _PruneSkip as e:
                print(f"[w2-prune] skip step={step}: {e}")
                floater_mask = None
            if floater_mask is not None:
                n_raw = int(floater_mask.sum().item())
                final_mask, over_aggressive = apply_safety_cap(floater_mask, safety_max_ratio)
                if over_aggressive:
                    print(f"[w2-prune] floater-prune over-aggressive "
                          f"(N_candidate={n_raw}, ratio={n_raw / max(1, floater_mask.numel()):.3f}, "
                          f"max_ratio={safety_max_ratio}, thresh_bin={thresh_bin_eff:.4f}); "
                          f"skipping at step={step}")
                else:
                    n_pruned = int(final_mask.sum().item())
                    if n_pruned > 0:
                        # PRE-prune snapshot only -- no in-training eval.
                        # The full test pass for every snapshot happens once,
                        # at the very end of training (see _test_pre_prune
                        # _ckpts), and the snapshots are deleted after that.
                        self._save_pre_prune_ckpt(step, kind="floater",
                                                  n_pruned=n_pruned,
                                                  thresh_bin=float(thresh_bin_eff))
                        self._log_eval(step, "pre_floater_prune", {},
                                       will_prune_n=n_pruned,
                                       thresh_bin=float(thresh_bin_eff))
                        _gs_remove(
                            params=self.gaussians.params,
                            optimizers=self.optimizers,
                            state=self.strategy_state,
                            mask=final_mask,
                        )
                        print(f"[w2-prune] floater pruned N={n_pruned} (step={step}, "
                              f"remain={self.gaussians.num_points}, "
                              f"thresh_bin={thresh_bin_eff:.4f})")
                        if self.tb is not None:
                            self.tb.add_scalar("stats/prune_floater_n", float(n_pruned), step)
                    else:
                        print(f"[w2-prune] floater: no floater gaussian (step={step})")

    @staticmethod
    def _prune_due(cfg: Dict[str, Any], step: int) -> bool:
        start = int(cfg.get("start_iter", 0))
        stop = int(cfg.get("stop_iter", 0))
        every = max(int(cfg.get("every_iter", 1)), 1)
        if step < start or step > stop:
            return False
        return ((step - start) % every) == 0

    @torch.no_grad()
    def _collect_visibility_for_unseen(self) -> torch.Tensor:
        """Render every training camera, accumulate per-Gaussian visibility.

        Returns a (N,) long tensor with the count of cameras that saw each
        Gaussian. Raises ``_PruneSkip`` if no view yields usable visibility.
        """
        N = self.gaussians.num_points
        device = self.gaussians.params["means"].device
        count = torch.zeros(N, dtype=torch.long, device=device)
        any_ok = False
        for cam in self._train_cams_gpu:
            out = self.renderer.render(
                self.gaussians,
                viewmat=cam.viewmat, K=cam.K,
                width=cam.width, height=cam.height,
                active_sh_degree=self.gaussians.active_sh_degree,
                background=self.background,
            )
            info = out["info"]
            if self.renderer.packed:
                gids = info.get("gaussian_ids", None)
                if gids is None:
                    continue
                # Some Gaussians may appear multiple times across cameras after
                # we OR-accumulate; here we just want presence/absence per cam.
                # ``unique`` keeps the count semantically "number of cams".
                gids_unique = torch.unique(gids.to(device, dtype=torch.long))
                count.index_add_(
                    0, gids_unique,
                    torch.ones_like(gids_unique, dtype=torch.long),
                )
                any_ok = True
            else:
                radii = info.get("radii", None)
                if radii is None:
                    continue
                # gsplat 1.5.3 packed=False: radii is (B, N, 2). A Gaussian is
                # visible iff either axis > 0.
                r = radii
                if r.dim() == 3:
                    r = r[0]
                if r.dim() == 2 and r.shape[-1] == 2:
                    vis = (r > 0).any(dim=-1).to(device)
                else:
                    vis = (r.reshape(-1) > 0).to(device)
                if vis.numel() != N:
                    continue
                count += vis.long()
                any_ok = True
        if not any_ok:
            raise _PruneSkip("no visibility info from any camera")
        return count

    @torch.no_grad()
    def _collect_floater_pixels_for_floater(
        self, *, alpha_thresh: float, thresh_bin: float,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Render every training camera, OR-accumulate floater Gaussian masks.

        If PVD is enabled, trained, and its ``floater_guidance`` is active
        (warmup passed + non-trivial real/fake gap), we additionally render
        a few **pseudo views** and use the discriminator's per-pixel
        "fakeness" map to OR more candidate floater pixels into the depth-
        based mask. The two signals are complementary: depth catches
        near-camera spikes, D catches view-inconsistent ghosting.

        Returns a (N,) bool tensor. Raises ``_PruneSkip`` if no view yields
        usable depth/alpha/info.
        """
        N = self.gaussians.num_points
        device = self.gaussians.params["means"].device
        out_mask = torch.zeros(N, dtype=torch.bool, device=device)
        any_ok = False
        for cam in self._train_cams_gpu:
            out = self.renderer.render(
                self.gaussians,
                viewmat=cam.viewmat, K=cam.K,
                width=cam.width, height=cam.height,
                active_sh_degree=self.gaussians.active_sh_degree,
                background=self.background,
            )
            depth = out.get("depth", None)
            alpha = out.get("alpha", None)
            info = out.get("info", None)
            if depth is None or alpha is None or info is None:
                continue
            means2d = info.get("means2d", None)
            gids = info.get("gaussian_ids", None)
            radii = info.get("radii", None)

            view_mask = compute_floater_mask(
                depth=depth.detach(),
                alpha=alpha.detach(),
                n_gaussians=N,
                gaussian_ids=gids.detach() if gids is not None else None,
                radii=radii.detach() if radii is not None else None,
                means2d=means2d.detach() if means2d is not None else None,
                alpha_thresh=alpha_thresh,
                thresh_bin=thresh_bin,
                packed=self.renderer.packed,
            )
            out_mask |= view_mask
            any_ok = True

        # ---- PVD-guided pseudo-view floater candidates ----
        # On a few sampled pseudo views, ask D where it thinks the render
        # looks fake. Convert that to a per-Gaussian mask via the same
        # back-projection logic and OR into out_mask.
        if (
            self.pvd is not None
            and step is not None
            and self.pvd.guidance_active(step)
            and len(self._train_cams_gpu) >= 2
        ):
            n_pseudo = max(1, int(self.pvd.cfg.get(
                "floater_guidance", {}).get("n_pseudo_views", 4)))
            gate_thresh = float(self.pvd.guidance_thresh)
            for _ in range(n_pseudo):
                pcam = sample_pseudo_camera(self._train_cams_gpu, device=device)
                pout = self.renderer.render(
                    self.gaussians,
                    viewmat=pcam.viewmat, K=pcam.K,
                    width=pcam.width, height=pcam.height,
                    active_sh_degree=self.gaussians.active_sh_degree,
                    background=self.background,
                )
                rgb = pout.get("rgb", None)
                pinfo = pout.get("info", None)
                if rgb is None or pinfo is None:
                    continue
                pmeans2d = pinfo.get("means2d", None)
                pgids = pinfo.get("gaussian_ids", None)
                pradii = pinfo.get("radii", None)
                if pmeans2d is None:
                    continue
                # Per-pixel fakeness in [0,1]
                fake_prob = self.pvd.floater_score_map(
                    rgb.detach(), target_h=pcam.height, target_w=pcam.width,
                )
                fake_pix = (fake_prob > gate_thresh)             # (H, W) bool
                if int(fake_pix.sum().item()) == 0:
                    continue
                # Back-project fake pixels -> per-Gaussian mask, reusing
                # the same mechanism as compute_floater_mask. We emulate it
                # here on the bool grid directly to avoid re-deriving depth.
                Hh, Ww = fake_pix.shape
                if self.renderer.packed:
                    if pgids is None or pmeans2d.dim() != 2 or pmeans2d.shape[0] != pgids.shape[0]:
                        continue
                    xs = pmeans2d[:, 0].long().clamp(0, Ww - 1)
                    ys = pmeans2d[:, 1].long().clamp(0, Hh - 1)
                    hits = fake_pix[ys, xs]
                    if int(hits.sum().item()) == 0:
                        continue
                    hit_global = pgids[hits].long()
                    out_mask[hit_global] = True
                else:
                    if pradii is None or pmeans2d.dim() != 2 or pmeans2d.shape[0] != N:
                        continue
                    visible = (pradii > 0)
                    xs = pmeans2d[:, 0].long().clamp(0, Ww - 1)
                    ys = pmeans2d[:, 1].long().clamp(0, Hh - 1)
                    hits = fake_pix[ys, xs] & visible
                    out_mask |= hits

        if not any_ok:
            raise _PruneSkip("no depth/alpha/info from any camera")
        return out_mask

    # ------------------------------------------------------------------
    # SH-degree warmup: gsplat uses min(active, sh_degree), so we ramp it.
    # ------------------------------------------------------------------
    def _active_sh_degree(self, step: int, total: int) -> int:
        # Start at 0, +1 every (total / (sh_degree+1)) steps.
        max_deg = self.gaussians.sh_degree
        if max_deg == 0:
            return 0
        frac = step / max(total, 1)
        return int(min(max_deg, math.floor(frac * (max_deg + 1))))

    # ------------------------------------------------------------------
    # optional PatchGAN appearance regularization
    # ------------------------------------------------------------------
    def _patch_gan_term(self, step: int, cam: Camera, active_sh: int, gt_rgb: torch.Tensor,
                        bg: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Dict[str, float]]:
        if self.patch_gan is None or not self.patch_gan.active(step):
            return torch.zeros((), device=self.device), {}

        bg_use = bg if bg is not None else self.background
        adv_out = self.renderer.render(
            self.gaussians,
            viewmat=cam.viewmat,
            K=cam.K,
            width=cam.width,
            height=cam.height,
            active_sh_degree=active_sh,
            background=bg_use,
            detach_geometry=self.patch_gan.detach_geometry,
        )
        fake_rgb = adv_out["rgb"]
        if fake_rgb is None:
            return torch.zeros((), device=self.device), {}

        _, d_logs = self.patch_gan.discriminator_step(gt_rgb, fake_rgb)
        g_term, g_logs = self.patch_gan.generator_loss(fake_rgb)
        logs = {**d_logs, **g_logs}
        return g_term, logs

    # ------------------------------------------------------------------
    # random background helper (v8_hard)
    # ------------------------------------------------------------------
    def _sample_train_bg_and_gt(self, cam: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the per-step background and the corresponding GT image.

        When random_background_aug is OFF (or alpha is missing), returns
        ``(self.background, cam.image)`` -- identical to the previous behaviour.

        When ON, with probability ``self._rand_bg_prob`` we sample a random RGB
        background ``bg`` and re-composite GT to that background via
            gt_new = rgb*a + bg*(1-a)
                   = (cam.image - 1*(1-a)) + bg*(1-a)        # since cam.image = rgb*a + 1*(1-a)
                   = cam.image + (bg - 1) * (1 - a)
        This is exact (no division by alpha) and lossless.
        """
        if not self._rand_bg_active:
            return self.background, cam.image
        if cam.alpha is None:
            return self.background, cam.image
        if random.random() >= self._rand_bg_prob:
            return self.background, cam.image
        bg = torch.rand(3, device=self.device, dtype=cam.image.dtype)
        # cam.alpha is (H, W, 1); broadcasts correctly against (H, W, 3) image.
        gt_new = cam.image + (bg - 1.0) * (1.0 - cam.alpha)
        return bg, gt_new

    # ------------------------------------------------------------------
    # one training step
    # ------------------------------------------------------------------
    def train_step(self, step: int, total: int) -> Dict[str, float]:
        cam: Camera = random.choice(self._train_cams_gpu)
        active_sh = self._active_sh_degree(step, total)
        self.gaussians.active_sh_degree = active_sh

        # v8_hard: optionally swap to a random background for this step.
        bg_step, gt_step = self._sample_train_bg_and_gt(cam)

        out = self.renderer.render(
            self.gaussians,
            viewmat=cam.viewmat,
            K=cam.K,
            width=cam.width,
            height=cam.height,
            active_sh_degree=active_sh,
            background=bg_step,
        )
        info = out["info"]
        # gsplat expects step_pre_backward to be called before .backward()
        self.strategy.step_pre_backward(
            params=self.gaussians.params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # base photometric loss
        ssim_lambda = float(self.cfg["train"].get("ssim_lambda", 0.2))
        gt_rgb = gt_step
        photo = photometric_loss(out["rgb"], gt_rgb, ssim_lambda=ssim_lambda)

        # optional SSL losses (no-ops in the baseline)
        ssl_term, ssl_logs = self.ssl_bank(
            step=step,
            total_steps=total,
            rendered=out,
            gt_rgb=gt_rgb,
            camera=cam,
            pose_pool=self._train_cams_gpu,
            teacher=self.teacher,
            gaussians=self.gaussians,
            renderer=self.renderer,
            background=bg_step,
        )

        adv_term, adv_logs = self._patch_gan_term(step, cam, active_sh, gt_rgb, bg=bg_step)

        # PVD generator term (optional; weight may be 0 for D-only mode).
        # Note: pvd D step happens *outside* this autograd graph below.
        pvd_g_term = torch.zeros((), device=self.device, dtype=out["rgb"].dtype)
        pvd_logs: Dict[str, float] = {}
        if (
            self.pvd is not None
            and self.pvd.active(step)
            and self.pvd.weight > 0.0
            and len(self._train_cams_gpu) >= 2
        ):
            pcam_g = sample_pseudo_camera(self._train_cams_gpu, device=self.device)
            pgen_out = self.renderer.render(
                self.gaussians,
                viewmat=pcam_g.viewmat, K=pcam_g.K,
                width=pcam_g.width, height=pcam_g.height,
                active_sh_degree=active_sh,
                background=bg_step,
                detach_geometry=self.pvd.detach_geometry,
            )
            if pgen_out.get("rgb", None) is not None:
                g_term, g_logs = self.pvd.generator_loss(pgen_out["rgb"])
                pvd_g_term = g_term
                pvd_logs.update(g_logs)

        if self.perceptual is not None:
            perc_term, perc_logs = self.perceptual(step, out["rgb"], gt_rgb)
        else:
            perc_term = torch.zeros((), device=self.device, dtype=out["rgb"].dtype)
            perc_logs = {}

        loss = photo + ssl_term + adv_term + pvd_g_term + perc_term
        loss.backward()

        # PVD discriminator step (separate autograd graph).
        if (
            self.pvd is not None
            and self.pvd.active(step)
            and len(self._train_cams_gpu) >= 2
        ):
            with torch.no_grad():
                pcam_d = sample_pseudo_camera(
                    self._train_cams_gpu, device=self.device,
                )
                pd_out = self.renderer.render(
                    self.gaussians,
                    viewmat=pcam_d.viewmat, K=pcam_d.K,
                    width=pcam_d.width, height=pcam_d.height,
                    active_sh_degree=active_sh,
                    background=bg_step,
                )
            if pd_out.get("rgb", None) is not None:
                _, d_logs = self.pvd.discriminator_step(gt_rgb, pd_out["rgb"])
                pvd_logs.update(d_logs)

        self._opt_step()
        if self.lr_sched is not None:
            self.lr_sched.step(step)

        # densify / prune (in-place mutation of params + optimizers)
        n_before = self.gaussians.num_points
        self.strategy.step_post_backward(
            params=self.gaussians.params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
            packed=self.renderer.packed,
        )
        # Enforce optional hard cap on Gaussian count (see ``densify.py``).
        # We do this *after* gsplat's own densify/prune so we always trim to
        # the cap on the final state. Cheap when cap is not exceeded.
        self._enforce_cap_max()
        # W2: periodic post-prune (DNGaussian-style unseen + SparseGS-style
        # lightweight floater). No-op unless cfg.strategy.prune.* is enabled.
        self._periodic_unseen_floater_prune(step)
        n_after = self.gaussians.num_points
        did_densify = (n_after != n_before)

        # Build / update EMA teacher (sparse-view SSL).
        if self.need_teacher:
            if self.teacher is None and step >= self.teacher_build_iter:
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
            mse = (out["rgb"].detach() - gt_rgb).pow(2).mean()
            ps = float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))
        logs = {
            "loss/total": float(loss.detach().item()),
            "loss/photo": float(photo.detach().item()),
            "train/psnr": ps,
            "stats/n_gaussians": float(self.gaussians.num_points),
            "stats/active_sh": float(active_sh),
        }
        logs.update(ssl_logs)
        logs.update(perc_logs)
        logs.update(adv_logs)
        return logs

    # ------------------------------------------------------------------
    # evaluation: PSNR/SSIM over a list of cameras
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, cameras: List[Camera], tag: str = "val",
                 save_renders_to: Optional[Path] = None,
                 max_views: Optional[int] = None,
                 use_lpips: Optional[bool] = None) -> Dict[str, float]:
        if max_views is not None:
            cameras = cameras[:max_views]
        if not cameras:
            return {}
        psnrs, ssims, lpipss = [], [], []
        # 显式 False 时强制不跑 LPIPS（用于训练期 val 加速）；
        # None 时回退到老行为（lpips_metric 可用就跑）。
        if use_lpips is None:
            use_lpips = self.lpips_metric is not None and self.lpips_metric.available
        else:
            use_lpips = bool(use_lpips) and self.lpips_metric is not None and self.lpips_metric.available
        for i, cam in enumerate(cameras):
            cam = cam.to(self.device)
            out = self.renderer.render(
                self.gaussians,
                viewmat=cam.viewmat, K=cam.K,
                width=cam.width, height=cam.height,
                active_sh_degree=self.gaussians.sh_degree,
                background=self.background,
            )
            pred = out["rgb"].clamp(0, 1)
            gt = cam.image
            psnrs.append(float(psnr(pred, gt).item()))
            ssims.append(float(ssim(hwc_to_bchw(pred), hwc_to_bchw(gt)).item()))
            if use_lpips:
                lp = self.lpips_metric(pred, gt)
                if lp is not None:
                    lpipss.append(lp)
            if save_renders_to is not None:
                save_image(pred, Path(save_renders_to) / f"{tag}_{i:03d}_{cam.image_name}.png")
        result = {f"{tag}/psnr": float(np.mean(psnrs)),
                  f"{tag}/ssim": float(np.mean(ssims))}
        if lpipss:
            result[f"{tag}/lpips"] = float(np.mean(lpipss))
        return result

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        total = int(self.cfg["train"]["iterations"])
        log_every = int(self.cfg["train"].get("log_every", 50))
        eval_every = int(self.cfg["train"].get("eval_every", 2000))
        save_every = int(self.cfg["train"].get("save_every", total))

        ckpt_dir = self.output_dir / "ckpts"
        render_dir = self.output_dir / "renders"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # tqdm 配置说明：
        # * file=sys.stderr  ：进度条只走 stderr，stdout (被 Tee-Object 捕获的那条流)
        #                     里只剩干净的 print，log 文件不会再被刷成几十兆。
        # * ascii=True       ：用 '#' 代替 '█▏'，避免 PowerShell GBK 终端把
        #                     unicode block 字符显示成 '鈻?' 乱码。
        # * mininterval=0.5  ：哪怕 25 it/s，也最多 0.5s 才刷一次。
        # * miniters=10      ：至少累计 10 步才刷新一次。
        # * leave=True       ：最后一行保留。
        pbar = tqdm(
            range(1, total + 1),
            dynamic_ncols=True,
            file=sys.stderr,
            ascii=True,
            mininterval=0.5,
            miniters=10,
            leave=True,
            desc="train",
        )
        t0 = time.time()
        last_ok_step = 0
        for step in pbar:
            try:
                logs = self.train_step(step, total)
            except RuntimeError as e:
                # On Windows + recent CUDA drivers, transient TDR resets surface
                # here as ``RuntimeError: CUDA error: unknown error`` (or
                # ``cudaErrorIllegalAddress``). The CUDA context is unusable
                # afterwards, so we just save the last good checkpoint and
                # bail out gracefully instead of losing all training progress.
                msg = str(e)
                print(f"\n[fatal @ step {step}] {type(e).__name__}: {msg}")
                if last_ok_step > 0:
                    panic_path = ckpt_dir / f"panic_iter_{last_ok_step:06d}.pt"
                    try:
                        self._save_ckpt(panic_path, last_ok_step)
                        print(f"[fatal] saved panic checkpoint -> {panic_path}")
                    except Exception as save_err:  # pragma: no cover
                        print(f"[fatal] panic save failed: {save_err}")
                raise
            last_ok_step = step
            if step % log_every == 0 or step == 1:
                pbar.set_postfix({
                    "loss": f"{logs['loss/total']:.4f}",
                    "psnr": f"{logs['train/psnr']:.2f}",
                    "N": int(logs["stats/n_gaussians"]),
                })
                if self.tb is not None:
                    for k, v in logs.items():
                        self.tb.add_scalar(k, v, step)
            if eval_every > 0 and step % eval_every == 0:
                metrics = self._eval_val()
                msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"\n[eval @ {step}] {msg}")
                if self.tb is not None:
                    for k, v in metrics.items():
                        self.tb.add_scalar(k, v, step)
                self._log_eval(step, "regular", metrics)
            # Extra eval probes scheduled by floater-prune. We pop everything
            # whose target step has been reached (handles offset=0 / 1 / out-
            # of-order safely). Skip if it coincides with an eval_every step
            # we just ran above, to avoid duplicate work.
            if self._post_prune_eval_pending:
                kept = []
                fired = False
                for target, anchor, off in self._post_prune_eval_pending:
                    if step < target:
                        kept.append((target, anchor, off))
                        continue
                    # target == step (we never let it fall behind because we
                    # check every step). If we already eval'd at this step
                    # via eval_every, just drop it silently.
                    if eval_every > 0 and step % eval_every == 0:
                        continue
                    if not fired:
                        metrics = self._eval_val()
                        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                        print(f"\n[eval @ {step} (post-floater +{off}, anchor={anchor})] {msg}")
                        if self.tb is not None:
                            for k, v in metrics.items():
                                self.tb.add_scalar(k, v, step)
                        self._log_eval(step, "post_floater_offset", metrics,
                                       anchor_step=int(anchor), offset=int(off))
                        fired = True  # don't run eval twice in same step
                    # else: another offset hit the same step; we already ran eval
                self._post_prune_eval_pending = kept
            if save_every > 0 and step % save_every == 0:
                self._save_ckpt(ckpt_dir / f"iter_{step:06d}.pt", step)

            # ---- periodic snapshots in the *fine-tune tail* ----
            # 复用 pre-prune snapshot 通道（save -> end-of-train test -> delete），
            # 用于在精修阶段（prune 已停）按固定步长保存中间状态，便于事后分析
            # 平台期。仅在 train.save_pre_prune_ckpt=True 时生效（与 prune 快照
            # 共用底层开关）；关闭时 _save_pre_prune_ckpt 自身会 early-return。
            train_cfg_main = self.cfg.get("train", {}) or {}
            pe = int(train_cfg_main.get("periodic_ckpt_every", 0) or 0)
            if pe > 0:
                ps = int(train_cfg_main.get("periodic_ckpt_start", 0) or 0)
                # `step >= ps` 与 `(step - ps) % pe == 0` 联合保证从 ps 开始
                # 每 pe 步存一次（ps 本身也存）；step==total 时下方还会再存
                # last.pt + 跑 final test，互不冲突。
                if step >= ps and ((step - ps) % pe == 0):
                    self._save_pre_prune_ckpt(step, kind="periodic", n_pruned=0)

        # final
        self._save_ckpt(ckpt_dir / "last.pt", total)
        if bool(self.cfg.get("eval", {}).get("render_test_at_end", True)):
            metrics = self.evaluate(
                self.dataset.test, tag="test",
                save_renders_to=render_dir,
                max_views=self.cfg.get("eval", {}).get("num_test_renders"),
            )
            wall = time.time() - t0
            msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"\n[test] {msg}  (elapsed {wall:.1f}s, #G={self.gaussians.num_points})")
            if self.tb is not None:
                for k, v in metrics.items():
                    self.tb.add_scalar(k, v, total)
            self._log_eval(total, "test", metrics, wall_clock_sec=float(wall))
            # Persist final metrics so we don't have to dig in TB later.
            try:
                import json
                payload = {
                    "experiment": self.cfg.get("experiment", {}).get("name"),
                    "scene": self.cfg.get("data", {}).get("scene"),
                    "n_train_views": int(self.cfg.get("data", {}).get("n_train_views", 0)),
                    "iterations": int(total),
                    "wall_clock_sec": float(wall),
                    "num_gaussians": int(self.gaussians.num_points),
                    "num_test_views_used": int(
                        len(self.dataset.test)
                        if self.cfg.get("eval", {}).get("num_test_renders") in (None, 0, "null")
                        else min(int(self.cfg.get("eval", {}).get("num_test_renders")), len(self.dataset.test))
                    ),
                    "num_test_views_total": int(len(self.dataset.test)),
                    "train_view_ids": list(self.dataset.train_view_ids),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                }
                with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"[test] metrics -> {self.output_dir / 'metrics.json'}")
            except Exception as save_err:  # pragma: no cover
                print(f"[test] failed to write metrics.json: {save_err}")

            # ----- After final test: replay every pre-prune snapshot -----
            # Loads each saved ckpt, runs the same full test pass (with
            # LPIPS), logs to eval_log.jsonl, then deletes the file.
            # Updates ``metrics.json`` in place with ``best_*`` fields and
            # the per-snapshot table so post-hoc selection is one-shot.
            try:
                final_n_g = int(self.gaussians.num_points)
                summary = self._test_pre_prune_ckpts(
                    render_dir=render_dir,
                    final_metrics=metrics,
                    final_step=int(total),
                    final_n_gauss=final_n_g,
                )
                if summary:
                    try:
                        import json
                        payload["pre_prune_test"] = {
                            "best": summary["best"],
                            "all": summary["all"],
                        }
                        # Promote best to top-level for quick scanning.
                        payload["best_label"] = summary["best"]["label"]
                        payload["best_step"] = int(summary["best"]["step"])
                        payload["best_n_gaussians"] = int(summary["best"]["n_gaussians"])
                        payload["best_metrics"] = {
                            k: float(v) for k, v in summary["best"]["metrics"].items()
                        }
                        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[pre-prune-test] metrics.json updated with best="
                              f"{summary['best']['label']}")
                    except Exception as e:  # pragma: no cover
                        print(f"[pre-prune-test] metrics.json merge failed: {e}")
            except Exception as e:  # pragma: no cover
                print(f"[pre-prune-test] failed: {e}")

        if self.tb is not None:
            self.tb.flush()
        # Close the structured eval log on a normal exit.
        try:
            if self._eval_log_fp is not None:
                self._eval_log_fp.flush()
                self._eval_log_fp.close()
                self._eval_log_fp = None
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------
    def _save_ckpt(self, path: Path, step: int) -> None:
        save_checkpoint({
            "step": step,
            "sh_degree": self.gaussians.sh_degree,
            "active_sh_degree": self.gaussians.active_sh_degree,
            "params": {k: v.detach().cpu() for k, v in self.gaussians.params.items()},
            "cfg": self.cfg,
            "scene_scale": self.dataset.scene_scale,
            "train_view_ids": self.dataset.train_view_ids,
        }, path)

    # ------------------------------------------------------------------
    # pre-prune snapshot helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _save_pre_prune_ckpt(self, step: int, kind: str,
                             n_pruned: int,
                             thresh_bin: Optional[float] = None) -> None:
        """Persist a *minimal* GS-only checkpoint right before a prune.

        Gated by ``train.save_pre_prune_ckpt`` (default False).  The whole
        pipeline (save -> later test -> delete) is opt-in so existing runs
        are unaffected.

        Files land in ``<output_dir>/ckpts_pre_prune/pre_{kind}_iter_{step:06d}.pt``
        and are deleted by ``_test_pre_prune_ckpts`` once we've recorded
        their test metrics.
        """
        train_cfg = self.cfg.get("train", {}) or {}
        if not bool(train_cfg.get("save_pre_prune_ckpt", False)):
            return
        try:
            ckpt_dir = self.output_dir / "ckpts_pre_prune"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"pre_{kind}_iter_{step:06d}.pt"
            self._save_ckpt(path, step)
            entry = {
                "path": str(path),
                "step": int(step),
                "kind": str(kind),
                "n_pruned": int(n_pruned),
                "n_gaussians": int(self.gaussians.num_points),
            }
            if thresh_bin is not None:
                entry["thresh_bin"] = float(thresh_bin)
            self._pre_prune_ckpts.append(entry)
            print(f"[pre-prune-ckpt] saved {path.name} (N={entry['n_gaussians']})")
        except Exception as e:  # pragma: no cover
            print(f"[pre-prune-ckpt] save failed at step={step} kind={kind}: {e}")

    @torch.no_grad()
    def _test_pre_prune_ckpts(self, render_dir: Path,
                              final_metrics: Dict[str, float],
                              final_step: int,
                              final_n_gauss: int) -> Dict[str, Any]:
        """After final training+test, reload each pre-prune snapshot, run
        the *same* full test pass, log metrics, then delete the ckpt file.

        Returns a dict with ``best`` info (PSNR-max across {final, snapshots}).
        """
        if not self._pre_prune_ckpts:
            return {}

        eval_cfg = self.cfg.get("eval", {}) or {}
        max_views = eval_cfg.get("num_test_renders")
        # Save current GS params so we can swap back after we're done (we
        # don't really need them again -- training is over -- but it's
        # cheap and makes the function side-effect-free w.r.t. the model).
        original_params = {k: v.detach().clone()
                           for k, v in self.gaussians.params.items()}

        # Tabulate: start with the final-train snapshot.
        records: list = [{
            "label": "final",
            "step": int(final_step),
            "kind": "final",
            "n_gaussians": int(final_n_gauss),
            "metrics": dict(final_metrics),
        }]

        print(f"\n[pre-prune-test] {len(self._pre_prune_ckpts)} snapshots queued.")
        for idx, entry in enumerate(self._pre_prune_ckpts):
            ck_path = Path(entry["path"])
            if not ck_path.exists():
                print(f"[pre-prune-test] missing {ck_path}; skip")
                continue
            try:
                state = load_checkpoint(ck_path, map_location="cpu")
                # Move params onto the model's device, in-place into the
                # existing nn.ParameterDict so optimizers / strategy state
                # don't break (they won't be used again, but safer).
                for k, v in state["params"].items():
                    if k not in self.gaussians.params:
                        continue
                    self.gaussians.params[k].data = v.to(
                        self.gaussians.params[k].device
                    )
                t0 = time.time()
                metrics = self.evaluate(
                    self.dataset.test, tag="test",
                    save_renders_to=None,  # do NOT overwrite final renders
                    max_views=max_views,
                    use_lpips=True,
                )
                wall = time.time() - t0
                msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[pre-prune-test {idx + 1}/{len(self._pre_prune_ckpts)}] "
                      f"step={entry['step']} kind={entry['kind']} "
                      f"N={entry['n_gaussians']}  {msg}  ({wall:.1f}s)")
                self._log_eval(int(entry["step"]),
                               f"pre_prune_ckpt_test",
                               metrics,
                               kind=str(entry["kind"]),
                               n_pruned=int(entry.get("n_pruned", 0)),
                               n_gaussians=int(entry["n_gaussians"]),
                               thresh_bin=float(entry["thresh_bin"]) if "thresh_bin" in entry else None,
                               elapsed_sec=float(wall))
                records.append({
                    "label": f"pre_{entry['kind']}@{entry['step']}",
                    "step": int(entry["step"]),
                    "kind": str(entry["kind"]),
                    "n_gaussians": int(entry["n_gaussians"]),
                    "metrics": dict(metrics),
                })
            except Exception as e:  # pragma: no cover
                print(f"[pre-prune-test] step={entry['step']} kind={entry['kind']} "
                      f"failed: {e}")
            finally:
                # Always delete the snapshot to keep disk usage bounded,
                # regardless of whether the test pass succeeded.
                try:
                    ck_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # Restore original (final) params just in case downstream code
        # touches the model again.
        for k, v in original_params.items():
            if k in self.gaussians.params:
                self.gaussians.params[k].data = v.to(
                    self.gaussians.params[k].device
                )

        # Try to remove the (now empty) ckpts_pre_prune dir.
        try:
            (self.output_dir / "ckpts_pre_prune").rmdir()
        except OSError:
            pass

        # Pick the best by test/psnr (fallback: SSIM if PSNR missing).
        def _score(rec):
            m = rec["metrics"]
            return m.get("test/psnr", m.get("test/ssim", -1e9))
        best = max(records, key=_score)
        print(f"\n[pre-prune-test] BEST: {best['label']}  "
              + " | ".join(f"{k}={v:.4f}" for k, v in best["metrics"].items())
              + f"  (N={best['n_gaussians']})")
        return {"best": best, "all": records}


    # ------------------------------------------------------------------
    # mid-train test eval (pick best ckpt without keeping ckpt files)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _maybe_mid_test(self, step: int, kind: str) -> None:
        """Run a held-out *test* eval at ``step`` (just before a prune) and
        append it to ``eval_log.jsonl``. No checkpoint is kept on disk —
        we just record metrics so we can pick the val/test peak post-hoc.

        Controlled by ``train.test_pre_prune`` (default False).
        Other knobs:
          - ``train.test_pre_prune_min_step`` : skip earlier than this step
          - ``train.test_pre_prune_views``   : cap test views (null = all)
          - ``train.test_pre_prune_lpips``   : compute LPIPS (default True)
        """
        train_cfg = self.cfg.get("train", {})
        if not bool(train_cfg.get("test_pre_prune", False)):
            return
        min_step = int(train_cfg.get("test_pre_prune_min_step", 0))
        if step < min_step:
            return
        max_views = train_cfg.get("test_pre_prune_views", None)
        if max_views in (None, 0, "null"):
            max_views_eff = None
        else:
            max_views_eff = int(max_views)
        use_lpips = bool(train_cfg.get("test_pre_prune_lpips", True))
        try:
            t0 = time.time()
            metrics = self.evaluate(
                self.dataset.test, tag="mid_test",
                max_views=max_views_eff,
                use_lpips=use_lpips,
            )
            wall = time.time() - t0
        except Exception as e:  # pragma: no cover
            print(f"[mid-test] step={step} kind={kind} failed: {e}")
            return
        if not metrics:
            return
        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        n_views = max_views_eff if max_views_eff is not None else len(self.dataset.test)
        print(f"\n[mid-test @ {step} (pre_{kind})] {msg}  "
              f"(views={n_views}, elapsed {wall:.1f}s, N={self.gaussians.num_points})")
        if self.tb is not None:
            for k, v in metrics.items():
                # Use a distinct namespace so it doesn't collide with final test/.
                self.tb.add_scalar(f"mid_{k}", v, step)
        self._log_eval(step, f"mid_test_pre_{kind}", metrics,
                       n_views=int(n_views),
                       elapsed_sec=float(wall))


class _PruneSkip(RuntimeError):
    """Internal sentinel: post-prune cannot run on this step (e.g. missing info)."""
