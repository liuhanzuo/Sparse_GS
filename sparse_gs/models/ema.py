"""EMA / snapshot teacher for GaussianModel.

Sparse-view 3DGS is prone to overfitting its handful of training views.
Running a "frozen" teacher gives us a smoother target we can distill
against (pseudo-view rendering, same-view consistency, etc.).

Why not a strict tensor-wise EMA?
---------------------------------
gsplat's ``DefaultStrategy`` mutates the student's parameter *count* at
every densify/prune step (every 100 iters by default). Maintaining a
shape-matched EMA across those mutations would require replaying every
split/duplicate/prune on the teacher, which is brittle. Instead we offer
a **snapshot teacher**: a full copy of the student that is refreshed at
fixed intervals (``snapshot_every`` iterations). Between refreshes the
teacher is frozen — this is the classical "delayed-target" recipe used
in BYOL / mean-teacher style methods.

Optionally a tensor-wise EMA can run on top of the snapshot **only when
the parameter count has not changed since the last snapshot**, so that
the teacher tracks the student smoothly within a refine interval.
"""

from __future__ import annotations

import torch

from .gaussians import GaussianModel


class EMATeacher:
    """Snapshot teacher (with optional intra-window EMA).

    Args:
        student:        the student GaussianModel to copy from
        momentum:       EMA momentum for tensor-wise updates between snapshots.
                        Set to 1.0 to disable EMA (pure frozen snapshot).
        snapshot_every: hard refresh interval, in trainer steps.
                        If 0/None, the teacher is built once and only
                        re-snapshotted when the student's N changes.
    """

    def __init__(
        self,
        student: GaussianModel,
        momentum: float = 1.0,
        snapshot_every: int = 0,
    ):
        self.momentum = float(momentum)
        self.snapshot_every = int(snapshot_every) if snapshot_every else 0
        self.model = GaussianModel(sh_degree=student.sh_degree)
        self.model.active_sh_degree = student.active_sh_degree
        self._snapshot_from(student)
        self._last_snapshot_step = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_from(self, student: GaussianModel) -> None:
        """Full copy of student -> teacher. Drops grads."""
        p = student.params
        self.model._set_params(
            means=p["means"].detach().clone(),
            scales=p["scales"].detach().clone(),
            quats=p["quats"].detach().clone(),
            opacities=p["opacities"].detach().clone(),
            sh0=p["sh0"].detach().clone(),
            shN=p["shN"].detach().clone(),
            device=p["means"].device,
        )
        for param in self.model.params.values():
            param.requires_grad_(False)
        self.model.active_sh_degree = student.active_sh_degree

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, student: GaussianModel) -> None:
        """Tensor-wise EMA. Caller must ensure shapes match."""
        if self.momentum >= 1.0:
            return
        m = self.momentum
        s = student.params
        t = self.model.params
        for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            t[k].data.mul_(m).add_(s[k].detach(), alpha=1.0 - m)
        self.model.active_sh_degree = student.active_sh_degree

    # ------------------------------------------------------------------
    @torch.no_grad()
    def after_student_step(
        self,
        student: GaussianModel,
        step: int,
        did_densify: bool,
    ) -> None:
        """Call exactly once per training step, AFTER the student is updated.

        Logic:
            * If shapes mismatch (densify just changed N): re-snapshot.
            * Else if snapshot_every > 0 and we've waited long enough: re-snapshot.
            * Else: tensor-wise EMA (no-op when momentum=1.0).
        """
        size_changed = (self.model.num_points != student.num_points) or did_densify
        time_to_snapshot = (
            self.snapshot_every > 0
            and (step - self._last_snapshot_step) >= self.snapshot_every
        )
        if size_changed or time_to_snapshot:
            self._snapshot_from(student)
            self._last_snapshot_step = step
        else:
            self._ema_update(student)

    # ------------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.model.num_points

    def state_for_render(self):
        return self.model.state_for_render()
