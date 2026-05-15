"""Deterministic sparse-view selection.

We keep this in its own file so the same logic can later be reused for
COLMAP / DTU loaders. Only "uniformly spaced" indices are implemented for
the baseline; FVS / FPS-style selection is left for later.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


# Fixed train-view indices used by SOTA papers on the NeRF-Synthetic
# (Blender) 8-view setting. Both DNGaussian and CoR-GS hard-code this exact
# list; using the same list lets us compare apples-to-apples against their
# published numbers.
#
# Provenance:
#   * third_party/DNGaussian/scene/dataset_readers.py:331
#       train_cam_infos = [c for idx, c in enumerate(train_cam_infos)
#                          if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
#   * third_party/CoR-GS/scene/dataset_readers.py:511
#       train_cam_infos = [c for idx, c in enumerate(train_cam_infos)
#                          if idx in [2, 16, 26, 55, 73, 76, 86, 93]]
SOTA_NERF_SYNTHETIC_N8: List[int] = [2, 16, 26, 55, 73, 76, 86, 93]


def sparse_view_indices(
    n_total: int,
    n_train: int,
    explicit: Optional[Sequence[int]] = None,
    seed: int = 42,
    mode: str = "uniform",
) -> List[int]:
    """Return a sorted list of training-view indices.

    Args:
        n_total: number of cameras in the original train split.
        n_train: how many to keep.
        explicit: if given, this list is returned verbatim (after sort/uniq)
            and overrides ``n_train`` / ``mode``.
        seed: RNG seed for the ``random`` mode.
        mode: one of
            * ``"uniform"`` (default — equidistant on the index axis),
            * ``"random"`` (RNG-seeded subsample),
            * ``"sota_nerf_synthetic_n8"`` (the fixed 8-view list shared by
              DNGaussian and CoR-GS for the NeRF-Synthetic Blender split;
              ``n_train`` and ``n_total`` are validated but not used).
    """
    if explicit is not None:
        ids = sorted(set(int(i) for i in explicit))
        for i in ids:
            if not 0 <= i < n_total:
                raise ValueError(f"train_view_id {i} out of range [0,{n_total})")
        return ids

    if n_train >= n_total:
        return list(range(n_total))

    if mode == "uniform":
        # Equidistant in the index space, including endpoints.
        idx = np.linspace(0, n_total - 1, num=n_train).round().astype(int)
        idx = sorted(set(idx.tolist()))
        # If rounding caused collisions, fill from the front.
        i = 0
        while len(idx) < n_train and i < n_total:
            if i not in idx:
                idx.append(i)
            i += 1
        return sorted(idx)[:n_train]

    if mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n_total, size=n_train, replace=False).tolist())

    if mode == "sota_nerf_synthetic_n8":
        if n_train != 8:
            raise ValueError(
                "sota_nerf_synthetic_n8 is only defined for n_train=8 "
                f"(got n_train={n_train}). Either set n_train=8 or use a "
                "different sparse_mode."
            )
        for i in SOTA_NERF_SYNTHETIC_N8:
            if not 0 <= i < n_total:
                raise ValueError(
                    f"sota_nerf_synthetic_n8 expects n_total>=94 "
                    f"(NeRF-Synthetic train split has 100), got n_total={n_total}."
                )
        return list(SOTA_NERF_SYNTHETIC_N8)

    raise ValueError(f"unknown sparse-view mode: {mode}")
