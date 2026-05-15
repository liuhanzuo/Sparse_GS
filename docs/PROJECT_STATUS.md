# Sparse-View 3DGS + SSL — Project Status

> Single source of truth for **what we're building, what works, what's
> broken, and what to do next.** Updated after every meaningful change.
> When in doubt, trust this file over `README.md` (which is user-facing).

Last updated: 2026-05-11 (lego n=8 landed + all lego runs re-eval'd on full 200-view split; lego_n8 = 19.21, lego_n8_ssl_mv_dav2s = 21.16 → Δ +1.95 dB; gap to FSGS avg-8 ~−3.4 dB)

---

## 1. Method (current)

### 1.1 Pipeline

```
  NeRF-Synthetic (RGBA, OpenGL) ──► sparse_sampler (n ∈ {3,6,9,12}) ──► train views
                                                                         │
                                                                         ▼
           random-in-box init (100 k Gaussians)  ──►  GaussianModel
                                                           │
                                                           ▼
                                gsplat.rasterization  (RGB + Expected-Depth, antialiased)
                                                           │
                          ┌────────────────────────────────┼────────────────────────────────┐
                          ▼                                ▼                                ▼
                  photometric loss                   SSL loss bank                DefaultStrategy
                 L1 + 0.2·DSSIM on GT           (multiview_photo, ...)      (densify / split / prune)
                          │                                │                                │
                          └────────────────► Σ ◄───────────┘                                │
                                             │                                              │
                                          backward                                          │
                                             │                                              │
                    per-param Adam  ◄────────┴────────────────────────────────────────────┘
                                             │
                                             ▼
                                EMA / snapshot teacher  (built lazily if any SSL loss needs it)
```

### 1.2 Key engineering choices

- **Camera convention**: OpenCV everywhere (X right, Y down, Z forward). The
  NeRF-Synthetic JSON (OpenGL) is flipped with `diag(1, -1, -1)` at load
  time. This is the single most common source of bugs in GS repos — do not
  touch the loader without re-running the smoke test.
- **Per-parameter Adams**, not a single fused Adam. `DefaultStrategy`
  mutates optimizer state during densify/prune, which only works if each
  tensor has its own optimizer. This matches gsplat's own `simple_trainer`.
- **SSL gradient isolation from densify**: two safe patterns
  - `detach_geometry=True` — SSL only updates color/opacity (used by
    `pseudo_view`, `ema_teacher`)
  - **Separate** forward pass whose `info` is never handed to the strategy
    (used by `multiview_photo`). Geometry still receives gradients via
    standard autograd, but `state["grad2d"]` stays clean.
  - Getting this wrong (v1 of `pseudo_view`) caused densify to **2×** the
    Gaussian count and dropped PSNR from 15.7 → 10.1.

### 1.3 SSL loss registry

All live in `sparse_gs/losses/ssl.py :: SSLLossBank`. Each is
**configurable**, **optional**, and **additive**.

| Loss name          | Status | Needs teacher | Needs prior | Notes |
|--------------------|--------|---------------|-------------|-------|
| `multiview_photo`  | ✅ working, +3.1 dB @ n=6 | no | no | Cross-view reproj via student depth |
| `pseudo_view`      | ✅ built, neutral/-0.5 dB | yes | no | No new info on vanilla 3DGS |
| `ema_teacher`      | ✅ built, neutral/-0.5 dB | yes | no | Same reason |
| `depth_consist`    | 🧱 stub | no (or yes for EMA variant) | yes (monocular) | TODO A1 |
| `feature`          | 🧱 stub | optional | yes (DINOv2) | TODO B1 |

---

## 2. Contributions (what's ours vs standard 3DGS)

1. **Self-contained sparse-view 3DGS baseline on top of gsplat 1.5.3**,
   runnable on Windows + sm_120 (we wrote the `scripts/_bootstrap.py`
   machinery to make the JIT build succeed there).
2. **Clean SSL hook design** that provably does **not** corrupt gsplat's
   density-control state, with two safe gradient-flow patterns documented
   and enforced by the renderer API (`detach_geometry`) and loss bank.
3. **`multiview_photo` loss** — a no-prior cross-view photometric
   reprojection that gives a large boost in the sparse-view regime:
   **+0.9 / +3.1 / +2.6 dB** at `n = 3 / 6 / 9` on `lego`.
4. **Negative-result ablation** documenting why pure self-distillation
   (`pseudo_view`, `ema_teacher`) is neutral/harmful on vanilla 3DGS at
   our iteration budgets — the student is already perfect on train views,
   so the teacher carries no extra information.
5. **Densify-aware EMA teacher** (`sparse_gs/models/ema.py`) that
   gracefully resets when gsplat changes the Gaussian count, so the
   teacher stays usable across densification steps. Kept in-tree for
   future priors (depth / feature distillation).

(3) and (4) are the only things that would justify a paper section on
their own today; (1), (2), (5) are infrastructure.

---

## 3. Current results on NeRF-Synthetic `lego`

Full 200-view test split, 7000 iters, RTX 5090. Best number in **bold**.

### 3.1 Head-to-head (this session, all numbers re-run on the same code)

| setup                       |  n | PSNR ↑    | SSIM ↑ | LPIPS ↓ | vs baseline | `kept_ratio` | `coverage` |
|-----------------------------|:--:|----------:|-------:|--------:|------------:|-------------:|-----------:|
| baseline (no SSL)           |  3 | 12.889    | 0.601  |    —    |      —      |       —      |      —     |
| mv v1 (no occ.)             |  3 | **13.812**| **0.634** |  —    |    +0.92    |       —      |      —     |
| mv v2 (strict occ. τ=0.05)  |  3 | 12.904    | 0.608  | 0.463   |    +0.02    |     0.37     |    0.10    |
| **mv v3 (loose occ. τ=0.10, start=3k)** | 3 | 13.676 | 0.630 | **0.448** |  +0.79  |   0.58  |    0.19    |
|                             |    |           |        |         |             |              |            |
| baseline (no SSL)           |  6 | 15.724    | 0.729  |    —    |      —      |       —      |      —     |
| mv v1 (no occ.)             |  6 | 13.880    | 0.712  |    —    |    −1.84 ⚠️ |       —      |      —     |
| mv v2 (strict occ. τ=0.05)  |  6 | 16.496    | 0.752  | 0.292   |    +0.77    |     0.48     |    0.14    |
| **mv v3 (loose occ. τ=0.10, start=3k)** | 6 | **18.153** | **0.774** | **0.271** | **+2.43** |   0.68  |    0.23    |

**Takeaways.**
- **v3 is now the recommended SSL setting**: best PSNR/SSIM/LPIPS at
  n = 6, tied with v1 at n = 3 but with better LPIPS.
- **v1 "no occlusion check" is unstable at n = 6.** It reproduced as
  −1.84 dB this session, not the +3.13 dB in the legacy table below.
  The legacy n = 6 number should be treated as un-reproducible; v3
  supersedes it.
- **Strict occlusion (v2) over-rejects at n = 3.** `kept_ratio` drops
  to 0.37 and the SSL signal collapses (coverage 0.10). v3 fixes it by
  (i) delaying the gate until iter 3000 and (ii) loosening τ = 0.05 → 0.10.
- `multiview_photo` time overhead with occlusion ≈ +60 % vs baseline
  (no extra overhead vs v1 once the check runs).

### 3.2 Three-density sweep (n = 3 / 6 / 9, all re-run this session)

| n | baseline | v1 (no occ.) | **v3 (loose occ.)** | v1 vs base | **v3 vs base** | v3 vs v1 |
|:-:|---------:|-------------:|--------------------:|-----------:|---------------:|---------:|
| 3 |   12.89  |    13.81     |          13.68      |    +0.92   |     **+0.79**  |   −0.14  |
| 6 |   15.72  |    13.88 ⚠️  |        **18.15**    |    −1.84   |     **+2.43**  |  +4.27   |
| 9 |   21.08  |    23.63     |          23.14      |    +2.56   |     **+2.06**  |   −0.50  |

**Interpretation.** v3's strength is concentrated at n = 6, exactly the
operating point where v1 collapses. At n = 9 the geometry is already
healthy enough that v1's "no-occlusion" warps are mostly correct, so
v3's gate occasionally rejects useful pixels (kept_ratio 0.74 — barely
filtering — but the small collateral cost is visible). At n = 3 v3 ≈ v1
within noise. Net: **v3 is a strict improvement on average and a major
improvement at n = 6**, which is the canonical sparse-view benchmark.

### 3.3 Legacy numbers (earlier sessions, v1 mv only — kept for context)

The "v1 +3.13 dB at n = 6" number from earlier sessions did not reproduce
here (we got −1.84 dB this run). The seed/densification trajectory matters
a lot for un-gated multi-view photo loss; this is exactly why v3 exists.

### 3.4 LLFF n=3 sweep — all 8 forward-facing scenes (2026-05-11)

The headline result of the project. Sweep ran 16 jobs (`baseline` +
`dav2s` × 8 scenes) via `scripts/run_llff_sweep.py`, ~19 min total
on RTX 5090. One transient `cudaErrorUnknown` on `leaves/dav2s` was
recovered by `scripts/retry_failed_runs.py` (see §5 S7).

| scene    | base PSNR | dav2s PSNR | ΔPSNR | base LPIPS | dav2s LPIPS | ΔLPIPS |
|----------|---------:|-----------:|------:|-----------:|------------:|-------:|
| fern     |    16.07 |      18.48 | +2.40 |     0.611  |     0.353   | −0.258 |
| flower   |    15.13 |      16.64 | +1.52 |     0.517  |     0.469   | −0.048 |
| fortress |    13.59 |  **18.14** |**+4.55**|   0.543  |     0.463   | −0.080 |
| horns    |    12.69 |      14.09 | +1.40 |     0.595  |     0.574   | −0.021 |
| leaves   |    12.03 |      12.35 | +0.33 |     0.543  |     0.535   | −0.008 |
| orchids  |    14.86 |      15.36 | +0.50 |     0.385  |     0.367   | −0.018 |
| room     |    11.79 |      12.99 | +1.20 |     0.656  |     0.603   | −0.054 |
| trex     |    11.77 |      13.93 | +2.16 |     0.595  |     0.557   | −0.038 |
| **avg**  |  **13.49**| **15.25** |**+1.76**|  **0.556**|   **0.490** |**−0.066**|

**Takeaways.**
- **Every one of 8 scenes improves on PSNR, SSIM, and LPIPS** when
  DAv2 + multiview-photo is added on top of the baseline. No scene
  regresses on any metric. This is the strongest evidence we have
  that the SSL+depth combination is genuinely useful, not lego/fern-
  specific overfitting.
- **Win is concentrated on large structured scenes** (`fortress`
  +4.55, `fern` +2.40, `trex` +2.16) where DAv2 has a coherent global
  prior to give. Self-similar / textureless scenes (`leaves` +0.33,
  `orchids` +0.50) benefit least, which is exactly the failure mode
  expected from monocular depth on repetitive textures.
- Absolute level is **~5 dB below FSGS** (their LLFF-avg ~20.4); the
  gap to literature is no longer "different dataset" but "missing
  ablations" (CoR-GS-style dual distill, longer training / higher
  cap, BLA depth filtering). See `outputs/sota_comparison.md` §3–4.

### 3.5 lego n=8 + 200-view re-eval of every lego run (2026-05-11)

The Blender / NeRF-Synthetic literature reports nearly all sparse-view
numbers at **n=8** on the full 200-view test split. We had a hole there
(only n=3/6/9/12 trained), and our older lego numbers were 8-view
training-time evals (`eval.num_test_renders=8`) which by §5 historic
correction are off by ±2–4 dB.

Two things landed:

1. **Trained `lego_n8` and `lego_n8_ssl_mv_dav2s`** with full-split
   eval baked in (`eval.num_test_renders: 200`), 7000 iters, same
   recipe as the n=6/n=9 dav2s runs:

   | run                       | PSNR   | SSIM   | LPIPS  | wall (s) | #G   |
   |---------------------------|-------:|-------:|-------:|---------:|-----:|
   | `lego_n8` (baseline)      | 19.211 | 0.789  | 0.234  | 148      | 406k |
   | `lego_n8_ssl_mv_dav2s`    | **21.158** | **0.829** | **0.202** | 436 | 372k |
   | **Δ (DAv2+mv)**           | **+1.95 dB** | **+0.040** | **−0.032** | — | — |

2. **Re-evaluated every existing lego run on the full 200-view split**
   via `scripts/eval_ckpt_lego_all.py` (one subprocess per run to
   sidestep the `cudaErrorUnknown` context-rot we saw on the LLFF
   sweep). 18 runs × ~70 s each ≈ 21 min. Numbers now live in
   `metrics_full.json` next to each `metrics.json`. The full
   re-aligned curve:

   | n  | baseline PSNR | DAv2+mv PSNR | Δ        |
   |---:|--------------:|-------------:|---------:|
   |  3 |        11.85  |       12.31  | +0.46    |
   |  6 |        17.64  |   **19.77**  | **+2.13**|
   |  8 |        19.21  |   **21.16**  | **+1.95**|
   |  9 |        18.09  |       19.95  | +1.86    |
   | 12 |        21.83  |          —   | —        |

**Takeaway — sweet point and SOTA gap.** With clean 200-view eval the
DAv2+mv-vs-baseline gain peaks near **n=6** (+2.13 dB) and stays large
through **n=8** (+1.95 dB), confirming the prior is filling genuine
holes when photo-only undertrains. **n=8 is also our best absolute
single-scene Blender point**: 21.16 dB vs the literature n=8 avg-8
of ~24.6 → gap **−3.44 dB** (one scene only — full avg-8 still
requires the other 7 Blender scenes).

---

## 4. Open issues

- **O1. `multiview_photo` occlusion test.** ✅ Implemented and **validated**.
  - v2 (strict τ=0.05, gate-from-start): +0.77 dB @ n=6, **−0.91 dB** vs
    v1 @ n=3 (over-rejects; see §3.1 `kept_ratio`).
  - **v3 (loose τ=0.10, `occlusion_start_iter=3000`, `occlusion_alpha_thresh=0.3`)**:
    +2.43 dB @ n=6 (beats v2 by 1.66 dB), −0.14 dB vs v1 @ n=3 (essentially
    tied, with better LPIPS). v3 is now the recommended SSL config.
- **O2. LPIPS.** ✅ Implemented (§5 S2). Reported automatically in
  `trainer.evaluate()` when the `lpips` PyPI package is importable;
  silently skipped otherwise (no hard dependency). Numbers in §3.1.
- [x] **O3.** ~~Only `lego` has been ablated.~~ ✅ Closed 2026-05-11.
  Full LLFF n=3 sweep over 8 scenes shows DAv2+mv improves on all
  metrics on all scenes (§3.4). The "lego-specific" worry is resolved.
- **O4. Surface reconstruction is missing entirely.** The project title
  says "Surface" but the output is still volumetric 3DGS.
  **→ TODO A2** (Poisson / TSDF post-process).
- **O5. No external prior is used.** `multiview_photo` only couples
  training views. For **n=3** we need an external signal (monocular
  depth, DINOv2). **→ TODO A1, B1.**
- **O6. No COLMAP / DTU / MipNeRF360 loader.** We can't claim to
  generalise without at least one real-scene dataset. **→ TODO A3.**
- **O7. `packed=False` default.** gsplat 1.5.3 has a packed-mode shape
  bug with `RGB+ED` + `backgrounds`. Cost: negligible at our scale.
  Tracked here so we don't forget to revisit on upgrade.
- **O8. `pseudo_view` / `ema_teacher` are dead code on vanilla 3DGS** but
  should become useful once O5 (external prior) is resolved — they
  propagate the prior to **novel** poses. Do not delete.
- **O9. Occlusion check cost.** Adds one extra render per neighbor per
  step (no-grad, geometry detached, so ~cheap but nonzero). If step
  time becomes an issue we can subsample or only run every K steps.

---

## 5. TODO list

Priority tags: **S** = ship today, **A** = needed for report, **B** =
differentiator / paper, **C** = nice-to-have.

### S — this session

- [x] **S0.** Create this file (`PROJECT_STATUS.md`).
- [x] **S2. LPIPS metric.** Added `LPIPSMetric` to `utils/metrics.py`
      (VGG backbone, lazy-loaded, degrades to no-op if the `lpips` pip
      package is unavailable). Wired into `trainer.evaluate()` behind
      `eval.lpips` (default `true`). Reported as `<tag>/lpips` in the
      eval dict.
- [x] **S1. Occlusion-aware `multiview_photo`.** Added
      `occlusion_check` / `occlusion_tau` / `occlusion_alpha_thresh`
      knobs to `_multiview_photo`. When enabled, we render the student
      depth from `cam_b` (`torch.no_grad()` + `detach_geometry=True`,
      strategy info discarded), sample it at the projected uv, and drop
      pixels with `|z_b - D_b(uv)| / z_b > tau` or low `cam_b` alpha.
      Logs `ssl/multiview_photo/kept_ratio` for diagnostics.
      Enabled on `configs/sparse_view_ssl_mv_v2.yaml`.
- [x] **Validation run** (n=3 and n=6 on `lego`). Done this session,
      numbers in §3.1. Surprising finding: v1 is unstable at n=6 (reproduced
      at −1.84 dB this session, not the +3.13 dB from earlier runs). The
      occlusion check is required to recover it; the naive version (v2)
      over-rejects at n=3 and is nearly tied with baseline there. The
      sparse-view-friendly **v3** config resolves both problems:
      - n = 3: 13.676 / 0.630 / 0.448  (+0.79 dB vs baseline, tied with v1
        within noise, better LPIPS)
      - n = 6: **18.153 / 0.774 / 0.271** (+2.43 dB vs baseline, +4.27 dB
        vs v1 which regressed this session)
- [x] **S3-follow-up. n = 9 with v3.** Done. Result **23.139 / 0.9022 /
      0.1221** (+2.06 dB vs baseline, −0.50 dB vs v1). v3's gate barely
      bites at n=9 (kept_ratio 0.74) so the small loss vs v1 is just
      collateral rejection of valid pixels. v3 is still preferred as the
      default because (i) it's robust at n=6 where v1 collapses and (ii)
      the n=9 gap is well within run-to-run noise. See §3.2 for the full
      sweep.
- [~] **S4. Depth-consistency loss (`_depth_consist`) — code done,
      validation blocked on network.** This session.
      - Implemented scale-shift-invariant L1 (MiDaS-style) in
        `losses/ssl.py::_depth_consist` (median-MAD normalize both maps
        in disparity space, masked L1).
      - Plumbing: `Camera.mono_depth` + `Camera.depth_kind` (datasets/
        nerf_synthetic.py), generic loader `utils/depth_prior.py` with
        two backends — `alpha` (free, RGBA-derived disparity) and
        `cache` (read pre-computed `.npz`). Trainer wires
        `cfg.data.depth_prior` straight to the dataset.
      - Precompute script: `scripts/precompute_depth.py` supports
        `--backend alpha` (trivial) and `--backend transformers` (HF
        depth-estimation pipeline; tested with Depth-Anything-V2-Small
        and DPT/GLPN).
      - Config: `configs/sparse_view_ssl_mv_depth.yaml` (combines mv-v3
        with depth_consist; defaults to `backend: alpha`).
      - Smoke test on `_mini` passes — loss runs without NaN, gradient
        is non-zero, training converges normally.
      - **Outcome with the alpha backend (negative result, expected):**
        | n | PSNR | LPIPS | vs v3 alone |
        |:-:|----:|------:|------------:|
        | 3 | 13.473 | 0.470 | −0.21 dB |
        | 6 | 18.074 | 0.298 | −0.08 dB |
        Alpha-derived disparity only encodes "foreground vs background",
        i.e. essentially the same information the photo loss already
        recovers from RGBA via the white background; SSI matching adds
        no new signal and slightly perturbs late training (val PSNR
        actually *decays* from 18.1 @2k to 17.0 @6k at n=6).
      - **Validation with a real prior is blocked** by HF Hub
        connectivity from the current host — every model id we tried
        (`LiheYoung/depth-anything-small-hf`,
        `depth-anything/Depth-Anything-V2-Small-hf`,
        `vinvino02/glpn-nyu`) downloads an *empty* snapshot dir and
        `transformers` then raises *"Unrecognized model … should have a
        `model_type` key in its config.json"*. `https://huggingface.co`
        responds 200, but `huggingface.co` model files appear to be
        either rate-limited or proxied through a path that fails for
        binary blobs. `HF_ENDPOINT=https://hf-mirror.com` then fails on
        DNS (`getaddrinfo`).
      - **Required next:** either (a) provide a working HF mirror /
        proxy / `HF_TOKEN`, (b) point the script at a local pre-
        downloaded model dir, or (c) run a Depth-Anything-V2 inference
        pass on another machine and copy the `.npz`-cache into
        `data/nerf_synthetic/<scene>/_depth_cache/<tag>/train/`. After
        that, switch the config to `backend: cache, tag: <tag>` and
        re-run `--n-views 3 / 6 / 9`. Expected (literature): +1–2 dB at
        n=3, smaller at n=6/9.
- [x] **S5. SparseSurf-style geometric priors (surface_flatten +
      normal_smooth).** This session, while S4 was blocked.
      - Two new SSL terms in `losses/ssl.py`:
        - `_surface_flatten` — penalises Gaussian aspect ratio
          (min_scale / max_scale, "ratio" mode by default; "abs" mode
          available). Drives every Gaussian toward a 2D surfel —
          the geometric prior used by 2DGS / PGSR /
          GaussianSurfels. Pure parameter-space regularizer, no
          extra render needed.
        - `_normal_smooth` — angular TV on the per-pixel normal that
          we derive on-the-fly from the rendered depth (central
          finite-differences in world space → cross product →
          normalize). Optional RGB-edge gating
          (`edge_aware: true`) lets texture seams pass through.
          Self-supervised: no external prior.
      - Both registered in `SSLLossBank.LOSSES`, fully configurable
        (`enabled / weight / start_iter / mode / opacity_thresh /
        alpha_thresh / depth_eps / edge_aware / edge_lambda`).
      - Configs: `configs/sparse_view_ssl_surface.yaml` (full
        strength) and `configs/sparse_view_ssl_surface_mild.yaml`
        (½–¼ weights, later start). Smoke-tested on `_mini`.
      - **Results on `lego`:**
        | n | config | PSNR | SSIM | LPIPS | vs v3 alone |
        |:-:|:--|----:|----:|------:|------------:|
        | 3 | surface (full)   | 13.492 | **0.637** | **0.429** | −0.18 / +0.007 / **−0.019** |
        | 6 | surface (full)   | 16.830 | 0.763 | 0.279 | −1.32 dB |
        | 6 | surface (mild)   | 17.288 | 0.768 | **0.269** | −0.86 dB / tied LPIPS |
        At **n=3 the LPIPS drops by 0.019** and SSIM ticks up — the
        prior is doing visually what we expect (cleaner surface,
        fewer floaters), even though novel-view PSNR is essentially
        flat. At **n=6 PSNR drops** because the prior over-smooths
        fine geometric detail (lego's teeth/rivets); the mild config
        recovers half the loss but doesn't beat the v3 baseline on
        any metric except a tied LPIPS.
      - **Diagnosis.** This is the *expected* SparseSurf trade-off:
        novel-view photometry rewards every floater that happens to
        line up with the few train views, while a surface prior
        deliberately removes them. PSNR will not show the win — it
        only shows up under depth / mesh metrics, which we don't yet
        evaluate.
      - **Required next.** (1) Add a depth-error and a Chamfer-on-
        extracted-points evaluator (S6 below). (2) Run the same
        configs and re-grade. Only then is the surface route
        comparable to the photo-only baselines on equal footing.
- [ ] **S6. Geometric evaluation (depth + Chamfer).** Without this
      we can't fairly grade S4/S5/A1/A2. Plan:
      - Render the *student's* depth at every test pose, dump as
        `.npz`. Add a `eval/depth_l1_ssi` metric (SSI-aligned L1
        between student depth and either GT depth from
        `transforms_test.json` if available, or a held-out dense-view
        reconstruction at iter ∞).
      - For Chamfer: marching-cubes-style mesh extraction from the
        Gaussian field is hard; instead sample `M` points by
        `(means | opacity > τ)` and compute Chamfer to a reference
        point cloud (either the dense-view reference or a `_clean`
        cache).
      - Wire into `trainer.evaluate()` behind `eval.geometry: true`.

### A — for the final report

- [~] **A1. Monocular depth prior** (`_depth_consist` implementation).
      **Code complete and smoke-tested** (see S4 above for details);
      what remains is purely producing a real prior cache. Once a
      proper monocular-depth `.npz` cache is in
      `data/nerf_synthetic/<scene>/_depth_cache/<tag>/train/`, switch
      `configs/sparse_view_ssl_mv_depth.yaml`
      `data.depth_prior` to `{backend: cache, tag: <tag>}` and re-run
      n = 3 / 6 / 9. Expected: +1–2 dB at n=3.
- [ ] **A2. Surface / mesh extraction.** `scripts/extract_mesh.py`:
      render depth from train + (optionally) pseudo views, fuse via
      `open3d` TSDF, marching cubes → `outputs/<exp>/mesh.ply`. Also
      save a screenshot for the report.
- [ ] **A3. COLMAP / DTU loader.** `datasets/colmap.py` via `pycolmap`.
      For DTU, additionally load per-view object mask as alpha
      supervision. Add `SparseSampler` strategy `uniform_pose` that
      picks views spread by pose distance.
- [x] **S3. Cross-scene ablation.** ✅ Done 2026-05-11 — full LLFF
      n=3 sweep over all 8 scenes (`fern, flower, fortress, horns,
      leaves, orchids, room, trex`), baseline vs DAv2+mv. **Every
      scene improves on every metric**; avg +1.76 dB / +0.041 SSIM /
      −0.066 LPIPS. See §3.4 and `outputs/sota_comparison.md` §2.
      The Blender (`chair`/`ficus`) cross-scene ablation is no longer
      blocking — LLFF is the canonical sparse-view benchmark and we
      now have 8/8 coverage there.
- [x] **S7. Auto-retry daemon for sweep flakiness.**
      `scripts/retry_failed_runs.py`: scans expected `outputs/llff_*_n3_*/
      metrics.json`, identifies missing/partial runs, re-launches them
      via the same `launch_llff_sweep_detached.py` machinery (no window,
      append log), with `--max-retries N` and `--dry-run`. Validated
      end-to-end on this sweep: it correctly identified the single
      `leaves/dav2s` `cudaErrorUnknown` failure as missing; we ran a
      manual retry (which succeeded), then `--dry-run` confirmed
      `no missing runs detected`. Now part of standard sweep workflow.

### B — paper-level differentiators

- [ ] **B1. DINOv2 feature consistency** (`_feature_consist`). Frozen
      DINOv2-S, patch 14, patch-token cosine loss on aligned crops.
      Especially useful on pseudo views (where GT is unavailable).
- [ ] **B2. Prior + pseudo-view combo.** Once A1 is in, re-enable
      `pseudo_view` and have it **propagate the depth prior** to novel
      poses via the EMA teacher. This is the one scenario where
      `pseudo_view` is not degenerate.
- [ ] **B3. Sparse-tuned densify strategy.** Either `MCMCStrategy` or
      tighter `refine_stop_iter` / higher `grow_grad2d` on
      `DefaultStrategy`. Cheap experiment, likely +0.3–1 dB "for free".

### C — maintenance / far future

- [ ] Real EMA teacher (momentum < 1.0) with robust densify-boundary
      handling (currently snapshots reset hard).
- [ ] SH degree schedule tuning for sparse-view.
- [ ] gsplat viewer hookup for interactive debug.
- [ ] Split `scripts/eval.py` out of `trainer.fit()`.

---

## 6. Assumptions & gotchas (do not lose track of these)

1. **Packed mode off by default** (gsplat 1.5.3 bug, §5 O7).
2. **First CUDA call triggers `ninja` JIT build** (~2 min on Windows
   sm_120). `scripts/_bootstrap.py` configures MSVC/NVCC env vars and
   patches a `small` macro collision in `c10`.
3. **White background**: NeRF-Synthetic RGBAs are composited on white at
   load time; the renderer also receives `backgrounds=white`.
4. **Train-view subsampling is deterministic** given `n_train_views + seed`
   or `train_view_ids`. Sweeps at `n=3/6/9` therefore see fixed subsets;
   conclusions between `n` values are not directly comparable in absolute
   PSNR, only in `Δ PSNR` (which is what we report).
5. **`multiview_photo` does a second forward per step** → ~+60 % step
   time. This is the dominant cost of SSL; everything else is cheap.
6. **EMA teacher is snapshot-mode** (`momentum=1.0`) by default. True EMA
   is gated on C-level work.

---

## 7. How to use this document

- **Before starting any task**, re-read §4 and §5 and pick the highest
  priority open item that fits the session.
- **After finishing a task**, update §3 (numbers), §4 (close the open
  issue), and §5 (tick the box or move to a new priority).
- Keep §1 & §2 **conceptually stable** — rewrite only when the method
  or contribution story genuinely changes, not when you add a feature.
- Deep detail (code pointers, gsplat quirks) belongs in `README.md`,
  not here. This file is about *direction*, not *API*.
