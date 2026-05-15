"""Debug: render one chair view and dump info dict shapes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap  # noqa
from sparse_gs.utils.config import load_config
from sparse_gs.datasets.nerf_synthetic import NerfSyntheticDataset
from sparse_gs.models.gaussians import GaussianModel
from sparse_gs.rendering.gsplat_renderer import GSplatRenderer
import torch

cfg = load_config(r"configs/_w2_prune/blender_chair_n8_dav2s_depthv2_prune.yaml")
ds = NerfSyntheticDataset(
    root=cfg["data"]["root"], scene="chair",
    n_train_views=8, train_view_ids=None, image_downsample=1,
    white_background=True, seed=42, depth_prior=None,
)
gauss = GaussianModel(sh_degree=3).to("cuda")
gauss.init_random(num_points=10_000, extent=1.5, rgb_init=0.5,
                  scale_init_factor=0.01, device="cuda")
renderer = GSplatRenderer(
    sh_degree=3, near_plane=0.01, far_plane=1e10,
    rasterize_mode="antialiased", packed=False, absgrad=True,
    render_mode="RGB+ED",
)
cam = ds.train[0].to("cuda")
out = renderer.render(
    gauss, viewmat=cam.viewmat, K=cam.K,
    width=cam.width, height=cam.height,
    active_sh_degree=0, background=torch.tensor([1.0,1.0,1.0], device="cuda"),
)
info = out["info"]
print("packed:", renderer.packed)
print("N_gaussians:", gauss.num_points)
print("rgb:", None if out["rgb"] is None else tuple(out["rgb"].shape))
print("depth:", None if out["depth"] is None else tuple(out["depth"].shape))
print("alpha:", None if out["alpha"] is None else tuple(out["alpha"].shape))
print("info.keys:", list(info.keys()))
for k in ("radii", "means2d", "depths", "gaussian_ids", "opacities"):
    v = info.get(k, None)
    if v is None:
        print(f"  {k}: None")
    elif torch.is_tensor(v):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.min().item() if v.numel()>0 else 'n/a'} "
              f"max={v.max().item() if v.numel()>0 else 'n/a'}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
