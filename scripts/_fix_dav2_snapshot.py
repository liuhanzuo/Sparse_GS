"""Download DAv2-Small to a real local dir (Windows hub cache uses
symlinks that may end up empty)."""
from huggingface_hub import snapshot_download
from pathlib import Path
import time

target = Path(r"D:\SSL\sparse_gs\outputs\_models\Depth-Anything-V2-Small-hf")
target.mkdir(parents=True, exist_ok=True)

t0 = time.time()
p = snapshot_download(
    repo_id="depth-anything/Depth-Anything-V2-Small-hf",
    repo_type="model",
    local_dir=str(target),
)
dt = time.time() - t0
print(f"local dir: {p}  ({dt:.1f}s)")
files = sorted(Path(p).rglob("*"))
files = [f for f in files if f.is_file()]
print(f"{len(files)} files:")
for f in files:
    print(f"  {f.relative_to(target)}  {f.stat().st_size}")
