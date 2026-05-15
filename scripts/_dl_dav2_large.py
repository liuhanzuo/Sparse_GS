"""See _dl_dav2_base.py — same idea for the Large variant (~1.3GB)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download   # noqa: E402

REPO = "depth-anything/Depth-Anything-V2-Large-hf"
LOCAL = Path(r"d:\SSL\sparse_gs\models\depth_anything_v2_large")
LOG = LOCAL.parent / "_download_large.log"
LOCAL.mkdir(parents=True, exist_ok=True)

with open(LOG, "w", encoding="utf-8") as f:
    f.write(f"[start] repo={REPO}\n  local={LOCAL}\n")
    try:
        p = snapshot_download(
            repo_id=REPO,
            local_dir=str(LOCAL),
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors",
                            "preprocessor_config.json"],
        )
        f.write(f"[done] {p}\n")
    except Exception as e:                                       # noqa: BLE001
        f.write(f"[error] {type(e).__name__}: {e}\n")
        sys.exit(1)
