"""One-shot HF snapshot downloader for Depth-Anything-V2-Base.

Designed to be launched as a *detached* background process via
``Start-Process pythonw -ArgumentList ...`` so it survives the parent
PowerShell session exiting.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download   # noqa: E402

REPO = "depth-anything/Depth-Anything-V2-Base-hf"
_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get(
    "SPARSE_GS_DAV2_BASE_DIR",
    str(_REPO_ROOT / "models" / "depth_anything_v2_base"),
))
LOG = LOCAL.parent / "_download_base.log"
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
