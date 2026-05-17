"""一次性下载脚本：拉取 Depth-Anything-V2-Small 权重到 models/depth_anything_v2_small。"""
import os
import sys
import time
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

LOG = r"F:\Sparse_GS\data\_dl_dav2.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    from huggingface_hub import snapshot_download

    log("START download Depth-Anything-V2-Small")
    try:
        p = snapshot_download(
            repo_id="depth-anything/Depth-Anything-V2-Small-hf",
            local_dir=r"F:\Sparse_GS\models\depth_anything_v2_small",
            max_workers=4,
            ignore_patterns=["*.bin"],  # 仅保留 safetensors，避免重复
        )
        log(f"DONE -> {p}")
        return 0
    except Exception as e:  # noqa: BLE001
        log(f"FAIL {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
