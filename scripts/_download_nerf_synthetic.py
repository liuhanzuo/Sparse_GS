"""一次性下载脚本：从 HF 镜像拉取 NeRF-Synthetic 全 8 场景（仅 RGB+transforms）。

后台运行，日志写到 F:/Sparse_GS/data/_dl_nerf.log。
"""
import os
import sys
import time
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

LOG = r"F:\Sparse_GS\data\_dl_nerf.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    from huggingface_hub import snapshot_download

    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    allow = []
    for s in scenes:
        allow += [
            f"{s}/transforms_train.json",
            f"{s}/transforms_val.json",
            f"{s}/transforms_test.json",
        ]
        for sp in ("train", "val", "test"):
            allow.append(f"{s}/{sp}/*.png")
    ignore = ["*_depth_*.png", "*_normal_*.png", "trained/*", "README*", ".gitattributes"]

    log("START download nerf_synthetic from pablovela5620/nerf-synthetic-mirror")
    try:
        p = snapshot_download(
            repo_id="pablovela5620/nerf-synthetic-mirror",
            repo_type="dataset",
            local_dir=r"F:\Sparse_GS\data\nerf_synthetic",
            allow_patterns=allow,
            ignore_patterns=ignore,
            max_workers=8,
        )
        log(f"DONE -> {p}")
        return 0
    except Exception as e:  # noqa: BLE001
        log(f"FAIL {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
