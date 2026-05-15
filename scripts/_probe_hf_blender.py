"""Probe nerfbaselines HF dataset for the blender file layout."""
from huggingface_hub import HfApi

api = HfApi()
repo = "nerfbaselines/nerfbaselines-data"
files = list(api.list_repo_files(repo, repo_type="dataset"))
print(f"total files in repo: {len(files)}")
blender = [f for f in files if "blender" in f.lower()]
print(f"blender files: {len(blender)}")
for f in blender[:200]:
    print(" ", f)
