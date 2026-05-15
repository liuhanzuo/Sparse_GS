"""Download (and partially extract) the LLFF benchmark dataset.

The canonical LLFF release ships as a single ZIP file
``nerf_llff_data.zip`` (~3.7 GB) containing 8 scenes. We provide three
modes:

  ``--scene fern --partial``       (default)
      Stream-download the ZIP, extract *only* the requested scene to
      ``data/nerf_llff_data/<scene>/``, then delete the ZIP. Net disk
      cost ~ 400-700 MB / scene.

  ``--scene fern``                 (no --partial)
      Same, but keep the ZIP cached at ``data/_cache/nerf_llff_data.zip``
      so subsequent ``--scene <other>`` calls extract instantly.

  ``--all``
      Download the full ZIP and extract all 8 scenes.

Source URL
----------
The original Google-Drive-hosted ZIP requires authentication. We use
the public mirror Mildenhall maintains on his Berkeley homepage:

    https://people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip

If that URL goes down, edit ``LLFF_ZIP_URL`` below or pass
``--url <other>``.

Usage
-----

    # First time: download fern only (deletes ZIP after extraction).
    python scripts/download_llff.py --scene fern --partial

    # Add another scene later (re-downloads since --partial deleted ZIP).
    python scripts/download_llff.py --scene flower --partial

    # Or: keep the ZIP cached so future scenes are free.
    python scripts/download_llff.py --scene fern             # keeps ZIP
    python scripts/download_llff.py --scene flower           # uses cache

    # Or: just grab everything (~3.7 GB → ~3.7 GB extracted).
    python scripts/download_llff.py --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

# Bootstrap so we can be run as ``python scripts/download_llff.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import _bootstrap  # noqa: E402,F401

_ROOT = _bootstrap.PROJECT_ROOT

# Berkeley used to mirror this; the URL has been 404 since ~2025. Keep
# it as a primary attempt for back-compat, but fall back through:
#   1) Hugging Face mirror (re-host of the full LLFF benchmark, public)
#   2) Google-Drive original (canonical, but quota frequently exceeded)
LLFF_ZIP_URL = "https://people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip"
LLFF_HF_URL  = "https://huggingface.co/datasets/SlekLi/LLFF/resolve/main/archive.zip"
LLFF_GDRIVE_ID = "16VnMcF1KJYxN9QId6TClMsZRahHNMW5g"  # canonical nerf_llff_data.zip
LLFF_SCENES = ["fern", "flower", "fortress", "horns",
               "leaves", "orchids", "room", "trex"]


def _gdown_download(file_id: str, dest: Path) -> None:
    """Download a Google-Drive file via ``gdown`` (handles confirm tokens).

    We intentionally do *not* import gdown at module load time so that
    users without the dep can still run the rest of the script.
    """
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Google-Drive fallback requires `gdown`: pip install gdown\n"
            f"  (original error: {e})"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gdown] id={file_id} -> {dest}")
    # gdown >= 6.0 dropped `fuzzy` and prefers `id=` for confirm-token handling.
    # Older releases (<5) used positional URL + `fuzzy=True`. Try the new API
    # first and fall back gracefully.
    try:
        out = gdown.download(id=file_id, output=str(dest),
                             quiet=False, resume=True)
    except TypeError:
        url = f"https://drive.google.com/uc?id={file_id}"
        out = gdown.download(url, str(dest), quiet=False)  # type: ignore[call-arg]
    if out is None or not Path(out).is_file():
        raise SystemExit(
            "[gdown] download failed. Possible causes:\n"
            "  * GDrive quota exceeded for this file (try again later)\n"
            "  * network unable to reach drive.google.com\n"
            "Workarounds: download nerf_llff_data.zip manually and place\n"
            f"  it at {dest}, then re-run this script."
        )


def _http_stream(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Plain HTTP streaming download with tqdm. Raises on HTTP errors."""
    import requests
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)
    except ImportError:
        bar = None

    with open(dest, "wb") as f:
        downloaded = 0
        for buf in r.iter_content(chunk_size=chunk):
            if not buf:
                continue
            f.write(buf)
            downloaded += len(buf)
            if bar is not None:
                bar.update(len(buf))
            elif total > 0 and downloaded % (50 * chunk) == 0:
                print(f"  ... {downloaded/1e6:.1f} / {total/1e6:.1f} MB")
    if bar is not None:
        bar.close()


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Download ``url`` to ``dest``, with multi-source fallback.

    Order of attempts:
        1. ``url`` (whatever the caller passed; default = Berkeley mirror)
        2. Hugging Face mirror (``LLFF_HF_URL``)
        3. Google Drive (``LLFF_GDRIVE_ID``) via ``gdown``

    Anything < 200MB is treated as a sentinel "this isn't the real ZIP"
    and is removed before trying the next source.
    """
    try:
        import requests
    except ImportError as e:
        raise SystemExit(f"download requires `requests`: pip install requests ({e})")

    dest.parent.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, str]] = [("primary", url)]
    if url != LLFF_HF_URL:
        sources.append(("huggingface", LLFF_HF_URL))
    sources.append(("gdrive", "<gdown>"))

    last_err: Optional[Exception] = None
    for label, src in sources:
        if src == "<gdown>":
            print(f"[dl] trying {label}: GDrive id={LLFF_GDRIVE_ID}")
            try:
                _gdown_download(LLFF_GDRIVE_ID, dest)
                return
            except SystemExit as e:           # gdown reraises as SystemExit
                last_err = e
                if dest.exists() and dest.stat().st_size < 200 * (1 << 20):
                    dest.unlink(missing_ok=True)
                continue
            except Exception as e:
                last_err = e
                continue

        print(f"[dl] trying {label}: {src} -> {dest}")
        try:
            _http_stream(src, dest)
            # sanity check: real LLFF zip is ~1.7-3.7 GB.
            sz = dest.stat().st_size
            if sz < 200 * (1 << 20):
                print(f"[dl] {label} returned only {sz/1e6:.1f} MB - probably "
                      f"an HTML error page, trying next source")
                dest.unlink(missing_ok=True)
                continue
            return
        except requests.HTTPError as e:
            print(f"[dl] {label} HTTP {e.response.status_code if e.response else '?'}, trying next")
            last_err = e
            if dest.exists():
                dest.unlink(missing_ok=True)
            continue
        except Exception as e:
            print(f"[dl] {label} failed: {e!r}, trying next")
            last_err = e
            if dest.exists():
                dest.unlink(missing_ok=True)
            continue

    raise SystemExit(f"all download sources failed; last error: {last_err!r}")


def _extract_scenes(zip_path: Path, out_dir: Path,
                    scenes: Optional[List[str]] = None) -> None:
    """Extract the listed scenes from the LLFF ZIP into ``out_dir``.

    Different mirrors wrap the data with different top-level prefixes
    (``nerf_llff_data/<scene>/...``, ``LLFF/<scene>/...``, just
    ``<scene>/...``, etc.). We detect the prefix by looking for any
    ``poses_bounds.npy`` entry and stripping everything above the scene
    folder. The output is always normalised to ``out_dir/<scene>/...``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Auto-detect the prefix by finding any ``.../<scene>/poses_bounds.npy``
        # entry and treating everything above ``<scene>/`` as the prefix.
        # This is robust to arbitrary wrappers (HF re-hosts often wrap
        # the data inside an extra folder).
        top = ""
        for n in names:
            if n.endswith("/poses_bounds.npy") or n.endswith("\\poses_bounds.npy"):
                # canonicalise separators
                norm = n.replace("\\", "/")
                parts = norm.split("/")
                # parts[-1] = poses_bounds.npy, parts[-2] = scene name
                if len(parts) >= 2 and parts[-2] in LLFF_SCENES:
                    top = "/".join(parts[:-2])
                    if top:
                        top += "/"
                    break
            elif n == "poses_bounds.npy":
                # zip is rooted directly at one scene -- unusual but possible
                top = ""
                break

        if top:
            print(f"        detected zip prefix: {top!r}")
        else:
            print(f"        no prefix detected (zip is rooted at scenes)")

        wanted = set(scenes) if scenes else None
        if wanted is not None:
            print(f"        scenes filter: {sorted(wanted)}")

        n_written = 0
        for name in names:
            norm = name.replace("\\", "/")
            if top and not norm.startswith(top):
                continue
            rel = norm[len(top):] if top else norm
            if not rel:
                continue
            scene = rel.split("/", 1)[0]
            # Skip anything that isn't actually one of the LLFF scenes
            # (handles the HF re-host that bundles unrelated checkpoints).
            if scene not in LLFF_SCENES:
                continue
            if wanted is not None and scene not in wanted:
                continue

            dest = out_dir / rel
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            n_written += 1
    print(f"[unzip] done ({n_written} files)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None,
                    help=f"single scene to extract; one of: {LLFF_SCENES}")
    ap.add_argument("--all", action="store_true",
                    help="extract all 8 scenes")
    ap.add_argument("--partial", action="store_true",
                    help="delete the ZIP after extraction (saves ~3.7 GB "
                         "but re-downloads on the next run)")
    ap.add_argument("--out", default=str(_ROOT / "data" / "nerf_llff_data"),
                    help="output directory (default: data/nerf_llff_data)")
    ap.add_argument("--cache", default=str(_ROOT / "data" / "_cache"),
                    help="ZIP cache directory")
    ap.add_argument("--url", default=LLFF_ZIP_URL,
                    help="override the ZIP URL")
    args = ap.parse_args()

    if not args.all and args.scene is None:
        ap.error("either --scene <name> or --all is required")
    if args.scene is not None and args.scene not in LLFF_SCENES:
        ap.error(f"unknown scene: {args.scene}; valid: {LLFF_SCENES}")

    out_dir   = Path(args.out)
    cache_dir = Path(args.cache)
    zip_path  = cache_dir / "nerf_llff_data.zip"

    # Skip if the target scene(s) already exist.
    scenes_to_extract: Optional[List[str]] = None if args.all else [args.scene]
    if scenes_to_extract is not None:
        already = [s for s in scenes_to_extract
                   if (out_dir / s / "poses_bounds.npy").is_file()]
        if already == scenes_to_extract:
            print(f"[skip] scene(s) already extracted: {already}")
            return

    # Download (or reuse cache).
    if not zip_path.is_file():
        _stream_download(args.url, zip_path)
    else:
        print(f"[cache] reusing {zip_path}")

    _extract_scenes(zip_path, out_dir, scenes_to_extract)

    if args.partial:
        print(f"[clean] deleting {zip_path}")
        zip_path.unlink()


if __name__ == "__main__":
    main()
