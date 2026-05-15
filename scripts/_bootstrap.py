"""Common bootstrap for scripts.

This handles three Windows-specific pain points so that the rest of the
project doesn't have to:

1. Add the project root to ``sys.path`` so ``import sparse_gs`` works
   regardless of cwd.
2. On Windows, prepend the user-site ``Scripts/`` (where ``pip --user``
   puts ``ninja.exe``) to ``PATH``. Without this, ``torch.utils.cpp_extension``
   refuses to JIT-compile gsplat's CUDA kernels.
3. On Windows, activate the latest installed MSVC toolchain (vcvars64) so
   that ``cl.exe`` / ``link.exe`` and the matching INCLUDE / LIB env vars
   are available. PyTorch's auto-detection only knows about VS 2019/2022;
   newer builds (e.g. VS 18 preview) are missed.

Set ``SPARSE_GS_SKIP_BOOTSTRAP=1`` to disable any of this (e.g. on Linux
or if you've already activated a Developer Command Prompt yourself).
"""

from __future__ import annotations

import os
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. project root on sys.path
# ---------------------------------------------------------------------------
def _add_project_root() -> Path:
    here = Path(__file__).resolve().parent
    root = here.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# ---------------------------------------------------------------------------
# 2. ninja on PATH
# ---------------------------------------------------------------------------
def _ensure_ninja_on_path() -> None:
    if shutil.which("ninja") is not None:
        return
    candidates = []
    user_base = site.getuserbase()
    if user_base:
        candidates.append(Path(user_base) / "Scripts")
    if hasattr(site, "getsitepackages"):
        for sp in site.getsitepackages():
            candidates.append(Path(sp).parent / "Scripts")
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            py_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
            candidates.append(Path(appdata) / "Python" / py_tag / "Scripts")

    seen: set = set()
    for d in candidates:
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        ninja_exe = d / ("ninja.exe" if os.name == "nt" else "ninja")
        if ninja_exe.is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return


# ---------------------------------------------------------------------------
# 3. activate MSVC (vcvars64.bat) on Windows
# ---------------------------------------------------------------------------
_MSVC_KEYS_TO_KEEP = {
    "PATH", "INCLUDE", "LIB", "LIBPATH",
    "VCINSTALLDIR", "VCToolsInstallDir", "VCToolsVersion",
    "VSINSTALLDIR", "WindowsSdkDir", "WindowsSdkBinPath",
    "WindowsSdkVerBinPath", "WindowsSDKLibVersion",
    "WindowsSDKVersion", "UCRTVersion", "UniversalCRTSdkDir",
    "DevEnvDir",
}


def _find_vcvars64() -> Path | None:
    """Return path to ``vcvars64.bat`` of the latest installed VS, or None."""
    pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = pf86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            out = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest", "-prerelease", "-products", "*",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath",
                ],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode(errors="ignore").strip().splitlines()
            for line in out:
                p = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if p.is_file():
                    return p
        except Exception:                                                  # noqa: BLE001
            pass

    # Fallback: glob common roots.
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ]
    for r in roots:
        if not r.is_dir():
            continue
        # Try both modern (\18\Community\...) and 2019/2022 (\2022\Community\...) layouts.
        for vcvars in r.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"):
            if vcvars.is_file():
                return vcvars
    return None


def _ensure_msvc_env() -> None:
    if os.name != "nt":
        return
    # If user already has a developer prompt active, don't overwrite.
    if shutil.which("cl") is not None:
        return
    vcvars = _find_vcvars64()
    if vcvars is None:
        return  # nothing we can do; gsplat JIT will fail with a clear error
    try:
        # Run vcvars64 then dump env. Use ``set`` to print all env vars.
        cmd = f'cmd /s /c ""{vcvars}" >nul 2>&1 && set"'
        out = subprocess.check_output(cmd, shell=False, timeout=60).decode(
            "mbcs" if os.name == "nt" else "utf-8", errors="ignore"
        )
    except Exception:                                                      # noqa: BLE001
        return

    for line in out.splitlines():
        m = re.match(r"^([^=]+)=(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k.upper() == "PATH":
            # Merge new PATH with existing (new entries first).
            cur = os.environ.get("PATH", "")
            os.environ["PATH"] = v + (os.pathsep + cur if cur else "")
        elif k in _MSVC_KEYS_TO_KEEP:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Run on import (unless explicitly skipped).
# ---------------------------------------------------------------------------
PROJECT_ROOT = _add_project_root()

def _allow_unsupported_msvc() -> None:
    """Tell nvcc to accept whichever MSVC it finds, and to use the standard
    C++ preprocessor.

    1. CUDA 13.1's host_config.h only whitelists MSVC 14.2x/14.3x (VS
       2019/2022). On systems with VS 17.11+ / VS 18 (MSVC 14.4x/14.5x)
       compilation aborts with C1189 unless ``-allow-unsupported-compiler``
       is passed.
    2. PyTorch 2.10's ``c10/cuda/CUDACachingAllocator.h`` uses C++17
       attribute syntax that the *traditional* MSVC preprocessor (which
       nvcc invokes by default) mis-parses, producing
       ``error: invalid combination of type specifiers``. Forwarding
       ``/Zc:preprocessor`` to cl via ``-Xcompiler`` enables the standard
       conforming preprocessor and resolves it.

    Both flags are honoured by ``NVCC_PREPEND_FLAGS``, which nvcc reads on
    every invocation regardless of the build system on top.
    """
    if os.name != "nt":
        return
    needed = ["-allow-unsupported-compiler", "-Xcompiler", "/Zc:preprocessor"]
    cur = os.environ.get("NVCC_PREPEND_FLAGS", "")
    for tok in needed:
        if tok not in cur:
            cur = (cur + " " if cur else "") + tok
    os.environ["NVCC_PREPEND_FLAGS"] = cur


def _patch_torch_c10_header() -> None:
    """One-line patch for ``c10/cuda/CUDACachingAllocator.h`` on Windows.

    PyTorch 2.10 declares::

        StreamSegmentSize(cudaStream_t s, bool small, size_t sz)

    which collides with Windows' ``<rpcndr.h>`` that defines
    ``#define small char``. As soon as any other CUDA TU pulls in
    ``windows.h`` (gsplat does, via ``c10/util/Logging.h`` -> Windows
    runtime headers), the parameter name gets rewritten to ``bool char``
    and nvcc reports::

        invalid combination of type specifiers
        type name is not allowed

    The fix is to rename the parameter. We do it in place — exactly once —
    on first run, so users do not need to touch site-packages by hand.
    The patch is no-op if the header has already been fixed (e.g. by a
    later torch release).
    """
    if os.name != "nt":
        return
    try:
        import torch                                                       # noqa: PLC0415
        header = (
            Path(torch.__file__).parent / "include" / "c10" / "cuda" / "CUDACachingAllocator.h"
        )
    except Exception:                                                      # noqa: BLE001
        return
    if not header.is_file():
        return
    try:
        text = header.read_text(encoding="utf-8")
    except Exception:                                                      # noqa: BLE001
        return
    # Only the very specific signature; do not touch anything else.
    needle = "StreamSegmentSize(cudaStream_t s, bool small, size_t sz)"
    repl   = "StreamSegmentSize(cudaStream_t s, bool is_small, size_t sz)"
    init_needle = ": stream(s), is_small_pool(small), total_size(sz) {}"
    init_repl   = ": stream(s), is_small_pool(is_small), total_size(sz) {}"
    if needle not in text:
        return
    new_text = text.replace(needle, repl).replace(init_needle, init_repl)
    try:
        header.write_text(new_text, encoding="utf-8")
        print(f"[sparse_gs] patched torch header (rpcndr 'small' macro workaround): {header}")
    except PermissionError:
        # Read-only / no write permission: leave it; user will see the build error.
        pass


def _patch_jit_filter_gcc_only_flags() -> None:
    """Strip GCC-only cflags before they reach MSVC.

    gsplat's ``_backend.py`` passes ``extra_cflags=['-O3','-Wno-attributes']``
    unconditionally. ``-Wno-attributes`` is a GCC/clang flag — cl rejects
    it with ``D8021: invalid numeric argument``. We wrap
    ``torch.utils.cpp_extension._jit_compile`` to filter out such flags
    on Windows. This avoids modifying the gsplat package itself.
    """
    if os.name != "nt":
        return
    try:
        import torch.utils.cpp_extension as _cppext                        # noqa: PLC0415
    except Exception:                                                      # noqa: BLE001
        return
    if getattr(_cppext, "_sparse_gs_filtered", False):
        return

    _orig = _cppext._jit_compile

    def _filter(flags):
        if not flags:
            return flags
        bad_prefixes = ("-W", "-f")            # GCC/clang families
        whitelist = {"-Wall", "-Wextra"}        # cl understands these
        out = []
        for f in flags:
            if f in whitelist:
                out.append(f)
                continue
            if isinstance(f, str) and any(f.startswith(p) for p in bad_prefixes):
                continue
            out.append(f)
        return out

    def _wrapped(*args, **kwargs):
        # Positional layout (PyTorch 2.10):
        # 0:name 1:sources 2:extra_cflags 3:extra_cuda_cflags
        # 4:extra_sycl_cflags 5:extra_ldflags 6:extra_include_paths ...
        if len(args) >= 3:
            args = list(args)
            args[2] = _filter(args[2])
            args = tuple(args)
        if "extra_cflags" in kwargs:
            kwargs["extra_cflags"] = _filter(kwargs["extra_cflags"])
        return _orig(*args, **kwargs)

    _cppext._jit_compile = _wrapped
    _cppext._sparse_gs_filtered = True


if os.environ.get("SPARSE_GS_SKIP_BOOTSTRAP", "0") != "1":
    _ensure_ninja_on_path()
    _ensure_msvc_env()
    _allow_unsupported_msvc()
    _patch_torch_c10_header()
    _patch_jit_filter_gcc_only_flags()
