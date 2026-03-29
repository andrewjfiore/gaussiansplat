#!/usr/bin/env python3
"""
Diagnostic script — prints a summary of the ML/pipeline environment.
Run: python scripts/check_cuda.py
"""
import importlib
import shutil
import subprocess
import sys


def _version(pkg: str) -> str:
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "installed (no __version__)")
    except ImportError:
        return "NOT INSTALLED"


def _cmd_available(cmd: str) -> str:
    return "yes  (" + shutil.which(cmd) + ")" if shutil.which(cmd) else "no"


def _cuda_info():
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            mem_gb = props.total_memory / (1024 ** 3)
            return (
                True,
                torch.version.cuda or "unknown",
                props.name,
                f"{mem_gb:.1f} GB",
            )
        return False, "n/a", "n/a", "n/a"
    except ImportError:
        return None, "n/a", "n/a", "n/a"


def main():
    print()
    print("=== GaussianSplat Studio — Environment Check ===")
    print()

    # Python
    print(f"  Python version : {sys.version.split()[0]}")

    # PyTorch
    torch_ver = _version("torch")
    print(f"  PyTorch version: {torch_ver}")

    # CUDA
    cuda_avail, cuda_ver, gpu_name, gpu_mem = _cuda_info()
    if cuda_avail is None:
        print("  CUDA available : no  (torch not installed)")
    elif cuda_avail:
        print(f"  CUDA available : yes")
        print(f"  CUDA version   : {cuda_ver}")
        print(f"  GPU name       : {gpu_name}")
        print(f"  GPU memory     : {gpu_mem}")
    else:
        print("  CUDA available : no  (CPU-only torch or no NVIDIA GPU)")

    # gsplat
    print(f"  gsplat version : {_version('gsplat')}")

    # System tools
    print(f"  COLMAP         : {_cmd_available('colmap')}")
    print(f"  ffmpeg         : {_cmd_available('ffmpeg')}")

    print()

    # Verdict
    issues = []
    if torch_ver == "NOT INSTALLED":
        issues.append("torch is not installed — run: pip install -r backend/requirements-cuda.txt")
    elif cuda_avail is False:
        issues.append("torch is CPU-only — re-install with CUDA wheels (see backend/requirements-cuda.txt)")
    if _version("gsplat") == "NOT INSTALLED":
        issues.append("gsplat is not installed — run: pip install gsplat")
    if not shutil.which("colmap"):
        issues.append("COLMAP not found — install: sudo apt install colmap  (Linux) or brew install colmap (Mac)")
    if not shutil.which("ffmpeg"):
        issues.append("ffmpeg not found — install: sudo apt install ffmpeg  (Linux) or brew install ffmpeg (Mac)")

    if issues:
        print("  Issues detected:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  All checks passed — ready to train!")

    print()


if __name__ == "__main__":
    main()
