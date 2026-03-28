import asyncio
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import httpx

from ..config import settings
from ..models import SystemDepStatus, SystemStatus

logger = logging.getLogger(__name__)


# Type for progress callbacks: (downloaded_bytes, total_bytes, phase_label)
ProgressCallback = Callable[[int, int, str], None]


async def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return proc.returncode or 0, out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return -1, str(e)


async def check_ffmpeg() -> SystemDepStatus:
    # Use the same resolution logic as runtime (config.py)
    resolved = str(settings.ffmpeg_bin)
    candidates = [resolved]
    if resolved != "ffmpeg":
        candidates.append("ffmpeg")

    for c in candidates:
        rc, out = await _run([c, "-version"])
        if rc == 0:
            version = out.split("\n")[0] if out else "unknown"
            return SystemDepStatus(name="ffmpeg", installed=True, version=version, path=c)

    # File exists but can't run (DLL issues, etc.) — still report as detected
    if Path(resolved).exists():
        return SystemDepStatus(name="ffmpeg", installed=True, version="detected (path)", path=resolved)

    return SystemDepStatus(name="ffmpeg", installed=False, error="ffmpeg not found")


async def check_colmap() -> SystemDepStatus:
    # Use the same resolution logic as runtime (config.py)
    resolved = str(settings.colmap_bin)
    candidates = [resolved]
    if resolved != "colmap":
        candidates.append("colmap")

    for c in candidates:
        # Try several probes — different COLMAP builds respond differently
        for probe in ([c, "-h"], [c, "help"], [c]):
            rc, out = await _run(probe)
            if rc == 0 or "colmap" in out.lower():
                version = "detected"
                for line in out.split("\n"):
                    if "version" in line.lower() or "COLMAP" in line:
                        version = line.strip()
                        break
                return SystemDepStatus(name="colmap", installed=True, version=version, path=c)

    # File exists but can't run (DLL issues, etc.)
    if Path(resolved).exists():
        return SystemDepStatus(name="colmap", installed=True, version="detected (path)", path=resolved)

    return SystemDepStatus(name="colmap", installed=False, error="COLMAP not found")


async def check_cuda() -> tuple[bool, Optional[str], Optional[str], Optional[int]]:
    rc, out = await _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0:
        return False, None, None, None

    parts = out.split("\n")[0].split(",")
    if len(parts) >= 3:
        gpu_name = parts[0].strip()
        vram = int(float(parts[1].strip()))
        driver = parts[2].strip()
        return True, driver, gpu_name, vram

    return True, None, None, None


async def check_python_deps() -> SystemDepStatus:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        ver = torch.__version__
        if cuda_ok:
            return SystemDepStatus(name="python_deps", installed=True, version=f"PyTorch {ver} (CUDA)")
        return SystemDepStatus(name="python_deps", installed=True, version=f"PyTorch {ver} (CPU only)", error="CUDA not available in PyTorch")
    except ImportError:
        return SystemDepStatus(name="python_deps", installed=False, error="PyTorch not installed")


async def get_system_status() -> SystemStatus:
    logger.debug("Checking system dependencies...")
    ffmpeg_status, colmap_status, python_status = await asyncio.gather(
        check_ffmpeg(), check_colmap(), check_python_deps()
    )
    cuda_available, cuda_version, gpu_name, gpu_vram = await check_cuda()

    # Determine actual PyTorch CUDA usability
    torch_cuda = False
    torch_cuda_ver = None
    if python_status.installed and python_status.version:
        torch_cuda = "(CUDA)" in python_status.version
        if torch_cuda:
            try:
                import torch
                torch_cuda_ver = getattr(torch.version, "cuda", None)
            except Exception:
                pass

    logger.debug(
        "Dependency check complete: ffmpeg=%s colmap=%s python_deps=%s driver_cuda=%s torch_cuda=%s",
        ffmpeg_status.installed, colmap_status.installed,
        python_status.installed, cuda_available, torch_cuda,
    )

    return SystemStatus(
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        torch_cuda_available=torch_cuda,
        torch_cuda_version=torch_cuda_ver,
        ffmpeg=ffmpeg_status,
        colmap=colmap_status,
        python_deps=python_status,
    )


async def _stream_download(url: str, dest_path: Path, label: str) -> AsyncGenerator[str, None]:
    """Stream-download a file to disk, yielding SSE progress events."""
    timeout = httpx.Timeout(connect=30, read=30, write=30, pool=30)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded * 100 / total) if total > 0 else 0
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    yield json.dumps({
                        "phase": "downloading",
                        "label": label,
                        "percent": pct,
                        "downloaded_mb": round(mb_done, 1),
                        "total_mb": round(mb_total, 1),
                    })


async def install_ffmpeg_stream() -> AsyncGenerator[str, None]:
    """Install FFmpeg with streamed progress events (SSE)."""
    if not settings.is_windows:
        yield json.dumps({"phase": "error", "message": "On Linux, install ffmpeg via: sudo apt install ffmpeg"})
        return

    dest = settings.tools_dir / "ffmpeg"
    dest.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    zip_path = dest / "ffmpeg.zip"

    async for event in _stream_download(url, zip_path, "FFmpeg"):
        yield event

    yield json.dumps({"phase": "extracting", "label": "FFmpeg"})
    await asyncio.sleep(0)  # yield control

    # Extract in a thread to avoid blocking
    def _extract():
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith("ffmpeg.exe") or member.endswith("ffprobe.exe"):
                    data = zf.read(member)
                    (dest / Path(member).name).write_bytes(data)
        zip_path.unlink()

    await asyncio.to_thread(_extract)

    status = await check_ffmpeg()
    yield json.dumps({
        "phase": "complete",
        "label": "FFmpeg",
        "installed": status.installed,
        "version": status.version,
        "error": status.error,
    })


async def install_colmap_stream() -> AsyncGenerator[str, None]:
    """Install COLMAP with streamed progress events (SSE)."""
    if not settings.is_windows:
        yield json.dumps({"phase": "error", "message": "On Linux, install COLMAP via: sudo apt install colmap"})
        return

    dest = settings.tools_dir / "colmap"
    dest.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/colmap/colmap/releases/download/4.0.2/colmap-x64-windows-cuda.zip"
    zip_path = dest / "colmap.zip"

    async for event in _stream_download(url, zip_path, "COLMAP"):
        yield event

    yield json.dumps({"phase": "extracting", "label": "COLMAP"})
    await asyncio.sleep(0)

    def _extract():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()
        # Flatten nested dir
        subdirs = [d for d in dest.iterdir() if d.is_dir() and ("COLMAP" in d.name or "colmap" in d.name)]
        if subdirs:
            for item in subdirs[0].iterdir():
                target = dest / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(subdirs[0], ignore_errors=True)

    await asyncio.to_thread(_extract)

    status = await check_colmap()
    yield json.dumps({
        "phase": "complete",
        "label": "COLMAP",
        "installed": status.installed,
        "version": status.version,
        "error": status.error,
    })


# Keep simple versions for CLI/non-streaming use
async def install_ffmpeg() -> SystemDepStatus:
    if not settings.is_windows:
        return SystemDepStatus(name="ffmpeg", installed=False, error="On Linux, install ffmpeg via: sudo apt install ffmpeg")
    async for _ in install_ffmpeg_stream():
        pass
    return await check_ffmpeg()


async def install_colmap() -> SystemDepStatus:
    if not settings.is_windows:
        return SystemDepStatus(name="colmap", installed=False, error="On Linux, install COLMAP via: sudo apt install colmap")
    async for _ in install_colmap_stream():
        pass
    return await check_colmap()
