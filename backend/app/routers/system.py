import json
import logging
from collections import deque

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from ..config import settings
from ..services.deps import (
    get_system_status,
    install_ffmpeg_stream,
    install_colmap_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
async def system_status():
    logger.info("System status check requested")
    status = await get_system_status()
    logger.info(
        "System status: CUDA=%s, FFmpeg=%s, COLMAP=%s",
        status.cuda_available,
        status.ffmpeg.installed,
        status.colmap.installed,
    )
    return status


@router.post("/install/{dep}")
async def install_dependency(dep: str):
    logger.info("Dependency install requested: %s", dep)
    if dep == "ffmpeg":
        gen = install_ffmpeg_stream()
    elif dep == "colmap":
        gen = install_colmap_stream()
    else:
        raise HTTPException(400, f"Unknown dependency: {dep}")

    async def sse_stream():
        try:
            async for event_data in gen:
                yield f"data: {event_data}\n\n"
        except Exception as e:
            logger.exception("Dependency install failed: %s", dep)
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/logs")
async def get_logs(
    lines: int = Query(default=100, ge=1, le=500),
):
    """Return the last N lines of the application log file."""
    log_path = settings.log_file
    if not log_path.exists():
        return {"lines": [], "total_lines": 0, "file": str(log_path)}

    try:
        # Read efficiently: only keep the last N lines in memory
        result = deque(maxlen=lines)
        total = 0
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                result.append(line.rstrip("\n"))
                total += 1
        return {
            "lines": list(result),
            "total_lines": total,
            "file": str(log_path),
        }
    except Exception as e:
        logger.exception("Failed to read log file")
        raise HTTPException(500, f"Failed to read log file: {e}")


@router.get("/logs/download")
async def download_logs():
    """Download the full log file."""
    log_path = settings.log_file
    if not log_path.exists():
        raise HTTPException(404, "Log file not found")

    return FileResponse(
        path=str(log_path),
        filename="gaussiansplat.log",
        media_type="text/plain",
    )
