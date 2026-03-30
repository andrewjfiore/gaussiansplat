import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from ..config import settings
from ..database import get_db
from ..models import (
    ProjectCreate, ProjectSummary, ProjectDetail, PipelineStep,
    SAMPLE_VIDEOS,
)
from ..services.equirect import is_equirectangular
from ..services.compress import ensure_ksplat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _project_dir(project_id: str) -> Path:
    return settings.data_dir / project_id


def _detect_video_type(video_path: Path) -> str:
    """Use ffprobe to get video dimensions and detect equirectangular format."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if is_equirectangular(w, h):
                    logger.info("Detected equirectangular video: %dx%d", w, h)
                    return "equirectangular"
    except Exception as e:
        logger.debug("ffprobe detection failed: %s", e)
    return "standard"


@router.get("", response_model=list[ProjectSummary])
async def list_projects():
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            thumb = None
            frames_dir = _project_dir(row["id"]) / "frames"
            if frames_dir.exists():
                imgs = sorted(frames_dir.glob("*.jpg"))
                if imgs:
                    thumb = f"/api/projects/{row['id']}/frames/{imgs[0].name}"
            results.append(ProjectSummary(
                id=row["id"], name=row["name"], step=row["step"],
                created_at=row["created_at"], error=row["error"], thumbnail=thumb,
            ))
        return results
    finally:
        await db.close()


@router.post("", response_model=ProjectSummary)
async def create_project(body: ProjectCreate):
    project_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    proj_dir = _project_dir(project_id)
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "input").mkdir(exist_ok=True)
    (proj_dir / "frames").mkdir(exist_ok=True)
    (proj_dir / "colmap").mkdir(exist_ok=True)
    (proj_dir / "output").mkdir(exist_ok=True)

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO projects (id, name, step, created_at) VALUES (?, ?, ?, ?)",
            (project_id, body.name, PipelineStep.CREATED.value, now),
        )
        await db.commit()
    finally:
        await db.close()

    logger.info("Project created: id=%s name=%s", project_id, body.name)
    return ProjectSummary(id=project_id, name=body.name, step=PipelineStep.CREATED, created_at=now)


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Project not found")

        frames_dir = _project_dir(project_id) / "frames"
        frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0

        output_dir = _project_dir(project_id) / "output"
        has_output = any(output_dir.rglob("*.ply")) if output_dir.exists() else False

        thumb = None
        if frame_count > 0:
            imgs = sorted(frames_dir.glob("*.jpg"))
            thumb = f"/api/projects/{project_id}/frames/{imgs[0].name}"

        return ProjectDetail(
            id=row["id"], name=row["name"], step=row["step"],
            created_at=row["created_at"], error=row["error"],
            video_filename=row["video_filename"],
            video_type=row["video_type"] or "standard",
            frame_count=frame_count,
            sfm_points=row["sfm_points"],
            training_iterations=row["training_iterations"],
            has_output=has_output,
            thumbnail=thumb,
        )
    finally:
        await db.close()


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    import shutil
    db = await get_db()
    try:
        await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
    finally:
        await db.close()

    proj_dir = _project_dir(project_id)
    if proj_dir.exists():
        shutil.rmtree(proj_dir, ignore_errors=True)
    logger.info("Project deleted: id=%s", project_id)
    return {"ok": True}


@router.post("/{project_id}/upload")
async def upload_video(project_id: str, file: UploadFile = File(...)):
    proj_dir = _project_dir(project_id)
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")

    # Sanitize filename — strip path separators and special chars
    raw_name = file.filename or "video.mp4"
    safe_name = raw_name.replace("\\", "/").split("/")[-1]  # strip path
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")
    filename = safe_name.strip() or "video.mp4"
    dest = proj_dir / "input" / filename
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    video_type = _detect_video_type(dest)

    db = await get_db()
    try:
        await db.execute(
            "UPDATE projects SET video_filename = ?, video_type = ? WHERE id = ?",
            (filename, video_type, project_id),
        )
        await db.commit()
    finally:
        await db.close()

    file_size = dest.stat().st_size
    logger.info("Video uploaded: project=%s file=%s size=%d bytes type=%s",
                project_id, filename, file_size, video_type)
    return {"filename": filename, "size": file_size, "video_type": video_type}


@router.post("/{project_id}/sample")
async def download_sample(project_id: str, sample_id: str = Form(...)):
    proj_dir = _project_dir(project_id)
    if not proj_dir.exists():
        raise HTTPException(404, "Project not found")

    sample = next((s for s in SAMPLE_VIDEOS if s.id == sample_id), None)
    if not sample:
        raise HTTPException(400, "Unknown sample")

    filename = f"{sample_id}.mp4"
    dest = proj_dir / "input" / filename

    timeout = httpx.Timeout(connect=30, read=30, write=30, pool=30)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream("GET", sample.url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                    f.write(chunk)

    db = await get_db()
    try:
        await db.execute(
            "UPDATE projects SET video_filename = ? WHERE id = ?",
            (filename, project_id),
        )
        await db.commit()
    finally:
        await db.close()

    return {"filename": filename, "size": dest.stat().st_size}


@router.get("/{project_id}/frames")
async def list_frames(project_id: str):
    frames_dir = _project_dir(project_id) / "frames"
    if not frames_dir.exists():
        return []
    imgs = sorted(frames_dir.glob("*.jpg"))
    return [{"name": img.name, "url": f"/api/projects/{project_id}/frames/{img.name}"} for img in imgs]


@router.get("/{project_id}/frames/{filename}")
async def get_frame(project_id: str, filename: str):
    path = _project_dir(project_id) / "frames" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{project_id}/output/{path:path}")
async def get_output(project_id: str, path: str):
    file_path = _project_dir(project_id) / "output" / path
    if not file_path.exists():
        raise HTTPException(404)
    media = "application/octet-stream"
    if file_path.suffix == ".ply":
        media = "application/x-ply"
    return FileResponse(file_path, media_type=media)


@router.get("/samples/list")
async def list_samples():
    return SAMPLE_VIDEOS


@router.get("/{project_id}/ply")
async def get_ply(project_id: str):
    """Download the raw .ply splat file."""
    output_dir = _project_dir(project_id) / "output"
    ply_path = output_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(404, "PLY not found — complete training first")
    return FileResponse(
        path=str(ply_path),
        filename=f"{project_id}.ply",
        media_type="application/x-ply",
    )


@router.get("/{project_id}/splat")
async def get_splat(project_id: str):
    """
    Serve compressed .ksplat if available, otherwise fall back to .ply.
    Compresses on demand if ksplat-encoder is installed.
    """
    output_dir = _project_dir(project_id) / "output"
    ply_path = output_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(404, "Splat not found — complete training first")

    served = ensure_ksplat(ply_path)
    suffix  = served.suffix  # .ksplat or .ply
    media   = "application/octet-stream"
    fname   = f"{project_id}{suffix}"
    return FileResponse(path=str(served), filename=fname, media_type=media)


@router.get("/{project_id}/mesh")
async def get_mesh(project_id: str):
    """Download the exported .glb mesh (if mesh extraction has completed)."""
    output_dir = _project_dir(project_id) / "output"
    glb_path = output_dir / "mesh.glb"
    if not glb_path.exists():
        raise HTTPException(404, "Mesh not ready — trigger extraction first")
    return FileResponse(
        path=str(glb_path),
        filename=f"{project_id}.glb",
        media_type="model/gltf-binary",
    )
