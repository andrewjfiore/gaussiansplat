import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from ..config import settings
from ..database import db_session
from ..models import (
    ProjectCreate, ProjectSummary, ProjectDetail, PipelineStep,
    SAMPLE_VIDEOS, VideoInfo, PruneSettings,
)
from ..services.equirect import is_equirectangular
from ..services.compress import ensure_ksplat
from ..services.spz_export import ply_to_spz

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
    async with db_session() as db:
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

    async with db_session() as db:
        await db.execute(
            "INSERT INTO projects (id, name, step, created_at) VALUES (?, ?, ?, ?)",
            (project_id, body.name, PipelineStep.CREATED.value, now),
        )

    logger.info("Project created: id=%s name=%s", project_id, body.name)
    return ProjectSummary(id=project_id, name=body.name, step=PipelineStep.CREATED, created_at=now)


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str):
    async with db_session() as db:
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

        # Load video list from project_videos table
        vcursor = await db.execute(
            "SELECT video_index, filename, video_type FROM project_videos "
            "WHERE project_id = ? ORDER BY video_index", (project_id,)
        )
        vrows = await vcursor.fetchall()
        if vrows:
            videos = [VideoInfo(index=v["video_index"], filename=v["filename"],
                                video_type=v["video_type"] or "standard") for v in vrows]
        elif row["video_filename"]:
            # Legacy single-video project — synthesize a one-element list
            videos = [VideoInfo(index=0, filename=row["video_filename"],
                                video_type=row["video_type"] or "standard")]
        else:
            videos = []

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
            temporal_mode=row["temporal_mode"] or "static",
            videos=videos,
            video_count=max(len(videos), 1),
            mask_keywords=row["mask_keywords"],
            mask_count=row["mask_count"] or 0,
        )


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    import shutil
    async with db_session() as db:
        await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))

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

    now = datetime.now(timezone.utc).isoformat()
    async with db_session() as db:
        # Keep legacy video_filename updated (always points to latest upload)
        await db.execute(
            "UPDATE projects SET video_filename = ?, video_type = ? WHERE id = ?",
            (filename, video_type, project_id),
        )
        # Add to project_videos table (additive — supports multiple uploads)
        cursor = await db.execute(
            "SELECT COALESCE(MAX(video_index), -1) FROM project_videos WHERE project_id = ?",
            (project_id,),
        )
        max_idx = (await cursor.fetchone())[0]
        await db.execute(
            "INSERT INTO project_videos (project_id, video_index, filename, video_type, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, max_idx + 1, filename, video_type, now),
        )

    file_size = dest.stat().st_size
    logger.info("Video uploaded: project=%s file=%s size=%d bytes type=%s video_index=%d",
                project_id, filename, file_size, video_type, max_idx + 1)
    return {"filename": filename, "size": file_size, "video_type": video_type,
            "video_index": max_idx + 1}


@router.get("/{project_id}/videos")
async def list_videos(project_id: str):
    """List all uploaded videos for a project."""
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT video_index, filename, video_type FROM project_videos "
            "WHERE project_id = ? ORDER BY video_index", (project_id,)
        )
        rows = await cursor.fetchall()
        if rows:
            return [{"index": r["video_index"], "filename": r["filename"],
                     "video_type": r["video_type"]} for r in rows]
        # Fallback: legacy project with just video_filename
        cursor = await db.execute("SELECT video_filename, video_type FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if row and row["video_filename"]:
            return [{"index": 0, "filename": row["video_filename"],
                     "video_type": row["video_type"] or "standard"}]
        return []


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

    async with db_session() as db:
        await db.execute(
            "UPDATE projects SET video_filename = ? WHERE id = ?",
            (filename, project_id),
        )

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


@router.get("/{project_id}/sparse-ply")
async def get_sparse_ply(project_id: str):
    """Serve the SfM sparse point cloud preview PLY."""
    ply_path = _project_dir(project_id) / "colmap" / "sparse_preview.ply"
    if not ply_path.exists():
        raise HTTPException(404, "Sparse preview not available — run SfM first")
    return FileResponse(
        path=str(ply_path),
        filename=f"{project_id}_sparse.ply",
        media_type="application/x-ply",
    )


@router.get("/{project_id}/spz")
async def get_spz(project_id: str):
    """Serve SPZ compressed gaussian splat (10-20x smaller than PLY)."""
    output_dir = _project_dir(project_id) / "output"
    ply_path = output_dir / "point_cloud.ply"
    spz_path = output_dir / "point_cloud.spz"

    if not ply_path.exists():
        raise HTTPException(404, "PLY not found — complete training first")

    # Compress on demand
    if not spz_path.exists():
        try:
            ply_to_spz(ply_path, spz_path)
        except Exception as e:
            logger.warning("SPZ export failed: %s", e)
            raise HTTPException(500, f"SPZ export failed: {e}")

    return FileResponse(
        path=str(spz_path),
        filename=f"{project_id}.spz",
        media_type="application/octet-stream",
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


@router.get("/{project_id}/temporal-info")
async def get_temporal_info(project_id: str):
    """Return temporal frame info for 4D splats."""
    output_dir = _project_dir(project_id) / "output" / "temporal_frames"
    if not output_dir.exists():
        return {"available": False, "frame_count": 0}
    frames = sorted(output_dir.glob("frame_*.ply"))
    return {
        "available": len(frames) > 0,
        "frame_count": len(frames),
    }


@router.get("/{project_id}/masks")
async def list_masks(project_id: str):
    """List generated mask images for preview."""
    masks_dir = _project_dir(project_id) / "masks"
    if not masks_dir.exists():
        return []
    imgs = sorted(masks_dir.glob("*.png"))
    return [{"name": img.name, "url": f"/api/projects/{project_id}/masks/{img.name}"} for img in imgs]


@router.get("/{project_id}/masks/{filename}")
async def get_mask(project_id: str, filename: str):
    path = _project_dir(project_id) / "masks" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@router.post("/{project_id}/prune-preview")
async def prune_preview(project_id: str, body: PruneSettings | None = None):
    """Preview how many Gaussians would be pruned — fast, no file writes."""
    body = body or PruneSettings()
    ply_path = _project_dir(project_id) / "output" / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(404, "No PLY found")

    import numpy as np

    # Read PLY header + data
    with open(ply_path, "rb") as f:
        props = []
        n_verts = 0
        while True:
            line = f.readline().decode("utf-8", errors="replace")
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith("property float"):
                props.append(line.split()[-1])
            if "end_header" in line:
                break
        n_floats = len(props)
        data = np.frombuffer(f.read(n_verts * n_floats * 4), dtype=np.float32).reshape(n_verts, n_floats)

    prop_idx = {name: i for i, name in enumerate(props)}

    opacity = 1.0 / (1.0 + np.exp(-data[:, prop_idx["opacity"]]))
    scales = np.stack([np.exp(data[:, prop_idx[f"scale_{i}"]]) for i in range(3)], axis=1)
    max_scale = scales.max(axis=1)
    med_scale = float(np.median(max_scale))
    positions = data[:, :3]
    dists = np.linalg.norm(positions - positions.mean(axis=0), axis=1)

    keep_op = opacity >= body.min_opacity
    keep_sc = max_scale <= med_scale * body.max_scale_mult
    keep_pos = dists <= np.percentile(dists, body.position_percentile)
    keep = keep_op & keep_sc & keep_pos

    return {
        "total": int(n_verts),
        "kept": int(keep.sum()),
        "pruned": int(n_verts - keep.sum()),
        "pruned_pct": round((n_verts - keep.sum()) / n_verts * 100, 1),
        "by_opacity": int((~keep_op).sum()),
        "by_scale": int((~keep_sc).sum()),
        "by_position": int((~keep_pos).sum()),
        "median_scale": round(med_scale, 6),
        "file_size_mb": round(ply_path.stat().st_size / 1024 / 1024, 1),
        "estimated_output_mb": round(ply_path.stat().st_size / 1024 / 1024 * keep.sum() / n_verts, 1),
    }


@router.post("/{project_id}/prune")
async def prune_splat(project_id: str, body: PruneSettings | None = None):
    """Apply pruning and overwrite the PLY (backs up original)."""
    import subprocess, sys
    body = body or PruneSettings()
    proj = _project_dir(project_id)
    ply_path = proj / "output" / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(404, "No PLY found")

    # Backup original if not already backed up
    backup = proj / "output" / "point_cloud_original.ply"
    if not backup.exists():
        import shutil
        shutil.copy2(ply_path, backup)

    prune_script = Path(__file__).resolve().parents[2] / "scripts" / "prune_splat.py"
    cmd = [
        sys.executable, str(prune_script),
        "--input", str(backup),
        "--output", str(ply_path),
        "--min_opacity", str(body.min_opacity),
        "--max_scale_mult", str(body.max_scale_mult),
        "--position_percentile", str(body.position_percentile),
    ]
    if body.bbox:
        cmd += ["--bbox", body.bbox]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise HTTPException(500, f"Prune failed: {result.stderr[-200:]}")

    return {"status": "ok", "output": result.stdout}


@router.post("/{project_id}/prune-reset")
async def prune_reset(project_id: str):
    """Restore the original unpruned PLY."""
    proj = _project_dir(project_id)
    backup = proj / "output" / "point_cloud_original.ply"
    ply_path = proj / "output" / "point_cloud.ply"
    if not backup.exists():
        raise HTTPException(404, "No backup found — PLY was never pruned")
    import shutil
    shutil.copy2(backup, ply_path)
    return {"status": "ok", "restored": True}


@router.get("/{project_id}/checkpoints")
async def list_checkpoints(project_id: str):
    """List available training checkpoints for comparison."""
    output_dir = _project_dir(project_id) / "output"
    if not output_dir.exists():
        return []
    checkpoints = []
    ckpt = output_dir / "checkpoint.pt"
    if ckpt.exists():
        checkpoints.append({
            "name": "checkpoint.pt",
            "size": ckpt.stat().st_size,
            "path": f"/api/projects/{project_id}/output/checkpoint.pt",
        })
    ply = output_dir / "point_cloud.ply"
    if ply.exists():
        checkpoints.append({
            "name": "point_cloud.ply",
            "size": ply.stat().st_size,
            "path": f"/api/projects/{project_id}/ply",
        })
    return checkpoints
