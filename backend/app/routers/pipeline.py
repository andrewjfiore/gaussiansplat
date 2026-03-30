import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..database import get_db
from ..models import PipelineStep, ExtractSettings, SfmSettings, TrainSettings
from ..pipeline.task_runner import task_runner
from ..services import ffmpeg as ffmpeg_svc
from ..services import colmap as colmap_svc
from ..services import trainer as trainer_svc
from ..services.denoise import denoise_point_cloud

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects/{project_id}/pipeline", tags=["pipeline"])


def _project_dir(pid: str) -> Path:
    return settings.data_dir / pid


async def _update_step(pid: str, step: PipelineStep, error: str | None = None):
    if error:
        logger.error("Pipeline step update: project=%s step=%s error=%s", pid, step.value, error)
    else:
        logger.info("Pipeline step update: project=%s step=%s", pid, step.value)
    db = await get_db()
    try:
        await db.execute(
            "UPDATE projects SET step = ?, error = ? WHERE id = ?",
            (step.value, error, pid),
        )
        await db.commit()
    finally:
        await db.close()


@router.post("/extract-frames")
async def extract_frames(project_id: str, body: ExtractSettings | None = None):
    body = body or ExtractSettings()
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    db = await get_db()
    try:
        cursor = await db.execute("SELECT video_filename FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        if not row or not row["video_filename"]:
            raise HTTPException(400, "No video uploaded")
    finally:
        await db.close()

    video_path = proj / "input" / row["video_filename"]
    frames_dir = proj / "frames"

    cmd = ffmpeg_svc.build_extract_cmd(video_path, frames_dir, fps=body.fps)

    async def run():
        try:
            await _update_step(project_id, PipelineStep.EXTRACTING_FRAMES)
            rc = await task_runner.run(
                project_id, cmd, step="extract_frames", substep="extracting",
                line_parser=ffmpeg_svc.parse_ffmpeg_line,
                timeout=600,  # 10 min max for frame extraction
            )
            if rc == 0:
                frame_count = sum(1 for _ in frames_dir.glob("*.jpg"))
                db2 = await get_db()
                try:
                    await db2.execute(
                        "UPDATE projects SET step = ?, frame_count = ? WHERE id = ?",
                        (PipelineStep.FRAMES_READY.value, frame_count, project_id),
                    )
                    await db2.commit()
                finally:
                    await db2.close()
            else:
                await _update_step(project_id, PipelineStep.FAILED, f"ffmpeg exited with code {rc}")
        except Exception as e:
            logger.exception(f"Frame extraction failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/sfm")
async def run_sfm(project_id: str, body: SfmSettings | None = None):
    body = body or SfmSettings()
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    frames_dir = proj / "frames"
    colmap_dir = proj / "colmap"
    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    dense_dir = colmap_dir / "dense"

    async def run():
        try:
            await _update_step(project_id, PipelineStep.RUNNING_SFM)

            # Step 1: Feature extraction
            cmd = colmap_svc.build_feature_extractor_cmd(db_path, frames_dir, body.single_camera)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="feature_extraction",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=1800,  # 30 min
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Feature extraction failed")
                return

            # Step 2: Matching
            cmd = colmap_svc.build_matcher_cmd(db_path, body.matcher_type)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="matching",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=3600,  # 1 hour
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Matching failed")
                return

            # Step 3: Mapping
            cmd = colmap_svc.build_mapper_cmd(db_path, frames_dir, sparse_dir)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="mapping",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=3600,  # 1 hour
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Mapping failed")
                return

            # Find the sparse model directory (usually sparse/0)
            sparse_model = sparse_dir / "0"
            if not sparse_model.exists():
                # Try to find any numbered subdirectory
                subdirs = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
                if subdirs:
                    sparse_model = subdirs[0]
                else:
                    await _update_step(project_id, PipelineStep.FAILED, "No sparse model found")
                    return

            # Step 4: Undistortion
            cmd = colmap_svc.build_undistorter_cmd(frames_dir, sparse_model, dense_dir)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="undistortion",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=1800,  # 30 min
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Undistortion failed")
                return

            await _update_step(project_id, PipelineStep.SFM_READY)
        except Exception as e:
            logger.exception(f"SfM failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/train")
async def train_splat(project_id: str, body: TrainSettings | None = None):
    body = body or TrainSettings()
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    data_dir = proj / "colmap" / "dense"
    result_dir = proj / "output"

    use_scaffold     = getattr(body, "use_scaffold", True)
    voxel_size       = getattr(body, "voxel_size", 0.001)
    denoise_strength = getattr(body, "denoise_strength", "off")

    # Apply point cloud denoising if requested
    points3d_path = data_dir / "sparse" / "points3D.bin"
    if denoise_strength != "off" and points3d_path.exists():
        denoised_path = data_dir / "sparse" / "points3D.bin.denoised"
        denoise_point_cloud(points3d_path, denoised_path, strength=denoise_strength)
        if denoised_path.exists():
            import shutil
            shutil.copy2(denoised_path, points3d_path)

    cmd = trainer_svc.build_train_cmd(data_dir, result_dir, body.max_steps,
                                      use_scaffold=use_scaffold, voxel_size=voxel_size)

    async def run():
        try:
            await _update_step(project_id, PipelineStep.TRAINING)
            rc = await task_runner.run(
                project_id, cmd, step="train", substep="training",
                line_parser=trainer_svc.parse_trainer_line,
                timeout=7200,  # 2 hours max for training
            )
            if rc == 0:
                await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
            else:
                await _update_step(project_id, PipelineStep.FAILED, f"Training failed (exit code {rc})")
        except Exception as e:
            logger.exception(f"Training failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/cancel")
async def cancel_pipeline(project_id: str):
    cancelled = await task_runner.cancel(project_id)
    if cancelled:
        await _update_step(project_id, PipelineStep.FAILED, "Cancelled by user")
    return {"cancelled": cancelled}


@router.post("/extract-mesh")
async def extract_mesh(project_id: str):
    """
    Kick off SuGaR mesh extraction in the background.
    The resulting .glb will be available at GET /api/projects/{id}/mesh.
    """
    import sys
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    ply_path = proj / "output" / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(400, "Training not complete — no .ply found")

    output_dir = proj / "output"
    glb_path = output_dir / "mesh.glb"

    async def run():
        try:
            sugar_script = Path(__file__).resolve().parents[3] / "scripts" / "extract_mesh.py"
            if not sugar_script.exists():
                # Fallback: use open3d ball-pivoting for a basic mesh
                sugar_script = None

            if sugar_script:
                cmd = [sys.executable, str(sugar_script),
                       "--ply", str(ply_path), "--output", str(glb_path)]
            else:
                # Simple fallback: convert PLY to GLB via open3d
                cmd = [sys.executable, "-c",
                       f"import open3d as o3d; m=o3d.io.read_point_cloud('{ply_path}'); "
                       f"mesh,_=m.compute_convex_hull(); o3d.io.write_triangle_mesh('{glb_path}', mesh)"]

            rc = await task_runner.run(
                project_id + "_mesh", cmd, step="mesh", substep="extract",
                timeout=3600,
            )
            logger.info("Mesh extraction %s for project=%s", "done" if rc == 0 else "failed", project_id)
        except Exception as e:
            logger.exception("Mesh extraction failed for %s", project_id)

    asyncio.create_task(run())
    return {"status": "started", "mesh_url": f"/api/projects/{project_id}/mesh"}
