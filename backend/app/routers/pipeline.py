import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..database import db_session
from ..models import PipelineStep, ExtractSettings, SfmSettings, TrainSettings, RefineSettings
from ..pipeline.task_runner import task_runner
from ..services import ffmpeg as ffmpeg_svc
from ..services import colmap as colmap_svc
from ..services import trainer as trainer_svc
from ..services.denoise import denoise_point_cloud
from ..services.disk import check_disk_space, SPACE_REQUIREMENTS_MB
from ..services.frame_quality import filter_frames
from ..services.sfm_quality import evaluate_sfm_quality, export_sparse_ply
from ..ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects/{project_id}/pipeline", tags=["pipeline"])


def _project_dir(pid: str) -> Path:
    return settings.data_dir / pid


# Pipeline step prerequisites: action → set of valid current steps
# Allows redo from any later step (e.g., re-extract frames from sfm_ready)
_STEP_PREREQUISITES: dict[str, set[PipelineStep]] = {
    "extract-frames": {
        PipelineStep.CREATED, PipelineStep.FRAMES_READY,
        PipelineStep.SFM_READY, PipelineStep.TRAINING_COMPLETE,
        PipelineStep.FAILED,
    },
    "sfm": {
        PipelineStep.FRAMES_READY, PipelineStep.SFM_READY,
        PipelineStep.TRAINING_COMPLETE, PipelineStep.FAILED,
    },
    "train": {
        PipelineStep.SFM_READY, PipelineStep.TRAINING_COMPLETE,
        PipelineStep.FAILED,
    },
    "extract-mesh": {PipelineStep.TRAINING_COMPLETE},
    "refine": {PipelineStep.TRAINING_COMPLETE},
}


async def _check_precondition(project_id: str, action: str) -> dict:
    """Validate project exists, is not busy, and meets step prerequisites."""
    async with db_session() as db:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Project not found")

    # Reject if a task is already running for this project
    if task_runner.is_running(project_id):
        raise HTTPException(409, "A pipeline task is already running for this project")

    current = PipelineStep(row["step"])
    allowed = _STEP_PREREQUISITES.get(action, set())
    if current not in allowed:
        raise HTTPException(
            422,
            f"Cannot run '{action}' — project is in step '{current.value}'. "
            f"Expected one of: {', '.join(s.value for s in allowed)}",
        )

    return row


async def _update_step(pid: str, step: PipelineStep, error: str | None = None):
    if error:
        logger.error("Pipeline step update: project=%s step=%s error=%s", pid, step.value, error)
    else:
        logger.info("Pipeline step update: project=%s step=%s", pid, step.value)
    async with db_session() as db:
        await db.execute(
            "UPDATE projects SET step = ?, error = ? WHERE id = ?",
            (step.value, error, pid),
        )


def _check_disk(project_id: str, action: str):
    """Raise HTTP 507 if insufficient disk space for the pipeline step."""
    required = SPACE_REQUIREMENTS_MB.get(action, 0)
    if required > 0:
        proj = _project_dir(project_id)
        ok, free_mb = check_disk_space(proj, required)
        if not ok:
            raise HTTPException(
                507,
                f"Insufficient disk space for '{action}': "
                f"{free_mb} MB free, {required} MB required",
            )


@router.post("/extract-frames")
async def extract_frames(project_id: str, body: ExtractSettings | None = None):
    body = body or ExtractSettings()
    row = await _check_precondition(project_id, "extract-frames")
    _check_disk(project_id, "extract-frames")

    proj = _project_dir(project_id)
    if not row["video_filename"]:
        raise HTTPException(400, "No video uploaded")

    frames_dir = proj / "frames"

    # Gather all videos for this project
    async with db_session() as db_v:
        cursor = await db_v.execute(
            "SELECT video_index, filename, video_type FROM project_videos "
            "WHERE project_id = ? ORDER BY video_index", (project_id,)
        )
        video_rows = await cursor.fetchall()

    # Fallback for legacy projects without project_videos rows
    if not video_rows:
        video_rows = [{"video_index": 0, "filename": row["video_filename"],
                       "video_type": row["video_type"] or "standard"}]

    multi_video = len(video_rows) > 1
    video_sources = []

    # Build extraction commands — one per video
    cmds = []
    for vr in video_rows:
        vid_idx = vr["video_index"]
        vid_file = vr["filename"]
        prefix = f"v{vid_idx}_" if multi_video else ""
        video_path = proj / "input" / vid_file
        cmds.append((prefix, ffmpeg_svc.build_extract_cmd(
            video_path, frames_dir, fps=body.fps,
            start_time=body.start_time, video_prefix=prefix,
        )))
        video_sources.append({
            "video_id": vid_idx, "filename": vid_file, "prefix": prefix,
        })

    async def run():
        try:
            await _update_step(project_id, PipelineStep.EXTRACTING_FRAMES)

            # Extract frames from each video sequentially
            for i, (prefix, cmd) in enumerate(cmds):
                substep = f"extracting video {i + 1}/{len(cmds)}" if multi_video else "extracting"
                if multi_video:
                    await manager.send_log(
                        project_id, f"[INFO] Extracting frames from video {i + 1}/{len(cmds)}: {video_sources[i]['filename']}"
                    )
                rc = await task_runner.run(
                    project_id, cmd, step="extract_frames", substep=substep,
                    line_parser=ffmpeg_svc.parse_ffmpeg_line,
                    timeout=600,
                )
                if rc != 0:
                    await _update_step(project_id, PipelineStep.FAILED,
                                       f"ffmpeg exited with code {rc} on video {i + 1}")
                    return

            # Apply blur filtering
            filter_blur = getattr(body, "filter_blur", True)
            if filter_blur:
                try:
                    result = filter_frames(
                        frames_dir,
                        min_blur_score=getattr(body, "min_blur_score", 50.0),
                    )
                    if result.rejected_frames > 0:
                        await manager.send_log(
                            project_id,
                            f"[INFO] Blur filter: rejected {result.rejected_frames}/"
                            f"{result.total_frames} frames "
                            f"(scores: min={result.min_score:.1f}, "
                            f"mean={result.mean_score:.1f}, max={result.max_score:.1f})",
                        )
                except Exception as e:
                    await manager.send_log(
                        project_id, f"[WARN] Blur filter skipped: {e}",
                    )

            frame_count = sum(1 for _ in frames_dir.glob("*.jpg"))
            # Generate frame manifest with video source tracking
            ffmpeg_svc.generate_frame_manifest(
                frames_dir, body.fps,
                video_sources=video_sources if multi_video else None,
            )

            async with db_session() as db2:
                await db2.execute(
                    "UPDATE projects SET step = ?, frame_count = ? WHERE id = ?",
                    (PipelineStep.FRAMES_READY.value, frame_count, project_id),
                )
        except Exception as e:
            logger.exception(f"Frame extraction failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/sfm")
async def run_sfm(project_id: str, body: SfmSettings | None = None):
    body = body or SfmSettings()
    await _check_precondition(project_id, "sfm")
    _check_disk(project_id, "sfm")
    proj = _project_dir(project_id)

    frames_dir = proj / "frames"
    colmap_dir = proj / "colmap"
    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    dense_dir = colmap_dir / "dense"

    # Multi-cam auto-detection: check how many videos this project has
    async with db_session() as db_mc:
        cursor = await db_mc.execute(
            "SELECT COUNT(*) FROM project_videos WHERE project_id = ?", (project_id,)
        )
        video_count = (await cursor.fetchone())[0]

    if video_count > 1:
        # Multiple cameras → disable single_camera, upgrade matcher
        body.single_camera = False
        if body.matcher_type == "sequential_matcher":
            body.matcher_type = "exhaustive_matcher"
            logger.info("Multi-video project %s: auto-switched to exhaustive_matcher", project_id)

    async def run():
        try:
            await _update_step(project_id, PipelineStep.RUNNING_SFM)

            if video_count > 1:
                await manager.send_log(
                    project_id,
                    f"[INFO] Multi-camera mode: {video_count} videos, "
                    f"single_camera=False, matcher={body.matcher_type}",
                )

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

            # Quality gate: validate SfM reconstruction
            quality = evaluate_sfm_quality(sparse_model, frames_dir)
            await manager.send_log(
                project_id,
                f"[INFO] SfM quality: {quality.registered_images}/{quality.total_frames} images "
                f"registered ({quality.registered_ratio:.0%}), "
                f"{quality.num_3d_points} 3D points, "
                f"reproj error={quality.mean_reprojection_error:.2f}px",
            )
            if quality.warnings:
                for warn in quality.warnings:
                    await manager.send_log(project_id, f"[WARN] {warn}")
            if not quality.passed:
                await _update_step(
                    project_id, PipelineStep.FAILED,
                    f"SfM quality too low: {quality.failure_reason}",
                )
                return

            # Store SfM quality metrics
            async with db_session() as db_q:
                await db_q.execute(
                    "UPDATE projects SET sfm_points = ?, sfm_registered_images = ?, "
                    "sfm_reprojection_error = ? WHERE id = ?",
                    (quality.num_3d_points, quality.registered_images,
                     quality.mean_reprojection_error, project_id),
                )

            # Export sparse point cloud as PLY for preview
            sparse_ply = colmap_dir / "sparse_preview.ply"
            try:
                export_sparse_ply(sparse_model, sparse_ply)
                await manager.send_log(project_id, f"[INFO] Sparse preview PLY exported")
            except Exception as e:
                await manager.send_log(project_id, f"[WARN] Sparse PLY export failed: {e}")

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

            # Optional: Dense reconstruction (patch_match_stereo + stereo_fusion)
            if getattr(body, "enable_dense", False):
                await manager.send_log(project_id, "[INFO] Running dense stereo matching...")
                cmd = colmap_svc.build_patch_match_stereo_cmd(dense_dir)
                rc = await task_runner.run(
                    project_id, cmd, step="sfm", substep="dense_stereo",
                    line_parser=colmap_svc.parse_colmap_line,
                    timeout=3600,  # 1 hour
                )
                if rc != 0:
                    await manager.send_log(project_id, "[WARN] Dense stereo failed, continuing with sparse")
                else:
                    fused_ply = dense_dir / "fused.ply"
                    cmd = colmap_svc.build_stereo_fusion_cmd(dense_dir, fused_ply)
                    rc = await task_runner.run(
                        project_id, cmd, step="sfm", substep="stereo_fusion",
                        line_parser=colmap_svc.parse_colmap_line,
                        timeout=1800,
                    )
                    if rc == 0:
                        await manager.send_log(project_id, f"[INFO] Dense fusion complete: {fused_ply}")

            await _update_step(project_id, PipelineStep.SFM_READY)
        except Exception as e:
            logger.exception(f"SfM failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/train")
async def train_splat(project_id: str, body: TrainSettings | None = None):
    body = body or TrainSettings()
    await _check_precondition(project_id, "train")
    _check_disk(project_id, "train")
    proj = _project_dir(project_id)

    data_dir = proj / "colmap" / "dense"
    result_dir = proj / "output"

    use_scaffold     = getattr(body, "use_scaffold", True)
    voxel_size       = getattr(body, "voxel_size", 0.001)
    denoise_strength = getattr(body, "denoise_strength", "off")

    # Apply point cloud denoising if requested
    points3d_path = data_dir / "sparse" / "points3D.bin"
    if denoise_strength != "off" and points3d_path.exists():
        denoised_path = data_dir / "sparse" / "points3D.bin.denoised"
        result = denoise_point_cloud(points3d_path, denoised_path, strength=denoise_strength)
        if result.skipped_reason:
            logger.warning(
                "Denoising skipped for project=%s: %s", project_id, result.skipped_reason,
            )
        elif result.success and denoised_path.exists():
            import shutil
            shutil.copy2(denoised_path, points3d_path)
            logger.info(
                "Denoising applied: %d → %d points",
                result.points_before, result.points_after,
            )

    enable_depth = getattr(body, "enable_depth", False)
    depth_weight = getattr(body, "depth_weight", 0.1)

    # Persist temporal mode so the view page knows what was trained
    temporal_mode_val = getattr(body, "temporal_mode", "static")
    async with db_session() as db_tm:
        await db_tm.execute(
            "UPDATE projects SET temporal_mode = ? WHERE id = ?",
            (temporal_mode_val, project_id),
        )

    # Bridge for depth estimation logs: callable from sync thread → async WS
    _loop = asyncio.get_event_loop()

    def _depth_log_sync(msg: str):
        asyncio.run_coroutine_threadsafe(manager.send_log(project_id, msg), _loop)

    async def run():
        try:
            await _update_step(project_id, PipelineStep.TRAINING)

            # Depth estimation runs inside the async task (not blocking the endpoint)
            depth_dir = None
            if enable_depth:
                depth_dir = proj / "depths"
                needs_estimation = not depth_dir.exists() or not any(depth_dir.glob("*.npy"))
                if needs_estimation:
                    await manager.send_log(project_id, "[INFO] Running depth estimation (first time may download model)...")
                    try:
                        from ..services.depth import estimate_depths
                        frames_dir = data_dir / "images"

                        def _run_depth():
                            return estimate_depths(
                                frames_dir, depth_dir, "cuda",
                                log_callback=_depth_log_sync,
                            )

                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, _run_depth)
                        await manager.send_log(project_id, "[INFO] Depth estimation complete")
                    except Exception as e:
                        await manager.send_log(project_id, f"[WARN] Depth estimation failed: {e} — continuing without")
                        depth_dir = None

            # Find frame manifest for 4D mode
            manifest_path = proj / "frames" / "manifest.json"

            cmd = trainer_svc.build_train_cmd(data_dir, result_dir, body.max_steps,
                                              use_scaffold=use_scaffold, voxel_size=voxel_size,
                                              resume=getattr(body, "resume", False),
                                              sh_degree=getattr(body, "sh_degree", 0),
                                              depth_dir=depth_dir,
                                              depth_weight=depth_weight,
                                              temporal_mode=getattr(body, "temporal_mode", "static"),
                                              temporal_smoothness=getattr(body, "temporal_smoothness", 0.01),
                                              manifest_path=manifest_path)
            rc = await task_runner.run(
                project_id, cmd, step="train", substep="training",
                line_parser=trainer_svc.parse_trainer_line,
                timeout=7200,  # 2 hours max for training
                requires_gpu=True,
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


@router.get("/health")
async def pipeline_health(project_id: str):
    """Check if a pipeline task is actually running for this project.

    The frontend polls this to detect silent failures — e.g. a subprocess
    crashed but the DB step was never updated from 'training'.
    """
    running = task_runner.is_running(project_id)
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT step, error FROM projects WHERE id = ?", (project_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Project not found")

    step = row["step"]
    # Stale = DB says an active step but no subprocess is alive
    active_steps = {
        PipelineStep.EXTRACTING_FRAMES.value,
        PipelineStep.RUNNING_SFM.value,
        PipelineStep.TRAINING.value,
    }
    stale = step in active_steps and not running

    # Auto-recover: mark stale projects as failed so the UI shows retry
    if stale:
        error_msg = "Process exited unexpectedly"
        await _update_step(project_id, PipelineStep.FAILED, error_msg)
        return {"running": False, "step": PipelineStep.FAILED.value, "stale": True, "error": error_msg}

    return {"running": running, "step": step, "stale": False, "error": row["error"]}


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
    await _check_precondition(project_id, "extract-mesh")
    proj = _project_dir(project_id)

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
                requires_gpu=True,
            )
            logger.info("Mesh extraction %s for project=%s", "done" if rc == 0 else "failed", project_id)
        except Exception as e:
            logger.exception("Mesh extraction failed for %s", project_id)

    asyncio.create_task(run())
    return {"status": "started", "mesh_url": f"/api/projects/{project_id}/mesh"}


@router.post("/refine")
async def refine_splat(project_id: str, body: RefineSettings | None = None):
    """
    Run multi-stage refinement: visibility transfer → confidence-aware training
    → (optional) diffusion inpainting → final training pass.

    Each stage runs as a separate subprocess — VRAM is fully freed between stages.
    """
    body = body or RefineSettings()
    row = await _check_precondition(project_id, "refine")
    proj = _project_dir(project_id)

    data_dir = proj / "colmap" / "dense"
    result_dir = proj / "output"
    ply_path = result_dir / "point_cloud.ply"
    ckpt_path = result_dir / "checkpoint.pt"
    visibility_dir = proj / "visibility"
    inpaint_dir = proj / "inpainted"

    checkpoint = ckpt_path if ckpt_path.exists() else ply_path
    if not ply_path.exists():
        raise HTTPException(400, "Training not complete — no .ply found")

    async def run():
        try:
            # Stage 1: Visibility transfer
            await manager.send_log(project_id, "[INFO] Stage 1: Cross-frame visibility transfer...")
            cmd = trainer_svc.build_visibility_transfer_cmd(
                data_dir, checkpoint, visibility_dir,
                alpha_low=body.alpha_low, alpha_high=body.alpha_high,
            )
            rc = await task_runner.run(
                project_id, cmd, step="refine", substep="visibility_transfer",
                line_parser=trainer_svc.parse_trainer_line,
                timeout=1800, requires_gpu=True,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Visibility transfer failed")
                return

            # Stage 2: Refinement training with pseudo-GT
            await manager.send_log(project_id, "[INFO] Stage 2: Confidence-aware refinement training...")
            cmd = trainer_svc.build_train_cmd(
                data_dir, result_dir, body.refine_steps,
                resume=True,
            )
            cmd += ["--pseudo_gt_dir", str(visibility_dir)]
            rc = await task_runner.run(
                project_id, cmd, step="refine", substep="refinement_training",
                line_parser=trainer_svc.parse_trainer_line,
                timeout=3600, requires_gpu=True,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Refinement training failed")
                return

            # Stage 3 (optional): Diffusion inpainting
            if body.diffusion_inpaint:
                await manager.send_log(project_id, "[INFO] Stage 3: Diffusion inpainting for unseen regions...")
                refined_ckpt = ckpt_path if ckpt_path.exists() else ply_path
                cmd = trainer_svc.build_inpaint_cmd(
                    data_dir, refined_ckpt, inpaint_dir,
                    num_novel_views=body.num_novel_views,
                    steps=body.diffusion_steps,
                    guidance=body.diffusion_guidance,
                )
                rc = await task_runner.run(
                    project_id, cmd, step="refine", substep="diffusion_inpaint",
                    line_parser=trainer_svc.parse_trainer_line,
                    timeout=3600, requires_gpu=True,
                )
                if rc != 0:
                    await manager.send_log(project_id, "[WARN] Inpainting failed — skipping final pass")
                else:
                    # Stage 4: Final training with inpainted views
                    await manager.send_log(project_id, "[INFO] Stage 4: Final training with inpainted views...")
                    cmd = trainer_svc.build_train_cmd(
                        data_dir, result_dir, body.refine_steps,
                        resume=True,
                    )
                    cmd += ["--novel_views_dir", str(inpaint_dir),
                            "--novel_view_weight", str(body.novel_view_weight)]
                    rc = await task_runner.run(
                        project_id, cmd, step="refine", substep="final_training",
                        line_parser=trainer_svc.parse_trainer_line,
                        timeout=3600, requires_gpu=True,
                    )

            await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
            await manager.send_log(project_id, "[INFO] Refinement complete!")
        except Exception as e:
            logger.exception(f"Refinement failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}
