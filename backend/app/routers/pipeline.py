import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..database import db_session, get_db
from ..models import (
    PipelineStep, ExtractSettings, SfmSettings, TrainSettings,
    RefineSettings, MaskSettings, MaskPreviewRequest, NovelViewSettings,
    CleanupSettings, SceneConfigResponse, SceneStatsResponse,
    PortraitSettings,
)
from ..pipeline.task_runner import task_runner
from ..services import ffmpeg as ffmpeg_svc
from ..services import colmap as colmap_svc
from ..services import trainer as trainer_svc
from ..services.denoise import denoise_point_cloud
from ..services.disk import check_disk_space, SPACE_REQUIREMENTS_MB
from ..services.frame_quality import filter_frames, select_sharpest_per_bucket
from ..services import masking as mask_svc
from ..services.sfm_quality import evaluate_sfm_quality, export_sparse_ply
from ..services import lod as lod_svc
from ..services import coverage as coverage_svc
from ..services import portrait as portrait_svc
from ..services.scene_analysis import analyze_scene
from ..services.cleanup import run_cleanup
from ..ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects/{project_id}/pipeline", tags=["pipeline"])


def _project_dir(pid: str) -> Path:
    return settings.data_dir / pid


# Pipeline step prerequisites: action -> set of valid current steps
# Allows redo from any later step (e.g., re-extract frames from sfm_ready)
_STEP_PREREQUISITES: dict[str, set[PipelineStep]] = {
    "extract-frames": {
        PipelineStep.CREATED, PipelineStep.FRAMES_READY,
        PipelineStep.MASKS_READY, PipelineStep.SFM_READY,
        PipelineStep.TRAINING_COMPLETE, PipelineStep.FAILED,
    },
    "mask": {
        PipelineStep.FRAMES_READY, PipelineStep.MASKS_READY,
        PipelineStep.SFM_READY, PipelineStep.TRAINING_COMPLETE,
        PipelineStep.FAILED,
    },
    "sfm": {
        PipelineStep.FRAMES_READY, PipelineStep.MASKS_READY,
        PipelineStep.SFM_READY, PipelineStep.TRAINING_COMPLETE,
        PipelineStep.FAILED,
    },
    "train": {
        PipelineStep.SFM_READY, PipelineStep.TRAINING_COMPLETE,
        PipelineStep.FAILED,
    },
    "extract-mesh": {PipelineStep.TRAINING_COMPLETE},
    "refine": {PipelineStep.TRAINING_COMPLETE},
    "portrait": {
        PipelineStep.CREATED, PipelineStep.FRAMES_READY,
        PipelineStep.TRAINING_COMPLETE, PipelineStep.FAILED,
    },
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
            f"Cannot run '{action}' -- project is in step '{current.value}'. "
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

    # Sharp-frame mode: sample more densely, then keep the sharpest frame
    # in each neighborhood bucket.
    sharp_mode = bool(getattr(body, "sharp_frame_selection", False))
    sharp_window = max(0, int(getattr(body, "sharp_window", 5)))
    bucket_size = (2 * sharp_window) + 1
    sample_fps = body.fps * bucket_size if sharp_mode else body.fps

    # Build extraction commands -- one per video
    cmds = []
    for vr in video_rows:
        vid_idx = vr["video_index"]
        vid_file = vr["filename"]
        prefix = f"v{vid_idx}_" if multi_video else ""
        video_path = proj / "input" / vid_file
        cmds.append((prefix, ffmpeg_svc.build_extract_cmd(
            video_path, frames_dir, fps=sample_fps,
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

            # Sharp-frame down-selection (optional)
            if sharp_mode:
                try:
                    sharp_res = select_sharpest_per_bucket(frames_dir, bucket_size=bucket_size)
                    await manager.send_log(
                        project_id,
                        f"[INFO] Sharp-frame selection: kept {sharp_res.selected_frames}/"
                        f"{sharp_res.total_frames} (window=+-{sharp_window}, bucket={bucket_size})",
                    )
                except Exception as e:
                    await manager.send_log(project_id, f"[WARN] Sharp-frame selection skipped: {e}")

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


@router.post("/mask")
async def run_masking(project_id: str, body: MaskSettings | None = None):
    body = body or MaskSettings()
    await _check_precondition(project_id, "mask")

    proj = _project_dir(project_id)
    frames_dir = proj / "frames"
    masks_dir = proj / "masks"

    if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
        raise HTTPException(400, "No extracted frames -- run frame extraction first")

    # Determine mode: point-prompted vs keyword vs external
    is_point_mode = body.points and body.reference_frame and body.point_labels

    if is_point_mode:
        cmd = mask_svc.build_point_mask_cmd(
            frames_dir, masks_dir,
            reference_frame=body.reference_frame,
            points=body.points,
            point_labels=body.point_labels,
            mode=body.mode, invert=body.invert,
            expand=body.expand, feather=body.feather,
        )
        log_msg = (
            f"[INFO] Point-prompted masking: {len(body.points)} points "
            f"on {body.reference_frame}"
        )
    elif body.use_external:
        exe = mask_svc.find_automasker_exe()
        if not exe:
            raise HTTPException(
                400,
                "AutoMasker executable not found. Place it in tools/automasker/ "
                "or disable 'use external' to use built-in masking."
            )
        cmd = mask_svc.build_external_mask_cmd(
            frames_dir, masks_dir, body.keywords,
            mode=body.mode, invert=body.invert,
            precision=body.precision, expand=body.expand, feather=body.feather,
        )
        log_msg = (
            f"[INFO] External masking with keywords: {body.keywords} "
            f"(precision={body.precision})"
        )
    else:
        cmd = mask_svc.build_builtin_mask_cmd(
            frames_dir, masks_dir, body.keywords,
            mode=body.mode, invert=body.invert,
            precision=body.precision, expand=body.expand, feather=body.feather,
        )
        log_msg = (
            f"[INFO] Keyword masking: {body.keywords} "
            f"(mode={body.mode}, precision={body.precision})"
        )

    mask_label = f"points:{body.reference_frame}" if is_point_mode else body.keywords

    async def run():
        try:
            await _update_step(project_id, PipelineStep.MASKING)
            await manager.send_log(project_id, log_msg)
            rc = await task_runner.run(
                project_id, cmd, step="mask", substep="generating",
                line_parser=mask_svc.parse_mask_line,
                timeout=3600,
                requires_gpu=True,
            )
            if rc == 0:
                mask_count = sum(1 for _ in masks_dir.glob("*.png")) if masks_dir.exists() else 0
                async with db_session() as db2:
                    await db2.execute(
                        "UPDATE projects SET step = ?, mask_keywords = ?, mask_count = ? "
                        "WHERE id = ?",
                        (PipelineStep.MASKS_READY.value, mask_label, mask_count, project_id),
                    )
                await manager.send_log(
                    project_id, f"[INFO] Masking complete: {mask_count} masks generated"
                )
            else:
                await _update_step(project_id, PipelineStep.FAILED, f"Masking failed (exit code {rc})")
        except Exception as e:
            logger.exception(f"Masking failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.post("/mask-preview")
async def mask_preview(project_id: str, body: MaskPreviewRequest):
    """Quick single-frame mask preview from click points. Returns base64 PNG overlay."""
    import base64
    import io

    proj = _project_dir(project_id)
    frames_dir = proj / "frames"
    frame_path = frames_dir / body.frame

    if not frame_path.exists():
        raise HTTPException(404, f"Frame not found: {body.frame}")

    if not body.points or not body.labels:
        raise HTTPException(400, "Must provide points and labels")

    loop = asyncio.get_event_loop()

    def _run_preview():
        import numpy as np
        from PIL import Image

        # Import SAM2 predictor inline (heavy import)
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch
        except ImportError:
            return None, "SAM2 not installed"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large", device=device,
        )

        img = np.array(Image.open(frame_path).convert("RGB"))
        predictor.set_image(img)

        pts = np.array(body.points, dtype=np.float32)
        lbls = np.array(body.labels, dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbls, multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        if mask.ndim == 3:
            mask = mask[0]

        # Create semi-transparent purple overlay
        H, W = mask.shape
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[mask, 0] = 147  # purple R
        overlay[mask, 1] = 51   # purple G
        overlay[mask, 2] = 234  # purple B
        overlay[mask, 3] = 128  # 50% opacity

        pil_overlay = Image.fromarray(overlay, "RGBA")
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Compute bounding box
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else []

        return b64, bbox

    result = await loop.run_in_executor(None, _run_preview)
    if result[0] is None:
        raise HTTPException(500, result[1])

    mask_b64, bbox = result
    return {"mask_b64": mask_b64, "bbox": bbox}


@router.post("/generate-novel-views")
async def generate_novel_views(project_id: str, body: NovelViewSettings | None = None):
    """Generate AI novel views from reference frames using selected model."""
    body = body or NovelViewSettings()
    proj = _project_dir(project_id)
    frames_dir = proj / "frames"
    novel_dir = proj / "novel_views"

    if not frames_dir.exists() or not any(frames_dir.glob("*.jpg")):
        raise HTTPException(400, "No frames -- extract frames first")

    cmd = trainer_svc.build_novel_view_cmd(
        input_dir=frames_dir,
        output_dir=novel_dir,
        model=body.model,
        num_refs=body.num_refs,
        output_size=body.output_size,
    )

    async def run():
        try:
            await manager.send_log(
                project_id,
                f"[INFO] Generating novel views with {body.model} "
                f"({body.num_refs} references, {body.output_size}px)",
            )
            rc = await task_runner.run(
                project_id, cmd, step="novel_views", substep="generating",
                line_parser=trainer_svc.parse_novel_view_line,
                timeout=600,
                requires_gpu=True,
            )
            if rc == 0:
                n_views = sum(1 for _ in novel_dir.glob("novel_*.jpg")) if novel_dir.exists() else 0
                await manager.send_log(
                    project_id, f"[INFO] Generated {n_views} novel views with {body.model}"
                )
            else:
                await manager.send_log(
                    project_id, f"[WARN] Novel view generation failed (exit code {rc})"
                )
        except Exception as e:
            logger.exception(f"Novel view generation failed for {project_id}")
            await manager.send_log(project_id, f"[ERROR] {e}")

    asyncio.create_task(run())
    return {"status": "started", "model": body.model}


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
        # Multiple cameras -> disable single_camera, upgrade matcher
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
            masks_dir = proj / "masks"
            mask_path = masks_dir if masks_dir.exists() and any(masks_dir.glob("*.png")) else None
            if mask_path:
                await manager.send_log(project_id, "[INFO] Using masks for feature extraction")
            cmd = colmap_svc.build_feature_extractor_cmd(
                db_path, frames_dir, body.single_camera, mask_path=mask_path
            )
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="feature_extraction",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=1800,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Feature extraction failed")
                return

            # Step 2: Matching
            cmd = colmap_svc.build_matcher_cmd(db_path, body.matcher_type)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="matching",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=3600,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Matching failed")
                return

            # Step 3: Mapping
            cmd = colmap_svc.build_mapper_cmd(db_path, frames_dir, sparse_dir)
            rc = await task_runner.run(
                project_id, cmd, step="sfm", substep="mapping",
                line_parser=colmap_svc.parse_colmap_line,
                timeout=3600,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Mapping failed")
                return

            # Find the sparse model directory (usually sparse/0)
            sparse_model = sparse_dir / "0"
            if not sparse_model.exists():
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
                timeout=1800,
            )
            if rc != 0:
                await _update_step(project_id, PipelineStep.FAILED, "Undistortion failed")
                return

            # Optional: Dense reconstruction
            if getattr(body, "enable_dense", False):
                await manager.send_log(project_id, "[INFO] Running dense stereo matching...")
                cmd = colmap_svc.build_patch_match_stereo_cmd(dense_dir)
                rc = await task_runner.run(
                    project_id, cmd, step="sfm", substep="dense_stereo",
                    line_parser=colmap_svc.parse_colmap_line,
                    timeout=3600,
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


@router.get("/scene-analysis")
async def scene_analysis(project_id: str):
    """Analyze COLMAP output and return recommended training parameters."""
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    loop = asyncio.get_event_loop()
    config = await loop.run_in_executor(None, analyze_scene, proj)

    stats_resp = None
    if config.stats is not None:
        stats_resp = SceneStatsResponse(
            num_points=config.stats.num_points,
            num_cameras=config.stats.num_cameras,
            num_images=config.stats.num_images,
            bbox_min=config.stats.bbox_min,
            bbox_max=config.stats.bbox_max,
            centroid=config.stats.centroid,
            scene_radius=config.stats.scene_radius,
            mean_point_density=config.stats.mean_point_density,
            camera_baseline=config.stats.camera_baseline,
        )

    return SceneConfigResponse(
        max_steps=config.max_steps,
        phase1_steps=config.phase1_steps,
        phase2_steps=config.phase2_steps,
        densify_grad_thresh=config.densify_grad_thresh,
        sh_degree=config.sh_degree,
        scene_complexity=config.scene_complexity,
        reasoning=config.reasoning,
        stats=stats_resp,
    )


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
            import shutil as _shutil
            _shutil.copy2(denoised_path, points3d_path)
            logger.info(
                "Denoising applied: %d -> %d points",
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

    # Bridge for depth estimation logs: callable from sync thread -> async WS
    _loop = asyncio.get_event_loop()

    def _depth_log_sync(msg: str):
        asyncio.run_coroutine_threadsafe(manager.send_log(project_id, msg), _loop)

    use_two_phase = body.two_phase
    phase1_steps = body.phase1_steps or int(body.max_steps * 0.8)
    phase2_steps = body.phase2_steps or (body.max_steps - phase1_steps)

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
                        await manager.send_log(project_id, f"[WARN] Depth estimation failed: {e} -- continuing without")
                        depth_dir = None

            # Find frame manifest for 4D mode
            manifest_path = proj / "frames" / "manifest.json"

            if use_two_phase and temporal_mode_val == "static":
                # Two-phase training (our branch feature)
                phases = trainer_svc.build_two_phase_cmds(
                    data_dir, result_dir,
                    phase1_steps=phase1_steps,
                    phase2_steps=phase2_steps,
                    sh_degree=body.sh_degree,
                    densify_grad_thresh=body.densify_grad_thresh,
                )

                for phase in phases:
                    await manager.send_log(
                        project_id, f"--- {phase.label} ---"
                    )
                    await manager.send_progress(
                        project_id, "train", phase.label,
                        (phase.step_offset / phase.total_steps) * 100,
                    )

                    phase_parser = trainer_svc.make_phase_line_parser(phase)
                    rc = await task_runner.run(
                        project_id, phase.cmd,
                        step="train", substep=phase.label,
                        line_parser=phase_parser,
                        timeout=7200,
                        requires_gpu=True,
                    )
                    if rc != 0:
                        await _update_step(
                            project_id, PipelineStep.FAILED,
                            f"{phase.label} failed (exit code {rc})",
                        )
                        return
            else:
                # Single-phase training (main branch path, also handles 4D)
                cmd = trainer_svc.build_train_cmd(
                    data_dir, result_dir, body.max_steps,
                    use_scaffold=use_scaffold, voxel_size=voxel_size,
                    resume=getattr(body, "resume", False),
                    sh_degree=getattr(body, "sh_degree", 0),
                    depth_dir=depth_dir,
                    depth_weight=depth_weight,
                    temporal_mode=getattr(body, "temporal_mode", "static"),
                    temporal_smoothness=getattr(body, "temporal_smoothness", 0.01),
                    manifest_path=manifest_path,
                    densify_grad_thresh=body.densify_grad_thresh,
                )
                rc = await task_runner.run(
                    project_id, cmd, step="train", substep="training",
                    line_parser=trainer_svc.parse_trainer_line,
                    timeout=7200,
                    requires_gpu=True,
                )
                if rc != 0:
                    await _update_step(
                        project_id, PipelineStep.FAILED,
                        f"Training failed (exit code {rc})",
                    )
                    return

            # Generate LOD versions as a post-training step
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lod_svc.generate_lod, result_dir
                )
            except Exception as lod_err:
                logger.warning("LOD generation failed for %s: %s",
                               project_id, lod_err)
            await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
        except Exception as e:
            logger.exception(f"Training failed for {project_id}")
            await _update_step(project_id, PipelineStep.FAILED, str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.get("/health")
async def pipeline_health(project_id: str):
    """Check if a pipeline task is actually running for this project."""
    running = task_runner.is_running(project_id)
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT step, error FROM projects WHERE id = ?", (project_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Project not found")

    step = row["step"]
    active_steps = {
        PipelineStep.EXTRACTING_FRAMES.value,
        PipelineStep.MASKING.value,
        PipelineStep.RUNNING_SFM.value,
        PipelineStep.TRAINING.value,
        PipelineStep.PORTRAIT.value,
    }
    stale = step in active_steps and not running

    if stale:
        error_msg = "Process exited unexpectedly"
        await _update_step(project_id, PipelineStep.FAILED, error_msg)
        return {"running": False, "step": PipelineStep.FAILED.value, "stale": True, "error": error_msg}

    return {"running": running, "step": step, "stale": False, "error": row["error"]}


@router.post("/cancel")
async def cancel_pipeline(project_id: str):
    async with db_session() as db_c:
        cursor = await db_c.execute("SELECT step FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
    is_training = row and row["step"] in (PipelineStep.TRAINING.value,)

    cancelled = await task_runner.cancel(project_id)
    if cancelled:
        if is_training:
            ply_path = _project_dir(project_id) / "output" / "point_cloud.ply"
            if ply_path.exists():
                await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
                await manager.send_log(
                    project_id, "[INFO] Training stopped early -- PLY exported successfully"
                )
            else:
                await _update_step(project_id, PipelineStep.FAILED, "Cancelled by user (no PLY exported)")
        else:
            await _update_step(project_id, PipelineStep.FAILED, "Cancelled by user")
    return {"cancelled": cancelled}


@router.post("/extract-mesh")
async def extract_mesh(project_id: str):
    """Kick off SuGaR mesh extraction in the background."""
    import sys
    await _check_precondition(project_id, "extract-mesh")
    if task_runner.is_running(project_id + "_mesh"):
        raise HTTPException(409, "Mesh extraction is already running for this project")
    proj = _project_dir(project_id)

    ply_path = proj / "output" / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(400, "Training not complete -- no .ply found")

    output_dir = proj / "output"
    glb_path = output_dir / "mesh.glb"

    async def run():
        try:
            sugar_script = Path(__file__).resolve().parents[3] / "scripts" / "extract_mesh.py"
            if not sugar_script.exists():
                sugar_script = None

            if sugar_script:
                cmd = [sys.executable, str(sugar_script),
                       "--ply", str(ply_path), "--output", str(glb_path)]
            else:
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
    """Run multi-stage refinement: visibility transfer -> confidence-aware training
    -> (optional) diffusion inpainting -> final training pass."""
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
        raise HTTPException(400, "Training not complete -- no .ply found")

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
                    await manager.send_log(project_id, "[WARN] Inpainting failed -- skipping final pass")
                else:
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


@router.post("/cleanup")
async def run_cleanup_endpoint(project_id: str, body: CleanupSettings | None = None):
    body = body or CleanupSettings()
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    output_dir = proj / "output"
    ply_files = list(output_dir.rglob("*.ply"))
    ply_files = [f for f in ply_files if f.stem != "point_cloud_original"]
    if not ply_files:
        raise HTTPException(400, "No training output (.ply) found")

    ply_path = ply_files[0]

    async def run():
        loop = asyncio.get_event_loop()
        try:
            await _update_step(project_id, PipelineStep.CLEANING)
            await manager.send_status(project_id, "cleanup", "running")

            async def on_progress(msg: str, pct: float):
                await manager.send_progress(project_id, "cleanup", "cleaning", pct)
                await manager.send_log(project_id, msg)

            def sync_progress(msg: str, pct: float):
                asyncio.run_coroutine_threadsafe(on_progress(msg, pct), loop)

            stats = await loop.run_in_executor(
                None,
                lambda: run_cleanup(
                    ply_path,
                    sor_k=body.sor_k,
                    sor_std=body.sor_std,
                    sparse_min_neighbors=body.sparse_min_neighbors,
                    large_splat_percentile=body.large_splat_percentile,
                    opacity_threshold=body.opacity_threshold,
                    bg_std_multiplier=body.bg_std_multiplier,
                    on_progress=sync_progress,
                ),
            )

            # Store cleanup stats in DB
            stats_json = json.dumps(stats.to_dict())
            async with db_session() as db_cl:
                await db_cl.execute(
                    "UPDATE projects SET cleanup_stats = ? WHERE id = ?",
                    (stats_json, project_id),
                )

            await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
            await manager.send_status(project_id, "cleanup", "completed")
            await manager.broadcast(project_id, {
                "type": "cleanup_complete",
                **stats.to_dict(),
            })

        except Exception as e:
            logger.exception("Cleanup failed for %s", project_id)
            await _update_step(project_id, PipelineStep.TRAINING_COMPLETE, None)
            await manager.send_status(project_id, "cleanup", "failed", str(e))

    asyncio.create_task(run())
    return {"status": "started"}


@router.get("/cleanup/stats")
async def get_cleanup_stats(project_id: str):
    async with db_session() as db:
        cursor = await db.execute(
            "SELECT cleanup_stats FROM projects WHERE id = ?", (project_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Project not found")
        raw = row["cleanup_stats"]
        if not raw:
            return {"has_stats": False}
        return {"has_stats": True, **json.loads(raw)}


@router.post("/cleanup/undo")
async def undo_cleanup(project_id: str):
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    output_dir = proj / "output"
    backups = list(output_dir.rglob("point_cloud_original.ply"))
    if not backups:
        raise HTTPException(400, "No cleanup backup found")

    backup_path = backups[0]
    ply_path = backup_path.parent / "point_cloud.ply"

    shutil.copy2(backup_path, ply_path)
    backup_path.unlink()

    async with db_session() as db:
        await db.execute(
            "UPDATE projects SET cleanup_stats = NULL WHERE id = ?", (project_id,)
        )

    logger.info("Cleanup undone for project %s", project_id)
    return {"status": "restored"}


@router.get("/coverage")
async def get_coverage(project_id: str):
    proj = _project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "Project not found")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, coverage_svc.analyze_coverage, proj
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("Coverage analysis failed for %s", project_id)
        raise HTTPException(500, f"Coverage analysis failed: {e}")

    return {
        "overall_score": result.overall_score,
        "direction_scores": result.direction_scores,
        "gaps": [
            {"direction": g.direction, "score": g.score, "recommendation": g.recommendation}
            for g in result.gaps
        ],
        "grid_data": result.grid_data,
    }


# ---------------------------------------------------------------------------
# Portrait Mode endpoints
# ---------------------------------------------------------------------------


def _find_portrait_image(proj: Path) -> Path | None:
    """Find a portrait image in the project's input/ directory."""
    input_dir = proj / "input"
    if not input_dir.exists():
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs = sorted(input_dir.glob(ext))
        if imgs:
            return imgs[0]
    # Fallback: check frames directory for an extracted frame
    frames_dir = proj / "frames"
    if frames_dir.exists():
        jpgs = sorted(frames_dir.glob("*.jpg"))
        if jpgs:
            return jpgs[0]
    return None


@router.post("/portrait")
async def run_portrait(project_id: str, body: PortraitSettings | None = None):
    """Run the portrait-to-Gaussian pipeline on a single portrait image."""
    body = body or PortraitSettings()
    await _check_precondition(project_id, "portrait")

    proj = _project_dir(project_id)
    image_path = _find_portrait_image(proj)
    if image_path is None:
        raise HTTPException(
            400,
            "No portrait image found. Upload a portrait image first "
            "(POST /api/projects/{id}/upload-portrait).",
        )

    output_dir = proj / "output"
    _loop = asyncio.get_event_loop()

    async def run():
        try:
            await _update_step(project_id, PipelineStep.PORTRAIT)
            await manager.send_status(project_id, "portrait", "running")
            await manager.send_log(
                project_id,
                f"[INFO] Starting portrait pipeline: {image_path.name} "
                f"(stride={body.stride}, depth_model={body.depth_model})",
            )

            async def on_progress(msg: str, pct: float):
                await manager.send_progress(project_id, "portrait", "processing", pct)
                await manager.send_log(project_id, msg)

            def sync_progress(msg: str, pct: float):
                asyncio.run_coroutine_threadsafe(on_progress(msg, pct), _loop)

            result = await _loop.run_in_executor(
                None,
                lambda: portrait_svc.run_portrait_pipeline(
                    image_path,
                    output_dir,
                    stride=body.stride,
                    focal_multiplier=body.focal_multiplier,
                    num_novel_views=body.num_novel_views,
                    include_background=body.include_background,
                    depth_model=body.depth_model,
                    on_progress=sync_progress,
                ),
            )

            # Copy PLY to standard location if not already there
            standard_ply = output_dir / "point_cloud.ply"
            if not standard_ply.exists() and result.ply_path.exists():
                shutil.copy2(result.ply_path, standard_ply)

            # Generate LOD versions
            try:
                await _loop.run_in_executor(
                    None, lod_svc.generate_lod, output_dir,
                )
            except Exception as lod_err:
                logger.warning("LOD generation failed for portrait %s: %s",
                               project_id, lod_err)

            await _update_step(project_id, PipelineStep.TRAINING_COMPLETE)
            await manager.send_status(project_id, "portrait", "completed")
            await manager.broadcast(project_id, {
                "type": "portrait_complete",
                "num_gaussians": result.num_gaussians,
                "novel_views": result.novel_view_count,
            })
            await manager.send_log(
                project_id,
                f"[INFO] Portrait pipeline complete: {result.num_gaussians} Gaussians, "
                f"{result.novel_view_count} novel views",
            )

        except Exception as e:
            logger.exception("Portrait pipeline failed for %s", project_id)
            await _update_step(project_id, PipelineStep.FAILED, str(e))
            await manager.send_status(project_id, "portrait", "failed", str(e))

    asyncio.create_task(run())
    return {"status": "started", "image": image_path.name}


@router.get("/portrait/depth-preview")
async def portrait_depth_preview(project_id: str):
    """Serve the depth map preview image generated by the portrait pipeline."""
    from fastapi.responses import FileResponse

    proj = _project_dir(project_id)
    depth_path = proj / "output" / "depth_preview.png"
    if not depth_path.exists():
        raise HTTPException(404, "Depth preview not available -- run portrait pipeline first")
    return FileResponse(depth_path, media_type="image/png")
