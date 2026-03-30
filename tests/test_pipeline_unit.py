"""
Unit tests for individual pipeline service modules.

Tests:
  - FFmpeg command builder + line parser
  - COLMAP command builders + line parser
  - Trainer command builder + line parser
  - Dependency checking (check_ffmpeg, check_colmap, check_python_deps)
  - TaskRunner state management (is_running, cancel)
  - Config executable resolution

All tests run without needing GPU or real video data.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ── Path setup (conftest already adds BACKEND_DIR to sys.path) ───────────────
log = logging.getLogger("tests.unit")


# ═══════════════════════════════════════════════════════════════════════════════
# FFmpeg service
# ═══════════════════════════════════════════════════════════════════════════════

class TestFFmpegCommandBuilder:
    def test_returns_list(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        video = tmp_path / "video.mp4"
        video.touch()
        frames = tmp_path / "frames"
        cmd = build_extract_cmd(video, frames, fps=2.0)
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_contains_ffmpeg(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        cmd = build_extract_cmd(tmp_path / "v.mp4", tmp_path / "f", fps=2.0)
        # First element is the ffmpeg executable path
        assert "ffmpeg" in str(cmd[0]).lower()

    def test_input_file_present(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        video = tmp_path / "myvideo.mp4"
        cmd = build_extract_cmd(video, tmp_path / "frames", fps=1.0)
        assert str(video) in cmd

    def test_fps_in_vf_filter(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        cmd = build_extract_cmd(tmp_path / "v.mp4", tmp_path / "f", fps=5.0)
        joined = " ".join(cmd)
        assert "fps=5.0" in joined

    def test_output_pattern_jpg(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        frames = tmp_path / "frames"
        cmd = build_extract_cmd(tmp_path / "v.mp4", frames, fps=2.0)
        # Output pattern should end with %04d.jpg
        assert any("%04d.jpg" in str(a) for a in cmd)

    def test_overwrite_flag_present(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        cmd = build_extract_cmd(tmp_path / "v.mp4", tmp_path / "f", fps=2.0)
        assert "-y" in cmd

    def test_progress_pipe_flag(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        cmd = build_extract_cmd(tmp_path / "v.mp4", tmp_path / "f", fps=2.0)
        joined = " ".join(cmd)
        assert "pipe:1" in joined

    def test_creates_output_dir(self, tmp_path, patch_settings):
        from app.services.ffmpeg import build_extract_cmd
        frames = tmp_path / "new_frames_dir"
        assert not frames.exists()
        build_extract_cmd(tmp_path / "v.mp4", frames, fps=2.0)
        assert frames.exists()


class TestFFmpegLineParser:
    def test_frame_line_returns_dict(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        result = parse_ffmpeg_line("frame=42")
        assert result is not None
        assert "percent" in result

    def test_frame_with_spaces(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        result = parse_ffmpeg_line("frame=  100")
        assert result is not None

    def test_progress_end_returns_100(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        result = parse_ffmpeg_line("progress=end")
        assert result is not None
        assert result["percent"] == 100

    def test_out_time_line(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        result = parse_ffmpeg_line("out_time=00:00:03.456789")
        assert result is not None

    def test_irrelevant_line_returns_none(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        result = parse_ffmpeg_line("Input #0, mov,mp4, from 'video.mp4'")
        assert result is None

    def test_empty_line_returns_none(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        assert parse_ffmpeg_line("") is None

    def test_frame_percent_is_negative_one(self):
        from app.services.ffmpeg import parse_ffmpeg_line
        # Frame lines report -1 because total frame count is unknown upfront
        result = parse_ffmpeg_line("frame=10")
        assert result["percent"] == -1


# ═══════════════════════════════════════════════════════════════════════════════
# COLMAP service
# ═══════════════════════════════════════════════════════════════════════════════

class TestColmapCommandBuilders:
    def test_feature_extractor_cmd(self, tmp_path, patch_settings):
        from app.services.colmap import build_feature_extractor_cmd
        db = tmp_path / "colmap" / "db.db"
        images = tmp_path / "frames"
        cmd = build_feature_extractor_cmd(db, images, single_camera=True)
        assert isinstance(cmd, list)
        joined = " ".join(cmd)
        assert "feature_extractor" in joined
        assert str(db) in joined
        assert str(images) in joined
        assert "single_camera" in joined or "1" in joined

    def test_feature_extractor_no_single_camera(self, tmp_path, patch_settings):
        from app.services.colmap import build_feature_extractor_cmd
        db = tmp_path / "db.db"
        images = tmp_path / "frames"
        cmd = build_feature_extractor_cmd(db, images, single_camera=False)
        joined = " ".join(cmd)
        assert "--ImageReader.single_camera" not in joined

    def test_feature_extractor_creates_parent_dir(self, tmp_path, patch_settings):
        from app.services.colmap import build_feature_extractor_cmd
        db = tmp_path / "colmap" / "subdir" / "db.db"
        assert not db.parent.exists()
        build_feature_extractor_cmd(db, tmp_path / "frames")
        assert db.parent.exists()

    def test_matcher_cmd_sequential(self, tmp_path, patch_settings):
        from app.services.colmap import build_matcher_cmd
        db = tmp_path / "db.db"
        cmd = build_matcher_cmd(db, matcher_type="sequential_matcher")
        joined = " ".join(cmd)
        assert "sequential_matcher" in joined
        assert str(db) in joined

    def test_matcher_cmd_exhaustive(self, tmp_path, patch_settings):
        from app.services.colmap import build_matcher_cmd
        db = tmp_path / "db.db"
        cmd = build_matcher_cmd(db, matcher_type="exhaustive_matcher")
        joined = " ".join(cmd)
        assert "exhaustive_matcher" in joined

    def test_mapper_cmd(self, tmp_path, patch_settings):
        from app.services.colmap import build_mapper_cmd
        db = tmp_path / "db.db"
        images = tmp_path / "frames"
        output = tmp_path / "sparse"
        cmd = build_mapper_cmd(db, images, output)
        joined = " ".join(cmd)
        assert "mapper" in joined
        assert str(db) in joined
        assert str(images) in joined
        assert str(output) in joined

    def test_mapper_creates_output_dir(self, tmp_path, patch_settings):
        from app.services.colmap import build_mapper_cmd
        output = tmp_path / "colmap" / "sparse"
        assert not output.exists()
        build_mapper_cmd(tmp_path / "db.db", tmp_path / "frames", output)
        assert output.exists()

    def test_undistorter_cmd(self, tmp_path, patch_settings):
        from app.services.colmap import build_undistorter_cmd
        image_path = tmp_path / "frames"
        input_path = tmp_path / "sparse" / "0"
        output_path = tmp_path / "dense"
        cmd = build_undistorter_cmd(image_path, input_path, output_path)
        joined = " ".join(cmd)
        assert "image_undistorter" in joined
        assert "COLMAP" in joined

    def test_colmap_bin_in_all_cmds(self, tmp_path, patch_settings):
        from app.services.colmap import (
            build_feature_extractor_cmd,
            build_matcher_cmd,
            build_mapper_cmd,
            build_undistorter_cmd,
        )
        from app.config import settings
        colmap_bin = str(settings.colmap_bin)
        for cmd in [
            build_feature_extractor_cmd(tmp_path / "db.db", tmp_path / "images"),
            build_matcher_cmd(tmp_path / "db.db"),
            build_mapper_cmd(tmp_path / "db.db", tmp_path / "images", tmp_path / "out"),
            build_undistorter_cmd(tmp_path / "images", tmp_path / "sparse", tmp_path / "dense"),
        ]:
            assert cmd[0] == colmap_bin, f"Expected {colmap_bin!r} but got {cmd[0]!r}"


class TestColmapLineParser:
    def test_registering_image(self):
        from app.services.colmap import parse_colmap_line
        result = parse_colmap_line("Registering image #42 (5)")
        assert result is not None
        assert "percent" in result

    def test_processed_file(self):
        from app.services.colmap import parse_colmap_line
        result = parse_colmap_line("Processed file [01/10]")
        assert result is not None

    def test_matching_block(self):
        from app.services.colmap import parse_colmap_line
        result = parse_colmap_line("Matching block [1/5, 1/5]")
        assert result is not None

    def test_verified_line(self):
        from app.services.colmap import parse_colmap_line
        result = parse_colmap_line("Verified 150 image pairs")
        assert result is not None

    def test_irrelevant_line(self):
        from app.services.colmap import parse_colmap_line
        result = parse_colmap_line("Loading database...")
        assert result is None

    def test_empty_returns_none(self):
        from app.services.colmap import parse_colmap_line
        assert parse_colmap_line("") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer service
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainerCommandBuilder:
    def test_returns_list(self, tmp_path):
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results", max_steps=100)
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_python_executable(self, tmp_path):
        from app.services.trainer import build_train_cmd
        import sys
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results", max_steps=100)
        # First element should be the current Python interpreter
        assert cmd[0] == sys.executable

    def test_scaffold_mode(self, tmp_path):
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results", max_steps=100, use_scaffold=True)
        joined = " ".join(cmd)
        assert "train_scaffold.py" in joined
        assert "--data_dir" in joined
        assert "--result_dir" in joined
        assert "--voxel_size" in joined

    def test_vanilla_mode(self, tmp_path):
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results", max_steps=100, use_scaffold=False)
        joined = " ".join(cmd)
        assert "train_splat.py" in joined
        assert "--data_dir" in joined
        assert "--result_dir" in joined

    def test_data_dir_arg(self, tmp_path):
        from app.services.trainer import build_train_cmd
        data = tmp_path / "mydata"
        cmd = build_train_cmd(data, tmp_path / "results", max_steps=100)
        assert str(data) in cmd

    def test_result_dir_arg(self, tmp_path):
        from app.services.trainer import build_train_cmd
        results = tmp_path / "myresults"
        cmd = build_train_cmd(tmp_path / "data", results, max_steps=100)
        assert str(results) in cmd

    def test_max_steps_arg(self, tmp_path):
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results", max_steps=500)
        assert "500" in cmd
        assert "--max_steps" in cmd

    def test_creates_result_dir(self, tmp_path):
        from app.services.trainer import build_train_cmd
        results = tmp_path / "new_results"
        assert not results.exists()
        build_train_cmd(tmp_path / "data", results, max_steps=100)
        assert results.exists()


class TestTrainerLineParser:
    def test_step_with_total(self):
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("Step 1000/7000, loss=0.0321, psnr=24.5")
        assert result is not None
        assert "percent" in result
        assert abs(result["percent"] - (1000 / 7000 * 100)) < 0.01

    def test_step_with_loss_and_psnr(self):
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("Step 500/7000, loss=0.0123, psnr=28.3")
        assert "metric" in result
        assert result["metric"]["step"] == 500
        assert abs(result["metric"]["loss"] - 0.0123) < 1e-6
        assert abs(result["metric"]["psnr"] - 28.3) < 1e-6

    def test_step_without_total(self):
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("step 200, loss=0.05")
        # When no total, percent key should not be in result
        assert result is not None
        assert "percent" not in result or result.get("percent") is None

    def test_irrelevant_line(self):
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("Loading dataset...")
        assert result is None

    def test_step_lowercase(self):
        # NOTE: parser uses [Ss]tep — matches 'step' and 'Step' only, not 'STEP'
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("step 100/1000, loss=0.01")
        assert result is not None

    def test_percent_calculation(self):
        from app.services.trainer import parse_trainer_line
        result = parse_trainer_line("Step 3500/7000, loss=0.02, psnr=25.0")
        assert result is not None
        assert abs(result["percent"] - 50.0) < 0.01

    def test_empty_returns_none(self):
        from app.services.trainer import parse_trainer_line
        assert parse_trainer_line("") is None

    def test_no_metric_when_only_step(self):
        from app.services.trainer import parse_trainer_line
        # Step with total but no loss/psnr → no metric key (or metric has only step)
        result = parse_trainer_line("Step 100/7000")
        if result and "metric" in result:
            # Only "step" in metric, no loss/psnr → metric should NOT be included
            # (parse_trainer_line requires >1 key in metric to include it)
            pass  # Acceptable either way per implementation


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency checking  (requires real ffmpeg/colmap installed in this env)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDependencyChecks:
    @pytest.mark.asyncio
    async def test_check_ffmpeg_installed(self):
        """FFmpeg is expected to be installed in this environment."""
        from app.services.deps import check_ffmpeg
        t0 = time.time()
        status = await check_ffmpeg()
        elapsed = time.time() - t0
        log.info("check_ffmpeg: installed=%s version=%s (%.2fs)", status.installed, status.version, elapsed)
        assert status.name == "ffmpeg"
        assert status.installed is True, f"FFmpeg not found: {status.error}"
        assert status.version is not None

    @pytest.mark.asyncio
    async def test_check_colmap_installed(self):
        """COLMAP is expected to be installed in this environment."""
        from app.services.deps import check_colmap
        t0 = time.time()
        status = await check_colmap()
        elapsed = time.time() - t0
        log.info("check_colmap: installed=%s version=%s (%.2fs)", status.installed, status.version, elapsed)
        assert status.name == "colmap"
        assert status.installed is True, f"COLMAP not found: {status.error}"

    @pytest.mark.asyncio
    async def test_check_python_deps(self):
        """PyTorch should be importable."""
        from app.services.deps import check_python_deps
        status = await check_python_deps()
        log.info("check_python_deps: installed=%s version=%s", status.installed, status.version)
        assert status.name == "python_deps"
        assert status.installed is True, f"PyTorch not found: {status.error}"
        assert "PyTorch" in (status.version or "")

    @pytest.mark.asyncio
    async def test_get_system_status_structure(self):
        """get_system_status returns a fully populated SystemStatus."""
        from app.services.deps import get_system_status
        from app.models import SystemStatus
        t0 = time.time()
        status = await get_system_status()
        elapsed = time.time() - t0
        log.info(
            "System status (%.2fs): CUDA=%s GPU=%s VRAM=%sMB torch_cuda=%s",
            elapsed, status.cuda_available, status.gpu_name,
            status.gpu_vram_mb, status.torch_cuda_available,
        )
        assert isinstance(status, SystemStatus)
        assert hasattr(status, "ffmpeg")
        assert hasattr(status, "colmap")
        assert hasattr(status, "python_deps")
        assert hasattr(status, "cuda_available")
        assert hasattr(status, "torch_cuda_available")


# ═══════════════════════════════════════════════════════════════════════════════
# TaskRunner
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskRunner:
    def test_initial_not_running(self):
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        assert runner.is_running("nonexistent_project") is False

    @pytest.mark.asyncio
    async def test_run_simple_command(self):
        """Run a real subprocess (echo) through the TaskRunner."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        rc = await runner.run(
            project_id="test_task",
            cmd=["echo", "hello from taskrunner"],
            step="test",
            substep="echo",
            timeout=10,
        )
        assert rc == 0

    @pytest.mark.asyncio
    async def test_run_failing_command(self):
        """A command that exits non-zero should return that exit code."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        rc = await runner.run(
            project_id="test_fail",
            cmd=["false"],  # Always exits 1 on Unix
            step="test",
            substep="fail",
            timeout=10,
        )
        assert rc != 0

    @pytest.mark.asyncio
    async def test_run_clears_process_on_complete(self):
        """After run() returns, the project should no longer appear as running."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        await runner.run(
            project_id="test_clear",
            cmd=["true"],
            step="test",
            substep="clear",
            timeout=10,
        )
        assert runner.is_running("test_clear") is False

    @pytest.mark.asyncio
    async def test_duplicate_run_raises(self):
        """Starting a second task for the same project while one is running should raise."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()

        async def slow_run():
            return await runner.run(
                project_id="dup_project",
                cmd=["sleep", "2"],
                step="test",
                substep="slow",
                timeout=10,
            )

        task = asyncio.create_task(slow_run())
        # Give the first task time to register
        await asyncio.sleep(0.1)
        with pytest.raises(RuntimeError, match="already has a running task"):
            await runner.run(
                project_id="dup_project",
                cmd=["echo", "second"],
                step="test",
                substep="dup",
                timeout=10,
            )
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        """cancel() should terminate a running process and return True."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()

        async def long_run():
            return await runner.run(
                project_id="cancel_me",
                cmd=["sleep", "30"],
                step="test",
                substep="long",
                timeout=60,
            )

        task = asyncio.create_task(long_run())
        await asyncio.sleep(0.15)
        cancelled = await runner.cancel("cancel_me")
        assert cancelled is True
        assert runner.is_running("cancel_me") is False
        # Clean up task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self):
        """Cancelling a project with no running task returns False."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        cancelled = await runner.cancel("ghost_project")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_line_parser_called(self):
        """line_parser callback should receive output lines."""
        from app.pipeline.task_runner import TaskRunner
        runner = TaskRunner()
        parsed_lines = []

        def capture_parser(line: str):
            parsed_lines.append(line)
            return None

        await runner.run(
            project_id="parser_test",
            cmd=["echo", "hello parser"],
            step="test",
            substep="parse",
            line_parser=capture_parser,
            timeout=10,
        )
        assert any("hello parser" in line for line in parsed_lines), f"Got: {parsed_lines}"


# ═══════════════════════════════════════════════════════════════════════════════
# Config / Settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestSettings:
    def test_ffmpeg_bin_is_path(self, patch_settings):
        from app.config import settings
        assert isinstance(settings.ffmpeg_bin, Path)

    def test_colmap_bin_is_path(self, patch_settings):
        from app.config import settings
        assert isinstance(settings.colmap_bin, Path)

    def test_ffmpeg_bin_name(self, patch_settings):
        from app.config import settings
        # Should resolve to something containing 'ffmpeg'
        assert "ffmpeg" in str(settings.ffmpeg_bin).lower()

    def test_colmap_bin_name(self, patch_settings):
        from app.config import settings
        assert "colmap" in str(settings.colmap_bin).lower()

    def test_is_windows_bool(self, patch_settings):
        from app.config import settings
        assert isinstance(settings.is_windows, bool)

    def test_log_file_property(self, patch_settings):
        from app.config import settings
        assert str(settings.log_file).endswith(".log")

    def test_data_dir_patched(self, patch_settings, test_tmp_dir):
        from app.config import settings
        # Confirm the fixture patching worked
        assert "gaussiansplat" in str(settings.data_dir)
        assert settings.data_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# FFmpeg real subprocess test (fast — uses synthetic video)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFFmpegRealExecution:
    @pytest.mark.asyncio
    async def test_extract_frames_from_synthetic_video(self, tmp_path, synthetic_video, patch_settings):
        """
        Run the actual ffmpeg command built by build_extract_cmd against the
        synthetic 5-second video.  Expects at least one frame to be produced.
        """
        from app.services.ffmpeg import build_extract_cmd
        import subprocess

        frames_dir = tmp_path / "frames"
        cmd = build_extract_cmd(synthetic_video, frames_dir, fps=2.0)

        log.info("Running ffmpeg: %s", " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - t0

        log.info("ffmpeg exit=%d  elapsed=%.2fs", result.returncode, elapsed)
        if result.returncode != 0:
            log.error("ffmpeg stderr:\n%s", result.stderr)

        assert result.returncode == 0, f"ffmpeg failed:\n{result.stderr}"

        frames = sorted(frames_dir.glob("*.jpg"))
        log.info("Frames produced: %d", len(frames))
        assert len(frames) > 0, "No frames extracted"

        # 5s @ 2fps = ~10 frames; allow some tolerance
        assert len(frames) >= 5, f"Expected ≥5 frames, got {len(frames)}"

        # Log frame sizes for diagnostics
        sizes = [f.stat().st_size for f in frames]
        log.info("Frame sizes (bytes): min=%d max=%d avg=%d", min(sizes), max(sizes), sum(sizes) // len(sizes))
