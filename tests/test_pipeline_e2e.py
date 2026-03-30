"""
End-to-end pipeline tests.

Quick tests (always run):
  - Frame extraction on synthetic video via the API

Slow tests (@pytest.mark.slow):
  - Full pipeline: frames → SfM (COLMAP) → 3DGS training
  - Requires real video in test_data/videos/ and GPU
  - Skipped in CI

Run slow tests with:
    pytest -m slow tests/test_pipeline_e2e.py -v
"""

import logging
import time
from pathlib import Path

import pytest

log = logging.getLogger("tests.e2e")

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data" / "videos"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_video(client, project_id: str, video_path: Path):
    with open(video_path, "rb") as f:
        resp = client.post(
            f"/api/projects/{project_id}/upload",
            files={"file": (video_path.name, f, "video/mp4")},
        )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()


def _create_and_upload(client, name: str, video_path: Path) -> str:
    resp = client.post("/api/projects", json={"name": name})
    assert resp.status_code == 200
    pid = resp.json()["id"]
    _upload_video(client, pid, video_path)
    return pid


# ═══════════════════════════════════════════════════════════════════════════════
# Quick E2E: Frame Extraction (synthetic video, fast)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrameExtractionE2E:
    """
    Tests that run the full frame extraction pipeline with the synthetic video.
    No GPU needed.  Should complete in <30 seconds.
    """

    def test_full_frame_extraction_pipeline(self, test_client, synthetic_video, patch_settings):
        """
        Create project → upload synthetic video → extract at 2fps →
        verify step=frames_ready, frames on disk match DB count.
        """
        from conftest import wait_for_step
        from app.config import settings

        t0 = time.time()
        pid = _create_and_upload(test_client, "e2e-extract", synthetic_video)

        # Verify initial state
        proj = test_client.get(f"/api/projects/{pid}").json()
        assert proj["step"] == "created"
        assert proj["video_filename"] is not None
        assert proj["frame_count"] == 0
        log.info("[E2E] Created project %s, video uploaded", pid)

        # Start extraction
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        assert resp.json()["status"] == "started"

        # Step should immediately transition to extracting_frames
        # (may be fast enough that it's already frames_ready by the time we poll)
        final = wait_for_step(test_client, pid, "frames_ready", timeout=60)
        elapsed = time.time() - t0

        log.info("[E2E] Extraction done in %.1fs: step=%s frames=%d",
                 elapsed, final["step"], final["frame_count"])

        assert final["step"] == "frames_ready", (
            f"Extraction failed: error={final.get('error')}"
        )
        assert final["frame_count"] >= 5, (
            f"Too few frames: {final['frame_count']}"
        )

        # Verify frames directory
        frames_dir = settings.data_dir / pid / "frames"
        on_disk = list(frames_dir.glob("*.jpg"))
        assert len(on_disk) == final["frame_count"], (
            f"DB says {final['frame_count']} frames but {len(on_disk)} on disk"
        )
        log.info("[E2E] Disk frames: %d, DB frame_count: %d ✓", len(on_disk), final["frame_count"])

    def test_frames_listed_via_api(self, test_client, synthetic_video):
        """After extraction, /frames endpoint should list the extracted frames."""
        from conftest import wait_for_step

        pid = _create_and_upload(test_client, "e2e-frames-api", synthetic_video)
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        wait_for_step(test_client, pid, "frames_ready", timeout=60)

        frames_resp = test_client.get(f"/api/projects/{pid}/frames")
        frames = frames_resp.json()
        log.info("[E2E] Frames listed via API: %d", len(frames))
        assert len(frames) > 0
        for frame in frames:
            assert "name" in frame
            assert "url" in frame
            assert frame["name"].endswith(".jpg")

    def test_frame_served_as_jpeg(self, test_client, synthetic_video):
        """The first extracted frame should be serveable as a JPEG."""
        from conftest import wait_for_step

        pid = _create_and_upload(test_client, "e2e-serve-frame", synthetic_video)
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        wait_for_step(test_client, pid, "frames_ready", timeout=60)

        frames = test_client.get(f"/api/projects/{pid}/frames").json()
        assert len(frames) > 0

        first = frames[0]
        img_resp = test_client.get(f"/api/projects/{pid}/frames/{first['name']}")
        assert img_resp.status_code == 200
        ct = img_resp.headers.get("content-type", "")
        assert "jpeg" in ct or "image" in ct
        # Verify JPEG magic bytes (FFD8)
        content = img_resp.content
        assert content[:2] == b"\xff\xd8", "Response is not a valid JPEG"

    def test_project_thumbnail_set_after_extraction(self, test_client, synthetic_video):
        """After extraction, ProjectDetail.thumbnail should be non-null."""
        from conftest import wait_for_step

        pid = _create_and_upload(test_client, "e2e-thumbnail", synthetic_video)
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        wait_for_step(test_client, pid, "frames_ready", timeout=60)

        proj = test_client.get(f"/api/projects/{pid}").json()
        log.info("[E2E] Thumbnail: %s", proj.get("thumbnail"))
        assert proj["thumbnail"] is not None

    def test_cancel_during_extraction(self, test_client, synthetic_video):
        """
        Cancel a running extraction — should mark the project as failed.
        NOTE: The synthetic video is tiny so the task may finish before we cancel.
        This test is best-effort; we assert consistent state either way.
        """
        from conftest import wait_for_step

        pid = _create_and_upload(test_client, "e2e-cancel-extract", synthetic_video)
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )

        # Immediately try to cancel
        cancel_resp = test_client.post(f"/api/projects/{pid}/pipeline/cancel")
        cancelled = cancel_resp.json()["cancelled"]
        log.info("[E2E] Cancel response: cancelled=%s", cancelled)

        if cancelled:
            # If we managed to cancel, project should eventually be marked failed
            final = wait_for_step(test_client, pid, "failed", timeout=10)
            assert final["step"] == "failed"
            log.info("[E2E] Extraction cancelled, project step=failed ✓")
        else:
            # Task finished before we could cancel — that's fine
            log.info("[E2E] Task completed before cancel could take effect")


# ═══════════════════════════════════════════════════════════════════════════════
# Slow E2E: Full pipeline (requires real video + GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_test_videos() -> list[Path]:
    """Return available real test videos from test_data/videos/."""
    if not TEST_DATA_DIR.exists():
        return []
    return list(TEST_DATA_DIR.glob("*.mp4")) + list(TEST_DATA_DIR.glob("*.mov"))


@pytest.mark.slow
class TestFullPipelineE2E:
    """
    Full pipeline tests: frame extraction + SfM + Gaussian Splat training.
    These are expensive (~hours) and require:
      - Real multi-angle video in test_data/videos/
      - GPU with sufficient VRAM for gsplat training
    Skip by default; run with: pytest -m slow
    """

    @pytest.fixture(params=_get_test_videos(), ids=lambda p: p.stem)
    def real_video(self, request) -> Path:
        video = request.param
        if not video.exists():
            pytest.skip(f"Test video not found: {video}")
        return video

    def test_frame_extraction_real_video(self, test_client, real_video, patch_settings):
        """Extract frames from a real video — verifies ffmpeg handles real content."""
        from conftest import wait_for_step
        from app.config import settings

        pid = _create_and_upload(test_client, f"e2e-real-extract-{real_video.stem}", real_video)
        t0 = time.time()
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        final = wait_for_step(test_client, pid, "frames_ready", timeout=300)
        elapsed = time.time() - t0

        log.info("[SLOW E2E] %s: extraction %.1fs, frames=%d",
                 real_video.name, elapsed, final["frame_count"])
        assert final["step"] == "frames_ready"
        assert final["frame_count"] > 10, "Real video should produce >10 frames at 2fps"

        # Log file sizes
        frames_dir = settings.data_dir / pid / "frames"
        sizes = [f.stat().st_size for f in frames_dir.glob("*.jpg")]
        if sizes:
            log.info("[SLOW E2E] Frame sizes: min=%dB max=%dB avg=%dB",
                     min(sizes), max(sizes), sum(sizes) // len(sizes))

    def test_sfm_real_video(self, test_client, real_video, patch_settings):
        """Run COLMAP SfM on a real video. Takes 5-30 minutes."""
        from conftest import wait_for_step
        from app.config import settings

        pid = _create_and_upload(test_client, f"e2e-real-sfm-{real_video.stem}", real_video)

        # Extract frames first
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        wait_for_step(test_client, pid, "frames_ready", timeout=300)

        # Run SfM
        t0 = time.time()
        test_client.post(
            f"/api/projects/{pid}/pipeline/sfm",
            json={"matcher_type": "sequential_matcher", "single_camera": True},
        )
        final = wait_for_step(test_client, pid, "sfm_ready", timeout=7200)  # 2 hours
        elapsed = time.time() - t0

        log.info("[SLOW E2E] SfM completed in %.1fs: step=%s sfm_points=%d",
                 elapsed, final["step"], final.get("sfm_points", 0))

        assert final["step"] == "sfm_ready", f"SfM failed: {final.get('error')}"

        # Verify COLMAP output on disk
        colmap_dir = settings.data_dir / pid / "colmap"
        assert (colmap_dir / "database.db").exists(), "COLMAP database not created"
        sparse_dir = colmap_dir / "sparse"
        assert sparse_dir.exists(), "COLMAP sparse directory not created"
        dense_dir = colmap_dir / "dense"
        assert dense_dir.exists(), "COLMAP dense directory not created (undistortion failed?)"

    def test_full_pipeline_with_training(self, test_client, real_video, patch_settings):
        """
        Full pipeline: extract → SfM → train (500 steps for speed).
        Requires GPU with enough VRAM for gsplat.
        """
        from conftest import wait_for_step
        from app.config import settings
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — cannot run training test")

        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        log.info("[SLOW E2E] GPU: %s  VRAM: %dMB", gpu_name, vram_mb)

        if vram_mb < 4000:
            pytest.skip(f"Insufficient VRAM ({vram_mb}MB < 4000MB) for training test")

        pid = _create_and_upload(test_client, f"e2e-real-full-{real_video.stem}", real_video)

        # Stage 1: Extract frames
        t_start = time.time()
        test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        frames_data = wait_for_step(test_client, pid, "frames_ready", timeout=300)
        log.info("[SLOW E2E] Frames extracted: %d (%.1fs)",
                 frames_data["frame_count"], time.time() - t_start)

        # Stage 2: SfM
        t_sfm = time.time()
        test_client.post(
            f"/api/projects/{pid}/pipeline/sfm",
            json={"matcher_type": "sequential_matcher", "single_camera": True},
        )
        sfm_data = wait_for_step(test_client, pid, "sfm_ready", timeout=7200)
        log.info("[SLOW E2E] SfM done: sfm_points=%d (%.1fs)",
                 sfm_data.get("sfm_points", 0), time.time() - t_sfm)
        assert sfm_data["step"] == "sfm_ready", f"SfM failed: {sfm_data.get('error')}"

        # Stage 3: Train (500 steps for speed)
        t_train = time.time()
        test_client.post(
            f"/api/projects/{pid}/pipeline/train",
            json={"max_steps": 500},
        )
        train_data = wait_for_step(test_client, pid, "training_complete", timeout=3600)
        log.info("[SLOW E2E] Training done: step=%s (%.1fs)",
                 train_data["step"], time.time() - t_train)

        assert train_data["step"] == "training_complete", (
            f"Training failed: {train_data.get('error')}"
        )

        # Verify output PLY
        output_dir = settings.data_dir / pid / "output"
        ply_files = list(output_dir.rglob("*.ply"))
        log.info("[SLOW E2E] PLY files: %s", [str(p) for p in ply_files])
        assert len(ply_files) > 0, "Training completed but no .ply file found"

        # Check PLY is non-trivial size
        for ply in ply_files:
            size_mb = ply.stat().st_size / (1024 * 1024)
            log.info("[SLOW E2E] PLY: %s (%.1fMB)", ply.name, size_mb)
            assert size_mb > 0.01, f"PLY file suspiciously small: {size_mb:.3f}MB"

        total_elapsed = time.time() - t_start
        log.info("[SLOW E2E] Full pipeline completed in %.1fs (%.1fmin)",
                 total_elapsed, total_elapsed / 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Equirectangular detection & perspective crop E2E
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquirectangularPipeline:
    """Tests for 360° equirectangular detection and perspective crop generation."""

    def test_equirect_detection_by_aspect_ratio(self, tmp_path):
        """2:1 image → is_equirectangular returns True."""
        from app.services.equirect import is_equirectangular
        assert is_equirectangular(3840, 1920) is True
        assert is_equirectangular(4096, 2048) is True
        assert is_equirectangular(1920, 1080) is False  # 16:9, not 2:1
        assert is_equirectangular(1920, 960) is True

    def test_equirect_tolerance(self):
        """Within 5% of 2.0 ratio is accepted."""
        from app.services.equirect import is_equirectangular
        # 1.96 → difference from 2.0 is 0.04, within 0.05
        assert is_equirectangular(196, 100) is True
        # 2.11 → difference is 0.11, exceeds 0.05
        assert is_equirectangular(211, 100) is False

    def test_perspective_crops_count(self, tmp_path):
        """One equirect frame → exactly 10 perspective crops."""
        import cv2
        import numpy as np
        from app.services.equirect import extract_perspective_crops

        # Create a synthetic 2:1 equirect image (1600x800)
        frame = tmp_path / "frame_0000.jpg"
        img = np.random.randint(0, 255, (800, 1600, 3), dtype=np.uint8)
        cv2.imwrite(str(frame), img)

        output_dir = tmp_path / "crops"
        crops = extract_perspective_crops(frame, output_dir, frame_index=0)

        assert len(crops) == 10, f"Expected 10 crops, got {len(crops)}"
        for p in crops:
            assert p.exists(), f"Crop missing: {p}"

    def test_perspective_crop_dimensions(self, tmp_path):
        """Each crop is 800x800 by default."""
        import cv2
        import numpy as np
        from app.services.equirect import extract_perspective_crops

        frame = tmp_path / "frame_0000.jpg"
        img = np.zeros((800, 1600, 3), dtype=np.uint8)
        cv2.imwrite(str(frame), img)

        output_dir = tmp_path / "crops"
        crops = extract_perspective_crops(frame, output_dir, frame_index=0)
        for p in crops:
            crop_img = cv2.imread(str(p))
            assert crop_img is not None
            h, w = crop_img.shape[:2]
            assert w == 800 and h == 800, f"Unexpected crop size {w}x{h}"

    def test_process_equirect_frames_batch(self, tmp_path):
        """process_equirect_frames produces 10 crops per frame."""
        import cv2
        import numpy as np
        from app.services.equirect import process_equirect_frames

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.random.randint(0, 255, (800, 1600, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), img)

        output_dir = tmp_path / "crops"
        total = process_equirect_frames(frames_dir, output_dir)
        assert total == 30, f"Expected 30 crops (3 frames × 10), got {total}"


# ═══════════════════════════════════════════════════════════════════════════════
# Point cloud denoising E2E
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointCloudDenoising:
    """Tests for statistical outlier removal on COLMAP point clouds."""

    def _make_points3d_bin(self, path, xyz, rgb):
        """Write a minimal points3D.bin for testing."""
        import struct
        n = len(xyz)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))
            for i in range(n):
                f.write(struct.pack("<Q", i + 1))
                f.write(struct.pack("<3d", *xyz[i]))
                f.write(struct.pack("<3B", *rgb[i]))
                f.write(struct.pack("<d", 0.0))   # error
                f.write(struct.pack("<Q", 0))      # empty track

    def test_denoise_off_returns_input(self, tmp_path):
        """strength='off' returns the input path unchanged."""
        from app.services.denoise import denoise_point_cloud
        import numpy as np
        xyz = np.random.randn(100, 3)
        rgb = np.zeros((100, 3), dtype=np.uint8)
        inp = tmp_path / "points3D.bin"
        out = tmp_path / "points3D_out.bin"
        self._make_points3d_bin(inp, xyz, rgb)
        result = denoise_point_cloud(inp, out, strength="off")
        assert result == inp

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("open3d"),
        reason="open3d not installed"
    )
    def test_denoise_removes_outliers(self, tmp_path):
        """Statistical outlier removal should remove injected outlier points."""
        import numpy as np
        from app.services.denoise import denoise_point_cloud

        rng = np.random.default_rng(42)
        # 200 inliers in a tight cluster
        inliers_xyz = rng.normal(loc=0.0, scale=0.05, size=(200, 3))
        # 20 outliers far away
        outliers_xyz = rng.uniform(low=50.0, high=100.0, size=(20, 3))
        xyz = np.vstack([inliers_xyz, outliers_xyz])
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

        inp = tmp_path / "points3D.bin"
        out = tmp_path / "points3D_denoised.bin"
        self._make_points3d_bin(inp, xyz, rgb)

        result = denoise_point_cloud(inp, out, strength="aggressive")
        assert result == out
        assert out.exists()

        # Read the output and verify outliers were removed
        import struct
        with open(out, "rb") as f:
            (n_out,) = struct.unpack("<Q", f.read(8))
        assert n_out < 220, f"Expected fewer than 220 points after denoising, got {n_out}"
        assert n_out >= 150, f"Expected at least 150 inliers retained, got {n_out}"


# ═══════════════════════════════════════════════════════════════════════════════
# Scaffold-GS trainer command E2E
# ═══════════════════════════════════════════════════════════════════════════════

class TestScaffoldTrainerE2E:
    """Tests for Scaffold-GS trainer command building with use_scaffold flag."""

    def test_scaffold_command_contains_train_scaffold(self, tmp_path):
        """use_scaffold=True → train_scaffold.py in command."""
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results",
                              max_steps=100, use_scaffold=True)
        joined = " ".join(cmd)
        assert "train_scaffold.py" in joined
        assert "--data_dir" in joined
        assert "--result_dir" in joined
        assert "--voxel_size" in joined

    def test_vanilla_command_contains_train_splat(self, tmp_path):
        """use_scaffold=False → train_splat.py in command."""
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results",
                              max_steps=100, use_scaffold=False)
        joined = " ".join(cmd)
        assert "train_splat.py" in joined
        assert "--data_dir" in joined
        assert "--result_dir" in joined
        # No voxel_size for vanilla
        assert "--voxel_size" not in joined

    def test_voxel_size_custom(self, tmp_path):
        """Custom voxel_size is passed through."""
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results",
                              max_steps=100, use_scaffold=True, voxel_size=0.005)
        joined = " ".join(cmd)
        assert "0.005" in joined

    def test_max_steps_in_command(self, tmp_path):
        """max_steps is passed as --max_steps argument."""
        from app.services.trainer import build_train_cmd
        cmd = build_train_cmd(tmp_path / "data", tmp_path / "results",
                              max_steps=5000, use_scaffold=True)
        assert "--max_steps" in cmd
        assert "5000" in cmd
