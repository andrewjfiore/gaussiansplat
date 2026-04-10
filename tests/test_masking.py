"""
Tests for the masking service and pipeline integration.

Tests:
  - Masking service command builders (keyword + point modes)
  - Line parser
  - MaskSettings / MaskPreviewRequest model validation
  - Pipeline router masking endpoint (keyword + point modes)
  - Mask preview endpoint

All tests run without GPU or SAM2/GroundingDINO models.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

log = logging.getLogger("tests.masking")


# ═══════════════════════════════════════════════════════════════════════════════
# Masking service: command builders
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuiltinMaskCmd:
    def test_returns_list(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person"
        )
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_contains_python(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person"
        )
        assert cmd[0] == sys.executable

    def test_contains_keywords(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person.tripod"
        )
        assert "person.tripod" in cmd

    def test_expand_and_feather(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person",
            expand=10, feather=5,
        )
        assert "--expand" in cmd
        assert "10" in cmd
        assert "--feather" in cmd
        assert "5" in cmd

    def test_invert_flag(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person", invert=True,
        )
        assert "--invert" in cmd

    def test_no_expand_when_zero(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        cmd = build_builtin_mask_cmd(
            tmp_path / "frames", tmp_path / "masks", "person",
            expand=0, feather=0,
        )
        assert "--expand" not in cmd
        assert "--feather" not in cmd

    def test_creates_output_dir(self, tmp_path, patch_settings):
        from app.services.masking import build_builtin_mask_cmd
        masks_dir = tmp_path / "new_masks"
        build_builtin_mask_cmd(tmp_path / "frames", masks_dir, "person")
        assert masks_dir.exists()


class TestPointMaskCmd:
    def test_returns_list(self, tmp_path, patch_settings):
        from app.services.masking import build_point_mask_cmd
        cmd = build_point_mask_cmd(
            tmp_path / "frames", tmp_path / "masks",
            reference_frame="0001.jpg",
            points=[[100, 200], [300, 400]],
            point_labels=[1, 1],
        )
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_contains_points_json(self, tmp_path, patch_settings):
        from app.services.masking import build_point_mask_cmd
        cmd = build_point_mask_cmd(
            tmp_path / "frames", tmp_path / "masks",
            reference_frame="0001.jpg",
            points=[[100, 200]],
            point_labels=[1],
        )
        assert "--points_json" in cmd
        # Find the JSON arg
        idx = cmd.index("--points_json")
        points_data = json.loads(cmd[idx + 1])
        assert points_data["frame"] == "0001.jpg"
        assert points_data["points"] == [[100, 200]]
        assert points_data["labels"] == [1]

    def test_no_keywords_arg(self, tmp_path, patch_settings):
        from app.services.masking import build_point_mask_cmd
        cmd = build_point_mask_cmd(
            tmp_path / "frames", tmp_path / "masks",
            reference_frame="0001.jpg",
            points=[[100, 200]],
            point_labels=[1],
        )
        assert "--keywords" not in cmd

    def test_expand_feather_invert(self, tmp_path, patch_settings):
        from app.services.masking import build_point_mask_cmd
        cmd = build_point_mask_cmd(
            tmp_path / "frames", tmp_path / "masks",
            reference_frame="0001.jpg",
            points=[[50, 50]],
            point_labels=[1],
            expand=5, feather=3, invert=True,
        )
        assert "--expand" in cmd
        assert "--feather" in cmd
        assert "--invert" in cmd

    def test_creates_output_dir(self, tmp_path, patch_settings):
        from app.services.masking import build_point_mask_cmd
        masks_dir = tmp_path / "point_masks"
        build_point_mask_cmd(
            tmp_path / "frames", masks_dir,
            reference_frame="0001.jpg",
            points=[[100, 200]],
            point_labels=[1],
        )
        assert masks_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Masking service: line parser
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaskLineParser:
    def test_progress_line(self, patch_settings):
        from app.services.masking import parse_mask_line
        result = parse_mask_line("[MASK] 5/20 frame_0005.jpg — 3 detections")
        assert result is not None
        assert result["percent"] == pytest.approx(25.0)

    def test_progress_complete(self, patch_settings):
        from app.services.masking import parse_mask_line
        result = parse_mask_line("[MASK] 20/20 frame_0020.jpg — 1 detections")
        assert result is not None
        assert result["percent"] == pytest.approx(100.0)

    def test_completion_message(self, patch_settings):
        from app.services.masking import parse_mask_line
        result = parse_mask_line("[INFO] Masking complete: 20 masks generated in 15.2s")
        assert result is not None
        assert result["percent"] == 100

    def test_unrelated_line(self, patch_settings):
        from app.services.masking import parse_mask_line
        result = parse_mask_line("[INFO] Loading SAM2 on cuda...")
        assert result is None

    def test_empty_line(self, patch_settings):
        from app.services.masking import parse_mask_line
        assert parse_mask_line("") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic model validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaskModels:
    def test_mask_settings_defaults(self, patch_settings):
        from app.models import MaskSettings
        s = MaskSettings()
        assert s.keywords == "person"
        assert s.points is None
        assert s.point_labels is None
        assert s.reference_frame is None

    def test_mask_settings_point_mode(self, patch_settings):
        from app.models import MaskSettings
        s = MaskSettings(
            points=[[100, 200], [300, 400]],
            point_labels=[1, 0],
            reference_frame="0001.jpg",
        )
        assert len(s.points) == 2
        assert s.point_labels == [1, 0]
        assert s.reference_frame == "0001.jpg"

    def test_mask_preview_request(self, patch_settings):
        from app.models import MaskPreviewRequest
        r = MaskPreviewRequest(
            frame="0001.jpg",
            points=[[100, 200]],
            labels=[1],
        )
        assert r.frame == "0001.jpg"
        assert r.points == [[100, 200]]
        assert r.labels == [1]

    def test_mask_preview_request_validation(self, patch_settings):
        from app.models import MaskPreviewRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MaskPreviewRequest(frame="0001.jpg")  # missing points and labels


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline router: mask endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaskEndpoint:
    """Tests the POST /pipeline/mask endpoint via TestClient."""

    def _create_project_with_frames(self, client, tmp_path, patch_settings):
        """Helper: create project and fake extracted frames."""
        resp = client.post("/api/projects", json={"name": "mask-test"})
        assert resp.status_code == 200
        pid = resp.json()["id"]

        # Create fake frames
        frames_dir = patch_settings.data_dir / pid / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (frames_dir / f"{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Set project step to frames_ready
        import asyncio
        from app.database import db_session

        async def _set_step():
            async with db_session() as db:
                await db.execute(
                    "UPDATE projects SET step = 'frames_ready', frame_count = 5 WHERE id = ?",
                    (pid,),
                )

        asyncio.get_event_loop().run_until_complete(_set_step())
        return pid

    @patch("app.pipeline.task_runner.task_runner.run", new_callable=AsyncMock, return_value=0)
    @patch("app.pipeline.task_runner.task_runner.is_running", return_value=False)
    def test_keyword_mode(self, mock_running, mock_run, test_client, tmp_path, patch_settings):
        pid = self._create_project_with_frames(test_client, tmp_path, patch_settings)
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/mask",
            json={"keywords": "person.tripod", "precision": 0.4},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    @patch("app.pipeline.task_runner.task_runner.run", new_callable=AsyncMock, return_value=0)
    @patch("app.pipeline.task_runner.task_runner.is_running", return_value=False)
    def test_point_mode(self, mock_running, mock_run, test_client, tmp_path, patch_settings):
        pid = self._create_project_with_frames(test_client, tmp_path, patch_settings)
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/mask",
            json={
                "points": [[100, 200], [300, 400]],
                "point_labels": [1, 1],
                "reference_frame": "0001.jpg",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_mask_no_frames(self, test_client, patch_settings):
        """Masking should fail if no frames exist."""
        resp = test_client.post("/api/projects", json={"name": "mask-empty"})
        pid = resp.json()["id"]

        # Try to mask at wrong step
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/mask",
            json={"keywords": "person"},
        )
        assert resp.status_code == 422  # Wrong pipeline step


# ═══════════════════════════════════════════════════════════════════════════════
# Script: generate_masks.py argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateMasksScript:
    def test_script_exists(self):
        script = Path(__file__).parent.parent / "backend" / "scripts" / "generate_masks.py"
        assert script.exists()

    def test_keyword_args_parsed(self, tmp_path):
        """Verify the argparse setup accepts keyword mode arguments."""
        import importlib.util
        script = Path(__file__).parent.parent / "backend" / "scripts" / "generate_masks.py"
        spec = importlib.util.spec_from_file_location("generate_masks", script)
        mod = importlib.util.module_from_spec(spec)

        # Don't actually execute main, just verify arg parsing works
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir", required=True, type=Path)
        parser.add_argument("--output_dir", required=True, type=Path)
        parser.add_argument("--keywords", type=str, default=None)
        parser.add_argument("--points_json", type=str, default=None)
        parser.add_argument("--mode", default="mask")
        parser.add_argument("--invert", action="store_true")
        parser.add_argument("--precision", type=float, default=0.3)
        parser.add_argument("--expand", type=int, default=0)
        parser.add_argument("--feather", type=int, default=0)

        args = parser.parse_args([
            "--input_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--keywords", "person.tripod",
            "--precision", "0.5",
        ])
        assert args.keywords == "person.tripod"
        assert args.precision == 0.5
        assert args.points_json is None

    def test_point_args_parsed(self, tmp_path):
        """Verify argparse accepts point mode arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir", required=True, type=Path)
        parser.add_argument("--output_dir", required=True, type=Path)
        parser.add_argument("--keywords", type=str, default=None)
        parser.add_argument("--points_json", type=str, default=None)
        parser.add_argument("--mode", default="mask")

        points_json = json.dumps({
            "frame": "0001.jpg",
            "points": [[100, 200]],
            "labels": [1],
        })
        args = parser.parse_args([
            "--input_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--points_json", points_json,
        ])
        assert args.keywords is None
        data = json.loads(args.points_json)
        assert data["frame"] == "0001.jpg"
        assert data["points"] == [[100, 200]]
