"""
FastAPI endpoint tests using Starlette TestClient.

Tests cover:
  - Health check
  - Project CRUD (create, list, get, delete)
  - Video upload
  - Frame listing / serving
  - Pipeline kick-off (extract-frames) + polling for completion
  - Pipeline cancel
  - System status
  - Error scenarios: 404, missing video, bad inputs
"""

import logging
import time
from pathlib import Path

import pytest

log = logging.getLogger("tests.api")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def create_project(client, name: str = "test-project") -> str:
    """Helper: POST /api/projects → returns project_id."""
    resp = client.post("/api/projects", json={"name": name})
    assert resp.status_code == 200, f"create_project failed: {resp.text}"
    data = resp.json()
    log.info("Created project: id=%s name=%s", data["id"], data["name"])
    return data["id"]


def delete_project(client, project_id: str):
    """Helper: DELETE /api/projects/{id}."""
    resp = client.delete(f"/api/projects/{project_id}")
    assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_ok(self, test_client):
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_is_fast(self, test_client):
        t0 = time.time()
        test_client.get("/api/health")
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# Project creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateProject:
    def test_create_returns_200(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "my-test-project"})
        assert resp.status_code == 200

    def test_create_response_has_id(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "id-check"})
        data = resp.json()
        assert "id" in data
        assert len(data["id"]) == 8  # UUID truncated to 8 chars

    def test_create_response_has_name(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "name-check"})
        assert resp.json()["name"] == "name-check"

    def test_create_initial_step_is_created(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "step-check"})
        assert resp.json()["step"] == "created"

    def test_create_builds_directory(self, test_client, patch_settings):
        resp = test_client.post("/api/projects", json={"name": "dir-check"})
        pid = resp.json()["id"]
        from app.config import settings
        proj_dir = settings.data_dir / pid
        assert proj_dir.exists()
        assert (proj_dir / "input").exists()
        assert (proj_dir / "frames").exists()
        assert (proj_dir / "colmap").exists()
        assert (proj_dir / "output").exists()

    def test_create_missing_name_returns_422(self, test_client):
        resp = test_client.post("/api/projects", json={})
        assert resp.status_code == 422

    def test_create_returns_created_at(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "ts-check"})
        assert "created_at" in resp.json()
        assert resp.json()["created_at"] is not None

    def test_create_no_error_initially(self, test_client):
        resp = test_client.post("/api/projects", json={"name": "no-error"})
        data = resp.json()
        # error field should be None or absent
        assert data.get("error") is None


# ═══════════════════════════════════════════════════════════════════════════════
# List projects
# ═══════════════════════════════════════════════════════════════════════════════

class TestListProjects:
    def test_list_returns_200(self, test_client):
        resp = test_client.get("/api/projects")
        assert resp.status_code == 200

    def test_list_returns_array(self, test_client):
        resp = test_client.get("/api/projects")
        assert isinstance(resp.json(), list)

    def test_created_project_appears_in_list(self, test_client):
        pid = create_project(test_client, "list-check")
        resp = test_client.get("/api/projects")
        ids = [p["id"] for p in resp.json()]
        assert pid in ids

    def test_list_items_have_required_fields(self, test_client):
        create_project(test_client, "fields-check")
        resp = test_client.get("/api/projects")
        for proj in resp.json():
            assert "id" in proj
            assert "name" in proj
            assert "step" in proj
            assert "created_at" in proj


# ═══════════════════════════════════════════════════════════════════════════════
# Get project
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetProject:
    def test_get_existing_project(self, test_client):
        pid = create_project(test_client, "get-test")
        resp = test_client.get(f"/api/projects/{pid}")
        assert resp.status_code == 200

    def test_get_nonexistent_returns_404(self, test_client):
        resp = test_client.get("/api/projects/00000000")
        assert resp.status_code == 404

    def test_get_returns_detail_fields(self, test_client):
        pid = create_project(test_client, "detail-test")
        resp = test_client.get(f"/api/projects/{pid}")
        data = resp.json()
        assert data["id"] == pid
        assert "video_filename" in data
        assert "frame_count" in data
        assert "sfm_points" in data
        assert "has_output" in data
        assert "training_iterations" in data

    def test_get_initial_frame_count_zero(self, test_client):
        pid = create_project(test_client, "frame-count-test")
        resp = test_client.get(f"/api/projects/{pid}")
        assert resp.json()["frame_count"] == 0

    def test_get_initial_has_output_false(self, test_client):
        pid = create_project(test_client, "has-output-test")
        resp = test_client.get(f"/api/projects/{pid}")
        assert resp.json()["has_output"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Delete project
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeleteProject:
    def test_delete_returns_ok(self, test_client):
        pid = create_project(test_client, "delete-me")
        resp = test_client.delete(f"/api/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json().get("ok") is True

    def test_deleted_project_not_in_list(self, test_client):
        pid = create_project(test_client, "delete-list-test")
        test_client.delete(f"/api/projects/{pid}")
        resp = test_client.get("/api/projects")
        ids = [p["id"] for p in resp.json()]
        assert pid not in ids

    def test_delete_removes_directory(self, test_client, patch_settings):
        from app.config import settings
        pid = create_project(test_client, "delete-dir-test")
        proj_dir = settings.data_dir / pid
        assert proj_dir.exists()
        test_client.delete(f"/api/projects/{pid}")
        assert not proj_dir.exists()

    def test_delete_nonexistent_returns_200(self, test_client):
        # Deleting a nonexistent project is idempotent (no 404)
        resp = test_client.delete("/api/projects/00000000")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Video upload
# ═══════════════════════════════════════════════════════════════════════════════

class TestVideoUpload:
    def test_upload_returns_filename_and_size(self, test_client, synthetic_video):
        pid = create_project(test_client, "upload-test")
        with open(synthetic_video, "rb") as f:
            resp = test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("test_video.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "filename" in data
        assert "size" in data
        assert data["size"] > 0
        log.info("Upload: %s → %d bytes", data["filename"], data["size"])

    def test_upload_stores_file_on_disk(self, test_client, synthetic_video, patch_settings):
        from app.config import settings
        pid = create_project(test_client, "upload-disk-test")
        with open(synthetic_video, "rb") as f:
            resp = test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("video.mp4", f, "video/mp4")},
            )
        filename = resp.json()["filename"]
        dest = settings.data_dir / pid / "input" / filename
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_upload_updates_project_video_filename(self, test_client, synthetic_video):
        pid = create_project(test_client, "upload-update-test")
        with open(synthetic_video, "rb") as f:
            test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("myvideo.mp4", f, "video/mp4")},
            )
        proj = test_client.get(f"/api/projects/{pid}").json()
        assert proj["video_filename"] is not None
        assert proj["video_filename"].endswith(".mp4")

    def test_upload_to_nonexistent_project_returns_404(self, test_client, synthetic_video):
        with open(synthetic_video, "rb") as f:
            resp = test_client.post(
                "/api/projects/00000000/upload",
                files={"file": ("video.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 404

    def test_upload_sanitizes_filename(self, test_client, synthetic_video):
        """Path traversal in filename should be stripped."""
        pid = create_project(test_client, "upload-sanitize-test")
        with open(synthetic_video, "rb") as f:
            resp = test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("../../evil.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 200
        filename = resp.json()["filename"]
        assert "/" not in filename
        assert "\\" not in filename
        assert "evil.mp4" in filename  # name preserved but path stripped


# ═══════════════════════════════════════════════════════════════════════════════
# Frame listing
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrameListing:
    def test_list_frames_empty_initially(self, test_client):
        pid = create_project(test_client, "frames-empty-test")
        resp = test_client.get(f"/api/projects/{pid}/frames")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_frames_returns_array(self, test_client):
        pid = create_project(test_client, "frames-array-test")
        resp = test_client.get(f"/api/projects/{pid}/frames")
        assert isinstance(resp.json(), list)

    def test_get_frame_nonexistent_returns_404(self, test_client):
        pid = create_project(test_client, "frame-404-test")
        resp = test_client.get(f"/api/projects/{pid}/frames/0001.jpg")
        assert resp.status_code == 404

    def test_manually_placed_frame_is_listed(self, test_client, patch_settings):
        """A frame placed directly in the frames dir should appear in the listing."""
        from app.config import settings
        pid = create_project(test_client, "frame-manual-test")
        frames_dir = settings.data_dir / pid / "frames"
        frame = frames_dir / "0001.jpg"
        # Write a minimal valid JPEG header
        frame.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\xff\xd9"
        )
        resp = test_client.get(f"/api/projects/{pid}/frames")
        names = [item["name"] for item in resp.json()]
        assert "0001.jpg" in names

    def test_get_existing_frame_returns_200(self, test_client, patch_settings):
        from app.config import settings
        pid = create_project(test_client, "frame-serve-test")
        frames_dir = settings.data_dir / pid / "frames"
        frame = frames_dir / "0001.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50 + b"\xff\xd9")
        resp = test_client.get(f"/api/projects/{pid}/frames/0001.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/jpeg")


# ═══════════════════════════════════════════════════════════════════════════════
# Samples list
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplesList:
    def test_samples_returns_200(self, test_client):
        resp = test_client.get("/api/projects/samples/list")
        assert resp.status_code == 200

    def test_samples_is_array(self, test_client):
        resp = test_client.get("/api/projects/samples/list")
        assert isinstance(resp.json(), list)

    def test_samples_have_required_fields(self, test_client):
        resp = test_client.get("/api/projects/samples/list")
        for sample in resp.json():
            assert "id" in sample
            assert "title" in sample
            assert "url" in sample


# ═══════════════════════════════════════════════════════════════════════════════
# System status
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemStatus:
    def test_status_returns_200(self, test_client):
        resp = test_client.get("/api/system/status")
        assert resp.status_code == 200

    def test_status_has_ffmpeg_field(self, test_client):
        resp = test_client.get("/api/system/status")
        data = resp.json()
        assert "ffmpeg" in data
        assert "installed" in data["ffmpeg"]

    def test_status_has_colmap_field(self, test_client):
        resp = test_client.get("/api/system/status")
        data = resp.json()
        assert "colmap" in data
        assert "installed" in data["colmap"]

    def test_status_has_cuda_field(self, test_client):
        resp = test_client.get("/api/system/status")
        data = resp.json()
        assert "cuda_available" in data
        assert isinstance(data["cuda_available"], bool)

    def test_status_has_torch_cuda(self, test_client):
        resp = test_client.get("/api/system/status")
        data = resp.json()
        assert "torch_cuda_available" in data
        log.info(
            "System status: cuda=%s torch_cuda=%s gpu=%s vram=%sMB",
            data.get("cuda_available"),
            data.get("torch_cuda_available"),
            data.get("gpu_name"),
            data.get("gpu_vram_mb"),
        )

    def test_ffmpeg_installed_in_env(self, test_client):
        """FFmpeg is expected to be available in this environment."""
        resp = test_client.get("/api/system/status")
        assert resp.json()["ffmpeg"]["installed"] is True

    def test_colmap_installed_in_env(self, test_client):
        """COLMAP is expected to be available in this environment."""
        resp = test_client.get("/api/system/status")
        assert resp.json()["colmap"]["installed"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# System logs
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemLogs:
    def test_logs_returns_200(self, test_client):
        resp = test_client.get("/api/system/logs")
        assert resp.status_code == 200

    def test_logs_response_structure(self, test_client):
        resp = test_client.get("/api/system/logs")
        data = resp.json()
        assert "lines" in data
        assert "total_lines" in data
        assert "file" in data
        assert isinstance(data["lines"], list)

    def test_logs_default_limit(self, test_client):
        resp = test_client.get("/api/system/logs")
        data = resp.json()
        assert len(data["lines"]) <= 100

    def test_logs_custom_limit(self, test_client):
        resp = test_client.get("/api/system/logs?lines=10")
        data = resp.json()
        assert len(data["lines"]) <= 10


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline: extract-frames (integration — runs real ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFramesPipeline:
    def test_extract_without_video_returns_400(self, test_client):
        pid = create_project(test_client, "extract-no-video")
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        assert resp.status_code == 400

    def test_extract_nonexistent_project_returns_404(self, test_client):
        resp = test_client.post(
            "/api/projects/00000000/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        assert resp.status_code == 404

    def test_extract_starts_successfully(self, test_client, synthetic_video):
        """Starting frame extraction returns {status: started}."""
        pid = create_project(test_client, "extract-start-test")
        with open(synthetic_video, "rb") as f:
            test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("video.mp4", f, "video/mp4")},
            )
        resp = test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_extract_completes_and_produces_frames(self, test_client, synthetic_video, patch_settings):
        """
        Full integration: upload synthetic video → extract frames → poll until
        frames_ready → verify frames exist on disk and are counted in DB.
        """
        from conftest import wait_for_step
        from app.config import settings

        pid = create_project(test_client, "extract-full-test")

        # Upload video
        with open(synthetic_video, "rb") as f:
            upload_resp = test_client.post(
                f"/api/projects/{pid}/upload",
                files={"file": ("video.mp4", f, "video/mp4")},
            )
        assert upload_resp.status_code == 200

        # Start extraction
        t0 = time.time()
        extract_resp = test_client.post(
            f"/api/projects/{pid}/pipeline/extract-frames",
            json={"fps": 2.0},
        )
        assert extract_resp.json()["status"] == "started"

        # Poll for completion
        final = wait_for_step(test_client, pid, "frames_ready", timeout=60)
        elapsed = time.time() - t0
        log.info("Frame extraction completed in %.1fs: step=%s frame_count=%d",
                 elapsed, final["step"], final["frame_count"])

        assert final["step"] == "frames_ready", f"Pipeline failed: {final.get('error')}"
        assert final["frame_count"] > 0

        # Verify frames on disk
        frames_dir = settings.data_dir / pid / "frames"
        jpg_files = sorted(frames_dir.glob("*.jpg"))
        log.info("Frames on disk: %d", len(jpg_files))
        assert len(jpg_files) > 0
        assert len(jpg_files) == final["frame_count"]

    def test_extract_uses_fps_param(self, test_client, synthetic_video, patch_settings):
        """
        Extracting at 1fps vs 4fps on a 5-second video should produce
        different frame counts.
        """
        from conftest import wait_for_step

        # 1 fps → ~5 frames
        pid1 = create_project(test_client, "extract-fps1")
        with open(synthetic_video, "rb") as f:
            test_client.post(f"/api/projects/{pid1}/upload",
                             files={"file": ("v.mp4", f, "video/mp4")})
        test_client.post(f"/api/projects/{pid1}/pipeline/extract-frames", json={"fps": 1.0})
        d1 = wait_for_step(test_client, pid1, "frames_ready", timeout=60)

        # 4 fps → ~20 frames
        pid4 = create_project(test_client, "extract-fps4")
        with open(synthetic_video, "rb") as f:
            test_client.post(f"/api/projects/{pid4}/upload",
                             files={"file": ("v.mp4", f, "video/mp4")})
        test_client.post(f"/api/projects/{pid4}/pipeline/extract-frames", json={"fps": 4.0})
        d4 = wait_for_step(test_client, pid4, "frames_ready", timeout=60)

        log.info("fps=1 → %d frames; fps=4 → %d frames", d1["frame_count"], d4["frame_count"])
        assert d4["frame_count"] > d1["frame_count"], (
            f"Expected more frames at 4fps ({d4['frame_count']}) than 1fps ({d1['frame_count']})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline: cancel
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCancel:
    def test_cancel_idle_project_returns_false(self, test_client):
        pid = create_project(test_client, "cancel-idle-test")
        resp = test_client.post(f"/api/projects/{pid}/pipeline/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False

    def test_cancel_nonexistent_project_returns_false(self, test_client):
        # NOTE: The cancel endpoint doesn't check project existence — it only
        # checks if a task is running.  Returns 200 {"cancelled": false}.
        resp = test_client.post("/api/projects/00000000/pipeline/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Error scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorScenarios:
    def test_sfm_on_project_without_frames_starts(self, test_client):
        """
        SfM on a project with no frames will start (returns 'started') but
        then fail asynchronously.  We verify the endpoint accepts it.
        NOTE: The endpoint does not validate pre-conditions synchronously.
        """
        pid = create_project(test_client, "sfm-no-frames")
        resp = test_client.post(f"/api/projects/{pid}/pipeline/sfm")
        # Endpoint starts the task; async failure happens later
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_train_on_project_without_sfm_starts(self, test_client):
        """Same pattern: train endpoint accepts request, fails async."""
        pid = create_project(test_client, "train-no-sfm")
        resp = test_client.post(f"/api/projects/{pid}/pipeline/train")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    def test_install_unknown_dep_returns_400(self, test_client):
        resp = test_client.post("/api/system/install/unknowntool")
        assert resp.status_code == 400

    def test_output_nonexistent_file_returns_404(self, test_client):
        pid = create_project(test_client, "output-404-test")
        resp = test_client.get(f"/api/projects/{pid}/output/nonexistent.ply")
        assert resp.status_code == 404
