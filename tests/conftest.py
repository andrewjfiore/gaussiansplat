"""
Shared pytest fixtures and configuration for GaussianSplat Studio test suite.

Fixtures:
  test_tmp_dir     — session-scoped temp directory for all test I/O
  patch_settings   — redirects backend settings to temp dirs (autouse)
  test_client      — FastAPI TestClient with DB initialized via lifespan
  synthetic_video  — path to a pre-generated 5-second synthetic MP4
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FIXTURES_DIR = TESTS_DIR / "fixtures"
LOGS_DIR = TESTS_DIR / "logs"

# Add backend to sys.path so tests can `from app.xxx import ...`
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ── Logging: console + timestamped log file ───────────────────────────────────
LOGS_DIR.mkdir(exist_ok=True)
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"test_run_{_timestamp}.log"

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s")
_file_handler.setFormatter(_fmt)
logging.getLogger().addHandler(_file_handler)
logging.getLogger().setLevel(logging.DEBUG)

log = logging.getLogger("tests.conftest")
log.info("=" * 70)
log.info("Test session starting. Log file: %s", LOG_FILE)
log.info("Project root: %s", PROJECT_ROOT)
log.info("Python: %s", sys.executable)
log.info("=" * 70)


# ── Custom marks ──────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (requires real video data or GPU compute)",
    )


def pytest_runtest_setup(item):
    log.info("── TEST START: %s", item.nodeid)


def pytest_runtest_teardown(item, nextitem):
    log.info("── TEST END:   %s", item.nodeid)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_tmp_dir(tmp_path_factory):
    """Session-scoped temp directory. All backend I/O is redirected here."""
    d = tmp_path_factory.mktemp("gaussiansplat")
    log.info("Temp dir: %s", d)
    return d


@pytest.fixture(scope="session", autouse=True)
def patch_settings(test_tmp_dir):
    """
    Redirect backend Settings singleton to a clean temp directory.
    Must run before anything imports `app.database` or starts the server.
    Returns the patched Settings instance.
    """
    from app.config import settings  # noqa: PLC0415

    data_dir = test_tmp_dir / "projects"
    db_path = test_tmp_dir / "test.db"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Pydantic v2 BaseSettings allows direct attribute assignment
    settings.data_dir = data_dir
    settings.db_path = db_path
    settings.log_dir = LOGS_DIR

    log.info(
        "Settings patched → data_dir=%s  db_path=%s",
        settings.data_dir,
        settings.db_path,
    )
    return settings


@pytest.fixture(scope="session")
def test_client(patch_settings):
    """
    Starlette TestClient for the FastAPI app.
    Using it as a context manager runs the lifespan (which calls init_db).
    Session-scoped so the DB is shared across all api tests.
    """
    from fastapi.testclient import TestClient
    from app.main import app  # noqa: PLC0415

    log.info("Starting FastAPI TestClient …")
    with TestClient(app, raise_server_exceptions=True) as client:
        log.info("TestClient ready")
        yield client
    log.info("TestClient closed")


@pytest.fixture(scope="session")
def synthetic_video():
    """
    Return a path to a 5-second synthetic test video.
    Uses the pre-generated file in tests/fixtures/ if available;
    generates it on-the-fly with ffmpeg otherwise.
    """
    target = FIXTURES_DIR / "small_video.mp4"
    if target.exists() and target.stat().st_size > 1000:
        log.info("Using cached synthetic video: %s", target)
        return target

    log.info("Generating synthetic video → %s", target)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=10",
        "-t", "5",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        pytest.fail(f"ffmpeg failed to generate synthetic video:\n{result.stderr}")
    size = target.stat().st_size
    log.info("Synthetic video generated in %.1fs: %s (%d bytes)", elapsed, target, size)
    return target


# ── Helpers (importable by tests) ─────────────────────────────────────────────

def wait_for_step(client, project_id: str, expected_step: str, timeout: float = 30.0, poll: float = 0.5) -> dict:
    """
    Poll GET /api/projects/{project_id} until the pipeline step reaches
    `expected_step` or 'failed'.  Returns the final project detail dict.
    Raises AssertionError on timeout.
    """
    deadline = time.time() + timeout
    last_step = None
    while time.time() < deadline:
        resp = client.get(f"/api/projects/{project_id}")
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}"
        data = resp.json()
        step = data["step"]
        if step != last_step:
            log.info("  project %s → step=%s", project_id, step)
            last_step = step
        if step == expected_step or step == "failed":
            return data
        time.sleep(poll)
    raise AssertionError(
        f"Timed out waiting for step={expected_step!r} after {timeout}s; last step={last_step!r}"
    )
