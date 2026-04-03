import os
import shutil
import sys
from pathlib import Path
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parents[2]


def _find_executable(base_dir: Path, names: list[str]) -> Path | None:
    """Search for an executable: direct children first, then recursively."""
    if not base_dir.exists():
        return None
    for name in names:
        p = base_dir / name
        if p.exists():
            return p
    lower_names = {n.lower() for n in names}
    for path in base_dir.rglob("*"):
        if path.is_file() and path.name.lower() in lower_names:
            return path
    return None


class Settings(BaseSettings):
    project_name: str = "GaussianSplat Studio"
    base_dir: Path = _ROOT
    data_dir: Path = _ROOT / "data" / "projects"
    tools_dir: Path = _ROOT / "tools"
    frontend_url: str = "http://localhost:3000"
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = _ROOT / "data" / "gaussiansplat.db"
    log_dir: Path = _ROOT / "data" / "logs"
    max_concurrent_gpu_tasks: int = 1
    automasker_exe: str = ""  # path to external AutoMasker.exe (optional)

    @property
    def log_file(self) -> Path:
        return self.log_dir / "gaussiansplat.log"

    @property
    def is_windows(self) -> bool:
        return sys.platform == "win32"

    @property
    def colmap_bin(self) -> Path:
        # 1. Check env var set by gs.py launcher
        env_path = os.environ.get("COLMAP_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # 2. Search tools dir recursively (handles bin/ nesting)
        if self.is_windows:
            found = _find_executable(
                self.tools_dir / "colmap",
                ["colmap.exe", "COLMAP.bat", "colmap.bat"]
            )
            if found:
                return found

        # 3. Check system PATH
        system = shutil.which("colmap")
        if system:
            return Path(system)

        # 4. Fallback (may not exist)
        if self.is_windows:
            return self.tools_dir / "colmap" / "COLMAP.bat"
        return Path("colmap")

    @property
    def ffmpeg_bin(self) -> Path:
        # 1. Check env var set by gs.py launcher
        env_path = os.environ.get("FFMPEG_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # 2. Search tools dir recursively
        if self.is_windows:
            found = _find_executable(
                self.tools_dir / "ffmpeg",
                ["ffmpeg.exe"]
            )
            if found:
                return found

        # 3. Check system PATH
        system = shutil.which("ffmpeg")
        if system:
            return Path(system)

        # 4. Fallback
        if self.is_windows:
            return self.tools_dir / "ffmpeg" / "ffmpeg.exe"
        return Path("ffmpeg")

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
