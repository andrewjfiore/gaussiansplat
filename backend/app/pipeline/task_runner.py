import asyncio
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from ..ws.manager import manager
from ..config import settings

# Maximum time (seconds) a single subprocess can run before being killed
DEFAULT_TIMEOUT = 3600  # 1 hour
READLINE_TIMEOUT = 300  # 5 min without any output = considered hung


def _build_gpu_env() -> dict[str, str]:
    """Build environment dict with MSVC + CUDA paths for JIT compilation on Windows."""
    env = os.environ.copy()
    if sys.platform != "win32":
        return env

    # Find MSVC
    vs_base = Path(r"C:\Program Files (x86)\Microsoft Visual Studio")
    msvc_bin = None
    msvc_root = None
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        for year in ("2022", "2019"):
            tools = vs_base / year / edition / "VC" / "Tools" / "MSVC"
            if tools.exists():
                versions = sorted(tools.iterdir(), reverse=True)
                if versions:
                    msvc_root = versions[0]
                    msvc_bin = msvc_root / "bin" / "Hostx64" / "x64"
                    break
        if msvc_bin:
            break

    # Find Windows SDK
    sdk_base = Path(r"C:\Program Files (x86)\Windows Kits\10")
    sdk_ver = None
    if (sdk_base / "Include").exists():
        versions = sorted((sdk_base / "Include").iterdir(), reverse=True)
        if versions:
            sdk_ver = versions[0].name

    # Find CUDA
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    cuda_home = None
    if cuda_base.exists():
        versions = sorted(cuda_base.iterdir(), reverse=True)
        if versions:
            cuda_home = versions[0]

    # Set PATH
    extra_paths = []
    if msvc_bin and msvc_bin.exists():
        extra_paths.append(str(msvc_bin))
    if cuda_home:
        extra_paths.append(str(cuda_home / "bin"))
    if extra_paths:
        env["PATH"] = ";".join(extra_paths) + ";" + env.get("PATH", "")

    # Set INCLUDE and LIB for MSVC
    if msvc_root and sdk_ver:
        env["INCLUDE"] = ";".join([
            str(msvc_root / "include"),
            str(sdk_base / "Include" / sdk_ver / "ucrt"),
            str(sdk_base / "Include" / sdk_ver / "shared"),
            str(sdk_base / "Include" / sdk_ver / "um"),
        ])
        env["LIB"] = ";".join([
            str(msvc_root / "lib" / "x64"),
            str(sdk_base / "Lib" / sdk_ver / "ucrt" / "x64"),
            str(sdk_base / "Lib" / sdk_ver / "um" / "x64"),
        ])

    if cuda_home:
        env["CUDA_HOME"] = str(cuda_home)

    # Target only the installed GPU arch to speed up compilation
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    return env


class TaskRunner:
    def __init__(self):
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._gpu_semaphore: asyncio.Semaphore | None = None

    def _get_gpu_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be created inside an event loop)."""
        if self._gpu_semaphore is None:
            self._gpu_semaphore = asyncio.Semaphore(settings.max_concurrent_gpu_tasks)
        return self._gpu_semaphore

    def is_running(self, project_id: str) -> bool:
        proc = self._processes.get(project_id)
        return proc is not None and proc.returncode is None

    async def run(
        self,
        project_id: str,
        cmd: list[str],
        cwd: Optional[Path] = None,
        step: str = "",
        substep: str = "",
        line_parser: Optional[Callable[[str], Optional[dict]]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        requires_gpu: bool = False,
    ) -> int:
        if self.is_running(project_id):
            raise RuntimeError(f"Project {project_id} already has a running task")

        # Acquire GPU semaphore if needed (prevents OOM from concurrent training)
        gpu_acquired = False
        if requires_gpu:
            sem = self._get_gpu_semaphore()
            if sem.locked():
                await manager.send_status(project_id, step, "queued")
                await manager.send_log(project_id, "[INFO] Waiting for GPU (another task is running)...")
            await sem.acquire()
            gpu_acquired = True

        await manager.send_log(project_id, f"$ {' '.join(cmd)}")
        await manager.send_status(project_id, step, "running")

        # Build env with MSVC + CUDA paths for GPU tasks that need JIT compilation
        proc_env = _build_gpu_env() if requires_gpu else None

        # On Windows, .bat/.cmd files must go through shell
        is_bat = sys.platform == "win32" and cmd[0].lower().endswith((".bat", ".cmd"))

        if is_bat:
            # create_subprocess_exec + shell=True is wrong; use create_subprocess_shell
            cmdline = subprocess.list2cmdline(cmd)
            proc = await asyncio.create_subprocess_shell(
                cmdline,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=proc_env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=proc_env,
            )
        self._processes[project_id] = proc

        try:
            assert proc.stdout is not None

            async def _read_lines():
                """Read lines with a per-line timeout to detect hangs."""
                while True:
                    try:
                        line_bytes = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=READLINE_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        await manager.send_log(
                            project_id,
                            f"[WARNING] No output for {READLINE_TIMEOUT}s — process may be hung",
                        )
                        # Don't break — give it more time, but log the warning
                        # Check if process is still alive
                        if proc.returncode is not None:
                            break
                        continue

                    if not line_bytes:
                        break

                    # Handle \r-only lines (progress bars) by splitting on both \r and \n
                    text = line_bytes.decode("utf-8", errors="replace")
                    for segment in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                        line = segment.rstrip()
                        if not line:
                            continue
                        await manager.send_log(project_id, line)
                        if line_parser:
                            parsed = line_parser(line)
                            if parsed:
                                if "percent" in parsed:
                                    await manager.send_progress(
                                        project_id, step, substep, parsed["percent"]
                                    )
                                if "metric" in parsed:
                                    await manager.send_metric(
                                        project_id, **parsed["metric"]
                                    )

            # Run with a global timeout
            try:
                await asyncio.wait_for(_read_lines(), timeout=timeout)
            except asyncio.TimeoutError:
                await manager.send_log(
                    project_id,
                    f"[ERROR] Process timed out after {timeout}s — killing",
                )
                proc.kill()
                await proc.wait()
                await manager.send_status(project_id, step, "failed", "Process timed out")
                return -1

            await proc.wait()
            rc = proc.returncode or 0

            if rc == 0:
                await manager.send_status(project_id, step, "completed")
            else:
                await manager.send_status(
                    project_id, step, "failed", f"Exit code {rc}"
                )

            return rc
        finally:
            self._processes.pop(project_id, None)
            if gpu_acquired:
                self._get_gpu_semaphore().release()

    async def cancel(self, project_id: str) -> bool:
        proc = self._processes.get(project_id)
        if proc is None or proc.returncode is not None:
            return False
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        self._processes.pop(project_id, None)
        await manager.send_status(project_id, "", "cancelled")
        return True


task_runner = TaskRunner()
