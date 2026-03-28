import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from ..ws.manager import manager

# Maximum time (seconds) a single subprocess can run before being killed
DEFAULT_TIMEOUT = 3600  # 1 hour
READLINE_TIMEOUT = 300  # 5 min without any output = considered hung


class TaskRunner:
    def __init__(self):
        self._processes: dict[str, asyncio.subprocess.Process] = {}

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
    ) -> int:
        if self.is_running(project_id):
            raise RuntimeError(f"Project {project_id} already has a running task")

        await manager.send_log(project_id, f"$ {' '.join(cmd)}")
        await manager.send_status(project_id, step, "running")

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
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
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
