#!/usr/bin/env python3
"""
GaussianSplat Studio — All-in-one CLI launcher.

Usage:
    python gs.py              # Full setup + launch (interactive wizard)
    python gs.py run          # Skip setup, just start servers
    python gs.py setup        # Run setup only (no launch)
    python gs.py doctor       # Check all dependencies
    python gs.py nuke         # Wipe all project data and tool downloads
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
import venv
import zipfile
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
TOOLS_DIR = ROOT / "tools"
DATA_DIR = ROOT / "data"
VENV_DIR = BACKEND_DIR / ".venv"
IS_WIN = sys.platform == "win32"
PY = str(VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python"))
PIP = str(VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip"))
NPM = shutil.which("npm")
NODE = shutil.which("node")

# ── ANSI helpers ─────────────────────────────────────────────────────────────
if IS_WIN:
    os.system("")  # enable VT100 on Windows
    # Fix Unicode output on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t: str)  -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str)    -> str: return _c("31", t)
def cyan(t: str)   -> str: return _c("36", t)
def bold(t: str)   -> str: return _c("1", t)
def dim(t: str)    -> str: return _c("2", t)

OK   = green("✓")
WARN = yellow("⚠")
FAIL = red("✗")
ARROW = cyan("→")

def banner():
    print()
    print(bold(cyan("  ╔══════════════════════════════════════════╗")))
    print(bold(cyan("  ║       GaussianSplat Studio  CLI         ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════╝")))
    print()


# ── Utility ──────────────────────────────────────────────────────────────────

def run_quiet(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        # On Windows, .cmd/.bat files (like npm) need shell=True
        use_shell = IS_WIN and cmd and any(cmd[0].lower().endswith(ext) for ext in (".cmd", ".bat"))
        # Also use shell on Windows if the command isn't a full path (let shell resolve it)
        if IS_WIN and cmd and not os.path.isabs(cmd[0]) and not Path(cmd[0]).suffix:
            use_shell = True
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=use_shell)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return -1, str(e)


def _find_local_executable(base: Path, names: list[str]) -> Optional[str]:
    """Search for an executable in a directory, checking direct children first,
    then recursively. Handles ZIP layouts that nest inside subdirectories."""
    if not base.exists():
        return None

    # Direct children first (fast path)
    for name in names:
        p = base / name
        if p.exists():
            return str(p)

    # Recursive search (handles bin/, nested dirs, etc.)
    lower_names = {n.lower() for n in names}
    for path in base.rglob("*"):
        if path.is_file() and path.name.lower() in lower_names:
            return str(path)

    return None


# ── Dependency checks ────────────────────────────────────────────────────────

def check_python() -> dict:
    v = sys.version_info
    ok = v >= (3, 10)
    return {"name": "Python", "ok": ok, "ver": f"{v.major}.{v.minor}.{v.micro}",
            "hint": "Install Python 3.10+ from https://python.org/downloads/"}

def check_node() -> dict:
    rc, out = run_quiet(["node", "--version"]) if NODE else (-1, "")
    ok = rc == 0
    ver = out.strip().lstrip("v") if ok else None
    return {"name": "Node.js", "ok": ok, "ver": ver,
            "hint": "Install Node.js 18+ from https://nodejs.org/"}

def check_npm() -> dict:
    # On Windows, npm is a .cmd — shutil.which may miss it, so just try running it
    rc, out = run_quiet(["npm", "--version"])
    return {"name": "npm", "ok": rc == 0, "ver": out.strip() if rc == 0 else None,
            "hint": "Comes with Node.js — install Node first"}

def check_git() -> dict:
    g = shutil.which("git")
    rc, out = run_quiet(["git", "--version"]) if g else (-1, "")
    return {"name": "Git", "ok": rc == 0, "ver": out.replace("git version ", "").strip() if rc == 0 else None,
            "hint": "Install Git from https://git-scm.com/downloads"}

def check_nvidia() -> dict:
    rc, out = run_quiet(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                         "--format=csv,noheader,nounits"])
    if rc != 0:
        return {"name": "NVIDIA Driver", "ok": False, "ver": None,
                "hint": "Install NVIDIA drivers from https://nvidia.com/drivers"}
    parts = out.split("\n")[0].split(",")
    gpu = parts[0].strip() if parts else "Unknown"
    vram = parts[1].strip() + " MB" if len(parts) > 1 else ""
    driver = parts[2].strip() if len(parts) > 2 else ""
    return {"name": "NVIDIA Driver", "ok": True, "ver": f"{gpu} ({vram}, driver {driver})"}

def check_ffmpeg_local() -> dict:
    candidates = []
    if IS_WIN:
        local = _find_local_executable(TOOLS_DIR / "ffmpeg", ["ffmpeg.exe"])
        if local:
            candidates.append(local)
    candidates.append("ffmpeg")

    for c in candidates:
        rc, out = run_quiet([c, "-version"])
        if rc == 0:
            ver = out.splitlines()[0] if out else "unknown"
            return {"name": "FFmpeg", "ok": True, "ver": ver, "path": c}

    return {"name": "FFmpeg", "ok": False, "ver": None,
            "hint": "Will be auto-installed" if IS_WIN else "sudo apt install ffmpeg"}

def check_colmap_local() -> dict:
    candidates = []
    if IS_WIN:
        local = _find_local_executable(
            TOOLS_DIR / "colmap",
            ["colmap.exe", "COLMAP.bat", "colmap.bat"]
        )
        if local:
            candidates.append(local)
    candidates.append("colmap")

    for c in candidates:
        # Try several probes — different COLMAP builds respond to different flags
        for probe in ([c, "-h"], [c, "help"], [c]):
            rc, out = run_quiet(probe)
            if rc == 0 or "colmap" in out.lower():
                return {"name": "COLMAP", "ok": True, "ver": "detected", "path": c}

    return {"name": "COLMAP", "ok": False, "ver": None,
            "hint": "Will be auto-installed" if IS_WIN else "sudo apt install colmap"}

def check_cuda_runtime() -> dict:
    """Check actual PyTorch CUDA usability — this is what matters for training."""
    if not Path(PY).exists():
        return {"name": "CUDA Runtime", "ok": False, "ver": None, "hint": "venv not created yet"}

    code = (
        'import json\n'
        'try:\n'
        '    import torch\n'
        '    out = {\n'
        '        "torch": torch.__version__,\n'
        '        "cuda_available": torch.cuda.is_available(),\n'
        '        "torch_cuda": getattr(torch.version, "cuda", None),\n'
        '        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,\n'
        '        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() else None,\n'
        '    }\n'
        '    print(json.dumps(out))\n'
        'except Exception as e:\n'
        '    print(json.dumps({"error": str(e)}))\n'
    )
    rc, out = run_quiet([PY, "-c", code], timeout=30)
    if rc != 0:
        return {"name": "CUDA Runtime", "ok": False, "ver": None,
                "hint": "PyTorch not installed or failed to query CUDA"}

    try:
        info = json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {"name": "CUDA Runtime", "ok": False, "ver": None,
                "hint": "Could not parse PyTorch CUDA output"}

    if info.get("error"):
        return {"name": "CUDA Runtime", "ok": False, "ver": None,
                "hint": info["error"]}

    if info.get("cuda_available"):
        ver = f"torch {info.get('torch')}, CUDA {info.get('torch_cuda')}, {info.get('device_name')}"
        return {"name": "CUDA Runtime", "ok": True, "ver": ver}

    return {"name": "CUDA Runtime", "ok": False,
            "ver": f"torch {info.get('torch')} (CPU only)",
            "hint": "torch.cuda.is_available() is False — training will fail"}

def check_gsplat_venv() -> dict:
    if not Path(PY).exists():
        return {"name": "gsplat", "ok": False, "ver": None, "hint": "venv not created yet"}
    rc, out = run_quiet([PY, "-c", "import gsplat; print(gsplat.__version__)"])
    if rc != 0:
        return {"name": "gsplat", "ok": False, "ver": None, "hint": "Will be installed during setup"}
    return {"name": "gsplat", "ok": True, "ver": out.strip()}

def print_check(chk: dict):
    icon = OK if chk["ok"] else FAIL
    ver = dim(f"({chk['ver']})") if chk.get("ver") else ""
    hint = dim(f"  {ARROW} {chk['hint']}") if chk.get("hint") and not chk["ok"] else ""
    print(f"  {icon} {chk['name']} {ver}{hint}")

def doctor():
    """Run all dependency checks and print a report."""
    print(bold("\n  System Requirements\n"))
    checks = [check_python(), check_node(), check_npm(), check_git(), check_nvidia()]
    for c in checks:
        print_check(c)

    print(bold("\n  Pipeline Tools\n"))
    tools = [check_ffmpeg_local(), check_colmap_local()]
    for c in tools:
        print_check(c)

    print(bold("\n  Python Packages (venv)\n"))
    py_checks = [check_cuda_runtime(), check_gsplat_venv()]
    for c in py_checks:
        print_check(c)

    # Runtime resolution — show exactly which executables will be used
    print(bold("\n  Runtime Resolution\n"))
    ff = check_ffmpeg_local()
    cm = check_colmap_local()
    if ff.get("path"):
        print(f"  {OK} FFmpeg  {dim(ff['path'])}")
    else:
        print(f"  {FAIL} FFmpeg  {dim('not resolved')}")
    if cm.get("path"):
        print(f"  {OK} COLMAP  {dim(cm['path'])}")
    else:
        print(f"  {FAIL} COLMAP  {dim('not resolved')}")

    all_ok = all(c["ok"] for c in checks + tools + py_checks)
    print()
    if all_ok:
        print(f"  {OK} {green('All dependencies satisfied!')}")
    else:
        n_fail = sum(1 for c in checks + tools + py_checks if not c["ok"])
        print(f"  {WARN} {yellow(f'{n_fail} issue(s) found — run')} {bold('python gs.py setup')} {yellow('to fix')}")
    print()
    return all_ok


# ── Install helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str = ""):
    """Download with progress bar using streaming to handle GitHub redirects."""
    print(f"  {ARROW} Downloading {label or url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "GaussianSplat-Studio/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    bar = "█" * (pct // 3) + "░" * (33 - pct // 3)
                    print(f"\r    [{bar}] {pct}%  ({downloaded // 1048576}/{total // 1048576} MB)", end="", flush=True)
                else:
                    print(f"\r    {downloaded // 1048576} MB downloaded...", end="", flush=True)
    # Ensure we show 100%
    if total > 0:
        bar = "█" * 33
        print(f"\r    [{bar}] 100%  ({total // 1048576}/{total // 1048576} MB)")
    else:
        print(f"\r    {downloaded // 1048576} MB downloaded — done.")
    print()

def install_ffmpeg():
    print(f"\n{bold('  Installing FFmpeg...')}")
    if IS_WIN:
        dest = TOOLS_DIR / "ffmpeg"
        dest.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zp = dest / "ffmpeg.zip"
        _download(url, zp, "FFmpeg")
        print(f"  {ARROW} Extracting...")
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if member.endswith("ffmpeg.exe") or member.endswith("ffprobe.exe"):
                    data = zf.read(member)
                    (dest / Path(member).name).write_bytes(data)
        zp.unlink()

        exe = _find_local_executable(dest, ["ffmpeg.exe"])
        if exe:
            print(f"  {OK} FFmpeg installed — {exe}")
        else:
            print(f"  {WARN} FFmpeg extracted but ffmpeg.exe not found in {dest}")
    else:
        print(f"  {ARROW} sudo apt install -y ffmpeg")
        subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        print(f"  {OK} FFmpeg installed")

def install_colmap():
    print(f"\n{bold('  Installing COLMAP...')}")
    if IS_WIN:
        dest = TOOLS_DIR / "colmap"
        dest.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/colmap/colmap/releases/download/4.0.2/colmap-x64-windows-cuda.zip"
        zp = dest / "colmap.zip"
        _download(url, zp, "COLMAP")
        print(f"  {ARROW} Extracting...")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
        zp.unlink()

        # Verify extraction succeeded
        exe = _find_local_executable(dest, ["colmap.exe"])
        if exe:
            print(f"  {OK} COLMAP installed — {exe}")
        else:
            print(f"  {FAIL} COLMAP extracted but colmap.exe not found in {dest}")
            raise RuntimeError("COLMAP extraction failed — colmap.exe not found")
    else:
        print(f"  {ARROW} sudo apt install -y colmap")
        subprocess.run(["sudo", "apt", "install", "-y", "colmap"], check=True)
        print(f"  {OK} COLMAP installed")


def setup_venv():
    print(f"\n{bold('  Setting up Python virtual environment...')}")
    if not VENV_DIR.exists():
        print(f"  {ARROW} Creating venv at {VENV_DIR}")
        venv.create(str(VENV_DIR), with_pip=True)
    else:
        print(f"  {OK} venv already exists")

    # Install backend deps
    print(f"  {ARROW} Installing backend requirements...")
    subprocess.run([PIP, "install", "-r", str(BACKEND_DIR / "requirements.txt")], check=True)
    print(f"  {OK} Backend dependencies installed")

def install_pytorch():
    print(f"\n{bold('  Installing PyTorch with CUDA...')}")
    chk = check_cuda_runtime()
    if chk["ok"]:
        ver = chk["ver"]
        print(f"  {OK} PyTorch + CUDA already working {dim(f'({ver})')}")
        return

    # Install PyTorch with CUDA 12.4 (latest stable)
    print(f"  {ARROW} Installing PyTorch (this may take several minutes)...")
    cmd = [PIP, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
    subprocess.run(cmd, check=True)
    print(f"  {OK} PyTorch installed")

def install_gsplat():
    print(f"\n{bold('  Installing gsplat...')}")
    chk = check_gsplat_venv()
    if chk["ok"]:
        ver = chk["ver"]
        print(f"  {OK} gsplat already installed {dim(f'({ver})')}")
        return
    print(f"  {ARROW} Installing gsplat...")
    subprocess.run([PIP, "install", "gsplat"], check=True)
    print(f"  {OK} gsplat installed")

def setup_frontend():
    print(f"\n{bold('  Setting up frontend...')}")
    if not NODE:
        print(f"  {FAIL} Node.js not found — skipping frontend setup")
        return False
    pkg_lock = FRONTEND_DIR / "node_modules"
    if pkg_lock.exists():
        print(f"  {OK} node_modules already exists")
    else:
        print(f"  {ARROW} Running npm install...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True, shell=IS_WIN)
        print(f"  {OK} Frontend dependencies installed")
    return True

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "projects").mkdir(exist_ok=True)


# ── Setup wizard ─────────────────────────────────────────────────────────────

def setup(skip_optional_prompts: bool = False):
    banner()
    print(bold("  Running setup wizard...\n"))

    ensure_dirs()

    # 1. Check hard requirements
    py = check_python()
    nd = check_node()
    if not py["ok"]:
        print(f"  {FAIL} {red('Python 3.10+ is required. Please install it and re-run.')}")
        sys.exit(1)
    if not nd["ok"]:
        print(f"  {FAIL} {red('Node.js is required. Please install it and re-run.')}")
        sys.exit(1)
    print(f"  {OK} Python {py['ver']}")
    print(f"  {OK} Node.js {nd['ver']}")

    # 2. Python venv + backend deps
    setup_venv()

    # 3. PyTorch
    install_pytorch()

    # 4. gsplat
    install_gsplat()

    # 5. FFmpeg
    ff = check_ffmpeg_local()
    if not ff["ok"]:
        install_ffmpeg()
    else:
        print(f"\n  {OK} FFmpeg already available")

    # 6. COLMAP
    cm = check_colmap_local()
    if not cm["ok"]:
        install_colmap()
    else:
        print(f"\n  {OK} COLMAP already available")

    # 7. Frontend
    setup_frontend()

    # Final report
    print(f"\n{bold(green('  ══════════════════════════════════════'))}")
    print(f"{bold(green('    Setup complete!'))}")
    print(f"{bold(green('  ══════════════════════════════════════'))}")
    print()
    doctor()
    print(f"  Run {bold('python gs.py run')} to start the app.\n")


# ── Runtime environment ──────────────────────────────────────────────────────

def build_runtime_env() -> dict:
    """Build an environment dict that prepends local tool dirs to PATH,
    so the backend subprocess can find ffmpeg/colmap without system installs."""
    env = os.environ.copy()
    extra_paths: list[str] = []

    ffmpeg_dir = TOOLS_DIR / "ffmpeg"
    colmap_dir = TOOLS_DIR / "colmap"

    if ffmpeg_dir.exists():
        extra_paths.append(str(ffmpeg_dir))

    if colmap_dir.exists():
        extra_paths.append(str(colmap_dir))
        # Also include nested bin/lib dirs (COLMAP 4.x puts exe in bin/)
        for p in colmap_dir.rglob("*"):
            if p.is_dir() and p.name.lower() in {"bin", "lib"}:
                extra_paths.append(str(p))

    # Prepend so local tools win over stale system PATH entries
    if extra_paths:
        env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])

    # Explicit hints the backend can use as fallback
    ffmpeg_exe = _find_local_executable(
        ffmpeg_dir, ["ffmpeg.exe"] if IS_WIN else ["ffmpeg"]
    )
    colmap_exe = _find_local_executable(
        colmap_dir,
        ["colmap.exe", "COLMAP.bat", "colmap.bat"] if IS_WIN else ["colmap"]
    )

    if ffmpeg_exe:
        env["FFMPEG_PATH"] = ffmpeg_exe
    if colmap_exe:
        env["COLMAP_PATH"] = colmap_exe

    return env


# ── Server launcher ─────────────────────────────────────────────────────────

def run():
    banner()
    print(bold("  Starting servers...\n"))

    # Quick sanity
    if not VENV_DIR.exists():
        print(f"  {WARN} venv not found. Running setup first...\n")
        setup()

    procs: list[subprocess.Popen] = []
    runtime_env = build_runtime_env()

    # Show what PATH additions we're making
    ff_path = runtime_env.get("FFMPEG_PATH")
    cm_path = runtime_env.get("COLMAP_PATH")
    if ff_path:
        print(f"  {OK} FFmpeg → {dim(ff_path)}")
    if cm_path:
        print(f"  {OK} COLMAP → {dim(cm_path)}")
    print()

    def _cleanup(*_):
        print(f"\n\n  {ARROW} Shutting down...")
        for p in procs:
            try:
                if IS_WIN:
                    p.terminate()
                else:
                    p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print(f"  {OK} All servers stopped.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Backend — pass runtime_env so it inherits the augmented PATH
    backend_cmd = [PY, "-m", "uvicorn", "backend.app.main:app",
                   "--reload", "--host", "0.0.0.0", "--port", "8000"]
    print(f"  {ARROW} Backend  → http://localhost:8000")
    bp = subprocess.Popen(backend_cmd, cwd=str(ROOT), env=runtime_env)
    procs.append(bp)

    time.sleep(2)

    # Frontend (shell=True needed on Windows because npm is a .cmd file)
    npm_cmd = ["npm", "run", "dev"]
    print(f"  {ARROW} Frontend → http://localhost:3000")
    fp = subprocess.Popen(npm_cmd, cwd=str(FRONTEND_DIR), shell=IS_WIN, env=runtime_env)
    procs.append(fp)

    print(f"\n  {OK} {bold('App is running!')}  Open {bold(cyan('http://localhost:3000'))} in your browser.")
    print(f"  {dim('Press Ctrl+C to stop both servers.')}\n")

    # Wait for either to exit
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    name = "Backend" if p == bp else "Frontend"
                    print(f"\n  {FAIL} {name} exited with code {ret}")
                    _cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        _cleanup()


# ── Nuke (data wipe) ────────────────────────────────────────────────────────

def nuke():
    banner()
    print(f"  {red(bold('WARNING:'))} This will delete ALL project data and downloaded tools.\n")
    print(f"    • {DATA_DIR}")
    print(f"    • {TOOLS_DIR}")
    print(f"    • {VENV_DIR}")
    print()
    confirm = input(f"  Type {bold('yes')} to confirm: ").strip().lower()
    if confirm != "yes":
        print(f"\n  {dim('Cancelled.')}\n")
        return
    for d in [DATA_DIR, TOOLS_DIR, VENV_DIR]:
        if d.exists():
            print(f"  {ARROW} Removing {d}...")
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n  {OK} {green('Everything wiped. Run')} {bold('python gs.py')} {green('to start fresh.')}\n")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GaussianSplat Studio — All-in-one CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Commands:
              (default)   Full setup wizard + launch both servers
              run         Start servers (skip setup if already done)
              setup       Run dependency setup only
              doctor      Check all dependencies and print report
              nuke        Wipe all data, tools, and venv
        """)
    )
    parser.add_argument("command", nargs="?", default="full",
                        choices=["full", "run", "setup", "doctor", "nuke"],
                        help="Command to run (default: full)")
    args = parser.parse_args()

    if args.command == "doctor":
        banner()
        doctor()
    elif args.command == "setup":
        setup()
    elif args.command == "run":
        run()
    elif args.command == "nuke":
        nuke()
    else:  # full
        setup()
        run()


if __name__ == "__main__":
    main()
