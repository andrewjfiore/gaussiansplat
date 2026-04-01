import re
import sys
from pathlib import Path
from typing import Optional

# __file__ = backend/app/services/trainer.py  →  parents[2] = backend/
_SCAFFOLD_SCRIPT    = Path(__file__).resolve().parents[2] / "scripts" / "train_scaffold.py"
_LEGACY_SCRIPT      = Path(__file__).resolve().parents[2] / "scripts" / "train_splat.py"
_4D_SCRIPT          = Path(__file__).resolve().parents[2] / "scripts" / "train_4d.py"
_VISIBILITY_SCRIPT  = Path(__file__).resolve().parents[2] / "scripts" / "visibility_transfer.py"
_INPAINT_SCRIPT     = Path(__file__).resolve().parents[2] / "scripts" / "diffusion_inpaint.py"


def build_train_cmd(
    data_dir: Path,
    result_dir: Path,
    max_steps: int = 7000,
    use_scaffold: bool = True,
    voxel_size: float = 0.001,
    resume: bool = False,
    sh_degree: int = 0,
    depth_dir: Path | None = None,
    depth_weight: float = 0.1,
    temporal_mode: str = "static",
    temporal_smoothness: float = 0.01,
    manifest_path: Path | None = None,
) -> list[str]:
    result_dir.mkdir(parents=True, exist_ok=True)

    if temporal_mode == "4d":
        cmd = [
            sys.executable, str(_4D_SCRIPT),
            "--data_dir", str(data_dir),
            "--result_dir", str(result_dir),
            "--max_steps", str(max_steps),
            "--sh_degree", str(sh_degree),
            "--temporal_smoothness", str(temporal_smoothness),
        ]
        if manifest_path and manifest_path.exists():
            cmd += ["--manifest", str(manifest_path)]
        if depth_dir and depth_dir.exists():
            cmd += ["--depth_dir", str(depth_dir), "--depth_weight", str(depth_weight)]
        cmd += ["--max_gaussians", "200000"]
        return cmd

    script = _SCAFFOLD_SCRIPT if use_scaffold else _LEGACY_SCRIPT
    cmd = [
        sys.executable, str(script),
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(max_steps),
        "--sh_degree", str(sh_degree),
    ]
    if use_scaffold:
        cmd += ["--voxel_size", str(voxel_size)]
    if resume:
        cmd.append("--resume")
    if depth_dir and depth_dir.exists():
        cmd += ["--depth_dir", str(depth_dir), "--depth_weight", str(depth_weight)]
    return cmd


def build_visibility_transfer_cmd(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    alpha_low: float = 0.5,
    alpha_high: float = 0.8,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, str(_VISIBILITY_SCRIPT),
        "--data_dir", str(data_dir),
        "--checkpoint", str(checkpoint),
        "--output_dir", str(output_dir),
        "--alpha_low", str(alpha_low),
        "--alpha_high", str(alpha_high),
    ]


def build_inpaint_cmd(
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    num_novel_views: int = 8,
    steps: int = 20,
    guidance: float = 3.0,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, str(_INPAINT_SCRIPT),
        "--data_dir", str(data_dir),
        "--checkpoint", str(checkpoint),
        "--output_dir", str(output_dir),
        "--num_novel_views", str(num_novel_views),
        "--steps", str(steps),
        "--guidance", str(guidance),
    ]


def parse_trainer_line(line: str) -> Optional[dict]:
    # Parse snapshot lines: [SNAPSHOT] 25 snapshot_25.jpg
    snap = re.match(r"\[SNAPSHOT\]\s+(\d+)\s+(\S+)", line)
    if snap:
        return {"metric": {"snapshot_pct": int(snap.group(1)), "snapshot_file": snap.group(2)}}

    # Parse lines like "Step 1000/7000, loss=0.0321, psnr=24.5"
    m = re.search(r"[Ss]tep\s*(\d+)(?:/(\d+))?", line)
    if m:
        step = int(m.group(1))
        total = int(m.group(2)) if m.group(2) else None
        result: dict = {}

        if total:
            result["percent"] = (step / total) * 100

        metric: dict = {"step": step}
        loss_m = re.search(r"loss[=:\s]+([0-9.e+-]+)", line, re.IGNORECASE)
        if loss_m:
            metric["loss"] = float(loss_m.group(1))

        psnr_m = re.search(r"psnr[=:\s]+([0-9.e+-]+)", line, re.IGNORECASE)
        if psnr_m:
            metric["psnr"] = float(psnr_m.group(1))

        if len(metric) > 1:
            result["metric"] = metric

        return result if result else None

    return None
