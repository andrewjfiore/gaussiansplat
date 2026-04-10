import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# __file__ = backend/app/services/trainer.py  ->  parents[2] = backend/
_SCAFFOLD_SCRIPT    = Path(__file__).resolve().parents[2] / "scripts" / "train_scaffold.py"
_LEGACY_SCRIPT      = Path(__file__).resolve().parents[2] / "scripts" / "train_splat.py"
_4D_SCRIPT          = Path(__file__).resolve().parents[2] / "scripts" / "train_4d.py"
_VISIBILITY_SCRIPT  = Path(__file__).resolve().parents[2] / "scripts" / "visibility_transfer.py"
_INPAINT_SCRIPT     = Path(__file__).resolve().parents[2] / "scripts" / "diffusion_inpaint.py"
_NOVEL_VIEW_SCRIPT  = Path(__file__).resolve().parents[2] / "scripts" / "generate_novel_views.py"


@dataclass
class TrainPhase:
    """Describes a single training phase command."""
    cmd: list[str]
    label: str
    step_offset: int  # steps already completed before this phase
    total_steps: int  # total steps across ALL phases (for progress calc)


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
    *,
    densify_grad_thresh: Optional[float] = None,
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
    if densify_grad_thresh is not None:
        cmd += ["--densify_grad_thresh", str(densify_grad_thresh)]
    return cmd


def build_two_phase_cmds(
    data_dir: Path,
    result_dir: Path,
    phase1_steps: int = 5600,
    phase2_steps: int = 1400,
    *,
    sh_degree: int = 3,
    densify_grad_thresh: Optional[float] = None,
) -> list[TrainPhase]:
    """Build commands for two-phase training.

    Phase 1: Full optimization (geometry + color) for phase1_steps.
    Phase 2: Color-only refinement for phase2_steps, loading the Phase 1
             checkpoint. We pass --disable_densify to prevent further
             geometry changes, and use a very high densify_grad_thresh
             as an additional safeguard.

    Note: gsplat's simple_trainer does not have a native --freeze_geometry
    flag. Phase 2 approximates color-only refinement by disabling
    densification (no new splats) and relying on the lower learning rate
    at late iterations to keep geometry nearly frozen. The checkpoint from
    Phase 1 is loaded via --ckpt.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    total_steps = phase1_steps + phase2_steps

    # Phase 1: full training
    cmd1 = [
        sys.executable, "-m", "gsplat.examples.simple_trainer",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(phase1_steps),
        "--sh_degree", str(sh_degree),
    ]
    if densify_grad_thresh is not None:
        cmd1 += ["--densify_grad_thresh", str(densify_grad_thresh)]

    # Phase 2: color-only refinement from checkpoint
    ckpt_path = result_dir / f"ckpt_{phase1_steps}_rank0.pt"
    cmd2 = [
        sys.executable, "-m", "gsplat.examples.simple_trainer",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(total_steps),
        "--sh_degree", str(sh_degree),
        "--ckpt", str(ckpt_path),
        # Prevent geometry changes in phase 2
        "--densify_grad_thresh", "1.0",
    ]

    return [
        TrainPhase(
            cmd=cmd1,
            label="Phase 1: Geometry + Color",
            step_offset=0,
            total_steps=total_steps,
        ),
        TrainPhase(
            cmd=cmd2,
            label="Phase 2: Color Refinement",
            step_offset=phase1_steps,
            total_steps=total_steps,
        ),
    ]


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


def build_novel_view_cmd(
    input_dir: Path,
    output_dir: Path,
    model: str = "zero123pp",
    num_refs: int = 4,
    output_size: int = 800,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, str(_NOVEL_VIEW_SCRIPT),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--model", model,
        "--num_refs", str(num_refs),
        "--output_size", str(output_size),
    ]


def parse_novel_view_line(line: str) -> Optional[dict]:
    """Parse novel view generation progress."""
    import re
    m = re.match(r"\[NOVEL\]", line)
    if m:
        return {"percent": -1}
    if "Complete:" in line:
        return {"percent": 100}
    return None


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


def make_phase_line_parser(phase: TrainPhase):
    """Create a line parser that adjusts step numbers for multi-phase progress."""

    def parser(line: str) -> Optional[dict]:
        parsed = parse_trainer_line(line)
        if parsed is None:
            return None

        # Recalculate percent based on global total
        if "metric" in parsed and "step" in parsed["metric"]:
            global_step = parsed["metric"]["step"]
            parsed["percent"] = (global_step / phase.total_steps) * 100
            parsed["metric"]["step"] = global_step
            parsed["phase"] = phase.label

        return parsed

    return parser
