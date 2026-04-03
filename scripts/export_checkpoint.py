#!/usr/bin/env python3
"""Export a PLY from a checkpoint.pt when training was stopped mid-run."""

import asyncio
import sys
from pathlib import Path

import aiosqlite
import torch
import torch.nn.functional as F
from gsplat.exporter import export_splats


async def stop_and_save(project_id: str):
    db_path = Path(__file__).resolve().parents[1] / "data" / "gaussiansplat.db"
    output_dir = Path(__file__).resolve().parents[1] / "data" / "projects" / project_id / "output"
    ckpt_path = output_dir / "checkpoint.pt"

    # Cancel running task if backend is up
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend.app.pipeline.task_runner import task_runner
        cancelled = await task_runner.cancel(project_id)
        print(f"Task cancel: {'cancelled' if cancelled else 'not running'}")
    except Exception as e:
        print(f"Could not cancel via task_runner (backend may not be running): {e}")

    # Mark as training_complete
    db = await aiosqlite.connect(str(db_path))
    await db.execute(
        "UPDATE projects SET step='training_complete', error=NULL WHERE id=?",
        (project_id,),
    )
    await db.commit()
    await db.close()
    print(f"Project {project_id} marked as training_complete")

    # Export PLY from checkpoint
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    print(f"Loading checkpoint ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
    params = ckpt["params"]
    step = ckpt.get("step", "?")

    means = params["means"]
    scales = params["scales"]
    quats = F.normalize(params["quats"], dim=-1)
    opacities = params["opacities"]
    sh_coeffs = params["sh_coeffs"]

    sh0 = sh_coeffs[:, :1, :]
    shN = sh_coeffs[:, 1:, :] if sh_coeffs.shape[1] > 1 else torch.zeros(means.shape[0], 0, 3, device=means.device)

    n_pts = means.shape[0]
    print(f"Checkpoint step {step}: {n_pts:,} Gaussians, SH degree {int(sh_coeffs.shape[1]**0.5) - 1}")

    ply_bytes = export_splats(means, scales, quats, opacities, sh0, shN)
    ply_path = output_dir / "point_cloud.ply"
    ply_path.write_bytes(ply_bytes)
    print(f"PLY exported: {ply_path} ({len(ply_bytes) / 1024 / 1024:.1f} MB)")

    # Also export deformation weights if present
    if "deform_mlp" in ckpt:
        deform_path = output_dir / "deformation.pt"
        torch.save({
            "mlp_state": ckpt["deform_mlp"],
            "n_gaussians": n_pts,
        }, deform_path)
        print(f"Deformation MLP exported: {deform_path}")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "gs_ultimate"
    asyncio.run(stop_and_save(pid))
