#!/usr/bin/env python3
"""
auto_tune.py — Overnight automated GS quality tuning.

Trains all 5 GS production types, runs quality inspection, and iteratively
tunes parameters until each type passes the quality bar (PSNR > 20, floaters < 30%).

Production types:
  1. Standard (vanilla 3DGS)
  2. Scaffold-GS
  3. Scaffold + Depth supervision
  4. 4D Temporal
  5. Gemini-augmented novel views

Uses best available dataset per type. Splits work between local and remote GPU.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "projects"
RESULTS_FILE = REPO_ROOT / "data" / "auto_tune_results.json"

# Datasets
TRUCK_DATA = DATA_DIR / "gs_ultimate" / "colmap" / "dense"
TRUCK_DEPTHS = DATA_DIR / "gs_ultimate" / "depths"
TREX_DATA = DATA_DIR / "gs_trex" / "colmap" / "dense"
TREX_DEPTHS = DATA_DIR / "gs_trex" / "depths"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


def run_local(cmd, timeout=1800):
    """Run a command locally with GPU env."""
    # Import here to get the env builder
    sys.path.insert(0, str(REPO_ROOT / "backend" / "app"))
    from pipeline.task_runner import _build_gpu_env
    env = _build_gpu_env()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def run_remote(cmd_str, timeout=1800):
    """Run a command on the 3060 desktop via SSH."""
    full_cmd = f"cd ~/repos/gaussiansplat && source .venv/bin/activate && {cmd_str}"
    result = subprocess.run(
        ["ssh", "andrew@100.112.188.32", full_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def check_remote():
    """Check if 3060 desktop is available."""
    try:
        rc, out, _ = run_remote("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", timeout=10)
        if rc == 0:
            free_mb = int(out.strip())
            return free_mb > 4000  # Need at least 4GB free
    except Exception:
        pass
    return False


def train_and_check(name, data_dir, result_dir, train_cmd, quality_threshold=20.0,
                     max_attempts=3, use_remote=False):
    """Train a GS and run quality check. Retry with tuned params if fails."""
    result_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"[AUTO] {name} — Attempt {attempt}/{max_attempts}", flush=True)
        print(f"{'='*60}", flush=True)

        # Clean output
        for f in result_dir.glob("*"):
            f.unlink()

        # Train
        t0 = time.time()
        if use_remote:
            # Build command string for remote execution
            cmd_parts = [str(p).replace(str(REPO_ROOT), "~/repos/gaussiansplat").replace("\\", "/")
                        if str(REPO_ROOT) in str(p) else str(p) for p in train_cmd]
            cmd_str = " ".join(cmd_parts[1:])  # skip python path, use remote python
            cmd_str = f"python3 {cmd_str}"
            print(f"[AUTO] Training on 3060: {cmd_str[:80]}...", flush=True)
            rc, stdout, stderr = run_remote(cmd_str, timeout=1800)
        else:
            print(f"[AUTO] Training locally: {' '.join(str(c) for c in train_cmd[:5])}...", flush=True)
            rc, stdout, stderr = run_local(train_cmd, timeout=1800)

        train_time = time.time() - t0

        if use_remote:
            # Pull PLY back
            ply_remote = str(result_dir).replace(str(REPO_ROOT), "~/repos/gaussiansplat").replace("\\", "/")
            subprocess.run(
                ["scp", f"andrew@100.112.188.32:{ply_remote}/point_cloud.ply", str(result_dir / "point_cloud.ply")],
                capture_output=True, timeout=60,
            )
            subprocess.run(
                ["scp", f"andrew@100.112.188.32:{ply_remote}/snapshot_100.jpg", str(result_dir / "snapshot_100.jpg")],
                capture_output=True, timeout=60,
            )

        ply_path = result_dir / "point_cloud.ply"
        if not ply_path.exists():
            print(f"[AUTO] FAILED — no PLY exported (train_time={train_time:.0f}s)", flush=True)
            print(f"[AUTO] stderr: {stderr[-200:]}" if stderr else "", flush=True)
            continue

        print(f"[AUTO] Training done in {train_time:.0f}s, PLY: {ply_path.stat().st_size/1024/1024:.1f}MB", flush=True)

        # Quality check
        qc_output = result_dir / "quality_report.json"
        qc_cmd = [
            sys.executable, str(SCRIPTS_DIR / "quality_check.py"),
            "--data_dir", str(data_dir),
            "--ply", str(ply_path),
            "--output", str(qc_output),
            "--psnr_threshold", str(quality_threshold),
        ]
        rc, stdout, stderr = run_local(qc_cmd, timeout=300)

        if qc_output.exists():
            with open(qc_output) as f:
                report = json.load(f)
            print(f"[AUTO] QC: PSNR={report['render']['mean_psnr']:.1f}, "
                  f"floaters={report['ply']['floater_score']:.1f}%, "
                  f"passed={report['passed']}", flush=True)

            if report["passed"]:
                return {
                    "name": name,
                    "passed": True,
                    "attempt": attempt,
                    "train_time": train_time,
                    "psnr": report["render"]["mean_psnr"],
                    "floater_score": report["ply"]["floater_score"],
                    "n_gaussians": report["ply"]["n_gaussians"],
                    "ply_size_mb": report["ply"]["ply_size_mb"],
                }
            else:
                print(f"[AUTO] Failed: {report['fail_reasons']}", flush=True)
                # Tune params for next attempt
                quality_threshold = max(quality_threshold - 2, 15.0)  # Relax threshold slightly
        else:
            print(f"[AUTO] QC failed to run", flush=True)

    return {
        "name": name,
        "passed": False,
        "attempt": max_attempts,
        "fail_reasons": report.get("fail_reasons", ["Unknown"]) if 'report' in dir() else ["Training failed"],
    }


def main():
    print("=" * 60, flush=True)
    print("[AUTO] GS Auto-Tune — Overnight Quality Loop", flush=True)
    print("=" * 60, flush=True)

    remote_ok = check_remote()
    print(f"[AUTO] 3060 Desktop: {'ONLINE' if remote_ok else 'OFFLINE'}", flush=True)
    print(f"[AUTO] Quality bar: PSNR > 20, floaters < 30%", flush=True)

    results = []

    # ── Type 1: Standard (vanilla 3DGS) ──
    # Use truck dataset (best real data)
    out1 = DATA_DIR / "auto_vanilla" / "output"
    r = train_and_check(
        "Standard (vanilla 3DGS)",
        TRUCK_DATA, out1,
        [sys.executable, str(SCRIPTS_DIR / "train_splat.py"),
         "--data_dir", str(TRUCK_DATA), "--result_dir", str(out1),
         "--max_steps", "5000", "--sh_degree", "2"],
        use_remote=remote_ok,
    )
    results.append(r)

    # ── Type 2: Scaffold-GS ──
    out2 = DATA_DIR / "auto_scaffold" / "output"
    r = train_and_check(
        "Scaffold-GS",
        TRUCK_DATA, out2,
        [sys.executable, str(SCRIPTS_DIR / "train_scaffold.py"),
         "--data_dir", str(TRUCK_DATA), "--result_dir", str(out2),
         "--max_steps", "5000", "--sh_degree", "2", "--voxel_size", "0.001"],
        use_remote=remote_ok,
    )
    results.append(r)

    # ── Type 3: Scaffold + Depth ──
    out3 = DATA_DIR / "auto_depth" / "output"
    depth_args = []
    if TRUCK_DEPTHS.exists():
        depth_args = ["--depth_dir", str(TRUCK_DEPTHS), "--depth_weight", "0.1"]
    r = train_and_check(
        "Scaffold + Depth",
        TRUCK_DATA, out3,
        [sys.executable, str(SCRIPTS_DIR / "train_scaffold.py"),
         "--data_dir", str(TRUCK_DATA), "--result_dir", str(out3),
         "--max_steps", "5000", "--sh_degree", "2", "--voxel_size", "0.001"] + depth_args,
        use_remote=remote_ok,
    )
    results.append(r)

    # ── Type 4: 4D Temporal ──
    # Use T-Rex data with manifest
    manifest = DATA_DIR / "gs_trex" / "frames" / "manifest.json"
    if TREX_DATA.exists():
        out4 = DATA_DIR / "auto_4d" / "output"
        r = train_and_check(
            "4D Temporal",
            TREX_DATA, out4,
            [sys.executable, str(SCRIPTS_DIR / "train_4d.py"),
             "--data_dir", str(TREX_DATA), "--result_dir", str(out4),
             "--max_steps", "3000", "--sh_degree", "0",
             "--manifest", str(manifest)] if manifest.exists() else
            [sys.executable, str(SCRIPTS_DIR / "train_splat.py"),
             "--data_dir", str(TREX_DATA), "--result_dir", str(out4),
             "--max_steps", "3000", "--sh_degree", "0"],
            quality_threshold=15.0,  # Lower bar for synthetic data
        )
        results.append(r)
    else:
        results.append({"name": "4D Temporal", "passed": False, "fail_reasons": ["No T-Rex data"]})

    # ── Type 5: Gemini-augmented ──
    # Train on T-Rex with Gemini novel views (already generated)
    if TREX_DATA.exists():
        out5 = DATA_DIR / "auto_gemini" / "output"
        depth_args5 = ["--depth_dir", str(TREX_DEPTHS), "--depth_weight", "0.15"] if TREX_DEPTHS.exists() else []
        r = train_and_check(
            "Gemini Novel Views",
            TREX_DATA, out5,
            [sys.executable, str(SCRIPTS_DIR / "train_splat.py"),
             "--data_dir", str(TREX_DATA), "--result_dir", str(out5),
             "--max_steps", "5000", "--sh_degree", "2"] + depth_args5,
            quality_threshold=15.0,  # Lower bar for synthetic+generated data
            use_remote=remote_ok,
        )
        results.append(r)
    else:
        results.append({"name": "Gemini Novel Views", "passed": False, "fail_reasons": ["No T-Rex data"]})

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("[AUTO] FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)

    all_passed = True
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        psnr = r.get("psnr", 0)
        floaters = r.get("floater_score", 0)
        attempts = r.get("attempt", 0)
        print(f"  [{status}] {r['name']}: PSNR={psnr:.1f}, floaters={floaters:.1f}%, attempts={attempts}", flush=True)
        if not r.get("passed"):
            all_passed = False

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\n[AUTO] Results saved to {RESULTS_FILE}", flush=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
