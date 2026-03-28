import re
import sys
from pathlib import Path
from typing import Optional


def build_train_cmd(data_dir: Path, result_dir: Path, max_steps: int = 7000) -> list[str]:
    result_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "-m", "gsplat.examples.simple_trainer",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(max_steps),
    ]


def parse_trainer_line(line: str) -> Optional[dict]:
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
