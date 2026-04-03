#!/usr/bin/env python3
"""Standalone PLY exporter — trains and exports with proper SH support."""
import sys, os, time, math, struct, argparse
from pathlib import Path
import numpy as np
import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = ""
sys.path.insert(0, str(Path(__file__).parent / "backend" / "scripts"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814

from train_scaffold import read_cameras_bin, read_images_bin, read_points3d_bin, qvec_to_rotmat, voxel_downsample

def write_ply(path, means, scales, quats, opacities, sh):
    n = len(means)
    num_sh = sh.shape[1] if sh.ndim == 3 else 1
    props = "property float x\nproperty float y\nproperty float z\n"
    props += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    props += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    props += "property float opacity\n"
    props += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    for k in range(1, num_sh):
        for c in range(3):
            props += f"property float f_rest_{(k-1)*3+c}\n"
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<3f", *means[i]))
            f.write(struct.pack("<3f", *scales[i]))
            f.write(struct.pack("<4f", *quats[i]))
            f.write(struct.pack("<f", opacities[i]))
            dc = sh[i][0] if sh.ndim == 3 else sh[i]
            f.write(struct.pack("<3f", *dc.flatten()[:3]))
            if num_sh > 1 and sh.ndim == 3:
                rest = sh[i][1:].flatten()
                f.write(struct.pack(f"<{len(rest)}f", *rest))
    print(f"Exported {n} splats ({num_sh} SH coeffs) to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--voxel_size", type=float, default=0.0005)
    parser.add_argument("--sh_degree", type=int, default=2)
    args = parser.parse_args()

    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    sparse_dir = args.data_dir / "sparse"
    images_dir = args.data_dir / "images"

    cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(xyz)} points")

    anchor_xyz, anchor_rgb = voxel_downsample(xyz, rgb, args.voxel_size)
    n_anchors = len(anchor_xyz)
    print(f"Scaffold anchors: {n_anchors} (voxel={args.voxel_size})")

    means = torch.tensor(anchor_xyz, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scales = torch.nn.Parameter(torch.full((n_anchors, 3), -4.0, device=DEVICE))
    quats = torch.nn.Parameter(torch.cat([torch.ones(n_anchors, 1, device=DEVICE),
                                           torch.zeros(n_anchors, 3, device=DEVICE)], dim=1))
    opacities = torch.nn.Parameter(torch.logit(torch.full((n_anchors,), 0.1, device=DEVICE)))

    num_sh = (args.sh_degree + 1) ** 2
    sh = torch.zeros(n_anchors, num_sh, 3, dtype=torch.float32, device=DEVICE)
    sh[:, 0, :] = (torch.tensor(anchor_rgb / 255.0, dtype=torch.float32, device=DEVICE) - 0.5) / C0
    sh = torch.nn.Parameter(sh)
    print(f"SH degree {args.sh_degree} ({num_sh} coefficients)")

    strategy = DefaultStrategy()
    state = strategy.initialize_state()

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4},
        {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3},
        {"params": [opacities], "lr": 5e-2},
        {"params": [sh], "lr": 2.5e-3},
    ], eps=1e-15)

    cam_list = []
    for img_data in images.values():
        cam = cameras[img_data["camera_id"]]
        img_path = images_dir / img_data["name"]
        if not img_path.exists():
            continue
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        p = cam["params"]
        fx = p[0]; fy = p[1] if len(p) > 1 else p[0]
        cx = p[2] if len(p) > 2 else cam["width"] / 2
        cy = p[3] if len(p) > 3 else cam["height"] / 2
        cam_list.append({"w2c": w2c, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                         "W": cam["width"], "H": cam["height"], "path": img_path})

    print(f"Training on {len(cam_list)} images for {args.max_steps} steps")

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        cam = cam_list[(step - 1) % len(cam_list)]
        img = np.array(Image.open(cam["path"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = torch.tensor(img, device=DEVICE).permute(2, 0, 1).unsqueeze(0)

        w2c_t = torch.tensor(cam["w2c"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        K = torch.tensor([[[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]],
                         dtype=torch.float32, device=DEVICE)

        render_colors, render_alphas, meta = rasterization(
            means, quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            torch.exp(scales), torch.sigmoid(opacities),
            sh, w2c_t, K, cam["W"], cam["H"],
            sh_degree=args.sh_degree, absgrad=True,
        )
        render = render_colors.permute(0, 3, 1, 2)
        loss = torch.nn.functional.l1_loss(render, gt)

        optimizer.zero_grad()
        loss.backward()

        strategy.step_pre_backward(
            params={"means": means, "scales": scales, "quats": quats,
                    "opacities": opacities, "sh0": sh},
            optimizers={"means": optimizer},
            state=state, step=step, info=meta,
        )
        optimizer.step()

        if step % 500 == 0 or step == args.max_steps:
            psnr = -10 * math.log10(max(torch.nn.functional.mse_loss(render, gt).item(), 1e-10))
            n_cur = len(means)
            print(f"Step {step}/{args.max_steps}, loss={loss.item():.4f}, psnr={psnr:.1f}, splats={n_cur}", flush=True)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    out_ply = args.result_dir / "point_cloud.ply"
    write_ply(str(out_ply),
              means.detach().cpu().numpy(),
              torch.exp(scales).detach().cpu().numpy(),
              (quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)).detach().cpu().numpy(),
              torch.sigmoid(opacities).detach().cpu().numpy().flatten(),
              sh.detach().cpu().numpy())
    print(f"Saved: {out_ply} ({out_ply.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
