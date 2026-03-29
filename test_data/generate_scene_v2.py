#!/usr/bin/env python3
"""
generate_scene_v2.py — Generates a heavily-textured synthetic 3D scene for COLMAP testing.
Renders a room with textured boxes from a camera orbiting the scene.
Uses numpy for high-frequency procedural textures to maximize SIFT keypoints.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1280, 720
FOV_DEG = 60
FX = FY = W / (2 * math.tan(math.radians(FOV_DEG / 2)))
CX, CY = W / 2, H / 2


# ─── Math ──────────────────────────────────────────────────────────────────

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def look_at(eye, target, up=np.array([0., 1., 0.])):
    """CV convention: z=forward, x=right, y=down."""
    z = normalize(target - eye)
    x = normalize(np.cross(up, z))
    y = np.cross(x, z)
    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R.astype(np.float32)
    mat[:3,  3] = t.astype(np.float32)
    return mat


def project_pts(pts, viewmat):
    """Project [N,3] world points → [N,2] pixel coords + z_cam."""
    N = len(pts)
    h = np.ones((N, 1), dtype=np.float32)
    p4 = np.hstack([pts.astype(np.float32), h])   # [N, 4]
    cam = (viewmat @ p4.T).T                       # [N, 4]
    xc, yc, zc = cam[:, 0], cam[:, 1], cam[:, 2]
    vis = zc > 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(vis, FX * xc / zc + CX, -1)
        v = np.where(vis, FY * yc / zc + CY, -1)
    return np.stack([u, v], 1), zc, vis


# ─── Procedural textures ────────────────────────────────────────────────────

def make_texture(seed: int, base_color, w=256, h=256):
    """Generate a high-frequency procedural texture as RGBA numpy array."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Base color
    img[:] = np.array(base_color, dtype=np.float32) / 255.0

    # Layer 1: coarse checkerboard pattern
    sq = 32
    for i in range(0, h, sq * 2):
        for j in range(0, w, sq * 2):
            img[i:i+sq, j:j+sq] *= 1.3
            img[i+sq:i+2*sq, j+sq:j+2*sq] *= 1.3

    # Layer 2: fine noise
    noise = rng.uniform(-0.15, 0.15, (h, w, 3)).astype(np.float32)
    img += noise

    # Layer 3: dots grid (strong SIFT blobs)
    dot_r = 4
    for i in range(dot_r * 2, h, 24):
        for j in range(dot_r * 2, w, 24):
            ys, xs = np.ogrid[-dot_r:dot_r+1, -dot_r:dot_r+1]
            mask = xs*xs + ys*ys <= dot_r*dot_r
            y0, y1 = max(0, i-dot_r), min(h, i+dot_r+1)
            x0, x1 = max(0, j-dot_r), min(w, j+dot_r+1)
            my = mask[:y1-y0, :x1-x0]
            c = rng.choice([0.0, 1.0])  # alternate black/white dots
            img[y0:y1, x0:x1][my] = c

    # Layer 4: horizontal and vertical lines
    for i in range(0, h, 16):
        img[i, :] = img[i, :] * 0.7
    for j in range(0, w, 16):
        img[:, j] = img[:, j] * 0.7

    return np.clip(img * 255, 0, 255).astype(np.uint8)


_TEXTURES = {}


def get_texture(seed, base_color):
    key = (seed, tuple(base_color))
    if key not in _TEXTURES:
        _TEXTURES[key] = make_texture(seed, base_color)
    return _TEXTURES[key]


# ─── Z-buffer rasterizer ────────────────────────────────────────────────────

def fill_triangle(zbuf, img, v0, v1, v2, z0, z1, z2, uv0, uv1, uv2, tex):
    """Rasterize a textured triangle using barycentric coordinates."""
    H_img, W_img = img.shape[:2]
    TH, TW = tex.shape[:2]

    # Bounding box
    xs = np.array([v0[0], v1[0], v2[0]])
    ys = np.array([v0[1], v1[1], v2[1]])
    x0 = max(0, int(xs.min()))
    x1 = min(W_img - 1, int(xs.max()) + 1)
    y0 = max(0, int(ys.min()))
    y1 = min(H_img - 1, int(ys.max()) + 1)
    if x0 >= x1 or y0 >= y1:
        return

    px, py = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    px = px.astype(np.float32)
    py = py.astype(np.float32)

    # Edge vectors
    ex01 = v1[0] - v0[0]; ey01 = v1[1] - v0[1]
    ex02 = v2[0] - v0[0]; ey02 = v2[1] - v0[1]
    denom = ex01 * ey02 - ey01 * ex02
    if abs(denom) < 1e-6:
        return

    dpx = px - v0[0]; dpy = py - v0[1]
    w1 = (dpx * ey02 - dpy * ex02) / denom
    w2 = (dpx * ey01 - dpy * ex01) / (-denom)  # note: fix sign
    w0 = 1.0 - w1 - w2

    # Actually recompute properly
    def cross2d(ax, ay, bx, by):
        return ax * by - ay * bx

    area = cross2d(v1[0]-v0[0], v1[1]-v0[1], v2[0]-v0[0], v2[1]-v0[1])
    if abs(area) < 1e-6:
        return

    w0 = cross2d(v1[0]-px, v1[1]-py, v2[0]-px, v2[1]-py) / area
    w1 = cross2d(v2[0]-px, v2[1]-py, v0[0]-px, v0[1]-py) / area
    w2 = 1.0 - w0 - w1

    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    # Interpolate depth
    z_interp = w0 * z0 + w1 * z1 + w2 * z2

    # Depth test
    curr_z = zbuf[y0:y1, x0:x1]
    visible = inside & (z_interp < curr_z)
    if not visible.any():
        return

    # Interpolate UV
    u_tex = np.clip((w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]) * TW, 0, TW - 1).astype(int)
    v_tex = np.clip((w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]) * TH, 0, TH - 1).astype(int)

    zbuf[y0:y1, x0:x1][visible] = z_interp[visible]
    img[y0:y1, x0:x1][visible] = tex[v_tex[visible], u_tex[visible]]


def draw_quad(zbuf, img, verts_3d, viewmat, tex, shade=1.0):
    """Project and rasterize a textured quad (4 vertices)."""
    proj, zc, vis = project_pts(verts_3d, viewmat)
    if not vis.all():
        return

    # Apply shading to texture
    t = (tex * shade).clip(0, 255).astype(np.uint8)

    # Split quad into 2 triangles
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for tri in [(0, 1, 2), (0, 2, 3)]:
        i0, i1, i2 = tri
        fill_triangle(
            zbuf, img,
            proj[i0], proj[i1], proj[i2],
            zc[i0], zc[i1], zc[i2],
            uvs[i0], uvs[i1], uvs[i2],
            t,
        )


# ─── Scene definition ────────────────────────────────────────────────────────

def box_quads(cx, cy_, cz, sx, sy, sz):
    """Return 6 quads (each 4 world-space corners) + face normals for a box."""
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy_ - sy/2, cy_ + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    return [
        # face,        normal
        (np.array([[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]), np.array([0,0,1.])),   # front
        (np.array([[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]]), np.array([0,0,-1.])),  # back
        (np.array([[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]]), np.array([-1,0,0.])),  # left
        (np.array([[x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1]]), np.array([1,0,0.])),   # right
        (np.array([[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]]), np.array([0,-1,0.])),  # bottom
        (np.array([[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]]), np.array([0,1,0.])),   # top
    ]


BOXES = [
    # cx, cy, cz, sx, sy, sz, base_color, tex_seed
    (0,    0,   0,  1.5, 1.5, 1.5, (220, 80,  80),  1),   # center red
    (-3,  -0.5, 1,  1.0, 1.0, 1.0, (80,  160, 220), 2),   # left blue
    (3,   -0.5,-1,  1.2, 1.0, 1.2, (80,  220, 120), 3),   # right green
    (1,    0.5,-3,  0.8, 2.5, 0.8, (220, 200, 80),  4),   # back tall yellow
    (-1.5,-0.7, 2,  0.6, 0.6, 0.6, (180, 80,  220), 5),   # front small purple
    (2,   -0.7, 2,  0.7, 0.6, 0.7, (220, 140, 60),  6),   # front-right orange
    # Ground (flat)
    (0,   -1.1, 0, 20.0, 0.2,20.0, (160, 140, 110), 7),
]

LIGHT_DIR = normalize(np.array([1., 2., 1.]))


def render_frame(viewmat):
    img  = np.full((H, W, 3), [135, 180, 220], dtype=np.float32)   # sky
    zbuf = np.full((H, W), 1e10, dtype=np.float32)

    eye_w = np.linalg.inv(viewmat)[:3, 3]

    # Collect all visible faces with their mean depth (for painter's order)
    faces = []
    for (cx, cy_, cz, sx, sy, sz, col, seed) in BOXES:
        for (verts, normal) in box_quads(cx, cy_, cz, sx, sy, sz):
            center = verts.mean(0)
            to_eye = normalize(eye_w - center)
            if np.dot(normal, to_eye) <= 0:
                continue   # back-face cull
            shade = max(0.35, np.dot(normal, LIGHT_DIR))
            tex   = get_texture(seed, col)
            depth = np.linalg.norm(center - eye_w)
            faces.append((depth, verts, tex, shade))

    faces.sort(key=lambda x: -x[0])   # back-to-front

    for (depth, verts, tex, shade) in faces:
        draw_quad(zbuf, img, verts, viewmat, tex, shade=shade)

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    out_dir   = Path(__file__).parent / "videos" / "scene_frames"
    video_out = Path(__file__).parent / "videos" / "textured_scene.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 120   # 4s at 30fps — 2 full 360° orbits → good for SfM
    radius   = 7.0
    height   = 1.5
    target   = np.zeros(3)

    print(f"Rendering {n_frames} frames at {W}×{H} …", flush=True)
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        eye   = np.array([radius * math.cos(angle), height, radius * math.sin(angle)])
        vm    = look_at(eye, target)
        frame = render_frame(vm)
        frame.save(out_dir / f"{i+1:04d}.jpg", quality=95)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_frames}", flush=True)

    print("Rendering done. Creating video…", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(out_dir / "%04d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        str(video_out),
    ], check=True, capture_output=True)

    import shutil
    shutil.rmtree(out_dir)
    print(f"Done: {video_out}", flush=True)


if __name__ == "__main__":
    main()
