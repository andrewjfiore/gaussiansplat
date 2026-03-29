#!/usr/bin/env python3
"""
Generate a synthetic 3D scene as an image sequence for testing the GaussianSplat pipeline.
Renders a textured room with colored boxes from a camera orbiting around the scene center.
Produces proper 3D parallax that COLMAP can reconstruct.
"""

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Output resolution
W, H = 1280, 720

# ─────────────────── 3D math ───────────────────

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def look_at(eye, target, up=np.array([0, 1, 0])):
    """Returns 4x4 world-to-camera matrix (CV convention: z=forward, x=right, y=down).
    Verified: world-up (y+) maps to screen-up (v < cy) via negative y_cam."""
    z = normalize(target - eye)         # camera forward axis in world coords
    x = normalize(np.cross(up, z))      # camera right axis
    y = np.cross(x, z)                  # camera down axis (world-down → positive y_cam → larger v)
    R = np.stack([x, y, z], axis=0)    # rows are camera basis vectors
    t = -R @ eye
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def project(points_world, viewmat, fx, fy, cx, cy, near=0.1):
    """
    Project world-space points to pixel coordinates.
    points_world: [N, 3]
    Returns pixel_xy [N, 2] (float), depth [N], visible mask [N].
    """
    N = len(points_world)
    ones = np.ones((N, 1))
    p4 = np.hstack([points_world, ones])  # [N, 4]
    cam = (viewmat @ p4.T).T             # [N, 4]
    x_c, y_c, z_c = cam[:, 0], cam[:, 1], cam[:, 2]
    visible = z_c > near
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(visible, fx * x_c / z_c + cx, 0)
        v = np.where(visible, fy * y_c / z_c + cy, 0)
    return np.stack([u, v], axis=1), z_c, visible


# ─────────────────── Scene definition ───────────────────

def make_checkerboard(size=10, squares=10, color_a=(200, 180, 140), color_b=(100, 80, 60)):
    """Generate a checkerboard texture image."""
    img = Image.new("RGB", (256, 256))
    sq = 256 // squares
    for i in range(squares):
        for j in range(squares):
            c = color_a if (i + j) % 2 == 0 else color_b
            x0, y0 = j * sq, i * sq
            for px in range(x0, min(x0 + sq, 256)):
                for py in range(y0, min(y0 + sq, 256)):
                    img.putpixel((px, py), c)
    return np.array(img)


def box_faces(cx, cy, cz, sx, sy, sz):
    """Return 6 faces (each a list of 4 corners) for a box."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    return [
        # front, back, left, right, bottom, top
        np.array([[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]),
        np.array([[x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0]]),
        np.array([[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]]),
        np.array([[x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1]]),
        np.array([[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]]),
        np.array([[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]]),
    ]


def face_normal(verts):
    """Compute face normal from 4 CCW vertices."""
    v0 = verts[1] - verts[0]
    v1 = verts[3] - verts[0]
    return normalize(np.cross(v0, v1))


def draw_face(draw, proj_pts, color, outline=(0, 0, 0)):
    """Draw a filled polygon for a face."""
    pts = [(float(p[0]), float(p[1])) for p in proj_pts]
    draw.polygon(pts, fill=color, outline=outline)


def render_frame(viewmat, fx, fy, cx_cam, cy_cam):
    """Render one frame of the synthetic scene."""
    img = Image.new("RGB", (W, H), (135, 180, 220))  # sky blue background
    draw = ImageDraw.Draw(img)

    # Scene objects: list of (box_params, color)
    boxes = [
        # Ground plane (flat box)
        dict(cx=0, cy=-1.0, cz=0, sx=20, sy=0.2, sz=20, color=(160, 140, 110)),
        # Main center box
        dict(cx=0, cy=0, cz=0, sx=1.5, sy=1.5, sz=1.5, color=(220, 80, 80)),
        # Left box
        dict(cx=-3, cy=-0.25, cz=1, sx=1.0, sy=1.5, sz=1.0, color=(80, 160, 220)),
        # Right box
        dict(cx=3, cy=-0.5, cz=-1, sx=1.2, sy=1.0, sz=1.2, color=(80, 220, 120)),
        # Back tall box
        dict(cx=1, cy=0.5, cz=-3, sx=0.8, sy=2.5, sz=0.8, color=(220, 200, 80)),
        # Small front box
        dict(cx=-1.5, cy=-0.6, cz=2, sx=0.6, sy=0.8, sz=0.6, color=(180, 80, 220)),
    ]

    # Collect all faces with depth (back-to-front painter's algorithm)
    faces_to_draw = []
    eye_world = np.linalg.inv(viewmat)[:3, 3]

    for box in boxes:
        cx_, cy_, cz_ = box['cx'], box['cy'], box['cz']
        sx_, sy_, sz_ = box.get('sx', 1), box.get('sy', 1), box.get('sz', 1)
        color = box['color']

        faces = box_faces(cx_, cy_, cz_, sx_, sy_, sz_)
        face_colors = [
            tuple(min(255, int(c * 1.0)) for c in color),   # front  - normal
            tuple(min(255, int(c * 0.7)) for c in color),   # back   - darker
            tuple(min(255, int(c * 0.85)) for c in color),  # left
            tuple(min(255, int(c * 0.9)) for c in color),   # right
            tuple(min(255, int(c * 0.6)) for c in color),   # bottom - darkest
            tuple(min(255, int(c * 1.1)) for c in color),   # top    - lightest
        ]

        for face_verts, face_col in zip(faces, face_colors):
            normal = face_normal(face_verts)
            center = face_verts.mean(axis=0)
            to_eye = normalize(eye_world - center)

            # Back-face culling
            if np.dot(normal, to_eye) < 0:
                continue

            # Simple diffuse shading
            light_dir = normalize(np.array([1, 2, 1], dtype=float))
            shade = max(0.3, np.dot(normal, light_dir))
            shaded = tuple(min(255, int(c * shade)) for c in face_col)

            depth = np.linalg.norm(center - eye_world)
            faces_to_draw.append((depth, face_verts, shaded))

    # Sort back-to-front
    faces_to_draw.sort(key=lambda x: -x[0])

    for depth, face_verts, color in faces_to_draw:
        proj, depths, visible = project(face_verts, viewmat, fx, fy, cx_cam, cy_cam)
        if not visible.all():
            continue
        # Clip to image bounds (rough)
        if proj[:, 0].min() > W or proj[:, 0].max() < 0:
            continue
        if proj[:, 1].min() > H or proj[:, 1].max() < 0:
            continue
        draw_face(draw, proj, color)

    # Draw ground grid lines for feature richness
    for i in range(-5, 6):
        p1 = np.array([[i * 1.0, -0.89, -5.0]])
        p2 = np.array([[i * 1.0, -0.89, 5.0]])
        q1 = np.array([[-5.0, -0.89, i * 1.0]])
        q2 = np.array([[5.0, -0.89, i * 1.0]])
        for a, b in [(p1, p2), (q1, q2)]:
            pa, da, va = project(a, viewmat, fx, fy, cx_cam, cy_cam)
            pb, db, vb = project(b, viewmat, fx, fy, cx_cam, cy_cam)
            if va.all() and vb.all():
                draw.line([(pa[0, 0], pa[0, 1]), (pb[0, 0], pb[0, 1])],
                          fill=(120, 100, 80), width=1)

    # Draw dots at grid intersections for keypoint-rich texture
    import random
    rng = random.Random(42)
    for i in range(-4, 5):
        for j in range(-4, 5):
            pts = np.array([[i * 1.0, -0.88, j * 1.0]])
            pp, dp, vp = project(pts, viewmat, fx, fy, cx_cam, cy_cam)
            if vp.all() and 0 < pp[0, 0] < W and 0 < pp[0, 1] < H:
                u, v_ = int(pp[0, 0]), int(pp[0, 1])
                c = ((i + j) % 2 == 0)
                col = (180, 150, 100) if c else (80, 60, 40)
                draw.ellipse([u-3, v_-3, u+3, v_+3], fill=col)

    return img


def main():
    out_dir = Path(__file__).parent / "videos" / "synthetic_scene"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Camera intrinsics (matching 1280x720 frame)
    fov_deg = 60
    fx = fy = W / (2 * math.tan(math.radians(fov_deg / 2)))
    cx_cam, cy_cam = W / 2, H / 2

    print(f"Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx_cam}, cy={cy_cam}")

    # Generate 60 frames of camera orbiting the scene at 30fps = 2s per revolution
    n_frames = 90  # 3 seconds at 30fps
    radius = 7.0
    height = 1.5
    target = np.array([0.0, 0.0, 0.0])

    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        eye = np.array([radius * math.cos(angle), height, radius * math.sin(angle)])
        viewmat = look_at(eye, target)
        frame = render_frame(viewmat, fx, fy, cx_cam, cy_cam)
        out_path = out_dir / f"{i+1:04d}.jpg"
        frame.save(out_path, quality=95)
        if i % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    print("Rendering done. Creating video...")

    # Stitch to video with ffmpeg
    video_out = Path(__file__).parent / "videos" / "synthetic_scene.mp4"
    os.system(
        f"ffmpeg -y -framerate 30 -i {out_dir}/%04d.jpg "
        f"-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p {video_out}"
    )
    import shutil
    shutil.rmtree(out_dir)
    print(f"Done: {video_out}")


if __name__ == "__main__":
    main()
