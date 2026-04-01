"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Props {
  plyUrl: string;
  height?: number;
}

/**
 * Lightweight Three.js point cloud renderer for COLMAP sparse reconstruction preview.
 * Parses binary PLY (float32 xyz + uint8 rgb) and renders with THREE.Points.
 */
export default function PointCloudViewer({ plyUrl, height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pointCount, setPointCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState(2.0);
  const materialRef = useRef<THREE.PointsMaterial | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    let disposed = false;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.01,
      10000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(1);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Fetch and parse PLY
    fetch(plyUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load PLY: ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => {
        if (disposed) return;

        const { positions, colors, count } = parseBinaryPLY(buf);
        setPointCount(count);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        });
        materialRef.current = material;

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Auto-fit camera to bounding box
        geometry.computeBoundingBox();
        const box = geometry.boundingBox!;
        const center = new THREE.Vector3();
        box.getCenter(center);
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        camera.position.copy(center);
        camera.position.z += maxDim * 1.5;
        camera.position.y += maxDim * 0.3;
        controls.target.copy(center);
        controls.update();

        setLoading(false);
      })
      .catch((err) => {
        if (!disposed) {
          setError(String(err));
          setLoading(false);
        }
      });

    // Render loop
    const animate = () => {
      if (disposed) return;
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize
    const onResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      disposed = true;
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, [plyUrl]);

  // Update point size when slider changes
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.size = pointSize;
    }
  }, [pointSize]);

  return (
    <div style={{ position: "relative", width: "100%", height }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />

      {loading && (
        <div
          style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.7)", color: "white", fontSize: 14,
          }}
        >
          Loading point cloud...
        </div>
      )}

      {error && (
        <div
          style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.8)", color: "#f87171", fontSize: 13, padding: 24,
          }}
        >
          {error}
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Stats overlay */}
          <div
            style={{
              position: "absolute", top: 8, left: 8,
              background: "rgba(0,0,0,0.6)", color: "#9ca3af",
              padding: "4px 10px", borderRadius: 4, fontSize: 11,
            }}
          >
            {pointCount.toLocaleString()} points
          </div>

          {/* Point size slider */}
          <div
            style={{
              position: "absolute", bottom: 8, left: 8,
              background: "rgba(0,0,0,0.6)", color: "#9ca3af",
              padding: "4px 10px", borderRadius: 4, fontSize: 11,
              display: "flex", alignItems: "center", gap: 6,
            }}
          >
            <span>Size</span>
            <input
              type="range"
              min={0.5}
              max={8}
              step={0.5}
              value={pointSize}
              onChange={(e) => setPointSize(Number(e.target.value))}
              style={{ width: 80 }}
            />
          </div>

          {/* Label */}
          <div
            style={{
              position: "absolute", top: 8, right: 8,
              background: "rgba(59,130,246,0.3)", color: "#93c5fd",
              padding: "4px 10px", borderRadius: 4, fontSize: 11,
            }}
          >
            SfM Preview (not final quality)
          </div>
        </>
      )}
    </div>
  );
}

/** Parse a binary PLY with float32 xyz + uint8 rgb vertices. */
function parseBinaryPLY(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);

  // Find end_header
  let headerEnd = 0;
  const needle = "end_header\n";
  const decoder = new TextDecoder("ascii");
  for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
    if (decoder.decode(bytes.slice(i, i + needle.length)) === needle) {
      headerEnd = i + needle.length;
      break;
    }
  }

  // Parse header for vertex count
  const headerStr = decoder.decode(bytes.slice(0, headerEnd));
  const vertexMatch = headerStr.match(/element vertex (\d+)/);
  const count = vertexMatch ? parseInt(vertexMatch[1], 10) : 0;

  // Each vertex: 3 floats (12 bytes) + 3 uint8 (3 bytes) = 15 bytes
  const vertexSize = 15;
  const data = new DataView(buffer, headerEnd);

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const off = i * vertexSize;
    positions[i * 3] = data.getFloat32(off, true);
    positions[i * 3 + 1] = data.getFloat32(off + 4, true);
    positions[i * 3 + 2] = data.getFloat32(off + 8, true);
    colors[i * 3] = data.getUint8(off + 12) / 255;
    colors[i * 3 + 1] = data.getUint8(off + 13) / 255;
    colors[i * 3 + 2] = data.getUint8(off + 14) / 255;
  }

  return { positions, colors, count };
}
