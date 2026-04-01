"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CameraPath, CameraKeyframe } from "@/lib/cameraPath";
import type { SplatRenderer } from "@/lib/renderers";

interface Props {
  renderer: SplatRenderer | null;
  visible: boolean;
}

export default function CameraPathEditor({ renderer, visible }: Props) {
  const [path] = useState(() => new CameraPath());
  const [keyframes, setKeyframes] = useState<CameraKeyframe[]>([]);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(5); // seconds
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef(0);

  const addKeyframe = useCallback(() => {
    if (!renderer) return;
    const cam = renderer.getCamera();
    if (!cam) return;

    const pos = cam.position;
    // Compute lookAt direction from camera
    const dir = new (await_THREE()).Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
    const lookAt = pos.clone().add(dir.multiplyScalar(5));

    const t = path.length === 0 ? 0 : 1;
    const kf: CameraKeyframe = {
      position: [pos.x, pos.y, pos.z],
      lookAt: [lookAt.x, lookAt.y, lookAt.z],
      fov: (cam as any).fov ?? 75,
      t: path.length === 0 ? 0 : path.keyframes[path.length - 1].t + 1,
    };

    path.addKeyframe(kf);
    // Normalize t values to [0, 1]
    const maxT = Math.max(...path.keyframes.map((k) => k.t));
    if (maxT > 0) {
      path.keyframes.forEach((k) => (k.t = k.t / maxT));
    }
    setKeyframes([...path.keyframes]);
  }, [renderer, path]);

  const removeKeyframe = useCallback(
    (idx: number) => {
      path.removeKeyframe(idx);
      setKeyframes([...path.keyframes]);
    },
    [path]
  );

  const play = useCallback(() => {
    if (path.length < 2 || !renderer) return;
    setPlaying(true);
    startTimeRef.current = performance.now();

    const animate = () => {
      const elapsed = (performance.now() - startTimeRef.current) / 1000;
      const t = Math.min(elapsed / duration, 1);
      const state = path.interpolate(t);
      if (state && renderer) {
        renderer.setCameraTransform(
          [state.position.x, state.position.y, state.position.z],
          [state.lookAt.x, state.lookAt.y, state.lookAt.z]
        );
      }
      if (t < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        setPlaying(false);
      }
    };
    rafRef.current = requestAnimationFrame(animate);
  }, [path, renderer, duration]);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    setPlaying(false);
  }, []);

  const exportPath = useCallback(() => {
    const json = JSON.stringify(path.toJSON(), null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "camera_path.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [path]);

  useEffect(() => {
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  if (!visible) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        background: "rgba(0, 0, 0, 0.85)",
        color: "#fff",
        padding: "10px 16px",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        gap: 12,
        fontSize: 12,
      }}
    >
      <button
        onClick={addKeyframe}
        disabled={playing}
        style={{
          background: "#3b82f6",
          color: "white",
          border: "none",
          borderRadius: 4,
          padding: "4px 12px",
          cursor: "pointer",
          opacity: playing ? 0.5 : 1,
        }}
      >
        + Keyframe
      </button>

      <div style={{ display: "flex", gap: 4, flex: 1, overflowX: "auto" }}>
        {keyframes.map((kf, i) => (
          <div
            key={i}
            style={{
              background: "rgba(255,255,255,0.1)",
              borderRadius: 4,
              padding: "2px 8px",
              display: "flex",
              alignItems: "center",
              gap: 4,
              whiteSpace: "nowrap",
            }}
          >
            <span>KF {i + 1}</span>
            <button
              onClick={() => removeKeyframe(i)}
              style={{
                background: "none",
                border: "none",
                color: "#f87171",
                cursor: "pointer",
                fontSize: 14,
                padding: 0,
              }}
            >
              x
            </button>
          </div>
        ))}
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
        Duration:
        <input
          type="number"
          value={duration}
          onChange={(e) => setDuration(Math.max(1, Number(e.target.value)))}
          style={{
            width: 40,
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: 4,
            color: "white",
            padding: "2px 4px",
            textAlign: "center",
          }}
        />
        s
      </label>

      {!playing ? (
        <button
          onClick={play}
          disabled={path.length < 2}
          style={{
            background: path.length >= 2 ? "#10b981" : "rgba(255,255,255,0.1)",
            color: "white",
            border: "none",
            borderRadius: 4,
            padding: "4px 12px",
            cursor: path.length >= 2 ? "pointer" : "default",
          }}
        >
          Play
        </button>
      ) : (
        <button
          onClick={stop}
          style={{
            background: "#ef4444",
            color: "white",
            border: "none",
            borderRadius: 4,
            padding: "4px 12px",
            cursor: "pointer",
          }}
        >
          Stop
        </button>
      )}

      <button
        onClick={exportPath}
        disabled={path.length < 2}
        style={{
          background: "rgba(255,255,255,0.1)",
          color: "white",
          border: "none",
          borderRadius: 4,
          padding: "4px 12px",
          cursor: path.length >= 2 ? "pointer" : "default",
          opacity: path.length >= 2 ? 1 : 0.5,
        }}
      >
        Export JSON
      </button>
    </div>
  );
}

// Lazy THREE import helper
function await_THREE() {
  // THREE is already imported as a peer dep via the renderer
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("three") as typeof import("three");
}
