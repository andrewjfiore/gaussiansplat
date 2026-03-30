"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { createRenderer, SplatRenderer } from "@/lib/renderers";

interface Props {
  plyUrl: string;
  ksplatUrl?: string;
}

export function SplatViewerFPS({ plyUrl, ksplatUrl }: Props) {
  const mountRef    = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<SplatRenderer | null>(null);
  const [rendererType, setRendererType] = useState<"webgpu" | "webgl">("webgl");
  const [fpsMode, setFpsMode]           = useState(false);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState<string | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const url = ksplatUrl || plyUrl;
    let disposed = false;

    createRenderer(url, mountRef.current)
      .then(async (r) => {
        if (disposed) { r.dispose(); return; }
        rendererRef.current = r;
        setRendererType(r.type);
        await r.load(url);
        setLoading(false);
      })
      .catch((err) => {
        if (!disposed) setError(String(err));
        setLoading(false);
      });

    return () => {
      disposed = true;
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, [plyUrl, ksplatUrl]);

  const toggleFPS = useCallback(() => {
    const r = rendererRef.current;
    if (!r) return;
    const next = !fpsMode;
    r.setFPSControls(next);
    setFpsMode(next);
  }, [fpsMode]);

  const enterVR = useCallback(() => {
    rendererRef.current?.enterVR();
  }, []);

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh" }}>
      {/* Canvas mount */}
      <div ref={mountRef} style={{ width: "100%", height: "100%" }} />

      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex",
          alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.7)", color: "white", fontSize: 16,
        }}>
          Loading splat…
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div style={{
          position: "absolute", inset: 0, display: "flex",
          alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.8)", color: "#f87171", fontSize: 14, padding: 24,
        }}>
          {error}
        </div>
      )}

      {/* Toolbar */}
      {!loading && !error && (
        <div style={{
          position: "absolute", top: 12, left: 12, display: "flex",
          gap: 8, alignItems: "center",
        }}>
          <span style={{
            background: "rgba(0,0,0,0.55)", color: "#9ca3af",
            fontSize: 11, padding: "3px 8px", borderRadius: 4,
          }}>
            Renderer: {rendererType === "webgpu" ? "WebGPU" : "WebGL"}
          </span>
          <button
            onClick={toggleFPS}
            style={{
              background: fpsMode ? "#3b82f6" : "rgba(0,0,0,0.55)",
              color: "white", border: "none", borderRadius: 4,
              padding: "4px 12px", fontSize: 12, cursor: "pointer",
            }}
          >
            {fpsMode ? "Orbit" : "Free Cam"}
          </button>
          <button
            onClick={enterVR}
            style={{
              background: "rgba(0,0,0,0.55)", color: "white",
              border: "none", borderRadius: 4,
              padding: "4px 12px", fontSize: 12, cursor: "pointer",
            }}
          >
            Enter VR
          </button>
        </div>
      )}

      {/* WASD hint */}
      {fpsMode && (
        <div style={{
          position: "absolute", bottom: 16, left: "50%",
          transform: "translateX(-50%)",
          color: "white", fontSize: 11, opacity: 0.65,
          background: "rgba(0,0,0,0.45)", padding: "5px 12px", borderRadius: 4,
          whiteSpace: "nowrap",
        }}>
          Click to capture mouse · WASD/arrows: move · Space/Shift: up/down · Esc: release
        </div>
      )}
    </div>
  );
}
