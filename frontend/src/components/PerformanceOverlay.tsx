"use client";

import { useEffect, useRef, useState } from "react";
import type { RendererStats } from "@/lib/renderers/mkkellogg";

interface Props {
  getStats: () => RendererStats | null;
  visible: boolean;
}

export default function PerformanceOverlay({ getStats, visible }: Props) {
  const [stats, setStats] = useState<RendererStats | null>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!visible) return;

    let running = true;
    const tick = () => {
      if (!running) return;
      const s = getStats();
      if (s) setStats(s);
      rafRef.current = requestAnimationFrame(tick);
    };
    // Update at ~4Hz to avoid excessive re-renders
    const interval = setInterval(() => {
      const s = getStats();
      if (s) setStats(s);
    }, 250);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      clearInterval(interval);
    };
  }, [getStats, visible]);

  if (!visible || !stats) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: 12,
        left: 12,
        background: "rgba(0, 0, 0, 0.7)",
        color: "#fff",
        padding: "8px 12px",
        borderRadius: 6,
        fontSize: 12,
        fontFamily: "monospace",
        lineHeight: 1.6,
        pointerEvents: "none",
        zIndex: 50,
      }}
    >
      <div>FPS: {stats.fps}</div>
      <div>Splats: {stats.splatCount.toLocaleString()}</div>
      <div>Draw calls: {stats.drawCalls}</div>
      <div>Renderer: {stats.rendererType}</div>
    </div>
  );
}
