"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  leftUrl: string;
  rightUrl: string;
  leftLabel?: string;
  rightLabel?: string;
}

/**
 * A/B comparison slider: two splat viewers side-by-side with a draggable divider.
 * Uses CSS clip-path to split a shared viewport.
 */
export default function ComparisonSlider({
  leftUrl,
  rightUrl,
  leftLabel = "A",
  rightLabel = "B",
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sliderPos, setSliderPos] = useState(50); // percentage
  const [dragging, setDragging] = useState(false);

  const onPointerDown = useCallback(() => setDragging(true), []);

  useEffect(() => {
    if (!dragging) return;

    const onMove = (e: PointerEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      setSliderPos(Math.max(5, Math.min(95, x)));
    };
    const onUp = () => setDragging(false);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [dragging]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "#000",
      }}
    >
      {/* Left viewer (clipped to left side) */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 ${100 - sliderPos}% 0 0)`,
        }}
      >
        <iframe
          src={`${leftUrl}&embed=1`}
          style={{ width: "100%", height: "100%", border: "none" }}
          title="Left splat"
        />
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            background: "rgba(0,0,0,0.6)",
            color: "white",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          {leftLabel}
        </div>
      </div>

      {/* Right viewer (clipped to right side) */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 0 0 ${sliderPos}%)`,
        }}
      >
        <iframe
          src={`${rightUrl}&embed=1`}
          style={{ width: "100%", height: "100%", border: "none" }}
          title="Right splat"
        />
        <div
          style={{
            position: "absolute",
            top: 12,
            right: 12,
            background: "rgba(0,0,0,0.6)",
            color: "white",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          {rightLabel}
        </div>
      </div>

      {/* Draggable divider */}
      <div
        onPointerDown={onPointerDown}
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${sliderPos}%`,
          width: 4,
          background: "white",
          cursor: "col-resize",
          zIndex: 10,
          transform: "translateX(-50%)",
          boxShadow: "0 0 8px rgba(0,0,0,0.5)",
        }}
      >
        {/* Handle grip */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: 24,
            height: 40,
            background: "white",
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
          }}
        >
          <span style={{ color: "#666", fontSize: 14, userSelect: "none" }}>||</span>
        </div>
      </div>
    </div>
  );
}
