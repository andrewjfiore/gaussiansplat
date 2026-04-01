"use client";

import { useCallback, useState } from "react";
import { Sun, Moon } from "lucide-react";

interface Props {
  visible: boolean;
  onLightChange: (direction: [number, number, number], intensity: number, ambient: number) => void;
}

/**
 * Simple relighting controls — adjusts a directional light's
 * azimuth, elevation, intensity, and ambient level.
 */
export default function RelightControls({ visible, onLightChange }: Props) {
  const [azimuth, setAzimuth] = useState(45);    // degrees
  const [elevation, setElevation] = useState(45); // degrees
  const [intensity, setIntensity] = useState(1.0);
  const [ambient, setAmbient] = useState(0.3);

  const update = useCallback(
    (az: number, el: number, int_: number, amb: number) => {
      const azRad = (az * Math.PI) / 180;
      const elRad = (el * Math.PI) / 180;
      const dir: [number, number, number] = [
        Math.cos(elRad) * Math.sin(azRad),
        Math.sin(elRad),
        Math.cos(elRad) * Math.cos(azRad),
      ];
      onLightChange(dir, int_, amb);
    },
    [onLightChange]
  );

  if (!visible) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: 50,
        right: 8,
        background: "rgba(0, 0, 0, 0.8)",
        color: "#fff",
        padding: "10px 14px",
        borderRadius: 8,
        fontSize: 11,
        zIndex: 50,
        width: 180,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
        <Sun style={{ width: 14, height: 14 }} />
        <span style={{ fontWeight: 600 }}>Relighting</span>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 6 }}>
        <span style={{ width: 55 }}>Azimuth</span>
        <input
          type="range" min={0} max={360} value={azimuth}
          onChange={(e) => { const v = Number(e.target.value); setAzimuth(v); update(v, elevation, intensity, ambient); }}
          style={{ flex: 1 }}
        />
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 6 }}>
        <span style={{ width: 55 }}>Elevation</span>
        <input
          type="range" min={-90} max={90} value={elevation}
          onChange={(e) => { const v = Number(e.target.value); setElevation(v); update(azimuth, v, intensity, ambient); }}
          style={{ flex: 1 }}
        />
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 6 }}>
        <span style={{ width: 55 }}>Intensity</span>
        <input
          type="range" min={0} max={3} step={0.1} value={intensity}
          onChange={(e) => { const v = Number(e.target.value); setIntensity(v); update(azimuth, elevation, v, ambient); }}
          style={{ flex: 1 }}
        />
      </label>

      <label style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ width: 55 }}>Ambient</span>
        <input
          type="range" min={0} max={1} step={0.05} value={ambient}
          onChange={(e) => { const v = Number(e.target.value); setAmbient(v); update(azimuth, elevation, intensity, v); }}
          style={{ flex: 1 }}
        />
      </label>
    </div>
  );
}
