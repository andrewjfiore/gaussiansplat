"use client";
import { useCallback, useEffect, useRef, useState } from "react";

function formatTime(seconds: number): string {
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}m ${rem}s`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

/**
 * Tracks elapsed time and estimates remaining time based on progress percentage.
 *
 * @param running - whether the timer should be ticking
 * @param percent - current progress 0-100 (optional, for ETA calculation)
 * @returns { elapsed, eta, elapsedStr, etaStr, resetTimer }
 */
export function useElapsedTimer(running: boolean, percent?: number) {
  const startTimeRef = useRef<number | null>(null);
  const prevRunningRef = useRef(false);
  const [elapsed, setElapsed] = useState(0);

  // Detect false→true transition to record startTime
  useEffect(() => {
    if (running && !prevRunningRef.current) {
      startTimeRef.current = Date.now();
      setElapsed(0);
    }
    prevRunningRef.current = running;
  }, [running]);

  // Tick every second while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      if (startTimeRef.current !== null) {
        setElapsed((Date.now() - startTimeRef.current) / 1000);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [running]);

  const resetTimer = useCallback(() => {
    startTimeRef.current = running ? Date.now() : null;
    setElapsed(0);
  }, [running]);

  // ETA calculation
  let eta: number | null = null;
  if (percent != null && percent > 0 && elapsed > 0) {
    const total = elapsed / (percent / 100);
    eta = Math.max(0, total - elapsed);
  }

  const elapsedStr = formatTime(elapsed);
  const etaStr = eta !== null ? `~${formatTime(eta)}` : null;

  return { elapsed, eta, elapsedStr, etaStr, resetTimer };
}
