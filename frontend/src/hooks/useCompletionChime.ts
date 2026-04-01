"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Hook that plays a chime sound when a task completes.
 * Uses Web Audio API — no sound files needed.
 */
export function useCompletionChime() {
  const [muted, setMuted] = useState(false);
  const prevStepRef = useRef<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const playChime = useCallback(() => {
    if (muted) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      const now = ctx.currentTime;

      // Two-tone chime: C5 then E5
      const notes = [523.25, 659.25];
      notes.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, now + i * 0.15);
        gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.15 + 0.4);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + i * 0.15);
        osc.stop(now + i * 0.15 + 0.5);
      });
    } catch {
      // Audio not available
    }
  }, [muted]);

  const onStepChange = useCallback(
    (step: string | undefined) => {
      if (!step) return;
      const prev = prevStepRef.current;
      prevStepRef.current = step;

      // Chime on completion or failure transitions
      if (prev && prev !== step) {
        const completions = [
          "frames_ready",
          "sfm_ready",
          "training_complete",
        ];
        if (completions.includes(step) || step === "failed") {
          playChime();
        }
      }
    },
    [playChime]
  );

  return { muted, setMuted, onStepChange, playChime };
}
