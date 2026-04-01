"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Repeat,
  Repeat1,
} from "lucide-react";

interface Props {
  projectId: string;
  frameCount: number;
}

function framePlyUrl(projectId: string, frame: number) {
  return `/api/projects/${projectId}/output/temporal_frames/frame_${String(frame).padStart(4, "0")}.ply`;
}

/**
 * Temporal viewer for 4D Gaussian Splats.
 * Loads per-frame PLYs with a persistent Three.js viewer that preserves camera
 * state across frame changes. Provides timeline scrubbing, playback, and looping.
 */
export default function TemporalViewer({ projectId, frameCount }: Props) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1.0);
  const [loop, setLoop] = useState(true);
  const fps = 10;
  const intervalRef = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const loadedFrameRef = useRef<number>(-1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const plyUrl = framePlyUrl(projectId, currentFrame);

  // Init viewer once
  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    async function init() {
      try {
        const GS3D = await import("@mkkellogg/gaussian-splats-3d");
        if (disposed || !containerRef.current) return;

        const viewer = new GS3D.Viewer({
          cameraUp: [0, -1, 0],
          rootElement: containerRef.current,
          sharedMemoryForWorkers: false,
          dynamicScene: true,
          antialiased: false,
          devicePixelRatio: 1,
        });

        viewerRef.current = viewer;

        await viewer.addSplatScene(framePlyUrl(projectId, 0), {
          showLoadingUI: false,
          progressiveLoad: true,
        });
        loadedFrameRef.current = 0;

        if (!disposed) {
          viewer.start();
          setLoading(false);
        }
      } catch (err: any) {
        if (!disposed) {
          setError(err.message || "Failed to load temporal viewer");
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposed = true;
      if (viewerRef.current) {
        try { viewerRef.current.dispose(); } catch { /* */ }
        viewerRef.current = null;
      }
    };
  }, [projectId]);

  // Load new frame when currentFrame changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || loadedFrameRef.current === currentFrame) return;

    let cancelled = false;

    async function loadFrame() {
      try {
        // Remove existing scene(s) and load the new frame
        try { await viewer.removeSplatScene(0); } catch { /* may not exist */ }
        await viewer.addSplatScene(plyUrl, {
          showLoadingUI: false,
          progressiveLoad: false,
        });
        if (!cancelled) {
          loadedFrameRef.current = currentFrame;
        }
      } catch {
        // Frame load failed — skip silently during playback
      }
    }

    loadFrame();
    return () => { cancelled = true; };
  }, [currentFrame, plyUrl]);

  // Playback loop
  useEffect(() => {
    if (!isPlaying) {
      clearInterval(intervalRef.current);
      return;
    }

    intervalRef.current = window.setInterval(() => {
      setCurrentFrame((prev) => {
        const next = prev + 1;
        if (next >= frameCount) {
          if (loop) return 0;
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, 1000 / (fps * playSpeed));

    return () => clearInterval(intervalRef.current);
  }, [isPlaying, fps, playSpeed, frameCount, loop]);

  const togglePlay = useCallback(() => setIsPlaying((p) => !p), []);

  const stepForward = useCallback(() => {
    setCurrentFrame((prev) => {
      if (prev + 1 >= frameCount) return loop ? 0 : prev;
      return prev + 1;
    });
  }, [frameCount, loop]);

  const stepBack = useCallback(() => {
    setCurrentFrame((prev) => {
      if (prev - 1 < 0) return loop ? frameCount - 1 : 0;
      return prev - 1;
    });
  }, [frameCount, loop]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      switch (e.code) {
        case "Space":
          e.preventDefault();
          togglePlay();
          break;
        case "ArrowRight":
          e.preventDefault();
          stepForward();
          break;
        case "ArrowLeft":
          e.preventDefault();
          stepBack();
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlay, stepForward, stepBack]);

  const progress = frameCount > 1 ? (currentFrame / (frameCount - 1)) * 100 : 0;

  return (
    <div className="relative w-full h-full">
      {/* Viewer mount */}
      <div ref={containerRef} className="w-full" style={{ height: "calc(100% - 56px)" }} />

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-white z-40">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-3" />
            <p>Loading 4D Gaussian Splat...</p>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-red-400 z-40">
          <p>{error}</p>
        </div>
      )}

      {/* Timeline controls */}
      <div className="absolute bottom-0 left-0 right-0 h-14 bg-black/90 backdrop-blur-sm flex items-center px-4 gap-3 z-50">
        {/* Playback buttons */}
        <button onClick={stepBack} className={btnClass}>
          <SkipBack className="w-4 h-4" />
        </button>
        <button onClick={togglePlay} className={btnClass}>
          {isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>
        <button onClick={stepForward} className={btnClass}>
          <SkipForward className="w-4 h-4" />
        </button>

        {/* Loop toggle */}
        <button
          onClick={() => setLoop((l) => !l)}
          className={loop ? btnClassActive : btnClass}
          title={loop ? "Loop: on" : "Loop: off"}
        >
          {loop ? <Repeat className="w-4 h-4" /> : <Repeat1 className="w-4 h-4" />}
        </button>

        {/* Frame counter */}
        <span className="text-gray-400 text-xs font-mono min-w-[64px] text-center">
          {currentFrame + 1} / {frameCount}
        </span>

        {/* Timeline scrubber */}
        <div className="flex-1 relative group">
          {/* Progress fill */}
          <div className="absolute top-1/2 left-0 right-0 h-1 -translate-y-1/2 bg-gray-700 rounded-full overflow-hidden pointer-events-none">
            <div
              className="h-full bg-purple-500 transition-[width] duration-75"
              style={{ width: `${progress}%` }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={frameCount - 1}
            value={currentFrame}
            onChange={(e) => setCurrentFrame(Number(e.target.value))}
            className="relative w-full h-5 opacity-0 cursor-pointer"
          />
        </div>

        {/* Speed control */}
        <select
          value={playSpeed}
          onChange={(e) => setPlaySpeed(Number(e.target.value))}
          className="bg-white/10 text-white border border-white/20 rounded px-2 py-1 text-xs"
        >
          <option value={0.25}>0.25x</option>
          <option value={0.5}>0.5x</option>
          <option value={1}>1x</option>
          <option value={2}>2x</option>
        </select>

        {/* 4D badge */}
        <span className="bg-purple-500/30 text-purple-300 px-2 py-0.5 rounded text-xs font-semibold">
          4D
        </span>
      </div>
    </div>
  );
}

const btnClass =
  "bg-white/10 hover:bg-white/20 text-white border-none rounded p-1.5 cursor-pointer flex items-center transition-colors";
const btnClassActive =
  "bg-purple-500/30 hover:bg-purple-500/40 text-purple-300 border-none rounded p-1.5 cursor-pointer flex items-center transition-colors";
