"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  RotateCcw,
  Maximize2,
  Minimize2,
  Camera,
  ZoomIn,
  ZoomOut,
  Share2,
  Download,
  Copy,
  Palette,
} from "lucide-react";

/* ---------- types ---------- */

interface Props {
  plyUrl: string;
  onScreenshot?: (dataUrl: string) => void;
}

type BgMode = "black" | "darkgray" | "white" | "checker";

const BG_CYCLE: BgMode[] = ["black", "darkgray", "white", "checker"];

const BG_COLORS: Record<BgMode, string> = {
  black: "#000000",
  darkgray: "#1a1a1a",
  white: "#ffffff",
  checker: "#1a1a1a", // base for the checker pattern
};

const BG_LABELS: Record<BgMode, string> = {
  black: "Black",
  darkgray: "Dark Gray",
  white: "White",
  checker: "Checkerboard",
};

const INITIAL_CAMERA_POSITION = [0, 0, 5] as const;
const INITIAL_CAMERA_LOOK_AT = [0, 0, 0] as const;

/* ---------- tiny tooltip wrapper ---------- */

function IconButton({
  label,
  onClick,
  children,
  className = "",
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={
        "relative p-1.5 rounded-md text-white/80 hover:text-white hover:bg-white/15 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-white/50 " +
        className
      }
    >
      {children}
    </button>
  );
}

/* ---------- main component ---------- */

export function SplatViewer({ plyUrl, onScreenshot }: Props) {
  /* refs */
  const outerRef = useRef<HTMLDivElement>(null); // fullscreen target
  const containerRef = useRef<HTMLDivElement>(null); // viewer root
  const viewerRef = useRef<any>(null);
  const threeRendererRef = useRef<any>(null);

  /* state */
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bgMode, setBgMode] = useState<BgMode>("black");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  /* ---------- toast helper ---------- */
  const toast = useCallback((msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 2000);
  }, []);

  /* ---------- init viewer ---------- */
  useEffect(() => {
    if (!containerRef.current) return;

    let disposed = false;

    async function init() {
      try {
        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
        if (disposed || !containerRef.current) return;

        const viewer = new GaussianSplats3D.Viewer({
          cameraUp: [0, -1, 0],
          initialCameraPosition: [...INITIAL_CAMERA_POSITION],
          initialCameraLookAt: [...INITIAL_CAMERA_LOOK_AT],
          rootElement: containerRef.current,
          sharedMemoryForWorkers: false,
        });

        viewerRef.current = viewer;

        // Try to grab the Three.js renderer for screenshots / bg color
        try {
          const renderer =
            (viewer as any).renderer ??
            (viewer as any).threeRenderer ??
            (viewer as any).renderModule?.renderer;
          if (renderer) threeRendererRef.current = renderer;
        } catch {
          /* non-critical */
        }

        await viewer.addSplatScene(plyUrl, {
          showLoadingUI: true,
          progressiveLoad: true,
        });

        if (!disposed) {
          viewer.start();
          setLoading(false);
        }
      } catch (err: any) {
        if (!disposed) {
          setError(err.message || "Failed to load splat viewer");
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposed = true;
      if (viewerRef.current) {
        try {
          viewerRef.current.dispose();
        } catch {
          /* swallow */
        }
        viewerRef.current = null;
      }
    };
  }, [plyUrl]);

  /* ---------- background color sync ---------- */
  useEffect(() => {
    if (!threeRendererRef.current) return;
    try {
      const THREE = (window as any).THREE ?? require("three");
      const hex = BG_COLORS[bgMode];
      threeRendererRef.current.setClearColor(new THREE.Color(hex), 1);
    } catch {
      /* three may not be on window; fallback is CSS bg */
    }
  }, [bgMode]);

  /* ---------- fullscreen listener ---------- */
  useEffect(() => {
    function onFs() {
      setIsFullscreen(!!document.fullscreenElement);
    }
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  /* ---------- pinch-to-zoom & two-finger pan ---------- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let lastDist = 0;
    let lastMid = { x: 0, y: 0 };
    let activeTouches = 0;

    function dist(a: Touch, b: Touch) {
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    }

    function mid(a: Touch, b: Touch) {
      return {
        x: (a.clientX + b.clientX) / 2,
        y: (a.clientY + b.clientY) / 2,
      };
    }

    function onTouchStart(e: TouchEvent) {
      activeTouches = e.touches.length;
      if (e.touches.length === 2) {
        lastDist = dist(e.touches[0], e.touches[1]);
        lastMid = mid(e.touches[0], e.touches[1]);
      }
    }

    function onTouchMove(e: TouchEvent) {
      if (e.touches.length !== 2) return;
      const target = containerRef.current;
      if (!target) return;

      const d = dist(e.touches[0], e.touches[1]);
      const m = mid(e.touches[0], e.touches[1]);

      // pinch-to-zoom: synthesize wheel events
      const delta = lastDist - d;
      if (Math.abs(delta) > 1) {
        const synth = new WheelEvent("wheel", {
          deltaY: delta * 2,
          clientX: m.x,
          clientY: m.y,
          bubbles: true,
        });
        target.dispatchEvent(synth);
      }

      // two-finger pan: synthesize pointer-move with right button (viewer uses this for pan)
      const dx = m.x - lastMid.x;
      const dy = m.y - lastMid.y;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
        const down = new PointerEvent("pointerdown", {
          clientX: lastMid.x,
          clientY: lastMid.y,
          button: 2,
          buttons: 2,
          bubbles: true,
        });
        const move = new PointerEvent("pointermove", {
          clientX: m.x,
          clientY: m.y,
          button: 2,
          buttons: 2,
          bubbles: true,
        });
        const up = new PointerEvent("pointerup", {
          clientX: m.x,
          clientY: m.y,
          button: 2,
          buttons: 0,
          bubbles: true,
        });
        target.dispatchEvent(down);
        target.dispatchEvent(move);
        target.dispatchEvent(up);
      }

      lastDist = d;
      lastMid = m;
    }

    function onTouchEnd() {
      activeTouches = 0;
      lastDist = 0;
    }

    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd, { passive: true });

    return () => {
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
    };
  }, []);

  /* ---------- toolbar actions ---------- */

  const resetCamera = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    try {
      // The library exposes camera through the Three.js orbit controls
      const camera =
        (viewer as any).camera ??
        (viewer as any).perspectiveCamera ??
        (viewer as any).controls?.object;
      if (camera) {
        camera.position.set(...INITIAL_CAMERA_POSITION);
        camera.lookAt(...INITIAL_CAMERA_LOOK_AT);
      }
      const controls =
        (viewer as any).controls ?? (viewer as any).orbitControls;
      if (controls) {
        controls.target?.set(...INITIAL_CAMERA_LOOK_AT);
        controls.update?.();
      }
    } catch {
      /* best-effort */
    }
  }, []);

  const cycleBg = useCallback(() => {
    setBgMode((prev) => {
      const idx = BG_CYCLE.indexOf(prev);
      return BG_CYCLE[(idx + 1) % BG_CYCLE.length];
    });
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = outerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }, []);

  const takeScreenshot = useCallback(() => {
    try {
      const canvas = containerRef.current?.querySelector("canvas");
      if (!canvas) {
        toast("No canvas found");
        return;
      }
      const dataUrl = canvas.toDataURL("image/png");

      onScreenshot?.(dataUrl);

      // Copy to clipboard via blob
      canvas.toBlob(async (blob) => {
        if (!blob) return;
        try {
          await navigator.clipboard.write([
            new ClipboardItem({ "image/png": blob }),
          ]);
          toast("Screenshot copied to clipboard");
        } catch {
          // Fallback: open in new tab
          const url = URL.createObjectURL(blob);
          window.open(url, "_blank");
          toast("Screenshot opened in new tab");
        }
      }, "image/png");
    } catch {
      toast("Screenshot failed");
    }
  }, [onScreenshot, toast]);

  const zoom = useCallback((direction: "in" | "out") => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const synth = new WheelEvent("wheel", {
      deltaY: direction === "in" ? -120 : 120,
      clientX: cx,
      clientY: cy,
      bubbles: true,
    });
    el.dispatchEvent(synth);
  }, []);

  const copyEmbedCode = useCallback(() => {
    const base =
      typeof window !== "undefined" ? window.location.href : "https://example.com";
    const snippet = `<iframe src="${base}/embed" width="800" height="600" frameborder="0"></iframe>`;
    navigator.clipboard
      .writeText(snippet)
      .then(() => toast("Embed code copied"))
      .catch(() => toast("Copy failed"));
    setExportOpen(false);
  }, [toast]);

  /* ---------- derived styles ---------- */

  const bgCss =
    bgMode === "checker"
      ? {
          backgroundImage:
            "repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%)",
          backgroundSize: "24px 24px",
        }
      : { backgroundColor: BG_COLORS[bgMode] };

  /* ---------- render ---------- */
  return (
    <div
      ref={outerRef}
      className="relative w-full h-full min-h-[500px] rounded-lg overflow-hidden select-none"
      style={bgCss}
    >
      {/* Three.js mount target */}
      <div ref={containerRef} className="w-full h-full" />

      {/* ───── Toolbar ───── */}
      {!loading && !error && (
        <div className="absolute top-0 inset-x-0 z-20 flex items-center gap-1 px-2 py-1.5 bg-black/60 backdrop-blur-sm text-sm flex-wrap">
          {/* Reset camera */}
          <IconButton label="Reset Camera" onClick={resetCamera}>
            <RotateCcw className="w-4 h-4" />
          </IconButton>

          {/* Background */}
          <IconButton
            label={`Background: ${BG_LABELS[bgMode]}`}
            onClick={cycleBg}
          >
            <Palette className="w-4 h-4" />
          </IconButton>

          {/* Fullscreen */}
          <IconButton
            label={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            onClick={toggleFullscreen}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </IconButton>

          {/* Screenshot */}
          <IconButton label="Screenshot to Clipboard" onClick={takeScreenshot}>
            <Camera className="w-4 h-4" />
          </IconButton>

          {/* Zoom */}
          <IconButton label="Zoom In" onClick={() => zoom("in")}>
            <ZoomIn className="w-4 h-4" />
          </IconButton>
          <IconButton label="Zoom Out" onClick={() => zoom("out")}>
            <ZoomOut className="w-4 h-4" />
          </IconButton>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Export / Share dropdown */}
          <div className="relative">
            <IconButton
              label="Share / Export"
              onClick={() => setExportOpen((v) => !v)}
            >
              <Share2 className="w-4 h-4" />
            </IconButton>

            {exportOpen && (
              <div className="absolute right-0 top-full mt-1 w-52 rounded-lg bg-gray-900/95 backdrop-blur border border-white/10 shadow-xl py-1 z-30">
                {/* Download .ply */}
                <a
                  href={plyUrl}
                  download
                  onClick={() => setExportOpen(false)}
                  className="flex items-center gap-2 px-3 py-2 text-sm text-white/80 hover:text-white hover:bg-white/10 transition-colors"
                >
                  <Download className="w-4 h-4 shrink-0" />
                  Download .ply
                </a>

                {/* Copy embed */}
                <button
                  type="button"
                  onClick={copyEmbedCode}
                  className="flex items-center gap-2 px-3 py-2 text-sm text-white/80 hover:text-white hover:bg-white/10 transition-colors w-full text-left"
                >
                  <Copy className="w-4 h-4 shrink-0" />
                  Copy Embed Code
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ───── Close export panel on outside click ───── */}
      {exportOpen && (
        <div
          className="absolute inset-0 z-10"
          onClick={() => setExportOpen(false)}
        />
      )}

      {/* ───── Mobile touch hint ───── */}
      {!loading && !error && (
        <div className="absolute bottom-2 inset-x-0 text-center pointer-events-none">
          <span className="text-[11px] text-white/40 sm:hidden">
            Pinch to zoom &middot; Two-finger drag to pan
          </span>
        </div>
      )}

      {/* ───── Toast ───── */}
      {toastMsg && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 px-4 py-2 rounded-lg bg-black/80 text-white text-sm backdrop-blur shadow-lg animate-[fadeIn_0.15s_ease]">
          {toastMsg}
        </div>
      )}

      {/* ───── Loading overlay ───── */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-white z-40">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-3" />
            <p>Loading Gaussian Splat...</p>
          </div>
        </div>
      )}

      {/* ───── Error overlay ───── */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-red-400 z-40">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}
