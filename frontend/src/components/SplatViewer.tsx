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
  Layers,
  ChevronDown,
} from "lucide-react";
import { api, type LodInfo } from "@/lib/api";

/* ---------- types ---------- */

interface Props {
  plyUrl: string;
  ksplatUrl?: string;
  projectId?: string;
  onScreenshot?: (dataUrl: string) => void;
}

type BgMode = "black" | "darkgray" | "white" | "checker";
type LodMode = "auto" | "preview" | "medium" | "full";

const BG_CYCLE: BgMode[] = ["black", "darkgray", "white", "checker"];

const BG_COLORS: Record<BgMode, string> = {
  black: "#000000",
  darkgray: "#1a1a1a",
  white: "#ffffff",
  checker: "#1a1a1a",
};

const BG_LABELS: Record<BgMode, string> = {
  black: "Black",
  darkgray: "Dark Gray",
  white: "White",
  checker: "Checkerboard",
};

const LOD_LABELS: Record<number, string> = {
  0: "Preview",
  1: "Medium",
  2: "Full Quality",
};

const LOD_MODE_LABELS: Record<LodMode, string> = {
  auto: "Auto (Progressive)",
  preview: "Preview Only",
  medium: "Medium",
  full: "Full Quality",
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

/* ---------- LOD badge ---------- */

function LodBadge({
  currentLod,
  targetLod,
  upgrading,
}: {
  currentLod: number;
  targetLod: number;
  upgrading: boolean;
}) {
  const label = LOD_LABELS[currentLod] ?? "Loading";
  const isPreview = currentLod < 2;

  return (
    <div
      className={
        "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium select-none " +
        (currentLod === 2
          ? "bg-green-600/80 text-white"
          : currentLod === 1
          ? "bg-blue-600/80 text-white"
          : "bg-amber-600/80 text-white")
      }
    >
      <Layers className="w-3 h-3" />
      <span>{label}</span>
      {upgrading && isPreview && (
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
      )}
    </div>
  );
}

/* ---------- LOD quality selector dropdown ---------- */

function LodSelector({
  mode,
  onChange,
  hasLod,
}: {
  mode: LodMode;
  onChange: (mode: LodMode) => void;
  hasLod: boolean;
}) {
  const [open, setOpen] = useState(false);

  if (!hasLod) return null;

  const modes: LodMode[] = ["auto", "preview", "medium", "full"];

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-white/80 hover:text-white hover:bg-white/15 transition-colors"
        title="Quality setting"
      >
        <Layers className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">{LOD_MODE_LABELS[mode].split(" ")[0]}</span>
        <ChevronDown className="w-3 h-3" />
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 top-full mt-1 w-44 rounded-lg bg-gray-900/95 backdrop-blur border border-white/10 shadow-xl py-1 z-50">
            {modes.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => {
                  onChange(m);
                  setOpen(false);
                }}
                className={
                  "flex items-center gap-2 px-3 py-2 text-sm w-full text-left transition-colors " +
                  (m === mode
                    ? "text-white bg-white/10"
                    : "text-white/70 hover:text-white hover:bg-white/5")
                }
              >
                {LOD_MODE_LABELS[m]}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ---------- main component ---------- */

export function SplatViewer({ plyUrl, ksplatUrl, projectId, onScreenshot }: Props) {
  /* refs */
  const outerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const threeRendererRef = useRef<any>(null);
  const disposedRef = useRef(false);
  const currentLodLoadedRef = useRef(-1);

  /* state */
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bgMode, setBgMode] = useState<BgMode>("black");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  /* LOD state */
  const [lodInfo, setLodInfo] = useState<LodInfo | null>(null);
  const [currentLod, setCurrentLod] = useState(-1);
  const [targetLod, setTargetLod] = useState(2);
  const [lodMode, setLodMode] = useState<LodMode>("auto");
  const [upgrading, setUpgrading] = useState(false);

  /* ---------- toast helper ---------- */
  const toast = useCallback((msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 2000);
  }, []);

  /* ---------- fetch LOD info ---------- */
  useEffect(() => {
    if (!projectId) return;
    api.getLodInfo(projectId).then(setLodInfo).catch(() => {
      // LOD not available, use fallback
      setLodInfo(null);
    });
  }, [projectId]);

  /* ---------- create / destroy viewer ---------- */
  const createViewer = useCallback(
    async (url: string): Promise<any> => {
      if (!containerRef.current) return null;

      const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");

      const viewer = new GaussianSplats3D.Viewer({
        cameraUp: [0, -1, 0],
        initialCameraPosition: [...INITIAL_CAMERA_POSITION],
        initialCameraLookAt: [...INITIAL_CAMERA_LOOK_AT],
        rootElement: containerRef.current,
        sharedMemoryForWorkers: false,
        dynamicScene: false,
        antialiased: false,
        devicePixelRatio: 1,
      });

      // Grab the Three.js renderer
      try {
        const renderer =
          (viewer as any).renderer ??
          (viewer as any).threeRenderer ??
          (viewer as any).renderModule?.renderer;
        if (renderer) threeRendererRef.current = renderer;
      } catch {
        /* non-critical */
      }

      await viewer.addSplatScene(url, {
        showLoadingUI: false,
        progressiveLoad: true,
      });

      viewer.start();
      return viewer;
    },
    []
  );

  const disposeViewer = useCallback(() => {
    if (viewerRef.current) {
      try {
        viewerRef.current.dispose();
      } catch {
        /* swallow */
      }
      viewerRef.current = null;
      threeRendererRef.current = null;
    }
  }, []);

  /* ---------- save & restore camera state between LOD swaps ---------- */
  const saveCameraState = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return null;
    try {
      const camera =
        (viewer as any).camera ??
        (viewer as any).perspectiveCamera ??
        (viewer as any).controls?.object;
      const controls =
        (viewer as any).controls ?? (viewer as any).orbitControls;
      if (camera && controls?.target) {
        return {
          position: [camera.position.x, camera.position.y, camera.position.z],
          target: [controls.target.x, controls.target.y, controls.target.z],
        };
      }
    } catch {
      /* best-effort */
    }
    return null;
  }, []);

  const restoreCameraState = useCallback(
    (state: { position: number[]; target: number[] } | null) => {
      if (!state) return;
      const viewer = viewerRef.current;
      if (!viewer) return;
      try {
        const camera =
          (viewer as any).camera ??
          (viewer as any).perspectiveCamera ??
          (viewer as any).controls?.object;
        const controls =
          (viewer as any).controls ?? (viewer as any).orbitControls;
        if (camera) {
          camera.position.set(...state.position);
        }
        if (controls?.target) {
          controls.target.set(...state.target);
          controls.update?.();
        }
      } catch {
        /* best-effort */
      }
    },
    []
  );

  /* ---------- load a specific LOD level ---------- */
  const loadLod = useCallback(
    async (level: number) => {
      if (disposedRef.current || !containerRef.current) return;
      if (currentLodLoadedRef.current === level) return;

      let url: string;
      if (projectId && lodInfo?.has_lod) {
        const lodLevel = lodInfo.levels.find(
          (l) => l.level === level && l.available
        );
        if (lodLevel) {
          url = api.getLodUrl(projectId, level);
        } else if (level === 2) {
          // Full quality fallback -- use ksplat if available
          url = ksplatUrl || plyUrl;
        } else {
          // Requested LOD not available, skip
          return;
        }
      } else {
        // No LOD available, only load full (prefer ksplat)
        if (level < 2) return;
        url = ksplatUrl || plyUrl;
      }

      // Save camera before swap (if viewer exists)
      const cameraState = saveCameraState();

      setUpgrading(true);

      try {
        // Dispose old viewer
        disposeViewer();

        // Clear container (remove old canvases)
        if (containerRef.current) {
          containerRef.current.innerHTML = "";
        }

        // Create new viewer with this LOD
        const viewer = await createViewer(url);
        if (disposedRef.current) {
          try { viewer?.dispose(); } catch { /* */ }
          return;
        }

        viewerRef.current = viewer;
        currentLodLoadedRef.current = level;
        setCurrentLod(level);
        setLoading(false);

        // Restore camera after a small delay to let viewer initialize
        if (cameraState) {
          setTimeout(() => restoreCameraState(cameraState), 100);
        }
      } catch (err: any) {
        if (!disposedRef.current) {
          setError(err.message || "Failed to load splat viewer");
          setLoading(false);
        }
      } finally {
        setUpgrading(false);
      }
    },
    [
      plyUrl,
      ksplatUrl,
      projectId,
      lodInfo,
      createViewer,
      disposeViewer,
      saveCameraState,
      restoreCameraState,
    ]
  );

  /* ---------- progressive loading orchestration ---------- */
  useEffect(() => {
    disposedRef.current = false;
    currentLodLoadedRef.current = -1;
    setCurrentLod(-1);
    setLoading(true);
    setError(null);

    // Determine what to load based on LOD mode and availability
    const hasLod = lodInfo?.has_lod ?? false;

    async function progressiveLoad() {
      if (!hasLod || lodMode === "full") {
        // No LOD or user wants full: load full directly
        await loadLod(2);
        return;
      }

      if (lodMode === "preview") {
        await loadLod(0);
        return;
      }

      if (lodMode === "medium") {
        const hasLod1 = lodInfo?.levels.find(
          (l) => l.level === 1 && l.available
        );
        if (hasLod1) {
          await loadLod(1);
        } else {
          await loadLod(2);
        }
        return;
      }

      // Auto mode: progressive upgrade 0 -> 1 -> 2
      const hasLod0 = lodInfo?.levels.find(
        (l) => l.level === 0 && l.available
      );
      const hasLod1 = lodInfo?.levels.find(
        (l) => l.level === 1 && l.available
      );

      if (hasLod0) {
        await loadLod(0);
        if (disposedRef.current) return;

        // Wait a moment for user to start interacting, then upgrade
        await new Promise((r) => setTimeout(r, 1500));
        if (disposedRef.current) return;

        if (hasLod1) {
          await loadLod(1);
          if (disposedRef.current) return;
          await new Promise((r) => setTimeout(r, 2000));
          if (disposedRef.current) return;
        }

        await loadLod(2);
      } else if (hasLod1) {
        await loadLod(1);
        if (disposedRef.current) return;
        await new Promise((r) => setTimeout(r, 2000));
        if (disposedRef.current) return;
        await loadLod(2);
      } else {
        await loadLod(2);
      }
    }

    progressiveLoad();

    return () => {
      disposedRef.current = true;
      disposeViewer();
    };
    // We intentionally run this when lodInfo or lodMode changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plyUrl, ksplatUrl, lodInfo, lodMode]);

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

      canvas.toBlob(async (blob) => {
        if (!blob) return;
        try {
          await navigator.clipboard.write([
            new ClipboardItem({ "image/png": blob }),
          ]);
          toast("Screenshot copied to clipboard");
        } catch {
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

  const hasLod = lodInfo?.has_lod ?? false;

  /* ---------- render ---------- */
  return (
    <div
      ref={outerRef}
      className="relative w-full h-full min-h-[500px] rounded-lg overflow-hidden select-none"
      style={bgCss}
    >
      {/* Three.js mount target */}
      <div ref={containerRef} className="w-full h-full" />

      {/* ---- Toolbar ---- */}
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

          {/* LOD quality selector */}
          <LodSelector mode={lodMode} onChange={setLodMode} hasLod={hasLod} />

          {/* Spacer */}
          <div className="flex-1" />

          {/* LOD badge */}
          {hasLod && currentLod >= 0 && (
            <LodBadge
              currentLod={currentLod}
              targetLod={targetLod}
              upgrading={upgrading}
            />
          )}

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

      {/* ---- Close export panel on outside click ---- */}
      {exportOpen && (
        <div
          className="absolute inset-0 z-10"
          onClick={() => setExportOpen(false)}
        />
      )}

      {/* ---- Mobile touch hint ---- */}
      {!loading && !error && (
        <div className="absolute bottom-2 inset-x-0 text-center pointer-events-none">
          <span className="text-[11px] text-white/40 sm:hidden">
            Pinch to zoom &middot; Two-finger drag to pan
          </span>
        </div>
      )}

      {/* ---- Toast ---- */}
      {toastMsg && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 px-4 py-2 rounded-lg bg-black/80 text-white text-sm backdrop-blur shadow-lg animate-[fadeIn_0.15s_ease]">
          {toastMsg}
        </div>
      )}

      {/* ---- Loading overlay ---- */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-white z-40">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-3" />
            <p>Loading Gaussian Splat...</p>
            {hasLod && (
              <p className="text-white/50 text-xs mt-1">
                Loading preview for instant interactivity
              </p>
            )}
          </div>
        </div>
      )}

      {/* ---- Error overlay ---- */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-red-400 z-40">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}
