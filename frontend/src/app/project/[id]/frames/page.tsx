"use client";
import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useElapsedTimer } from "@/hooks/useElapsedTimer";
import { LogStream } from "@/components/LogStream";
import { FrameGrid } from "@/components/FrameGrid";
import { SplatViewer } from "@/components/SplatViewer";
import type { FrameInfo, ProjectDetail, QuickPreviewStatus, QuickPreviewComplete } from "@/lib/types";
import { Play, ArrowRight, ArrowLeft, Loader2, XCircle, RotateCcw, Clock, Settings2, Volume2, VolumeX, Scissors, Eye, Sparkles, Maximize2 } from "lucide-react";
import { useCompletionChime } from "@/hooks/useCompletionChime";

const FPS_OPTIONS = [
  { value: 1, label: "1 fps", desc: "Fewer frames, faster processing" },
  { value: 2, label: "2 fps", desc: "Balanced (recommended)" },
  { value: 3, label: "3 fps", desc: "More frames, better quality" },
  { value: 5, label: "5 fps", desc: "Many frames, slow SfM" },
];

export default function FramesPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const { logs, progress, status, clearLogs } = useWebSocket(id);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [frames, setFrames] = useState<FrameInfo[]>([]);
  const [starting, setStarting] = useState(false);
  const [fps, setFps] = useState(2);
  const [showSettings, setShowSettings] = useState(false);
  const [sharpMode, setSharpMode] = useState(false);
  const [sharpWindow, setSharpWindow] = useState(5);
  const { muted, setMuted, onStepChange } = useCompletionChime();

  // Masking state
  const [maskKeywords, setMaskKeywords] = useState("person");
  const [maskPrecision, setMaskPrecision] = useState(0.3);
  const [maskExpand, setMaskExpand] = useState(5);
  const [maskFeather, setMaskFeather] = useState(3);
  const [maskInvert, setMaskInvert] = useState(false);
  const [showMaskSettings, setShowMaskSettings] = useState(false);
  const [maskPreviews, setMaskPreviews] = useState<{ name: string; url: string }[]>([]);

  // Quick preview state
  const [previewStatus, setPreviewStatus] = useState<QuickPreviewStatus | null>(null);
  const [previewGenerating, setPreviewGenerating] = useState(false);
  const [previewResult, setPreviewResult] = useState<QuickPreviewComplete | null>(null);

  useEffect(() => { onStepChange(project?.step); }, [project?.step, onStepChange]);

  const isExtracting = project?.step === "extracting_frames";
  const isMasking = project?.step === "masking";
  const isBusy = isExtracting || isMasking;
  const isFailed = project?.step === "failed";

  const framesReady =
    project?.step === "frames_ready" ||
    project?.step === "masks_ready" ||
    project?.step === "running_sfm" ||
    project?.step === "sfm_ready" ||
    project?.step === "training" ||
    project?.step === "training_complete";

  const masksReady =
    project?.step === "masks_ready" ||
    project?.step === "running_sfm" ||
    project?.step === "sfm_ready" ||
    project?.step === "training" ||
    project?.step === "training_complete";

  const { elapsedStr, etaStr, resetTimer } = useElapsedTimer(
    isBusy || false,
    progress?.percent ?? undefined
  );

  const refresh = () => {
    api.getProject(id).then(setProject);
    api.listFrames(id).then(setFrames);
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [id]);

  // Load mask previews when masks are ready
  useEffect(() => {
    if (masksReady) {
      api.listMasks(id).then(setMaskPreviews).catch(() => {});
    }
  }, [id, masksReady]);

  // Check for existing quick preview when frames are ready
  useEffect(() => {
    if (framesReady) {
      api.getQuickPreviewStatus(id).then(setPreviewStatus).catch(() => {});
    }
  }, [id, framesReady]);

  // Listen for quick_preview completion via WebSocket status updates
  useEffect(() => {
    if (!previewGenerating) return;
    if (
      status?.step === "quick_preview" &&
      (status?.state === "completed" || status?.state === "failed")
    ) {
      setPreviewGenerating(false);
      if (status.state === "completed") {
        api.getQuickPreviewStatus(id).then(setPreviewStatus).catch(() => {});
      }
    }
  }, [status, previewGenerating, id]);

  const handleQuickPreview = useCallback(async () => {
    setPreviewGenerating(true);
    setPreviewResult(null);
    try {
      await api.generateQuickPreview(id);
    } catch (err: any) {
      alert(err.message);
      setPreviewGenerating(false);
    }
  }, [id]);

  const handleExtract = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.extractFrames(id, {
        fps,
        sharp_frame_selection: sharpMode,
        sharp_window: sharpWindow,
      });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  const handleMask = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.runMasking(id, {
        keywords: maskKeywords,
        precision: maskPrecision,
        expand: maskExpand,
        feather: maskFeather,
        invert: maskInvert,
      });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  const handleCancel = async () => {
    await api.cancelPipeline(id);
  };

  return (
    <div className="space-y-6">
      {/* Extraction Settings */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-white transition"
        >
          <Settings2 className="w-4 h-4" />
          Frame Extraction Settings
        </button>
        {showSettings && (
          <div className="mt-3 space-y-3">
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 whitespace-nowrap">
                Frame Rate:
              </label>
              <select
                value={fps}
                onChange={(e) => setFps(Number(e.target.value))}
                disabled={isBusy}
                className="bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white"
              >
                {FPS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label} — {o.desc}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sharpMode}
                  onChange={(e) => setSharpMode(e.target.checked)}
                  disabled={isBusy}
                  className="rounded border-gray-600"
                />
                Sharp Frame Selection
              </label>
              {sharpMode && (
                <>
                  <label className="text-sm text-gray-400">Window:</label>
                  <input
                    type="range"
                    min={2}
                    max={15}
                    value={sharpWindow}
                    onChange={(e) => setSharpWindow(Number(e.target.value))}
                    disabled={isBusy}
                    className="w-20"
                  />
                  <span className="text-sm text-white w-8">{sharpWindow}</span>
                </>
              )}
            </div>
            {sharpMode && (
              <p className="text-xs text-gray-500">
                Extracts {2 * sharpWindow + 1}x more frames, then keeps the sharpest
                per interval. Better quality but slower extraction.
              </p>
            )}
            <p className="text-xs text-gray-500">
              Higher FPS extracts more frames. More frames give better 3D
              quality but make SfM slower.
            </p>
          </div>
        )}
      </div>

      {/* Masking Settings — visible after frames are extracted */}
      {framesReady && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <button
            onClick={() => setShowMaskSettings(!showMaskSettings)}
            className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-white transition"
          >
            <Scissors className="w-4 h-4" />
            Object Masking
            {masksReady && project?.mask_count ? (
              <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                {project.mask_count} masks
              </span>
            ) : null}
          </button>
          {showMaskSettings && (
            <div className="mt-3 space-y-3">
              <div className="flex items-center gap-4">
                <label className="text-sm text-gray-400 whitespace-nowrap">
                  Keywords:
                </label>
                <input
                  type="text"
                  value={maskKeywords}
                  onChange={(e) => setMaskKeywords(e.target.value)}
                  disabled={isBusy}
                  placeholder="person.tripod.camera"
                  className="flex-1 bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white placeholder-gray-500"
                />
              </div>
              <p className="text-xs text-gray-500">
                Separate keywords with dots. Objects matching any keyword will be masked out
                (removed from reconstruction). Common: person, tripod, camera, car
              </p>
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-400">Precision:</label>
                  <input
                    type="range" min={0.1} max={0.9} step={0.05}
                    value={maskPrecision}
                    onChange={(e) => setMaskPrecision(Number(e.target.value))}
                    disabled={isBusy} className="w-20"
                  />
                  <span className="text-sm text-white w-8">{maskPrecision.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-400">Expand:</label>
                  <input
                    type="range" min={0} max={30} step={1}
                    value={maskExpand}
                    onChange={(e) => setMaskExpand(Number(e.target.value))}
                    disabled={isBusy} className="w-20"
                  />
                  <span className="text-sm text-white w-6">{maskExpand}px</span>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-400">Feather:</label>
                  <input
                    type="range" min={0} max={20} step={1}
                    value={maskFeather}
                    onChange={(e) => setMaskFeather(Number(e.target.value))}
                    disabled={isBusy} className="w-20"
                  />
                  <span className="text-sm text-white w-6">{maskFeather}px</span>
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input
                    type="checkbox" checked={maskInvert}
                    onChange={(e) => setMaskInvert(e.target.checked)}
                    disabled={isBusy} className="rounded border-gray-600"
                  />
                  Invert
                </label>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleMask}
                  disabled={starting || isBusy || !maskKeywords.trim()}
                  className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition text-sm"
                >
                  {isMasking ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Scissors className="w-4 h-4" />
                  )}
                  {isMasking ? "Masking..." : masksReady ? "Re-mask" : "Generate Masks"}
                </button>
                <button
                  onClick={() => router.push(`/project/${id}/sfm`)}
                  className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition text-sm"
                >
                  Skip Masking <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Mask previews */}
          {maskPreviews.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-400 mb-2">Mask previews (first 6):</p>
              <div className="flex gap-2 overflow-x-auto">
                {maskPreviews.slice(0, 6).map((m) => (
                  <img
                    key={m.name}
                    src={m.url}
                    alt={m.name}
                    className="w-24 h-16 object-cover rounded border border-gray-600"
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Quick 3D Preview */}
      {framesReady && !isBusy && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-amber-400" />
                <h3 className="text-sm font-medium text-white">Quick 3D Preview</h3>
                {previewStatus?.exists && !previewGenerating && (
                  <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                    Ready
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {previewStatus?.exists && (
                  <button
                    onClick={() => router.push(`/project/${id}/view`)}
                    className="text-xs text-gray-400 hover:text-white flex items-center gap-1 transition"
                  >
                    <Maximize2 className="w-3 h-3" /> Full Screen
                  </button>
                )}
                {!previewGenerating && (
                  <button
                    onClick={handleQuickPreview}
                    disabled={previewGenerating}
                    className="bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 disabled:from-gray-600 disabled:to-gray-600 text-white text-sm px-4 py-1.5 rounded-lg flex items-center gap-2 transition font-medium shadow-lg shadow-amber-500/20"
                  >
                    <Sparkles className="w-4 h-4" />
                    {previewStatus?.exists ? "Regenerate" : "Quick 3D Preview"}
                  </button>
                )}
                {previewGenerating && (
                  <div className="flex items-center gap-2 text-sm text-amber-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Generating preview...
                  </div>
                )}
              </div>
            </div>
            {!previewStatus?.exists && !previewGenerating && (
              <p className="text-xs text-gray-500 mt-2">
                Generate an instant rough 3D preview from a single frame while the full pipeline runs.
              </p>
            )}
          </div>

          {/* Inline SplatViewer for preview */}
          {previewStatus?.exists && !previewGenerating && (
            <div>
              <div className="h-[400px] border-t border-gray-700">
                <SplatViewer
                  plyUrl={api.getQuickPreviewUrl(id)}
                  projectId={id}
                />
              </div>
              <div className="px-4 py-2 bg-gray-900/50 border-t border-gray-700 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-amber-400 font-medium">
                    Quick Preview
                  </span>
                  {previewStatus.size_mb && (
                    <span className="text-xs text-gray-500">
                      {previewStatus.size_mb.toFixed(1)} MB
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500">
                  This is a rough preview from a single frame. Continue to SfM and Training for full quality.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-3">
        {!framesReady && !isBusy && !isFailed && (
          <button
            onClick={handleExtract}
            disabled={starting || !project?.video_filename}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Extract Frames
          </button>
        )}
        {isFailed && (
          <button
            onClick={handleExtract}
            disabled={starting}
            className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RotateCcw className="w-4 h-4" />
            )}
            Retry Extraction
          </button>
        )}
        {isBusy && (
          <>
            <button
              onClick={handleCancel}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              <XCircle className="w-4 h-4" /> Cancel
            </button>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Clock className="w-4 h-4" />
              <span>{elapsedStr}</span>
              {etaStr && (
                <span className="text-gray-500">• ETA {etaStr}</span>
              )}
            </div>
          </>
        )}
        {framesReady && !isBusy && (
          <>
            <button
              onClick={handleExtract}
              disabled={starting}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              <RotateCcw className="w-4 h-4" /> Redo Extraction
            </button>
            <button
              onClick={() => router.push(`/project/${id}/sfm`)}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              Continue to SfM <ArrowRight className="w-4 h-4" />
            </button>
          </>
        )}
      </div>

      {(isBusy || logs.length > 0) && <LogStream logs={logs} />}

      <FrameGrid frames={frames} />
    </div>
  );
}
