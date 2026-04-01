"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useElapsedTimer } from "@/hooks/useElapsedTimer";
import { LogStream } from "@/components/LogStream";
import { FrameGrid } from "@/components/FrameGrid";
import type { FrameInfo, ProjectDetail } from "@/lib/types";
import { Play, ArrowRight, ArrowLeft, Loader2, XCircle, RotateCcw, Clock, Settings2, Volume2, VolumeX } from "lucide-react";
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
  const { muted, setMuted, onStepChange } = useCompletionChime();

  useEffect(() => { onStepChange(project?.step); }, [project?.step, onStepChange]);

  const isExtracting = project?.step === "extracting_frames";
  const isFailed = project?.step === "failed";
  const isDone =
    project?.step === "frames_ready" ||
    project?.step === "running_sfm" ||
    project?.step === "sfm_ready" ||
    project?.step === "training" ||
    project?.step === "training_complete";

  const { elapsedStr, etaStr, resetTimer } = useElapsedTimer(
    isExtracting || false,
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

  const handleExtract = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.extractFrames(id, { fps });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  const handleCancel = async () => {
    await api.cancelPipeline(id);
  };

  const handleRetry = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.extractFrames(id, { fps });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  return (
    <div className="space-y-6">
      {/* Settings panel */}
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
                disabled={isExtracting}
                className="bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white"
              >
                {FPS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label} — {o.desc}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-xs text-gray-500">
              Higher FPS extracts more frames. More frames give better 3D
              quality but make SfM slower.
            </p>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-3">
        {!isDone && !isExtracting && !isFailed && (
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
            onClick={handleRetry}
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
        {isExtracting && (
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
        {isDone && (
          <>
            <button
              onClick={handleRetry}
              disabled={starting}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              <RotateCcw className="w-4 h-4" /> Redo
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

      {(isExtracting || logs.length > 0) && <LogStream logs={logs} />}

      <FrameGrid frames={frames} />
    </div>
  );
}
