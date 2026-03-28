"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useElapsedTimer } from "@/hooks/useElapsedTimer";
import { LogStream } from "@/components/LogStream";
import type { ProjectDetail } from "@/lib/types";
import { Play, ArrowRight, Loader2, XCircle, RotateCcw, Clock } from "lucide-react";

export default function SfmPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const { logs, progress, status, clearLogs } = useWebSocket(id);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [matcherType, setMatcherType] = useState("sequential_matcher");
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    const refresh = () => api.getProject(id).then(setProject);
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [id]);

  const isRunning = project?.step === "running_sfm";
  const isFailed = project?.step === "failed";
  const isDone =
    project?.step === "sfm_ready" ||
    project?.step === "training" ||
    project?.step === "training_complete";

  const { elapsedStr, etaStr, resetTimer } = useElapsedTimer(
    isRunning || false,
    progress?.percent ?? undefined
  );

  const handleRun = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.runSfm(id, { matcher_type: matcherType });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          SfM Settings
        </h3>
        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400">Matcher:</label>
          <select
            value={matcherType}
            onChange={(e) => setMatcherType(e.target.value)}
            disabled={isRunning}
            className="bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white"
          >
            <option value="sequential_matcher">
              Sequential (fast, for video sequences)
            </option>
            <option value="exhaustive_matcher">
              Exhaustive (slow, more robust)
            </option>
          </select>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {!isDone && !isRunning && !isFailed && (
          <button
            onClick={handleRun}
            disabled={starting}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Run SfM Pipeline
          </button>
        )}
        {isFailed && (
          <button
            onClick={handleRun}
            disabled={starting}
            className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RotateCcw className="w-4 h-4" />
            )}
            Retry SfM
          </button>
        )}
        {isRunning && (
          <>
            <button
              onClick={() => api.cancelPipeline(id)}
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
          <button
            onClick={() => router.push(`/project/${id}/train`)}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            Continue to Training <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {progress && (
        <div className="text-sm text-gray-300">
          Step:{" "}
          <span className="text-white font-medium">{progress.substep}</span>
          {progress.percent > 0 && (
            <div className="mt-1 w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          )}
        </div>
      )}

      {(isRunning || logs.length > 0) && <LogStream logs={logs} />}
    </div>
  );
}
