"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useElapsedTimer } from "@/hooks/useElapsedTimer";
import { LogStream } from "@/components/LogStream";
import type { ProjectDetail } from "@/lib/types";
import {
  Play,
  ArrowRight,
  Loader2,
  XCircle,
  AlertTriangle,
  RotateCcw,
  Clock,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function TrainPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const { logs, progress, metrics, clearLogs } = useWebSocket(id);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [maxSteps, setMaxSteps] = useState(7000);
  const [starting, setStarting] = useState(false);
  const [sysStatus, setSysStatus] = useState<any>(null);

  useEffect(() => {
    const refresh = () => api.getProject(id).then(setProject);
    refresh();
    api.systemStatus().then(setSysStatus);
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [id]);

  const isTraining = project?.step === "training";
  const isFailed = project?.step === "failed";
  const isDone = project?.step === "training_complete";
  // Check PyTorch CUDA (what actually matters), falling back to driver CUDA
  const noCuda = sysStatus && !(sysStatus.torch_cuda_available ?? sysStatus.cuda_available);

  const { elapsedStr, etaStr, resetTimer } = useElapsedTimer(
    isTraining || false,
    progress?.percent ?? undefined
  );

  const lossData = metrics
    .filter((m) => m.loss !== undefined)
    .map((m) => ({ step: m.step, loss: m.loss, psnr: m.psnr }));

  const handleTrain = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.train(id, { max_steps: maxSteps });
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  return (
    <div className="space-y-6">
      {noCuda && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />
          <div>
            <p className="text-yellow-400 font-medium">CUDA not available</p>
            <p className="text-yellow-500/80 text-sm mt-1">
              Gaussian splat training requires an NVIDIA GPU with CUDA. Check
              the Settings page for details.
            </p>
          </div>
        </div>
      )}

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-300 mb-3">
          Training Settings
        </h3>
        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400">Max Iterations:</label>
          <input
            type="range"
            min={1000}
            max={30000}
            step={1000}
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number(e.target.value))}
            disabled={isTraining}
            className="flex-1"
          />
          <span className="text-sm text-white w-16 text-right">
            {maxSteps.toLocaleString()}
          </span>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          7,000 = quick preview (~5 min), 30,000 = high quality (~30 min)
        </p>
      </div>

      <div className="flex items-center gap-3">
        {!isDone && !isTraining && !isFailed && (
          <button
            onClick={handleTrain}
            disabled={starting || noCuda}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Start Training
          </button>
        )}
        {isFailed && (
          <button
            onClick={handleTrain}
            disabled={starting || noCuda}
            className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            {starting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RotateCcw className="w-4 h-4" />
            )}
            Retry Training
          </button>
        )}
        {isTraining && (
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
            onClick={() => router.push(`/project/${id}/view`)}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
          >
            View Splat <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {progress && progress.percent > 0 && (
        <div>
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>Progress</span>
            <span>{Math.round(progress.percent)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div
              className="bg-blue-500 h-2.5 rounded-full transition-all"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
        </div>
      )}

      {lossData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 mb-3">
            Training Loss
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
              <YAxis stroke="#9ca3af" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
                labelStyle={{ color: "#9ca3af" }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#3b82f6"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {(isTraining || logs.length > 0) && <LogStream logs={logs} />}
    </div>
  );
}
