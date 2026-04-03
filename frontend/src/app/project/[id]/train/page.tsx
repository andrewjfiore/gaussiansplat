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
  ArrowLeft,
  Loader2,
  XCircle,
  AlertTriangle,
  RotateCcw,
  Clock,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useCompletionChime } from "@/hooks/useCompletionChime";
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
  const [shDegree, setShDegree] = useState(0);
  const [enableDepth, setEnableDepth] = useState(false);
  const [depthWeight, setDepthWeight] = useState(0.1);
  const [temporalMode, setTemporalMode] = useState<"static" | "4d">("static");
  const [temporalSmoothness, setTemporalSmoothness] = useState(0.01);
  const [starting, setStarting] = useState(false);
  const [sysStatus, setSysStatus] = useState<any>(null);
  const [snapshots, setSnapshots] = useState<{ pct: number; url: string }[]>([]);
  const { muted, setMuted, onStepChange } = useCompletionChime();

  useEffect(() => {
    const refresh = () => api.getProject(id).then(setProject);
    refresh();
    api.systemStatus().then(setSysStatus);
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [id]);

  // Chime on step completion
  useEffect(() => {
    onStepChange(project?.step);
  }, [project?.step, onStepChange]);

  const isTraining = project?.step === "training";
  const isFailed = project?.step === "failed";
  const isDone = project?.step === "training_complete";
  // Check PyTorch CUDA (what actually matters), falling back to driver CUDA
  const noCuda = sysStatus && !(sysStatus.torch_cuda_available ?? sysStatus.cuda_available);

  // Periodic health check: detect silent training failures
  useEffect(() => {
    if (!isTraining) return;
    const check = async () => {
      try {
        const health = await api.pipelineHealth(id);
        if (health.stale) {
          // Backend detected and auto-recovered stale state — refresh project
          const p = await api.getProject(id);
          setProject(p);
        }
      } catch {
        // Health endpoint unavailable — ignore
      }
    };
    const interval = setInterval(check, 10_000);
    return () => clearInterval(interval);
  }, [id, isTraining]);

  const { elapsedStr, etaStr, resetTimer } = useElapsedTimer(
    isTraining || false,
    progress?.percent ?? undefined
  );

  const lossData = metrics
    .filter((m) => m.loss !== undefined)
    .map((m) => ({ step: m.step, loss: m.loss, psnr: m.psnr }));

  // Watch for snapshot metrics
  useEffect(() => {
    const latest = metrics[metrics.length - 1];
    if (latest?.snapshot_pct && latest?.snapshot_file) {
      setSnapshots((prev) => {
        if (prev.some((s) => s.pct === latest.snapshot_pct)) return prev;
        return [
          ...prev,
          {
            pct: latest.snapshot_pct,
            url: `/api/projects/${id}/output/${latest.snapshot_file}`,
          },
        ].sort((a, b) => a.pct - b.pct);
      });
    }
  }, [metrics, id]);

  const handleTrain = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.train(id, {
            max_steps: maxSteps,
            sh_degree: shDegree,
            enable_depth: enableDepth,
            depth_weight: depthWeight,
            temporal_mode: temporalMode,
            temporal_smoothness: temporalMode === "4d" ? temporalSmoothness : undefined,
          });
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
        <div className="flex items-center gap-4 mt-3">
          <label className="text-sm text-gray-400">SH Degree:</label>
          <select
            value={shDegree}
            onChange={(e) => setShDegree(Number(e.target.value))}
            disabled={isTraining}
            className="bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white"
          >
            <option value={0}>0 — Flat color (fast)</option>
            <option value={1}>1 — Mild reflections</option>
            <option value={2}>2 — Rich reflections (recommended)</option>
            <option value={3}>3 — Maximum quality (slow)</option>
          </select>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Higher SH degrees capture view-dependent color (reflections, specular highlights).
          Degree grows progressively during training.
        </p>
        <div className="flex items-center gap-4 mt-3">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={enableDepth}
              onChange={(e) => setEnableDepth(e.target.checked)}
              disabled={isTraining}
              className="rounded border-gray-600"
            />
            Depth Supervision
          </label>
          {enableDepth && (
            <>
              <label className="text-sm text-gray-400">Weight:</label>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={depthWeight}
                onChange={(e) => setDepthWeight(Number(e.target.value))}
                disabled={isTraining}
                className="w-24"
              />
              <span className="text-sm text-white w-10">{depthWeight.toFixed(2)}</span>
            </>
          )}
        </div>
        {enableDepth && (
          <p className="text-xs text-gray-500 mt-1">
            Runs Depth Anything V2 on all frames, then uses depth maps to reduce floaters.
            First run downloads a ~25MB model.
          </p>
        )}
        <div className="flex items-center gap-4 mt-3">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={temporalMode === "4d"}
              onChange={(e) => setTemporalMode(e.target.checked ? "4d" : "static")}
              disabled={isTraining}
              className="rounded border-gray-600"
            />
            <span className={temporalMode === "4d" ? "text-purple-400" : ""}>
              4D Mode (Temporal)
            </span>
          </label>
          {temporalMode === "4d" && (
            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              4D
            </span>
          )}
        </div>
        {temporalMode === "4d" && (
          <div className="mt-2 ml-6 space-y-2 border-l-2 border-purple-500/30 pl-3">
            <p className="text-xs text-purple-400/80">
              Trains a deformation network that moves Gaussians over time.
              Produces an animated splat you can scrub through. ~2x training time.
            </p>
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 whitespace-nowrap">
                Smoothness:
              </label>
              <input
                type="range"
                min={0.001}
                max={0.1}
                step={0.001}
                value={temporalSmoothness}
                onChange={(e) => setTemporalSmoothness(Number(e.target.value))}
                disabled={isTraining}
                className="w-32"
              />
              <span className="text-sm text-white w-12">
                {temporalSmoothness.toFixed(3)}
              </span>
            </div>
            <p className="text-xs text-gray-500">
              Controls deformation regularization. Lower = more motion allowed,
              higher = smoother/stiffer movement. Default 0.01 works for most scenes.
            </p>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => router.push(`/project/${id}/sfm`)}
          className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
        >
          <ArrowLeft className="w-4 h-4" /> Back to SfM
        </button>
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
            <button
              onClick={() => setMuted((m) => !m)}
              className="text-gray-400 hover:text-white transition p-1"
              title={muted ? "Unmute completion chime" : "Mute completion chime"}
            >
              {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
          </>
        )}
        {isDone && (
          <>
            <button
              onClick={handleTrain}
              disabled={starting || noCuda}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              <RotateCcw className="w-4 h-4" /> Redo
            </button>
            <button
              onClick={async () => {
                setStarting(true);
                clearLogs();
                resetTimer();
                try {
                  await api.refineSplat(id, { refine_steps: Math.round(maxSteps * 0.4) });
                } catch (err: any) { alert(err.message); }
                setStarting(false);
              }}
              disabled={starting || noCuda}
              className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
              Refine (Fill Gaps)
            </button>
            <button
              onClick={() => router.push(`/project/${id}/view`)}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              View Splat <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => router.push(`/project/${id}/postprocess`)}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
            >
              Post-Process
            </button>
          </>
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

      {snapshots.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 mb-3">
            Training Snapshots
          </h3>
          <div className="flex gap-3 overflow-x-auto">
            {snapshots.map((snap) => (
              <a
                key={snap.pct}
                href={snap.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 group"
              >
                <div className="relative">
                  <img
                    src={snap.url}
                    alt={`${snap.pct}% training`}
                    className="w-40 h-24 object-cover rounded border border-gray-600 group-hover:border-blue-500 transition"
                  />
                  <span className="absolute bottom-1 right-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
                    {snap.pct}%
                  </span>
                </div>
              </a>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Click to view full size. Snapshots taken at 25% training intervals.
          </p>
        </div>
      )}

      {(isTraining || logs.length > 0) && <LogStream logs={logs} />}
    </div>
  );
}
