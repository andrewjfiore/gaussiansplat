"use client";
import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useElapsedTimer } from "@/hooks/useElapsedTimer";
import { LogStream } from "@/components/LogStream";
import type { ProjectDetail, SceneConfig, TrainSettings, AugmentSettings, SyntheticViewList } from "@/lib/types";
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
  Sparkles,
  Settings,
  ChevronDown,
  ChevronUp,
  Zap,
  Box,
  Eye,
  Images,
  Trash2,
  Info,
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

const COMPLEXITY_COLORS: Record<string, string> = {
  low: "text-green-400",
  medium: "text-yellow-400",
  high: "text-orange-400",
};

const COMPLEXITY_BG: Record<string, string> = {
  low: "bg-green-500/10 border-green-500/30",
  medium: "bg-yellow-500/10 border-yellow-500/30",
  high: "bg-orange-500/10 border-orange-500/30",
};

export default function TrainPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const { logs, progress, metrics, clearLogs } = useWebSocket(id);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [starting, setStarting] = useState(false);
  const [sysStatus, setSysStatus] = useState<any>(null);
  const [snapshots, setSnapshots] = useState<{ pct: number; url: string }[]>([]);
  const { muted, setMuted, onStepChange } = useCompletionChime();

  // Scene analysis state
  const [sceneConfig, setSceneConfig] = useState<SceneConfig | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // View augmentation state
  const [augmentSettings, setAugmentSettings] = useState<AugmentSettings>({
    num_views_per_frame: 2,
    angle_range: 15,
    max_source_frames: 10,
  });
  const [augmenting, setAugmenting] = useState(false);
  const [syntheticViews, setSyntheticViews] = useState<SyntheticViewList | null>(null);
  const [augmentError, setAugmentError] = useState<string | null>(null);

  // Training settings (populated from scene analysis or defaults)
  const [trainSettings, setTrainSettings] = useState<TrainSettings>({
    max_steps: 7000,
    two_phase: true,
    sh_degree: 3,
  });

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
  const noCuda =
    sysStatus &&
    !(sysStatus.torch_cuda_available ?? sysStatus.cuda_available);

  // Periodic health check: detect silent training failures
  useEffect(() => {
    if (!isTraining) return;
    const check = async () => {
      try {
        const health = await api.pipelineHealth(id);
        if (health.stale) {
          // Backend detected and auto-recovered stale state -- refresh project
          const p = await api.getProject(id);
          setProject(p);
        }
      } catch {
        // Health endpoint unavailable -- ignore
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

  // Determine current training phase from the progress substep
  const currentPhase = progress?.substep || "";
  const isPhase1 = currentPhase.includes("Phase 1");
  const isPhase2 = currentPhase.includes("Phase 2");
  const phaseLabel = isPhase2
    ? "Phase 2: Color Refinement"
    : isPhase1
      ? "Phase 1: Geometry + Color"
      : "";

  const handleAnalyze = useCallback(async () => {
    setAnalyzing(true);
    try {
      const config = await api.analyzeScene(id);
      setSceneConfig(config);
      setTrainSettings({
        max_steps: config.max_steps,
        two_phase: true,
        phase1_steps: config.phase1_steps,
        phase2_steps: config.phase2_steps,
        sh_degree: config.sh_degree,
        densify_grad_thresh: config.densify_grad_thresh,
      });
      setAnalyzed(true);
    } catch (err: any) {
      console.error("Scene analysis failed:", err);
      // Fall back to defaults
      setAnalyzed(true);
    }
    setAnalyzing(false);
  }, [id]);

  // Auto-analyze when page loads for projects at sfm_ready stage
  useEffect(() => {
    if (
      project &&
      (project.step === "sfm_ready" || project.step === "failed") &&
      !analyzed &&
      !analyzing
    ) {
      handleAnalyze();
    }
  }, [project, analyzed, analyzing, handleAnalyze]);

  // Load existing synthetic views on mount
  const refreshSyntheticViews = useCallback(async () => {
    try {
      const result = await api.listSyntheticViews(id);
      setSyntheticViews(result);
    } catch {
      // ignore -- no synthetic views yet
    }
  }, [id]);

  useEffect(() => {
    refreshSyntheticViews();
  }, [refreshSyntheticViews]);

  // Track augmentation progress from WebSocket
  useEffect(() => {
    if (progress?.step === "augment_views") {
      if (progress.substep === "complete") {
        setAugmenting(false);
        refreshSyntheticViews();
      } else if (progress.substep === "failed") {
        setAugmenting(false);
        setAugmentError("View augmentation failed -- check logs for details");
      }
    }
  }, [progress, refreshSyntheticViews]);

  const handleAugment = async () => {
    setAugmenting(true);
    setAugmentError(null);
    try {
      await api.augmentViews(id, augmentSettings);
    } catch (err: any) {
      setAugmentError(err.message);
      setAugmenting(false);
    }
  };

  const handleClearSynthetic = async () => {
    try {
      await api.clearSyntheticViews(id);
      setSyntheticViews(null);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleTrain = async () => {
    setStarting(true);
    clearLogs();
    resetTimer();
    try {
      await api.train(id, trainSettings);
    } catch (err: any) {
      alert(err.message);
    }
    setStarting(false);
  };

  const updateSetting = <K extends keyof TrainSettings>(
    key: K,
    value: TrainSettings[K]
  ) => {
    setTrainSettings((prev) => {
      const next = { ...prev, [key]: value };
      // Keep phase steps in sync when max_steps changes
      if (key === "max_steps") {
        const ms = value as number;
        next.phase1_steps = Math.round(ms * 0.8);
        next.phase2_steps = ms - next.phase1_steps;
      }
      return next;
    });
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

      {/* Scene Analysis Card */}
      {analyzed && sceneConfig && !isTraining && !isDone && (
        <div
          className={`rounded-lg p-4 border ${COMPLEXITY_BG[sceneConfig.scene_complexity]}`}
        >
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-5 h-5 text-blue-400" />
            <h3 className="text-sm font-medium text-gray-200">
              Scene Analysis
            </h3>
            <span
              className={`text-xs font-semibold uppercase px-2 py-0.5 rounded ${COMPLEXITY_COLORS[sceneConfig.scene_complexity]}`}
            >
              {sceneConfig.scene_complexity} complexity
            </span>
          </div>

          {sceneConfig.stats && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-xs text-gray-500">3D Points</div>
                <div className="text-sm text-white font-medium">
                  {sceneConfig.stats.num_points.toLocaleString()}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-xs text-gray-500">Views</div>
                <div className="text-sm text-white font-medium">
                  {sceneConfig.stats.num_images}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-xs text-gray-500">Scene Radius</div>
                <div className="text-sm text-white font-medium">
                  {sceneConfig.stats.scene_radius.toFixed(2)}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-xs text-gray-500">Camera Baseline</div>
                <div className="text-sm text-white font-medium">
                  {sceneConfig.stats.camera_baseline.toFixed(3)}
                </div>
              </div>
            </div>
          )}

          <div className="space-y-1">
            {sceneConfig.reasoning.map((r, i) => (
              <p key={i} className="text-xs text-gray-400">
                {r}
              </p>
            ))}
          </div>

          <button
            onClick={handleAnalyze}
            className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition"
          >
            Re-analyze
          </button>
        </div>
      )}

      {analyzing && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
          <span className="text-sm text-gray-300">
            Analyzing scene characteristics...
          </span>
        </div>
      )}

      {/* Augment Views Card */}
      {!isTraining && !isDone && (project?.step === "sfm_ready" || project?.step === "frames_ready" || project?.step === "failed") && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Images className="w-4 h-4 text-indigo-400" />
              Augment Views
              {syntheticViews && syntheticViews.count > 0 && (
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                  {syntheticViews.count} synthetic views
                </span>
              )}
            </h3>
            {project?.frame_count && (
              <span className="text-xs text-gray-500">
                {project.frame_count} source frames
              </span>
            )}
          </div>

          <p className="text-xs text-gray-400 mb-3 flex items-start gap-1.5">
            <Info className="w-3.5 h-3.5 text-gray-500 mt-0.5 flex-shrink-0" />
            Synthetic views help fill coverage gaps for better 3D reconstruction.
            Uses depth estimation to warp existing frames to nearby viewpoints.
          </p>

          {sceneConfig?.stats && sceneConfig.stats.num_images < 30 && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded p-2 mb-3">
              <p className="text-xs text-amber-400">
                Coverage gaps detected -- augmenting views can help improve quality
                with only {sceneConfig.stats.num_images} registered views.
              </p>
            </div>
          )}

          {augmentError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded p-2 mb-3">
              <p className="text-xs text-red-400">{augmentError}</p>
            </div>
          )}

          {/* Settings */}
          {!augmenting && (
            <div className="space-y-2 mb-3">
              <div className="flex items-center gap-4">
                <label className="text-xs text-gray-400 w-36">Views per frame:</label>
                <input
                  type="range"
                  min={1}
                  max={4}
                  step={1}
                  value={augmentSettings.num_views_per_frame}
                  onChange={(e) =>
                    setAugmentSettings((s) => ({ ...s, num_views_per_frame: Number(e.target.value) }))
                  }
                  className="flex-1"
                />
                <span className="text-xs text-white w-6 text-right">
                  {augmentSettings.num_views_per_frame}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <label className="text-xs text-gray-400 w-36">Angle range:</label>
                <input
                  type="range"
                  min={5}
                  max={30}
                  step={1}
                  value={augmentSettings.angle_range}
                  onChange={(e) =>
                    setAugmentSettings((s) => ({ ...s, angle_range: Number(e.target.value) }))
                  }
                  className="flex-1"
                />
                <span className="text-xs text-white w-6 text-right">
                  {augmentSettings.angle_range}&deg;
                </span>
              </div>
              <div className="flex items-center gap-4">
                <label className="text-xs text-gray-400 w-36">Max source frames:</label>
                <input
                  type="range"
                  min={5}
                  max={20}
                  step={1}
                  value={augmentSettings.max_source_frames}
                  onChange={(e) =>
                    setAugmentSettings((s) => ({ ...s, max_source_frames: Number(e.target.value) }))
                  }
                  className="flex-1"
                />
                <span className="text-xs text-white w-6 text-right">
                  {augmentSettings.max_source_frames}
                </span>
              </div>
            </div>
          )}

          {/* Progress bar during augmentation */}
          {augmenting && progress?.step === "augment_views" && (
            <div className="mb-3">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{progress.substep || "Generating..."}</span>
                <span>{Math.round(progress.percent ?? 0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-indigo-500 h-2 rounded-full transition-all"
                  style={{ width: `${progress.percent ?? 0}%` }}
                />
              </div>
            </div>
          )}

          {/* Preview grid of synthetic views */}
          {syntheticViews && syntheticViews.count > 0 && !augmenting && (
            <div className="mb-3">
              <div className="flex gap-2 overflow-x-auto pb-2">
                {syntheticViews.views.slice(0, 8).map((view) => (
                  <img
                    key={view.name}
                    src={view.url}
                    alt={view.name}
                    className="w-24 h-16 object-cover rounded border border-gray-600 flex-shrink-0"
                  />
                ))}
                {syntheticViews.count > 8 && (
                  <div className="w-24 h-16 rounded border border-gray-600 flex-shrink-0 flex items-center justify-center bg-gray-700/50">
                    <span className="text-xs text-gray-400">
                      +{syntheticViews.count - 8} more
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleAugment}
              disabled={augmenting}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 text-white text-sm px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition"
            >
              {augmenting ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Images className="w-3.5 h-3.5" />
              )}
              {augmenting ? "Generating..." : syntheticViews?.count ? "Regenerate" : "Generate Synthetic Views"}
            </button>
            {syntheticViews && syntheticViews.count > 0 && !augmenting && (
              <button
                onClick={handleClearSynthetic}
                className="text-gray-400 hover:text-red-400 text-sm px-3 py-1.5 rounded-lg border border-gray-600 hover:border-red-500/50 flex items-center gap-1.5 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {/* Training Settings */}
      {!isTraining && !isDone && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Training Settings
              {sceneConfig && (
                <span className="text-xs text-gray-500 font-normal">
                  (auto-calibrated)
                </span>
              )}
            </h3>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition"
            >
              Advanced
              {showAdvanced ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>
          </div>

          {/* Max iterations slider */}
          <div className="flex items-center gap-4 mb-3">
            <label className="text-sm text-gray-400 w-32">
              Total Iterations:
            </label>
            <input
              type="range"
              min={1000}
              max={30000}
              step={1000}
              value={trainSettings.max_steps}
              onChange={(e) =>
                updateSetting("max_steps", Number(e.target.value))
              }
              className="flex-1"
            />
            <span className="text-sm text-white w-16 text-right">
              {trainSettings.max_steps.toLocaleString()}
            </span>
          </div>

          {/* Two-phase toggle */}
          <div className="flex items-center gap-3 mb-2">
            <label className="text-sm text-gray-400 w-32">
              Two-Phase Training:
            </label>
            <button
              onClick={() =>
                updateSetting("two_phase", !trainSettings.two_phase)
              }
              className={`relative w-10 h-5 rounded-full transition-colors ${
                trainSettings.two_phase ? "bg-blue-600" : "bg-gray-600"
              }`}
            >
              <div
                className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                  trainSettings.two_phase ? "translate-x-5" : ""
                }`}
              />
            </button>
            <span className="text-xs text-gray-500">
              {trainSettings.two_phase
                ? `Phase 1: ${trainSettings.phase1_steps?.toLocaleString() ?? "auto"} steps, Phase 2: ${trainSettings.phase2_steps?.toLocaleString() ?? "auto"} steps`
                : "Single-phase training"}
            </span>
          </div>

          {trainSettings.two_phase && (
            <p className="text-xs text-gray-500 ml-[8.5rem] mb-2">
              Phase 1 optimizes geometry + color. Phase 2 freezes geometry and
              refines colors for sharper results.
            </p>
          )}

          {/* Advanced settings */}
          {showAdvanced && (
            <div className="mt-3 pt-3 border-t border-gray-700 space-y-3">
              <div className="flex items-center gap-4">
                <label className="text-sm text-gray-400 w-32">SH Degree:</label>
                <select
                  value={trainSettings.sh_degree}
                  onChange={(e) =>
                    updateSetting("sh_degree", Number(e.target.value))
                  }
                  className="bg-gray-700 text-white text-sm rounded px-2 py-1 border border-gray-600"
                >
                  <option value={0}>0 -- Flat color (fast)</option>
                  <option value={1}>1 (fastest reflections)</option>
                  <option value={2}>2 (balanced)</option>
                  <option value={3}>3 (best quality)</option>
                </select>
                <span className="text-xs text-gray-500">
                  Higher = better view-dependent colors
                </span>
              </div>

              <div className="flex items-center gap-4">
                <label className="text-sm text-gray-400 w-32">
                  Densify Threshold:
                </label>
                <input
                  type="number"
                  step={0.00005}
                  min={0.00005}
                  max={0.001}
                  value={trainSettings.densify_grad_thresh ?? 0.0002}
                  onChange={(e) =>
                    updateSetting(
                      "densify_grad_thresh",
                      parseFloat(e.target.value)
                    )
                  }
                  className="bg-gray-700 text-white text-sm rounded px-2 py-1 w-28 border border-gray-600"
                />
                <span className="text-xs text-gray-500">
                  Lower = more splats (denser)
                </span>
              </div>

              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainSettings.enable_depth ?? false}
                    onChange={(e) => updateSetting("enable_depth", e.target.checked)}
                    disabled={isTraining}
                    className="rounded border-gray-600"
                  />
                  Depth Supervision
                </label>
                {trainSettings.enable_depth && (
                  <>
                    <label className="text-sm text-gray-400">Weight:</label>
                    <input
                      type="range"
                      min={0.01}
                      max={0.5}
                      step={0.01}
                      value={trainSettings.depth_weight ?? 0.1}
                      onChange={(e) => updateSetting("depth_weight", Number(e.target.value))}
                      disabled={isTraining}
                      className="w-24"
                    />
                    <span className="text-sm text-white w-10">{(trainSettings.depth_weight ?? 0.1).toFixed(2)}</span>
                  </>
                )}
              </div>
              {trainSettings.enable_depth && (
                <p className="text-xs text-gray-500 mt-1">
                  Runs Depth Anything V2 on all frames, then uses depth maps to reduce floaters.
                </p>
              )}

              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trainSettings.temporal_mode === "4d"}
                    onChange={(e) => updateSetting("temporal_mode", e.target.checked ? "4d" : "static")}
                    disabled={isTraining}
                    className="rounded border-gray-600"
                  />
                  <span className={trainSettings.temporal_mode === "4d" ? "text-purple-400" : ""}>
                    4D Mode (Temporal)
                  </span>
                </label>
                {trainSettings.temporal_mode === "4d" && (
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                    4D
                  </span>
                )}
              </div>
              {trainSettings.temporal_mode === "4d" && (
                <div className="ml-6 space-y-2 border-l-2 border-purple-500/30 pl-3">
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
                      value={trainSettings.temporal_smoothness ?? 0.01}
                      onChange={(e) => updateSetting("temporal_smoothness", Number(e.target.value))}
                      disabled={isTraining}
                      className="w-32"
                    />
                    <span className="text-sm text-white w-12">
                      {(trainSettings.temporal_smoothness ?? 0.01).toFixed(3)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          <p className="text-xs text-gray-500 mt-2">
            {trainSettings.max_steps <= 5000
              ? "Quick preview (~3-5 min)"
              : trainSettings.max_steps <= 10000
                ? "Standard quality (~5-15 min)"
                : "High quality (~15-30 min)"}
          </p>
        </div>
      )}

      {/* Action buttons */}
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
                <span className="text-gray-500">&#8226; ETA {etaStr}</span>
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
                  await api.refineSplat(id, { refine_steps: Math.round(trainSettings.max_steps * 0.4) });
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

      {/* Two-phase progress indicator */}
      {isTraining && progress && progress.percent > 0 && (
        <div>
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <div className="flex items-center gap-2">
              <span>Progress</span>
              {phaseLabel && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    isPhase2
                      ? "bg-purple-500/20 text-purple-400"
                      : "bg-blue-500/20 text-blue-400"
                  }`}
                >
                  {isPhase1 && <Box className="w-3 h-3 inline mr-1" />}
                  {isPhase2 && <Eye className="w-3 h-3 inline mr-1" />}
                  {phaseLabel}
                </span>
              )}
            </div>
            <span>{Math.round(progress.percent)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
            {trainSettings.two_phase && trainSettings.phase1_steps ? (
              <>
                {/* Phase 1 segment (blue) */}
                <div className="h-2.5 flex">
                  <div
                    className="bg-blue-500 h-2.5 transition-all"
                    style={{
                      width: `${Math.min(
                        progress.percent,
                        (trainSettings.phase1_steps / trainSettings.max_steps) *
                          100
                      )}%`,
                    }}
                  />
                  {/* Phase 2 segment (purple) */}
                  {progress.percent >
                    (trainSettings.phase1_steps / trainSettings.max_steps) *
                      100 && (
                    <div
                      className="bg-purple-500 h-2.5 transition-all"
                      style={{
                        width: `${
                          progress.percent -
                          (trainSettings.phase1_steps /
                            trainSettings.max_steps) *
                            100
                        }%`,
                      }}
                    />
                  )}
                </div>
              </>
            ) : (
              <div
                className="bg-blue-500 h-2.5 rounded-full transition-all"
                style={{ width: `${progress.percent}%` }}
              />
            )}
          </div>
          {trainSettings.two_phase && trainSettings.phase1_steps && (
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
                Phase 1 (geometry + color)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-purple-500 inline-block" />
                Phase 2 (color refinement)
              </span>
            </div>
          )}
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
