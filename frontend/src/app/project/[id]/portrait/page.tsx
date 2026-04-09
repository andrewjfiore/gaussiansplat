"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { SplatViewer } from "@/components/SplatViewer";
import type { ProjectDetail, PortraitSettings } from "@/lib/types";
import {
  Upload,
  Loader2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  ArrowRight,
  Eye,
  Settings,
  ImageIcon,
  Wand2,
  CheckCircle2,
  XCircle,
} from "lucide-react";

type Stage = "upload" | "processing" | "complete" | "error";

const PROCESSING_STEPS = [
  "Estimating depth...",
  "Segmenting subject...",
  "Generating 3D points...",
  "Creating novel views...",
  "Writing model...",
];

export default function PortraitPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { logs, progress, status } = useWebSocket(id);

  // Core state
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Settings
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [settings, setSettings] = useState<PortraitSettings>({
    stride: 2,
    include_background: false,
    num_novel_views: 6,
  });

  // Processing state
  const [processingStep, setProcessingStep] = useState<string>("");
  const [processingPercent, setProcessingPercent] = useState(0);
  const [depthPreviewUrl, setDepthPreviewUrl] = useState<string | null>(null);
  const [depthLoaded, setDepthLoaded] = useState(false);
  const [showDepth, setShowDepth] = useState(false);

  // Complete state
  const [gaussianCount, setGaussianCount] = useState<number | null>(null);

  // Load project info
  useEffect(() => {
    api.getProject(id).then((p) => {
      setProject(p);
      // If already complete, jump to complete stage
      if (p.step === "training_complete" && p.has_output) {
        setStage("complete");
      } else if (p.step === "portrait_processing") {
        setStage("processing");
      } else if (p.step === "failed") {
        setStage("error");
        setError(p.error || "Processing failed");
      }
    });
  }, [id]);

  // Watch WebSocket for processing updates
  useEffect(() => {
    if (stage !== "processing") return;

    if (progress) {
      setProcessingPercent(progress.percent || 0);
      if (progress.substep) {
        setProcessingStep(progress.substep);
      }
    }

    if (status) {
      if (status.state === "complete" || status.step === "training_complete") {
        setStage("complete");
        // Refresh project to get updated info
        api.getProject(id).then(setProject);
      } else if (status.state === "failed") {
        setStage("error");
        setError(status.error || "Processing failed");
      }
    }
  }, [progress, status, stage, id]);

  // Try loading depth preview once processing starts
  useEffect(() => {
    if (stage !== "processing") return;

    const interval = setInterval(() => {
      const url = api.getDepthPreview(id);
      const img = new Image();
      img.onload = () => {
        setDepthPreviewUrl(`${url}?t=${Date.now()}`);
        clearInterval(interval);
      };
      img.src = `${url}?t=${Date.now()}`;
    }, 2000);

    return () => clearInterval(interval);
  }, [stage, id]);

  // Extract gaussian count from logs
  useEffect(() => {
    for (const log of logs) {
      const match = log.match(/(\d[\d,]+)\s*gaussians?/i);
      if (match) {
        setGaussianCount(parseInt(match[1].replace(/,/g, ""), 10));
      }
    }
  }, [logs]);

  // File handling
  const handleFileSelect = useCallback((selectedFile: File) => {
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(selectedFile.type)) {
      setError("Please select a valid image file (JPG, PNG, or WebP)");
      return;
    }
    setFile(selectedFile);
    setError(null);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFileSelect(dropped);
    },
    [handleFileSelect]
  );

  const handleGenerate = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setUploadPercent(0);

    try {
      // Upload the portrait
      await api.uploadPortrait(id, file, (pct) => {
        setUploadPercent(pct);
      });

      setUploadPercent(null);
      setUploading(false);

      // Start portrait processing
      setStage("processing");
      setProcessingStep("Estimating depth...");
      setProcessingPercent(0);

      await api.runPortrait(id, settings);
    } catch (err: any) {
      setError(err.message || "Failed to process portrait");
      setUploading(false);
      setUploadPercent(null);
    }
  };

  const handleReset = () => {
    setStage("upload");
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    setProcessingStep("");
    setProcessingPercent(0);
    setDepthPreviewUrl(null);
    setDepthLoaded(false);
    setShowDepth(false);
    setGaussianCount(null);
  };

  // Render upload stage
  const renderUpload = () => (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-sm mb-4">
          <Wand2 className="w-4 h-4" />
          Portrait Mode
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">
          Turn a photo into 3D
        </h2>
        <p className="text-gray-400">
          Upload a single portrait photo and generate a 3D Gaussian splat in
          seconds
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`relative border-2 border-dashed rounded-xl transition-all duration-200 cursor-pointer overflow-hidden ${
          dragOver
            ? "border-violet-400 bg-violet-500/10"
            : previewUrl
              ? "border-gray-600 bg-gray-800/50"
              : "border-gray-700 hover:border-gray-500 bg-gray-800/30"
        }`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFileSelect(f);
          }}
        />

        {previewUrl ? (
          <div className="relative group">
            <img
              src={previewUrl}
              alt="Portrait preview"
              className="w-full max-h-[400px] object-contain mx-auto"
            />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center">
              <span className="text-white opacity-0 group-hover:opacity-100 transition-opacity text-sm">
                Click to change image
              </span>
            </div>
            {file && (
              <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded-md">
                {file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)
              </div>
            )}
          </div>
        ) : (
          <div className="py-16 px-8 text-center">
            <div className="w-16 h-16 rounded-full bg-violet-500/10 flex items-center justify-center mx-auto mb-4">
              <ImageIcon className="w-8 h-8 text-violet-400" />
            </div>
            <p className="text-gray-300 font-medium mb-1">
              Drag & drop your portrait photo
            </p>
            <p className="text-gray-500 text-sm">
              or click to browse -- .jpg, .png, .webp
            </p>
          </div>
        )}
      </div>

      {/* Advanced settings */}
      <div className="bg-gray-800/50 rounded-xl border border-gray-700/50">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-4 py-3 text-sm text-gray-400 hover:text-gray-300 transition"
        >
          <span className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Advanced Settings
          </span>
          {showAdvanced ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {showAdvanced && (
          <div className="px-4 pb-4 space-y-4 border-t border-gray-700/50 pt-3">
            {/* Stride */}
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 w-36">
                Density (Stride):
              </label>
              <input
                type="range"
                min={1}
                max={4}
                step={1}
                value={settings.stride}
                onChange={(e) =>
                  setSettings((s) => ({
                    ...s,
                    stride: Number(e.target.value),
                  }))
                }
                className="flex-1 accent-violet-500"
              />
              <span className="text-sm text-white w-8 text-right">
                {settings.stride}
              </span>
            </div>
            <p className="text-xs text-gray-500 ml-40">
              Lower stride = denser output (slower). 1 is maximum density.
            </p>

            {/* Novel views */}
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 w-36">Novel Views:</label>
              <div className="flex gap-2">
                {[4, 6, 8].map((n) => (
                  <button
                    key={n}
                    onClick={() =>
                      setSettings((s) => ({ ...s, num_novel_views: n }))
                    }
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                      settings.num_novel_views === n
                        ? "bg-violet-600 text-white"
                        : "bg-gray-700 text-gray-400 hover:text-white"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-xs text-gray-500 ml-40">
              More views improve 3D quality but take longer.
            </p>

            {/* Include background */}
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 w-36">
                Include Background:
              </label>
              <button
                onClick={() =>
                  setSettings((s) => ({
                    ...s,
                    include_background: !s.include_background,
                  }))
                }
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  settings.include_background ? "bg-violet-600" : "bg-gray-600"
                }`}
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    settings.include_background ? "translate-x-5" : ""
                  }`}
                />
              </button>
              <span className="text-xs text-gray-500">
                {settings.include_background
                  ? "Background included in 3D model"
                  : "Subject only (recommended for portraits)"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-lg p-3 flex items-center gap-2">
          <XCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={!file || uploading}
        className={`w-full py-4 rounded-xl font-semibold text-lg flex items-center justify-center gap-3 transition-all duration-200 ${
          file && !uploading
            ? "bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white shadow-lg shadow-violet-500/20"
            : "bg-gray-700 text-gray-400 cursor-not-allowed"
        }`}
      >
        {uploading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Uploading... {uploadPercent !== null ? `${uploadPercent}%` : ""}
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Generate 3D
          </>
        )}
      </button>

      {/* Upload progress */}
      {uploadPercent !== null && (
        <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-gradient-to-r from-violet-500 to-purple-500 h-full rounded-full transition-all duration-300"
            style={{ width: `${uploadPercent}%` }}
          />
        </div>
      )}
    </div>
  );

  // Render processing stage
  const renderProcessing = () => (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: portrait + depth preview */}
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden border border-gray-700 bg-gray-900">
            {/* Toggle between RGB and depth */}
            {depthPreviewUrl && (
              <div className="absolute top-3 right-3 z-10 flex bg-black/60 rounded-lg overflow-hidden backdrop-blur-sm">
                <button
                  onClick={() => setShowDepth(false)}
                  className={`px-3 py-1.5 text-xs font-medium transition ${
                    !showDepth
                      ? "bg-violet-600 text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Photo
                </button>
                <button
                  onClick={() => setShowDepth(true)}
                  className={`px-3 py-1.5 text-xs font-medium transition ${
                    showDepth
                      ? "bg-violet-600 text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Depth
                </button>
              </div>
            )}

            {/* Images */}
            <div className="relative">
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Portrait"
                  className={`w-full object-contain max-h-[400px] transition-opacity duration-500 ${
                    showDepth && depthPreviewUrl ? "opacity-0" : "opacity-100"
                  }`}
                />
              )}
              {depthPreviewUrl && (
                <img
                  src={depthPreviewUrl}
                  alt="Depth map"
                  className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-500 ${
                    showDepth ? "opacity-100" : "opacity-0"
                  }`}
                  onLoad={() => setDepthLoaded(true)}
                />
              )}
            </div>
          </div>

          {/* Depth preview notice */}
          {depthPreviewUrl && depthLoaded && (
            <div className="flex items-center gap-2 text-xs text-violet-400 animate-fade-in">
              <Eye className="w-3.5 h-3.5" />
              Depth map estimated -- toggle above to compare
            </div>
          )}
        </div>

        {/* Right: processing status */}
        <div className="space-y-6">
          <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-violet-400" />
              Processing your portrait
            </h3>

            {/* Progress bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1.5">
                <span className="text-gray-400">Progress</span>
                <span className="text-white font-medium">
                  {Math.round(processingPercent)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-violet-500 to-purple-500 h-full rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${processingPercent}%` }}
                />
              </div>
            </div>

            {/* Step indicators */}
            <div className="space-y-2">
              {PROCESSING_STEPS.map((step, i) => {
                const isActive =
                  processingStep.toLowerCase().includes(step.toLowerCase().replace("...", "").trim()) ||
                  (processingStep && PROCESSING_STEPS.indexOf(processingStep) === i);

                // Determine if this step is done based on the current step index
                const currentIdx = PROCESSING_STEPS.findIndex((s) =>
                  processingStep.toLowerCase().includes(s.toLowerCase().replace("...", "").trim())
                );
                const isDone = currentIdx > i;

                return (
                  <div
                    key={step}
                    className={`flex items-center gap-3 py-1.5 px-3 rounded-lg transition-all duration-300 ${
                      isActive
                        ? "bg-violet-500/10 border border-violet-500/20"
                        : isDone
                          ? "opacity-60"
                          : "opacity-30"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                    ) : isActive ? (
                      <Loader2 className="w-4 h-4 text-violet-400 animate-spin shrink-0" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-gray-600 shrink-0" />
                    )}
                    <span
                      className={`text-sm ${
                        isActive
                          ? "text-violet-300 font-medium"
                          : isDone
                            ? "text-gray-400"
                            : "text-gray-500"
                      }`}
                    >
                      {step}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Cancel */}
          <button
            onClick={() => {
              api.cancelPipeline(id);
              setStage("upload");
              setProcessingPercent(0);
              setProcessingStep("");
            }}
            className="text-sm text-gray-500 hover:text-red-400 transition flex items-center gap-1"
          >
            <XCircle className="w-3.5 h-3.5" />
            Cancel processing
          </button>
        </div>
      </div>
    </div>
  );

  // Render complete stage
  const renderComplete = () => (
    <div className="space-y-6">
      {/* Success banner */}
      <div className="text-center py-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 mb-3">
          <CheckCircle2 className="w-5 h-5" />
          <span className="font-medium">Portrait 3D model ready</span>
        </div>
        {gaussianCount && (
          <p className="text-gray-400 text-sm">
            Generated{" "}
            <span className="text-white font-medium">
              {gaussianCount.toLocaleString()}
            </span>{" "}
            Gaussians from your portrait
          </p>
        )}
      </div>

      {/* Viewer */}
      <div className="h-[500px] md:h-[600px] rounded-xl overflow-hidden border border-gray-700">
        <SplatViewer
          plyUrl={`/api/projects/${id}/output/point_cloud.ply`}
          projectId={id}
        />
      </div>

      {/* Action buttons */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          onClick={() => router.push(`/project/${id}/view`)}
          className="bg-violet-600 hover:bg-violet-700 text-white px-5 py-2.5 rounded-lg flex items-center gap-2 transition font-medium"
        >
          <Eye className="w-4 h-4" />
          View in Full Viewer
        </button>
        <button
          onClick={handleReset}
          className="bg-gray-700 hover:bg-gray-600 text-white px-5 py-2.5 rounded-lg flex items-center gap-2 transition"
        >
          <RotateCcw className="w-4 h-4" />
          Try Another
        </button>
        <button
          onClick={() => router.push(`/project/${id}/train`)}
          className="bg-gray-700 hover:bg-gray-600 text-white px-5 py-2.5 rounded-lg flex items-center gap-2 transition"
        >
          <Sparkles className="w-4 h-4" />
          Refine with Training
        </button>
      </div>
    </div>
  );

  // Render error stage
  const renderError = () => (
    <div className="max-w-lg mx-auto text-center py-12">
      <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mx-auto mb-4">
        <XCircle className="w-8 h-8 text-red-400" />
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">
        Processing Failed
      </h3>
      <p className="text-gray-400 mb-6">{error || "An unexpected error occurred"}</p>
      <button
        onClick={handleReset}
        className="bg-violet-600 hover:bg-violet-700 text-white px-6 py-2.5 rounded-lg flex items-center gap-2 transition mx-auto"
      >
        <RotateCcw className="w-4 h-4" />
        Try Again
      </button>
    </div>
  );

  return (
    <div className="py-4">
      {stage === "upload" && renderUpload()}
      {stage === "processing" && renderProcessing()}
      {stage === "complete" && renderComplete()}
      {stage === "error" && renderError()}
    </div>
  );
}
