"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import { SplatViewer } from "@/components/SplatViewer";
import type { ProjectDetail, FewViewSettings } from "@/lib/types";
import {
  Upload,
  Loader2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Eye,
  Settings,
  Images,
  CheckCircle2,
  XCircle,
  Trash2,
  CircleDot,
  ArrowRight,
  Layers,
} from "lucide-react";

type Stage = "upload" | "processing" | "complete" | "error";

const ARRANGEMENTS = [
  {
    id: "turntable" as const,
    label: "Turntable",
    desc: "Photos taken around the object",
    icon: "rotate",
  },
  {
    id: "forward" as const,
    label: "Forward-facing",
    desc: "All cameras face same direction",
    icon: "forward",
  },
  {
    id: "free" as const,
    label: "Free",
    desc: "Arbitrary camera positions",
    icon: "scatter",
  },
];

const PROCESSING_STEPS = [
  "Loading images",
  "Estimating depth",
  "Back-projecting to 3D",
  "Assigning camera poses",
  "Merging point clouds",
  "Filling gaps",
  "Converting to Gaussians",
  "Writing model",
];

export default function FewViewPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { logs, progress, status } = useWebSocket(id);

  // Core state
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Settings
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [settings, setSettings] = useState<FewViewSettings>({
    arrangement: "turntable",
    merge_resolution: 0.01,
    fill_gaps: true,
  });

  // Processing state
  const [processingStep, setProcessingStep] = useState<string>("");
  const [processingPercent, setProcessingPercent] = useState(0);

  // Complete state
  const [gaussianCount, setGaussianCount] = useState<number | null>(null);
  const [mergedCount, setMergedCount] = useState<number | null>(null);
  const [fillCount, setFillCount] = useState<number | null>(null);
  const [imagesUsed, setImagesUsed] = useState<number | null>(null);

  // Load project info
  useEffect(() => {
    api.getProject(id).then((p: ProjectDetail) => {
      setProject(p);
      if (p.step === "training_complete" && p.has_output) {
        setStage("complete");
      } else if (p.step === "fewview_processing") {
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
      if (status.state === "complete" || status.state === "completed" || status.step === "training_complete") {
        setStage("complete");
        api.getProject(id).then(setProject);
      } else if (status.state === "failed") {
        setStage("error");
        setError(status.error || "Processing failed");
      }
    }
  }, [progress, status, stage, id]);

  // Extract stats from logs
  useEffect(() => {
    for (const log of logs) {
      const gaussMatch = log.match(/(\d[\d,]+)\s*gaussians?/i);
      if (gaussMatch) {
        setGaussianCount(parseInt(gaussMatch[1].replace(/,/g, ""), 10));
      }
      const mergedMatch = log.match(/(\d+)\s*merged/i);
      if (mergedMatch) {
        setMergedCount(parseInt(mergedMatch[1], 10));
      }
      const fillMatch = log.match(/(\d+)\s*fill/i);
      if (fillMatch) {
        setFillCount(parseInt(fillMatch[1], 10));
      }
      const imgMatch = log.match(/from\s+(\d+)\s+images/i);
      if (imgMatch) {
        setImagesUsed(parseInt(imgMatch[1], 10));
      }
    }
  }, [logs]);

  // File handling
  const handleFilesSelect = useCallback((selectedFiles: File[]) => {
    const validFiles = selectedFiles.filter((f) =>
      ["image/jpeg", "image/png", "image/webp"].includes(f.type)
    );
    if (validFiles.length === 0) {
      setError("Please select valid image files (JPG, PNG, or WebP)");
      return;
    }

    setFiles((prev) => {
      const combined = [...prev, ...validFiles].slice(0, 8);
      return combined;
    });
    setError(null);

    // Generate previews
    for (const f of validFiles) {
      const url = URL.createObjectURL(f);
      setPreviews((prev) => [...prev, url].slice(0, 8));
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = Array.from(e.dataTransfer.files);
      handleFilesSelect(dropped);
    },
    [handleFilesSelect]
  );

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => {
      const url = prev[index];
      if (url) URL.revokeObjectURL(url);
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleReconstruct = async () => {
    if (files.length < 2) {
      setError("At least 2 images are required");
      return;
    }

    setUploading(true);
    setError(null);
    setUploadPercent(0);

    try {
      // Upload all images
      await api.uploadFewView(id, files, (pct: number) => {
        setUploadPercent(pct);
      });

      setUploadPercent(null);
      setUploading(false);

      // Start fewview processing
      setStage("processing");
      setProcessingStep("Loading images...");
      setProcessingPercent(0);

      await api.runFewView(id, settings);
    } catch (err: any) {
      setError(err.message || "Failed to process images");
      setUploading(false);
      setUploadPercent(null);
    }
  };

  const handleReset = () => {
    setStage("upload");
    setFiles([]);
    setPreviews((prev) => {
      prev.forEach(URL.revokeObjectURL);
      return [];
    });
    setError(null);
    setProcessingStep("");
    setProcessingPercent(0);
    setGaussianCount(null);
    setMergedCount(null);
    setFillCount(null);
    setImagesUsed(null);
  };

  // --- Render upload stage ---
  const renderUpload = () => (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm mb-4">
          <Images className="w-4 h-4" />
          Few-View Mode
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">
          Reconstruct 3D from photos
        </h2>
        <p className="text-gray-400">
          Upload 2-8 photos of an object or scene from different angles
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`relative border-2 border-dashed rounded-xl transition-all duration-200 cursor-pointer overflow-hidden ${
          dragOver
            ? "border-emerald-400 bg-emerald-500/10"
            : files.length > 0
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
          multiple
          className="hidden"
          onChange={(e) => {
            const selected = Array.from(e.target.files || []);
            if (selected.length > 0) handleFilesSelect(selected);
          }}
        />

        {files.length > 0 ? (
          <div className="p-4">
            {/* Image grid */}
            <div className="grid grid-cols-4 gap-3 mb-3">
              {previews.map((url, i) => (
                <div key={i} className="relative group aspect-square rounded-lg overflow-hidden border border-gray-600">
                  <img
                    src={url}
                    alt={`Photo ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(i);
                    }}
                    className="absolute top-1 right-1 w-6 h-6 rounded-full bg-black/60 text-white opacity-0 group-hover:opacity-100 transition flex items-center justify-center"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                  <div className="absolute bottom-1 left-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded">
                    {i + 1}
                  </div>
                </div>
              ))}
              {files.length < 8 && (
                <div className="aspect-square rounded-lg border-2 border-dashed border-gray-600 flex items-center justify-center text-gray-500 hover:border-emerald-500/50 hover:text-emerald-400 transition">
                  <Upload className="w-6 h-6" />
                </div>
              )}
            </div>
            {/* Counter badge */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">
                <span className="text-white font-medium">{files.length}</span> of 8 images
              </span>
              {files.length < 2 && (
                <span className="text-xs text-amber-400">Minimum 2 images required</span>
              )}
            </div>
          </div>
        ) : (
          <div className="py-16 px-8 text-center">
            <div className="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center mx-auto mb-4">
              <Images className="w-8 h-8 text-emerald-400" />
            </div>
            <p className="text-gray-300 font-medium mb-1">
              Drag & drop your photos
            </p>
            <p className="text-gray-500 text-sm">
              or click to browse -- .jpg, .png, .webp -- 2 to 8 images
            </p>
          </div>
        )}
      </div>

      {/* Arrangement selector */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Camera Arrangement
        </label>
        <div className="grid grid-cols-3 gap-3">
          {ARRANGEMENTS.map((arr) => (
            <button
              key={arr.id}
              onClick={() =>
                setSettings((s) => ({ ...s, arrangement: arr.id }))
              }
              className={`px-4 py-3 rounded-xl text-left transition border-2 ${
                settings.arrangement === arr.id
                  ? "bg-emerald-600/10 border-emerald-500 text-emerald-400"
                  : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600 hover:text-white"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                {arr.icon === "rotate" && <RotateCcw className="w-4 h-4" />}
                {arr.icon === "forward" && <ArrowRight className="w-4 h-4" />}
                {arr.icon === "scatter" && <CircleDot className="w-4 h-4" />}
                <span className="text-sm font-medium">{arr.label}</span>
              </div>
              <p className="text-xs opacity-60">{arr.desc}</p>
            </button>
          ))}
        </div>
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
            {/* Fill gaps toggle */}
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 w-36">
                Fill Gaps:
              </label>
              <button
                onClick={() =>
                  setSettings((s) => ({
                    ...s,
                    fill_gaps: !s.fill_gaps,
                  }))
                }
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  settings.fill_gaps ? "bg-emerald-600" : "bg-gray-600"
                }`}
              >
                <div
                  className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    settings.fill_gaps ? "translate-x-5" : ""
                  }`}
                />
              </button>
              <span className="text-xs text-gray-500">
                {settings.fill_gaps
                  ? "Fill gaps between views (recommended)"
                  : "No gap filling"}
              </span>
            </div>

            {/* Merge resolution */}
            <div className="flex items-center gap-4">
              <label className="text-sm text-gray-400 w-36">
                Merge Resolution:
              </label>
              <input
                type="range"
                min={0.005}
                max={0.05}
                step={0.005}
                value={settings.merge_resolution}
                onChange={(e) =>
                  setSettings((s) => ({
                    ...s,
                    merge_resolution: Number(e.target.value),
                  }))
                }
                className="flex-1 accent-emerald-500"
              />
              <span className="text-sm text-white w-14 text-right">
                {(settings.merge_resolution || 0.01).toFixed(3)}
              </span>
            </div>
            <p className="text-xs text-gray-500 ml-40">
              Lower = more detail, higher = fewer duplicates.
            </p>
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
        onClick={handleReconstruct}
        disabled={files.length < 2 || uploading}
        className={`w-full py-4 rounded-xl font-semibold text-lg flex items-center justify-center gap-3 transition-all duration-200 ${
          files.length >= 2 && !uploading
            ? "bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-500 hover:to-green-500 text-white shadow-lg shadow-emerald-500/20"
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
            Reconstruct 3D
          </>
        )}
      </button>

      {/* Upload progress */}
      {uploadPercent !== null && (
        <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-gradient-to-r from-emerald-500 to-green-500 h-full rounded-full transition-all duration-300"
            style={{ width: `${uploadPercent}%` }}
          />
        </div>
      )}
    </div>
  );

  // --- Render processing stage ---
  const renderProcessing = () => (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: thumbnail strip */}
        <div className="space-y-4">
          <div className="rounded-xl border border-gray-700 bg-gray-900 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
              <Images className="w-4 h-4" />
              Input Images
            </h3>
            <div className="grid grid-cols-4 gap-2">
              {previews.map((url, i) => (
                <div key={i} className="aspect-square rounded-lg overflow-hidden border border-gray-700">
                  <img
                    src={url}
                    alt={`Photo ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: processing status */}
        <div className="space-y-6">
          <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-emerald-400" />
              Reconstructing 3D scene
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
                  className="bg-gradient-to-r from-emerald-500 to-green-500 h-full rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${processingPercent}%` }}
                />
              </div>
            </div>

            {/* Step indicators */}
            <div className="space-y-2">
              {PROCESSING_STEPS.map((step, i) => {
                const stepLower = step.toLowerCase();
                const currentLower = processingStep.toLowerCase();
                const isActive = currentLower.includes(stepLower) ||
                  (stepLower.includes("depth") && currentLower.includes("depth")) ||
                  (stepLower.includes("merging") && currentLower.includes("merg")) ||
                  (stepLower.includes("filling") && currentLower.includes("fill")) ||
                  (stepLower.includes("converting") && currentLower.includes("gaussian")) ||
                  (stepLower.includes("writing") && currentLower.includes("ply"));

                const currentIdx = PROCESSING_STEPS.findIndex((s) => {
                  const sLower = s.toLowerCase();
                  return currentLower.includes(sLower) ||
                    (sLower.includes("depth") && currentLower.includes("depth")) ||
                    (sLower.includes("merging") && currentLower.includes("merg")) ||
                    (sLower.includes("filling") && currentLower.includes("fill")) ||
                    (sLower.includes("converting") && currentLower.includes("gaussian")) ||
                    (sLower.includes("writing") && currentLower.includes("ply"));
                });
                const isDone = currentIdx > i;

                return (
                  <div
                    key={step}
                    className={`flex items-center gap-3 py-1.5 px-3 rounded-lg transition-all duration-300 ${
                      isActive
                        ? "bg-emerald-500/10 border border-emerald-500/20"
                        : isDone
                          ? "opacity-60"
                          : "opacity-30"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                    ) : isActive ? (
                      <Loader2 className="w-4 h-4 text-emerald-400 animate-spin shrink-0" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-gray-600 shrink-0" />
                    )}
                    <span
                      className={`text-sm ${
                        isActive
                          ? "text-emerald-300 font-medium"
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

  // --- Render complete stage ---
  const renderComplete = () => (
    <div className="space-y-6">
      {/* Success banner */}
      <div className="text-center py-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 mb-3">
          <CheckCircle2 className="w-5 h-5" />
          <span className="font-medium">3D reconstruction complete</span>
        </div>
        <p className="text-gray-400 text-sm">
          {imagesUsed && (
            <>
              Reconstructed from{" "}
              <span className="text-white font-medium">{imagesUsed}</span> images
            </>
          )}
          {gaussianCount && (
            <>
              {imagesUsed ? " -- " : ""}
              <span className="text-white font-medium">
                {gaussianCount.toLocaleString()}
              </span>{" "}
              Gaussians
            </>
          )}
          {mergedCount != null && fillCount != null && (
            <>
              {" -- "}
              <span className="text-white font-medium">
                {mergedCount.toLocaleString()}
              </span>{" "}
              merged from overlaps
            </>
          )}
        </p>
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
          className="bg-emerald-600 hover:bg-emerald-700 text-white px-5 py-2.5 rounded-lg flex items-center gap-2 transition font-medium"
        >
          <Eye className="w-4 h-4" />
          View Full Screen
        </button>
        <button
          onClick={handleReset}
          className="bg-gray-700 hover:bg-gray-600 text-white px-5 py-2.5 rounded-lg flex items-center gap-2 transition"
        >
          <RotateCcw className="w-4 h-4" />
          Try Another
        </button>
      </div>
    </div>
  );

  // --- Render error stage ---
  const renderError = () => (
    <div className="max-w-lg mx-auto text-center py-12">
      <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mx-auto mb-4">
        <XCircle className="w-8 h-8 text-red-400" />
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">
        Reconstruction Failed
      </h3>
      <p className="text-gray-400 mb-6">{error || "An unexpected error occurred"}</p>
      <button
        onClick={handleReset}
        className="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-2.5 rounded-lg flex items-center gap-2 transition mx-auto"
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
