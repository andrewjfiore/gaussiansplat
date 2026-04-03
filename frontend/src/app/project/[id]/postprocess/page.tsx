"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { SplatViewer } from "@/components/SplatViewer";
import type { ProjectDetail } from "@/lib/types";
import {
  ArrowLeft,
  ArrowRight,
  Loader2,
  RotateCcw,
  Scissors,
  Eye,
  Download,
  Check,
  RefreshCw,
} from "lucide-react";

interface PreviewResult {
  total: number;
  kept: number;
  pruned: number;
  pruned_pct: number;
  by_opacity: number;
  by_scale: number;
  by_position: number;
  median_scale: number;
  file_size_mb: number;
  estimated_output_mb: number;
}

export default function PostProcessPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [project, setProject] = useState<ProjectDetail | null>(null);

  // Prune params
  const [minOpacity, setMinOpacity] = useState(0.1);
  const [maxScaleMult, setMaxScaleMult] = useState(8.0);
  const [positionPctl, setPositionPctl] = useState(99.0);

  // Preview
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewerKey, setViewerKey] = useState(0); // bump to force viewer reload
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Novel view generation
  const [novelModel, setNovelModel] = useState("zero123pp");
  const [novelRefs, setNovelRefs] = useState(4);
  const [novelGenerating, setNovelGenerating] = useState(false);
  const [novelDone, setNovelDone] = useState(false);

  useEffect(() => {
    api.getProject(id).then(setProject);
  }, [id]);

  // Debounced preview — fires 300ms after last slider change
  const fetchPreview = useCallback(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.prunePreview(id, {
          min_opacity: minOpacity,
          max_scale_mult: maxScaleMult,
          position_percentile: positionPctl,
        });
        setPreview(result);
      } catch (err: any) {
        setError(err.message);
      }
      setLoading(false);
    }, 300);
  }, [id, minOpacity, maxScaleMult, positionPctl]);

  // Fetch preview on mount and when params change
  useEffect(() => {
    if (project?.has_output) fetchPreview();
  }, [project?.has_output, fetchPreview]);

  const handleApply = async () => {
    setApplying(true);
    setApplied(false);
    try {
      await api.pruneSplat(id, {
        min_opacity: minOpacity,
        max_scale_mult: maxScaleMult,
        position_percentile: positionPctl,
      });
      setApplied(true);
      setViewerKey((k) => k + 1); // reload viewer with pruned PLY
      fetchPreview();
    } catch (err: any) {
      setError(err.message);
    }
    setApplying(false);
  };

  const handleReset = async () => {
    try {
      await api.pruneReset(id);
      setApplied(false);
      setViewerKey((k) => k + 1); // reload viewer with original PLY
      fetchPreview();
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (!project) {
    return <div className="text-gray-400 text-center py-12">Loading...</div>;
  }

  if (!project.has_output) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-400 text-lg">No splat output available</p>
        <p className="text-gray-500 text-sm mt-2">Complete training first</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Nav */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => router.push(`/project/${id}/train`)}
          className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
        >
          <ArrowLeft className="w-4 h-4 inline mr-1" /> Training
        </button>
        <h2 className="text-lg font-medium">Post-Processing</h2>
        <button
          onClick={() => router.push(`/project/${id}/view`)}
          className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700 ml-auto"
        >
          Full Viewer <ArrowRight className="w-4 h-4 inline ml-1" />
        </button>
      </div>

      {/* Embedded viewer — reloads when pruning is applied */}
      <div className="h-[400px] rounded-lg overflow-hidden border border-gray-700">
        <SplatViewer
          key={viewerKey}
          plyUrl={`/api/projects/${id}/output/point_cloud.ply`}
        />
      </div>

      {/* Live preview stats */}
      {preview && (
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">
              <Eye className="w-4 h-4 inline mr-1" /> Live Preview
            </h3>
            {loading && <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-white">
                {preview.total.toLocaleString()}
              </div>
              <div className="text-xs text-gray-400">Total Gaussians</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-400">
                {preview.pruned.toLocaleString()}
              </div>
              <div className="text-xs text-gray-400">
                Will Remove ({preview.pruned_pct}%)
              </div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">
                {preview.kept.toLocaleString()}
              </div>
              <div className="text-xs text-gray-400">Will Keep</div>
            </div>
          </div>
          <div className="mt-3 flex gap-4 text-xs text-gray-500">
            <span>Opacity: {preview.by_opacity.toLocaleString()}</span>
            <span>Scale: {preview.by_scale.toLocaleString()}</span>
            <span>Position: {preview.by_position.toLocaleString()}</span>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            File: {preview.file_size_mb} MB → ~{preview.estimated_output_mb} MB
          </div>
        </div>
      )}

      {/* Sliders */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 space-y-5">
        <h3 className="text-sm font-medium text-gray-300">
          <Scissors className="w-4 h-4 inline mr-1" /> Pruning Parameters
        </h3>

        {/* Opacity */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-gray-400">Min Opacity</label>
            <span className="text-sm text-white font-mono">{minOpacity.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={minOpacity}
            onChange={(e) => setMinOpacity(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Remove near-invisible Gaussians. Higher = more aggressive. 0.1 is a safe default.
          </p>
        </div>

        {/* Scale */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-gray-400">Max Scale Multiplier</label>
            <span className="text-sm text-white font-mono">{maxScaleMult.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min={2}
            max={30}
            step={0.5}
            value={maxScaleMult}
            onChange={(e) => setMaxScaleMult(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Remove oversized blobs (larger than Nx the median scale). Lower = more aggressive.
          </p>
        </div>

        {/* Position */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm text-gray-400">Position Percentile</label>
            <span className="text-sm text-white font-mono">{positionPctl.toFixed(1)}%</span>
          </div>
          <input
            type="range"
            min={90}
            max={100}
            step={0.5}
            value={positionPctl}
            onChange={(e) => setPositionPctl(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-1">
            Remove distant outlier Gaussians. 99% keeps 99% closest to center. Lower = tighter crop.
          </p>
        </div>
      </div>

      {/* Novel View Generation */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 space-y-4">
        <h3 className="text-sm font-medium text-gray-300">
          <RefreshCw className="w-4 h-4 inline mr-1" /> AI Novel View Generation
        </h3>
        <p className="text-xs text-gray-500">
          Generate novel views from unseen angles using AI diffusion models.
          These can fill coverage gaps (e.g., back of objects never photographed).
        </p>

        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Model:</label>
            <select
              value={novelModel}
              onChange={(e) => setNovelModel(e.target.value)}
              disabled={novelGenerating}
              className="bg-gray-900 border border-gray-600 rounded px-3 py-1.5 text-sm text-white"
            >
              <option value="zero123pp">Zero123++ (6 views, ~5GB VRAM)</option>
              <option value="wonder3d">Wonder3D (6 views + normals, ~8GB)</option>
              <option value="era3d">Era3D (6 hi-res views, ~12GB)</option>
              <option value="sd_inpaint">SD Inpainting (4 views, ~4GB)</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">References:</label>
            <select
              value={novelRefs}
              onChange={(e) => setNovelRefs(Number(e.target.value))}
              disabled={novelGenerating}
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value={2}>2 (fast)</option>
              <option value={4}>4 (balanced)</option>
              <option value={8}>8 (thorough)</option>
              <option value={12}>12 (maximum)</option>
            </select>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={async () => {
              setNovelGenerating(true);
              setNovelDone(false);
              try {
                await api.generateNovelViews(id, {
                  model: novelModel,
                  num_refs: novelRefs,
                });
                setNovelDone(true);
              } catch (err: any) {
                setError(err.message);
              }
              setNovelGenerating(false);
            }}
            disabled={novelGenerating}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition text-sm"
          >
            {novelGenerating ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            {novelGenerating ? "Generating..." : novelDone ? "Regenerate" : "Generate Novel Views"}
          </button>
        </div>

        {novelDone && (
          <p className="text-xs text-green-400">
            Novel views generated. Run Refine from the Training page to incorporate them.
          </p>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          onClick={handleApply}
          disabled={applying || !preview}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
        >
          {applying ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : applied ? (
            <Check className="w-4 h-4" />
          ) : (
            <Scissors className="w-4 h-4" />
          )}
          {applying ? "Applying..." : applied ? "Applied!" : "Apply Pruning"}
        </button>
        <button
          onClick={handleReset}
          className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
        >
          <RotateCcw className="w-4 h-4" /> Restore Original
        </button>
        <a
          href={`/api/projects/${id}/ply`}
          download
          className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
        >
          <Download className="w-4 h-4" /> Download PLY
        </a>
      </div>

      {applied && (
        <div className="bg-green-900/20 border border-green-800 rounded-lg p-3 text-sm text-green-400">
          Pruning applied. Open the{" "}
          <button
            onClick={() => router.push(`/project/${id}/view`)}
            className="underline hover:text-green-300"
          >
            viewer
          </button>{" "}
          to see the result. Original PLY is backed up — click Restore to undo.
        </div>
      )}

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 text-sm text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
