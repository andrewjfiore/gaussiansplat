"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AOVResult, AOVImage, AOVStats } from "@/lib/types";

/* ---------- Tabs ---------- */

const CHANNEL_TABS = [
  { key: "depth", label: "Depth" },
  { key: "scale", label: "Scale" },
  { key: "opacity", label: "Opacity" },
  { key: "density", label: "Density" },
  { key: "stats", label: "Stats" },
] as const;

type ChannelTab = (typeof CHANNEL_TABS)[number]["key"];

/* ---------- Lightbox ---------- */

function Lightbox({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt: string;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="relative max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <img
          src={src}
          alt={alt}
          className="max-w-full max-h-[85vh] rounded-lg shadow-2xl"
        />
        <button
          type="button"
          onClick={onClose}
          className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-gray-800 border border-white/20 text-white text-sm flex items-center justify-center hover:bg-gray-700"
        >
          X
        </button>
        <p className="text-center text-xs text-gray-400 mt-2">{alt}</p>
      </div>
    </div>
  );
}

/* ---------- Image Grid ---------- */

const DIRECTION_ORDER = ["front", "back", "left", "right", "top", "bottom"];
const DENSITY_ORDER = ["XY", "XZ", "YZ"];

function ImageGrid({
  images,
  channel,
}: {
  images: AOVImage[];
  channel: string;
}) {
  const [lightbox, setLightbox] = useState<{ src: string; alt: string } | null>(
    null
  );

  const filtered = images.filter((img) => img.channel === channel);

  // Sort by canonical direction order
  const order = channel === "density" ? DENSITY_ORDER : DIRECTION_ORDER;
  const sorted = [...filtered].sort(
    (a, b) => order.indexOf(a.direction) - order.indexOf(b.direction)
  );

  if (sorted.length === 0) {
    return (
      <p className="text-sm text-gray-500 text-center py-4">
        No {channel} visualizations available for this model.
      </p>
    );
  }

  const gridCols = channel === "density" ? "grid-cols-3" : "grid-cols-3";

  return (
    <>
      <div className={`grid ${gridCols} gap-2`}>
        {sorted.map((img) => (
          <button
            key={img.filename}
            type="button"
            className="group relative rounded-lg overflow-hidden border border-white/10 hover:border-white/30 transition-colors bg-black"
            onClick={() =>
              setLightbox({
                src: img.url || "",
                alt: `${channel} - ${img.direction}`,
              })
            }
          >
            <img
              src={img.url}
              alt={`${channel} ${img.direction}`}
              className="w-full aspect-square object-cover"
              loading="lazy"
            />
            <span className="absolute bottom-0 inset-x-0 bg-black/70 text-[10px] text-gray-300 text-center py-0.5 capitalize">
              {img.direction}
            </span>
          </button>
        ))}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <Lightbox
          src={lightbox.src}
          alt={lightbox.alt}
          onClose={() => setLightbox(null)}
        />
      )}
    </>
  );
}

/* ---------- Stats Table ---------- */

function StatsTable({ stats }: { stats: AOVStats }) {
  const rows: [string, string][] = [
    ["Gaussian Count", stats.gaussian_count.toLocaleString()],
    ["Scale (mean)", stats.scale_mean.toFixed(6)],
    ["Scale (median)", stats.scale_median.toFixed(6)],
    ["Scale (std)", stats.scale_std.toFixed(6)],
    ["Opacity (mean)", stats.opacity_mean.toFixed(4)],
    ["Opacity (median)", stats.opacity_median.toFixed(4)],
    ["Opacity (std)", stats.opacity_std.toFixed(4)],
    [
      "Bounding Box Min",
      stats.bbox_min.map((v) => v.toFixed(3)).join(", "),
    ],
    [
      "Bounding Box Max",
      stats.bbox_max.map((v) => v.toFixed(3)).join(", "),
    ],
    [
      "Scene Extent",
      stats.bbox_extent.map((v) => v.toFixed(3)).join(", "),
    ],
    ["Density (mean/cell)", stats.density_mean.toFixed(1)],
    ["Density (max/cell)", stats.density_max.toFixed(0)],
  ];

  return (
    <div className="space-y-0.5">
      {rows.map(([label, value]) => (
        <div
          key={label}
          className="flex justify-between items-baseline py-1.5 px-2 rounded text-xs odd:bg-white/5"
        >
          <span className="text-gray-400">{label}</span>
          <span className="text-gray-200 font-mono text-[11px]">{value}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------- Color Legend ---------- */

function ColorLegend({ channel }: { channel: string }) {
  const legends: Record<string, { low: string; high: string; colors: string }> = {
    depth: {
      low: "Near",
      high: "Far",
      colors: "from-[#440154] via-[#21918c] to-[#fde725]",
    },
    scale: {
      low: "Small",
      high: "Large",
      colors: "from-[#000004] via-[#a03c5e] to-[#fcffa4]",
    },
    opacity: {
      low: "Transparent",
      high: "Opaque",
      colors: "from-[#000000] to-[#ffffff]",
    },
    density: {
      low: "Sparse",
      high: "Dense",
      colors: "from-[#000000] via-[#c80000] to-[#ffffff]",
    },
  };

  const legend = legends[channel];
  if (!legend) return null;

  return (
    <div className="flex items-center gap-2 text-[10px] text-gray-400 px-1">
      <span>{legend.low}</span>
      <div
        className={`flex-1 h-2 rounded-full bg-gradient-to-r ${legend.colors}`}
      />
      <span>{legend.high}</span>
    </div>
  );
}

/* ---------- Main Panel ---------- */

interface Props {
  projectId: string;
  visible: boolean;
}

export function AOVPanel({ projectId, visible }: Props) {
  const [data, setData] = useState<AOVResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ChannelTab>("depth");

  const fetchAov = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getAovImages(projectId);
      setData(result);
    } catch (err: any) {
      setError(err.message || "Failed to load AOV data");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setError(null);
    try {
      await api.generateAov(projectId);
      // Poll until ready
      const poll = setInterval(async () => {
        try {
          const result = await api.getAovImages(projectId);
          if (result.available) {
            setData(result);
            setGenerating(false);
            clearInterval(poll);
          }
        } catch {
          // keep polling
        }
      }, 2000);
      // Safety timeout after 5 minutes
      setTimeout(() => {
        clearInterval(poll);
        setGenerating(false);
      }, 300_000);
    } catch (err: any) {
      setGenerating(false);
      setError(err.message || "Failed to start AOV generation");
    }
  }, [projectId]);

  useEffect(() => {
    if (visible && !data && !loading && !generating) {
      fetchAov();
    }
  }, [visible, data, loading, generating, fetchAov]);

  if (!visible) return null;

  return (
    <div className="bg-gray-900/80 backdrop-blur border border-white/10 rounded-xl p-4 space-y-3 w-full max-w-sm">
      <h3 className="text-sm font-semibold text-gray-200 tracking-wide uppercase">
        Scene Inspector (AOV)
      </h3>

      {/* Loading */}
      {(loading || generating) && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full" />
          <span className="ml-3 text-sm text-gray-400">
            {generating ? "Generating visualizations..." : "Loading..."}
          </span>
        </div>
      )}

      {/* Error */}
      {error && !loading && !generating && (
        <div className="text-sm text-red-400 bg-red-900/20 rounded-lg p-3">
          {error}
          <button
            onClick={fetchAov}
            className="ml-2 underline text-red-300 hover:text-red-200"
          >
            Retry
          </button>
        </div>
      )}

      {/* Generate button (if not yet available) */}
      {data && !data.available && !loading && !generating && (
        <div className="text-center py-4">
          <p className="text-sm text-gray-400 mb-3">
            AOV visualizations have not been generated yet.
          </p>
          <button
            type="button"
            onClick={handleGenerate}
            className="px-4 py-2 text-sm rounded-lg bg-purple-600 text-white hover:bg-purple-500 transition-colors"
          >
            Generate Visualizations
          </button>
        </div>
      )}

      {/* Results */}
      {data && data.available && !loading && !generating && (
        <>
          {/* Tab bar */}
          <div className="flex gap-0.5 bg-white/5 rounded-lg p-0.5">
            {CHANNEL_TABS.map((tab) => {
              // Only show tabs that have data
              if (tab.key !== "stats" && !data.images.some((img) => img.channel === tab.key)) return null;

              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  className={
                    "flex-1 px-2 py-1.5 text-xs font-medium rounded-md transition-colors " +
                    (activeTab === tab.key
                      ? "bg-purple-600 text-white"
                      : "text-gray-400 hover:text-gray-200 hover:bg-white/5")
                  }
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Color legend */}
          {activeTab !== "stats" && <ColorLegend channel={activeTab} />}

          {/* Tab content */}
          {activeTab === "stats" ? (
            data.stats ? (
              <StatsTable stats={data.stats} />
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                No statistics available.
              </p>
            )
          ) : (
            <ImageGrid images={data.images} channel={activeTab} />
          )}

          {/* Regenerate button */}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating}
            className="w-full px-3 py-1.5 text-xs rounded-lg border border-white/10 text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors disabled:opacity-50"
          >
            Regenerate
          </button>
        </>
      )}
    </div>
  );
}
