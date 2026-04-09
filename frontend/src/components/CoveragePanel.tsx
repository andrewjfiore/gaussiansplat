"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { CoverageResult } from "@/lib/types";

/* ---------- helpers ---------- */

function scoreColor(score: number): string {
  if (score >= 80) return "#22c55e"; // green-500
  if (score >= 60) return "#eab308"; // yellow-500
  if (score >= 40) return "#f97316"; // orange-500
  return "#ef4444"; // red-500
}

function scoreLabel(score: number): string {
  if (score >= 80) return "Excellent coverage";
  if (score >= 60) return "Good coverage, some gaps";
  if (score >= 40) return "Fair coverage, gaps detected";
  return "Poor coverage, significant gaps";
}

/* ---------- Radar Chart (pure SVG) ---------- */

const DIRECTIONS = ["Forward", "Right", "Backward", "Left", "Top", "Bottom"] as const;
const AXIS_COUNT = DIRECTIONS.length;
const CENTER = 100;
const RADIUS = 75;

function polarToXY(index: number, value: number): [number, number] {
  const angle = (Math.PI * 2 * index) / AXIS_COUNT - Math.PI / 2;
  const r = (value / 100) * RADIUS;
  return [CENTER + r * Math.cos(angle), CENTER + r * Math.sin(angle)];
}

function RadarChart({ scores }: { scores: Record<string, number> }) {
  // Build the polygon path for the data
  const points = DIRECTIONS.map((d, i) => {
    const val = scores[d] ?? 0;
    return polarToXY(i, val);
  });
  const dataPath = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";

  // Grid rings at 25%, 50%, 75%, 100%
  const rings = [25, 50, 75, 100];

  return (
    <svg viewBox="0 0 200 200" className="w-full max-w-[260px] mx-auto">
      {/* Grid rings */}
      {rings.map((pct) => {
        const ringPoints = DIRECTIONS.map((_, i) => polarToXY(i, pct));
        const ringPath =
          ringPoints.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";
        return (
          <path
            key={pct}
            d={ringPath}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={0.5}
          />
        );
      })}

      {/* Axis lines */}
      {DIRECTIONS.map((_, i) => {
        const [x, y] = polarToXY(i, 100);
        return (
          <line
            key={i}
            x1={CENTER}
            y1={CENTER}
            x2={x}
            y2={y}
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={0.5}
          />
        );
      })}

      {/* Data polygon */}
      <path d={dataPath} fill="rgba(59,130,246,0.2)" stroke="#3b82f6" strokeWidth={1.5} />

      {/* Data points */}
      {points.map(([x, y], i) => {
        const val = scores[DIRECTIONS[i]] ?? 0;
        const color = scoreColor(val);
        return <circle key={i} cx={x} cy={y} r={3} fill={color} />;
      })}

      {/* Axis labels */}
      {DIRECTIONS.map((d, i) => {
        const [x, y] = polarToXY(i, 118);
        const val = scores[d] ?? 0;
        return (
          <g key={d}>
            <text
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-gray-300"
              style={{ fontSize: "8px" }}
            >
              {d}
            </text>
            <text
              x={x}
              y={y + 10}
              textAnchor="middle"
              dominantBaseline="middle"
              style={{ fontSize: "7px", fill: scoreColor(val) }}
            >
              {val.toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- Score Badge ---------- */

function ScoreBadge({ score }: { score: number }) {
  const color = scoreColor(score);
  const circumference = 2 * Math.PI * 36;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 80 80" className="w-20 h-20">
        {/* Background circle */}
        <circle cx="40" cy="40" r="36" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="5" />
        {/* Progress arc */}
        <circle
          cx="40"
          cy="40"
          r="36"
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 40 40)"
        />
        {/* Score text */}
        <text
          x="40"
          y="38"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-white font-bold"
          style={{ fontSize: "16px" }}
        >
          {score.toFixed(0)}
        </text>
        <text
          x="40"
          y="50"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-gray-400"
          style={{ fontSize: "7px" }}
        >
          % coverage
        </text>
      </svg>
      <span className="text-xs" style={{ color }}>
        {scoreLabel(score)}
      </span>
    </div>
  );
}

/* ---------- Main Panel ---------- */

interface Props {
  projectId: string;
  visible: boolean;
}

export function CoveragePanel({ projectId, visible }: Props) {
  const [data, setData] = useState<CoverageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCoverage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getCoverage(projectId);
      setData(result);
    } catch (err: any) {
      setError(err.message || "Coverage analysis failed");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    if (visible && !data && !loading) {
      fetchCoverage();
    }
  }, [visible, data, loading, fetchCoverage]);

  if (!visible) return null;

  return (
    <div className="bg-gray-900/80 backdrop-blur border border-white/10 rounded-xl p-4 space-y-4 w-full max-w-sm">
      <h3 className="text-sm font-semibold text-gray-200 tracking-wide uppercase">
        Coverage Analysis
      </h3>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full" />
          <span className="ml-3 text-sm text-gray-400">Analyzing coverage...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-sm text-red-400 bg-red-900/20 rounded-lg p-3">
          {error}
          <button
            onClick={fetchCoverage}
            className="ml-2 underline text-red-300 hover:text-red-200"
          >
            Retry
          </button>
        </div>
      )}

      {/* Results */}
      {data && !loading && (
        <>
          {/* Score badge */}
          <ScoreBadge score={data.overall_score} />

          {/* Radar chart */}
          <RadarChart scores={data.direction_scores} />

          {/* Gap recommendations */}
          {data.gaps.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide">
                Gap Recommendations
              </h4>
              <ul className="space-y-1.5">
                {data.gaps.map((gap) => (
                  <li
                    key={gap.direction}
                    className="flex items-start gap-2 text-xs text-gray-300 bg-white/5 rounded-lg px-3 py-2"
                  >
                    <span
                      className="mt-0.5 w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: scoreColor(gap.score) }}
                    />
                    <span>{gap.recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {data.gaps.length === 0 && (
            <p className="text-xs text-green-400 text-center">
              No significant gaps detected -- great coverage!
            </p>
          )}
        </>
      )}
    </div>
  );
}
