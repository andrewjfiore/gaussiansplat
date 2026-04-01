"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { SplatViewer } from "@/components/SplatViewer";
import { SplatViewerFPS } from "@/components/SplatViewerFPS";
import TemporalViewer from "@/components/TemporalViewer";
import { QRCodeSVG } from "qrcode.react";
import type { ProjectDetail, TemporalInfo } from "@/lib/types";

type ViewMode = "orbit" | "fps";

export default function ViewPage() {
  const params = useParams();
  const router = useRouter();
  const id     = params.id as string;

  const [project, setProject]       = useState<ProjectDetail | null>(null);
  const [plyUrl, setPlyUrl]         = useState<string | null>(null);
  const [ksplatUrl, setKsplatUrl]   = useState<string | null>(null);
  const [viewMode, setViewMode]     = useState<ViewMode>("fps");
  const [localUrl, setLocalUrl]     = useState<string>("");
  const [showQuest, setShowQuest]   = useState(false);
  const [temporalInfo, setTemporalInfo] = useState<TemporalInfo | null>(null);
  const [show4D, setShow4D]         = useState(false);

  useEffect(() => {
    api.getProject(id).then((p) => {
      setProject(p);
      if (p.has_output) {
        // For 4D models, use a baked mid-frame PLY instead of the canonical
        // (canonical Gaussians are optimized for deformation, not direct viewing)
        if (p.temporal_mode === "4d") {
          api.getTemporalInfo(id).then((info) => {
            setTemporalInfo(info);
            if (info.available && info.frame_count > 0) {
              setShow4D(true);
              const midFrame = Math.floor(info.frame_count / 2);
              const padded = String(midFrame).padStart(4, "0");
              setPlyUrl(`/api/projects/${id}/output/temporal_frames/frame_${padded}.ply`);
            } else {
              setPlyUrl(`/api/projects/${id}/output/point_cloud.ply`);
            }
          }).catch(() => {
            setPlyUrl(`/api/projects/${id}/output/point_cloud.ply`);
            setTemporalInfo({ available: false, frame_count: 0 });
          });
        } else {
          setPlyUrl(`/api/projects/${id}/output/point_cloud.ply`);
        }
        // Check if ksplat file exists on disk (compressed version)
        fetch(`/api/projects/${id}/output/point_cloud.ksplat`, { method: "HEAD" })
          .then((r) => {
            if (r.ok) setKsplatUrl(`/api/projects/${id}/output/point_cloud.ksplat`);
          })
          .catch(() => {});
      }
    });

    // Fetch temporal info for non-4D projects
    if (!temporalInfo) {
      api.getTemporalInfo(id)
        .then((info) => {
          setTemporalInfo(info);
          if (info.available) setShow4D(true);
        })
        .catch(() => setTemporalInfo({ available: false, frame_count: 0 }));
    }

    fetch("/api/system/local-url")
      .then((r) => r.json())
      .then((d) => setLocalUrl(d.url))
      .catch(() => {});
  }, [id]);

  if (!project) {
    return <div className="text-gray-400 text-center py-12">Loading...</div>;
  }

  if (!project.has_output) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-400 text-lg">No splat output available yet</p>
        <p className="text-gray-500 text-sm mt-2">Complete the training step first</p>
      </div>
    );
  }

  const is4D = temporalInfo?.available && temporalInfo.frame_count > 0;
  const questViewUrl = localUrl ? `${localUrl}/project/${id}/view?vr=1` : "";

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push(`/project/${id}/train`)}
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            Back to Training
          </button>
          <h2 className="text-lg font-medium">3D Gaussian Splat Viewer</h2>
          {project.temporal_mode === "4d" && (
            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              4D
            </span>
          )}
        </div>
        <div className="flex gap-2">
          {!show4D && (
            <button
              onClick={() => setViewMode(viewMode === "fps" ? "orbit" : "fps")}
              className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
            >
              {viewMode === "fps" ? "Switch to Orbit" : "Switch to Free Cam"}
            </button>
          )}
          {is4D && (
            <button
              onClick={() => setShow4D((v) => !v)}
              className={`px-3 py-1 text-sm rounded border transition ${
                show4D
                  ? "border-purple-500 bg-purple-500/20 text-purple-300"
                  : "border-gray-600 text-gray-300 hover:bg-gray-700"
              }`}
            >
              {show4D ? "Static View" : "4D Timeline"}
            </button>
          )}
          {plyUrl && (
            <button
              onClick={() => setShowQuest((v) => !v)}
              className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
            >
              View on Quest
            </button>
          )}
        </div>
      </div>

      {/* Quest panel */}
      {showQuest && questViewUrl && (
        <div className="border border-gray-700 rounded-lg p-4 bg-gray-900 flex gap-6 items-start">
          <div>
            <QRCodeSVG value={questViewUrl} size={128} bgColor="#111827" fgColor="#f9fafb" />
          </div>
          <div className="space-y-2">
            <p className="font-medium text-sm">View on Meta Quest (72fps native)</p>
            <p className="text-xs text-gray-400 font-mono break-all">{questViewUrl}</p>
            <ol className="text-xs text-gray-400 space-y-1 list-decimal list-inside">
              <li>Open Quest Browser &rarr; scan QR code</li>
              <li>Or open Niantic Scaniverse on Quest</li>
              <li>In browser: tap the <strong>Enter VR</strong> button</li>
            </ol>
          </div>
        </div>
      )}

      {/* Export buttons */}
      {plyUrl && (
        <div className="flex gap-2 flex-wrap">
          <a
            href={`/api/projects/${id}/ply`}
            download
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            Download .ply
          </a>
          <a
            href={`/api/projects/${id}/splat`}
            download
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            Download .ksplat
          </a>
          <a
            href={`/api/projects/${id}/spz`}
            download
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            Download .spz
          </a>
          <button
            onClick={async () => {
              await fetch(`/api/projects/${id}/pipeline/extract-mesh`, { method: "POST" });
              alert("Mesh extraction started — check back in a few minutes for the .glb");
            }}
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            Extract + Download .glb
          </button>
        </div>
      )}

      {/* Viewer */}
      <div className={show4D || viewMode === "fps" ? "h-screen" : "h-[600px]"}>
        {show4D && is4D ? (
          <TemporalViewer projectId={id} frameCount={temporalInfo.frame_count} />
        ) : viewMode === "fps" ? (
          plyUrl && (
            <SplatViewerFPS
              plyUrl={plyUrl}
              ksplatUrl={ksplatUrl ?? undefined}
            />
          )
        ) : (
          plyUrl && <SplatViewer plyUrl={plyUrl} ksplatUrl={ksplatUrl ?? undefined} />
        )}
      </div>

      {viewMode === "orbit" && !show4D && (
        <div className="text-sm text-gray-500 text-center">
          Drag to rotate · Scroll to zoom · Right-drag to pan
        </div>
      )}
    </div>
  );
}
