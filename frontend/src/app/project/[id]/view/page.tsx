"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { SplatViewer } from "@/components/SplatViewer";
import { SplatViewerFPS } from "@/components/SplatViewerFPS";
import { QRCodeSVG } from "qrcode.react";
import type { ProjectDetail } from "@/lib/types";

type ViewMode = "orbit" | "fps";

export default function ViewPage() {
  const params = useParams();
  const id     = params.id as string;

  const [project, setProject]     = useState<ProjectDetail | null>(null);
  const [plyUrl, setPlyUrl]       = useState<string | null>(null);
  const [ksplatUrl, setKsplatUrl] = useState<string | null>(null);
  const [viewMode, setViewMode]   = useState<ViewMode>("fps");
  const [localUrl, setLocalUrl]   = useState<string>("");
  const [showQuest, setShowQuest] = useState(false);

  useEffect(() => {
    api.getProject(id).then((p) => {
      setProject(p);
      if (p.has_output) {
        setPlyUrl(`/api/projects/${id}/output/point_cloud.ply`);
        // Check if ksplat is available (falls back gracefully)
        setKsplatUrl(`/api/projects/${id}/splat`);
      }
    });

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

  const questViewUrl = localUrl ? `${localUrl}/project/${id}/view?vr=1` : "";

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">3D Gaussian Splat Viewer</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode(viewMode === "fps" ? "orbit" : "fps")}
            className="px-3 py-1 text-sm rounded border border-gray-600 hover:bg-gray-700"
          >
            {viewMode === "fps" ? "Switch to Orbit" : "Switch to Free Cam"}
          </button>
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
              <li>Open Quest Browser → scan QR code</li>
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
      <div className={viewMode === "fps" ? "h-screen" : "h-[600px]"}>
        {viewMode === "fps" ? (
          plyUrl && (
            <SplatViewerFPS
              plyUrl={plyUrl}
              ksplatUrl={ksplatUrl ?? undefined}
            />
          )
        ) : (
          plyUrl && <SplatViewer plyUrl={plyUrl} />
        )}
      </div>

      {viewMode === "orbit" && (
        <div className="text-sm text-gray-500 text-center">
          Drag to rotate · Scroll to zoom · Right-drag to pan
        </div>
      )}
    </div>
  );
}
