"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { SplatViewer } from "@/components/SplatViewer";
import type { ProjectDetail } from "@/lib/types";

export default function ViewPage() {
  const params = useParams();
  const id = params.id as string;
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [plyUrl, setPlyUrl] = useState<string | null>(null);

  useEffect(() => {
    api.getProject(id).then((p) => {
      setProject(p);
      if (p.has_output) {
        setPlyUrl(`/api/projects/${id}/output/ply/point_cloud.ply`);
      }
    });
  }, [id]);

  if (!project) {
    return (
      <div className="text-gray-400 text-center py-12">Loading...</div>
    );
  }

  if (!project.has_output) {
    return (
      <div className="text-center py-20">
        <p className="text-gray-400 text-lg">
          No splat output available yet
        </p>
        <p className="text-gray-500 text-sm mt-2">
          Complete the training step first
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium">3D Gaussian Splat Viewer</h2>

      <div className="h-[600px]">
        {plyUrl && <SplatViewer plyUrl={plyUrl} />}
      </div>

      <div className="text-sm text-gray-500 text-center">
        Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan
      </div>
    </div>
  );
}
