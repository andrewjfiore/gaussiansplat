"use client";
import { useEffect, useState } from "react";
import { useParams, usePathname } from "next/navigation";
import { api } from "@/lib/api";
import { StepIndicator } from "@/components/StepIndicator";
import type { ProjectDetail } from "@/lib/types";

export default function ProjectLayout({ children }: { children: React.ReactNode }) {
  const params = useParams();
  const pathname = usePathname();
  const id = params.id as string;
  const [project, setProject] = useState<ProjectDetail | null>(null);

  const isPortrait =
    pathname?.includes("/portrait") ||
    project?.step === "portrait_processing";

  useEffect(() => {
    api.getProject(id).then(setProject).catch(() => {});
    const interval = setInterval(() => {
      api.getProject(id).then(setProject).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, [id]);

  return (
    <div>
      <div className="mb-4">
        <h1 className="text-2xl font-bold">{project?.name || "Loading..."}</h1>
        {project && <StepIndicator currentStep={project.step} isPortrait={isPortrait} />}
        {project?.error && (
          <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-lg p-3 mt-2">
            {project.error}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}
