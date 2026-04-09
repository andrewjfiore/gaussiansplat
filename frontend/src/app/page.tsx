"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";
import { STEP_LABELS } from "@/lib/types";
import { Plus, Trash2, Image } from "lucide-react";
import { ConfirmDialog } from "@/components/ConfirmDialog";

export default function DashboardPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<ProjectSummary | null>(null);

  useEffect(() => {
    api.listProjects().then(setProjects).finally(() => setLoading(false));
  }, []);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await api.deleteProject(deleteTarget.id);
    setProjects((prev) => prev.filter((p) => p.id !== deleteTarget.id));
    setDeleteTarget(null);
  };

  const stepLink = (p: ProjectSummary) => {
    if (p.step === "portrait_processing") return `/project/${p.id}/portrait`;
    if (p.step === "training_complete") return `/project/${p.id}/view`;
    if (p.step === "sfm_ready" || p.step === "training") return `/project/${p.id}/train`;
    if (p.step === "frames_ready" || p.step === "running_sfm") return `/project/${p.id}/sfm`;
    return `/project/${p.id}/frames`;
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Projects</h1>
        <Link
          href="/new"
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition"
        >
          <Plus className="w-4 h-4" /> New Project
        </Link>
      </div>

      {loading ? (
        <div className="text-gray-400 text-center py-12">Loading...</div>
      ) : projects.length === 0 ? (
        <div className="text-center py-20">
          <Image className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h2 className="text-xl text-gray-400 mb-2">No projects yet</h2>
          <p className="text-gray-500 mb-6">
            Create your first Gaussian Splat from a video
          </p>
          <Link
            href="/new"
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition"
          >
            Get Started
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map((p) => (
            <div
              key={p.id}
              className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 hover:border-gray-600 transition"
            >
              <Link href={stepLink(p)}>
                <div className="aspect-video bg-gray-900 flex items-center justify-center">
                  {p.thumbnail ? (
                    <img
                      src={p.thumbnail}
                      alt={p.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <Image className="w-12 h-12 text-gray-700" />
                  )}
                </div>
              </Link>
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium truncate">{p.name}</h3>
                  <button
                    onClick={() => setDeleteTarget(p)}
                    className="text-gray-500 hover:text-red-400 transition"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <div className="mt-1 text-sm text-gray-400">
                  {STEP_LABELS[p.step] || p.step}
                  {p.error && (
                    <span className="text-red-400 ml-2">({p.error})</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Project"
        message={`Delete "${deleteTarget?.name}" and all its files? This cannot be undone.`}
        confirmLabel="Delete"
        confirmVariant="danger"
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
