const BASE = "";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { "Content-Type": "application/json", ...opts?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  // Projects
  listProjects: () => request<any[]>("/api/projects"),
  createProject: (name: string) =>
    request<any>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  getProject: (id: string) => request<any>(`/api/projects/${id}`),
  deleteProject: (id: string) =>
    request<any>(`/api/projects/${id}`, { method: "DELETE" }),

  // Upload
  uploadVideo: async (
    id: string,
    file: File,
    onProgress?: (percent: number) => void
  ) => {
    return new Promise<any>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      // Route through Next.js proxy (middlewareClientMaxBodySize handles large files)
      xhr.open("POST", `/api/projects/${id}/upload`);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            resolve({});
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
        }
      };

      xhr.onerror = () => reject(new Error("Upload failed: network error"));
      xhr.ontimeout = () => reject(new Error("Upload failed: timeout"));
      xhr.timeout = 600000; // 10 min timeout

      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    });
  },

  downloadSample: (id: string, sampleId: string) => {
    const form = new URLSearchParams();
    form.set("sample_id", sampleId);
    return fetch(`/api/projects/${id}/sample`, {
      method: "POST",
      body: form,
    }).then((r) => {
      if (!r.ok) throw new Error(`Download failed: ${r.status}`);
      return r.json();
    });
  },

  // Pipeline
  extractFrames: (id: string, opts?: {
    fps?: number;
    sharp_frame_selection?: boolean;
    sharp_window?: number;
  }) =>
    request<any>(`/api/projects/${id}/pipeline/extract-frames`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  runSfm: (id: string, opts?: { matcher_type?: string; enable_dense?: boolean }) =>
    request<any>(`/api/projects/${id}/pipeline/sfm`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  analyzeScene: (id: string) =>
    request<import("@/lib/types").SceneConfig>(
      `/api/projects/${id}/pipeline/scene-analysis`
    ),
  train: (id: string, opts?: Partial<import("@/lib/types").TrainSettings>) =>
    request<any>(`/api/projects/${id}/pipeline/train`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  getTemporalInfo: (id: string) =>
    request<import("@/lib/types").TemporalInfo>(`/api/projects/${id}/temporal-info`),
  refineSplat: (id: string, opts?: { diffusion_inpaint?: boolean; refine_steps?: number }) =>
    request<any>(`/api/projects/${id}/pipeline/refine`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  cancelPipeline: (id: string) =>
    request<any>(`/api/projects/${id}/pipeline/cancel`, { method: "POST" }),
  pipelineHealth: (id: string) =>
    request<{ running: boolean; step: string; stale: boolean; error: string | null }>(
      `/api/projects/${id}/pipeline/health`
    ),

  // Videos
  listVideos: (id: string) =>
    request<import("@/lib/types").VideoInfo[]>(`/api/projects/${id}/videos`),

  // Novel View Generation
  generateNovelViews: (id: string, opts?: {
    model?: string; num_refs?: number; output_size?: number;
  }) =>
    request<{ status: string; model: string }>(`/api/projects/${id}/pipeline/generate-novel-views`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),

  // Masking
  runMasking: (id: string, opts?: {
    keywords?: string; mode?: string; invert?: boolean;
    precision?: number; expand?: number; feather?: number;
    use_external?: boolean;
  }) =>
    request<any>(`/api/projects/${id}/pipeline/mask`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  listMasks: (id: string) =>
    request<{ name: string; url: string }[]>(`/api/projects/${id}/masks`),

  // Post-processing (pruning)
  prunePreview: (id: string, opts?: {
    min_opacity?: number; max_scale_mult?: number;
    position_percentile?: number; bbox?: string;
  }) =>
    request<{
      total: number; kept: number; pruned: number; pruned_pct: number;
      by_opacity: number; by_scale: number; by_position: number;
      median_scale: number; file_size_mb: number; estimated_output_mb: number;
    }>(`/api/projects/${id}/prune-preview`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  pruneSplat: (id: string, opts?: {
    min_opacity?: number; max_scale_mult?: number;
    position_percentile?: number; bbox?: string;
  }) =>
    request<{ status: string; output: string }>(`/api/projects/${id}/prune`, {
      method: "POST",
      body: JSON.stringify(opts || {}),
    }),
  pruneReset: (id: string) =>
    request<{ status: string }>(`/api/projects/${id}/prune-reset`, { method: "POST" }),

  // Coverage
  getCoverage: (id: string) =>
    request<any>(`/api/projects/${id}/pipeline/coverage`),

  // Cleanup
  runCleanup: (id: string) =>
    request<any>(`/api/projects/${id}/pipeline/cleanup`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  getCleanupStats: (id: string) =>
    request<any>(`/api/projects/${id}/pipeline/cleanup/stats`),
  undoCleanup: (id: string) =>
    request<any>(`/api/projects/${id}/pipeline/cleanup/undo`, {
      method: "POST",
    }),

  // LOD
  getLodInfo: (id: string) =>
    request<LodInfo>(`/api/projects/${id}/output/lod/info`),
  getLodUrl: (id: string, level: number) =>
    `/api/projects/${id}/output/lod/${level}`,

  // Frames
  listFrames: (id: string) => request<any[]>(`/api/projects/${id}/frames`),

  // Checkpoints (for comparison view)
  listCheckpoints: (id: string) => request<any[]>(`/api/projects/${id}/checkpoints`),

  // System
  systemStatus: () => request<any>("/api/system/status"),
  getSystemLogs: (lines: number = 100) =>
    request<{ lines: string[]; total_lines: number; file: string }>(
      `/api/system/logs?lines=${lines}`
    ),
  downloadLogsUrl: () => `/api/system/logs/download`,
  installDep: (dep: string) =>
    request<any>(`/api/system/install/${dep}`, { method: "POST" }),

  /**
   * Install a dependency with SSE progress streaming.
   * Calls onProgress for each event, resolves when complete.
   */
  installDepStream: async (
    dep: string,
    onProgress: (event: InstallProgress) => void
  ): Promise<void> => {
    const res = await fetch(`/api/system/install/${dep}`, {
      method: "POST",
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "Unknown error");
      throw new Error(`Install failed: ${text}`);
    }
    if (!res.body) throw new Error("No response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const event = JSON.parse(line.slice(6));
            onProgress(event);
            if (event.phase === "error") {
              throw new Error(event.message);
            }
          } catch (e: any) {
            if (e.message && !e.message.includes("JSON")) throw e;
          }
        }
      }
    }
  },
};

export interface LodLevelInfo {
  level: number;
  name: string;
  filename: string;
  size_bytes: number;
  available: boolean;
}

export interface LodInfo {
  levels: LodLevelInfo[];
  has_lod: boolean;
}

export interface InstallProgress {
  phase: "downloading" | "extracting" | "complete" | "error";
  label?: string;
  percent?: number;
  downloaded_mb?: number;
  total_mb?: number;
  installed?: boolean;
  version?: string;
  message?: string;
  error?: string;
}
