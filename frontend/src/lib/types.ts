export type PipelineStep =
  | "created"
  | "extracting_frames"
  | "frames_ready"
  | "running_sfm"
  | "sfm_ready"
  | "training"
  | "training_complete"
  | "failed";

export interface ProjectSummary {
  id: string;
  name: string;
  step: PipelineStep;
  created_at: string;
  error?: string | null;
  thumbnail?: string | null;
}

export interface VideoInfo {
  index: number;
  filename: string;
  video_type: string;
}

export interface ProjectDetail extends ProjectSummary {
  video_filename?: string | null;
  video_type?: "standard" | "equirectangular";
  frame_count: number;
  sfm_points: number;
  training_iterations: number;
  has_output: boolean;
  temporal_mode?: "static" | "4d";
  videos?: VideoInfo[];
  video_count?: number;
}

export interface TrainSettings {
  max_steps?: number;
  sh_degree?: number;
  enable_depth?: boolean;
  depth_weight?: number;
  temporal_mode?: "static" | "4d";
  temporal_smoothness?: number;
  resume?: boolean;
}

export interface TemporalInfo {
  available: boolean;
  frame_count: number;
  timestamps?: number[];
}

export interface SampleVideo {
  id: string;
  title: string;
  url: string;
  thumbnail: string;
  duration: string;
}

export interface SystemDepStatus {
  name: string;
  installed: boolean;
  version?: string | null;
  path?: string | null;
  error?: string | null;
}

export interface SystemStatus {
  cuda_available: boolean;
  cuda_version?: string | null;
  gpu_name?: string | null;
  gpu_vram_mb?: number | null;
  ffmpeg: SystemDepStatus;
  colmap: SystemDepStatus;
  python_deps: SystemDepStatus;
}

export interface WsMessage {
  type: "log" | "progress" | "status" | "metric";
  line?: string;
  step?: string;
  substep?: string;
  percent?: number;
  state?: string;
  error?: string | null;
  [key: string]: unknown;
}

export interface FrameInfo {
  name: string;
  url: string;
}

export const STEP_ORDER: PipelineStep[] = [
  "created",
  "extracting_frames",
  "frames_ready",
  "running_sfm",
  "sfm_ready",
  "training",
  "training_complete",
];

export const STEP_LABELS: Record<string, string> = {
  created: "Upload",
  extracting_frames: "Extracting Frames...",
  frames_ready: "Frames Ready",
  running_sfm: "Running SfM...",
  sfm_ready: "SfM Ready",
  training: "Training...",
  training_complete: "Complete",
  failed: "Failed",
};
