export type PipelineStep =
  | "created"
  | "extracting_frames"
  | "frames_ready"
  | "masking"
  | "masks_ready"
  | "running_sfm"
  | "sfm_ready"
  | "training"
  | "training_complete"
  | "cleaning"
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
  mask_keywords?: string | null;
  mask_count?: number;
}

// Merged TrainSettings: includes fields from both main and our branch
export interface TrainSettings {
  max_steps: number;
  // Main branch fields
  sh_degree?: number;
  enable_depth?: boolean;
  depth_weight?: number;
  temporal_mode?: "static" | "4d";
  temporal_smoothness?: number;
  resume?: boolean;
  // Our branch fields
  two_phase?: boolean;
  phase1_steps?: number;
  phase2_steps?: number;
  densify_grad_thresh?: number;
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
  "masking",
  "masks_ready",
  "running_sfm",
  "sfm_ready",
  "training",
  "training_complete",
];

export interface CoverageGap {
  direction: string;
  score: number;
  recommendation: string;
}

export interface CoverageGridEntry {
  azimuth: number;
  elevation: number;
  score: number;
  grid: number[][];
}

export interface CoverageResult {
  overall_score: number;
  direction_scores: Record<string, number>;
  gaps: CoverageGap[];
  grid_data: CoverageGridEntry[];
}

export const STEP_LABELS: Record<string, string> = {
  created: "Upload",
  extracting_frames: "Extracting Frames...",
  frames_ready: "Frames Ready",
  masking: "Masking...",
  masks_ready: "Masks Ready",
  running_sfm: "Running SfM...",
  sfm_ready: "SfM Ready",
  training: "Training...",
  training_complete: "Complete",
  cleaning: "Cleaning Up...",
  failed: "Failed",
};

export interface SceneStats {
  num_points: number;
  num_cameras: number;
  num_images: number;
  bbox_min: number[];
  bbox_max: number[];
  centroid: number[];
  scene_radius: number;
  mean_point_density: number;
  camera_baseline: number;
}

export interface SceneConfig {
  max_steps: number;
  phase1_steps: number;
  phase2_steps: number;
  densify_grad_thresh: number;
  sh_degree: number;
  scene_complexity: "low" | "medium" | "high";
  reasoning: string[];
  stats: SceneStats | null;
}

export interface CleanupFilterStats {
  name: string;
  removed: number;
}

export interface CleanupStats {
  has_stats: boolean;
  original_count?: number;
  final_count?: number;
  total_removed?: number;
  removal_pct?: number;
  filters?: CleanupFilterStats[];
}
