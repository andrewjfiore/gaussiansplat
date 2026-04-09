from __future__ import annotations
from enum import Enum
from pydantic import BaseModel
from typing import Optional


class PipelineStep(str, Enum):
    CREATED = "created"
    EXTRACTING_FRAMES = "extracting_frames"
    FRAMES_READY = "frames_ready"
    MASKING = "masking"
    MASKS_READY = "masks_ready"
    RUNNING_SFM = "running_sfm"
    SFM_READY = "sfm_ready"
    TRAINING = "training"
    TRAINING_COMPLETE = "training_complete"
    CLEANING = "cleaning"
    PORTRAIT = "portrait_processing"
    FAILED = "failed"


class ProjectCreate(BaseModel):
    name: str


class ProjectSummary(BaseModel):
    id: str
    name: str
    step: PipelineStep
    created_at: str
    error: Optional[str] = None
    thumbnail: Optional[str] = None


class VideoInfo(BaseModel):
    index: int
    filename: str
    video_type: str = "standard"


class ProjectDetail(ProjectSummary):
    video_filename: Optional[str] = None
    video_type: str = "standard"  # "standard" | "equirectangular"
    frame_count: int = 0
    sfm_points: int = 0
    training_iterations: int = 0
    has_output: bool = False
    temporal_mode: str = "static"  # "static" | "4d"
    videos: list[VideoInfo] = []
    video_count: int = 1
    mask_keywords: Optional[str] = None
    mask_count: int = 0


class SampleVideo(BaseModel):
    id: str
    title: str
    url: str
    thumbnail: str
    duration: str


class ExtractSettings(BaseModel):
    fps: float = 2.0
    start_time: Optional[float] = None  # seconds offset into video
    filter_blur: bool = True
    min_blur_score: float = 50.0
    # Sharp-frame mode: extract denser candidates then keep sharpest frame per bucket.
    # window=5 means 11 candidates per target frame (~+-5 neighboring samples).
    sharp_frame_selection: bool = False
    sharp_window: int = 5


class MaskSettings(BaseModel):
    keywords: str = "person"  # dot-separated keywords, e.g. "person.camera.tripod"
    mode: str = "mask"  # "mask" | "transparent" | "combined"
    invert: bool = False  # invert the mask (keep the masked region instead)
    precision: float = 0.3  # detection confidence threshold (0-1, lower = more detections)
    expand: int = 0  # expand mask by N pixels
    feather: int = 0  # feather/blur mask edges by N pixels
    use_external: bool = False  # use external AutoMasker exe instead of built-in


class PruneSettings(BaseModel):
    min_opacity: float = 0.1      # remove Gaussians with sigmoid(opacity) < this
    max_scale_mult: float = 8.0   # remove Gaussians with scale > N * median
    position_percentile: float = 99.0  # keep only within this distance percentile
    bbox: Optional[str] = None    # crop bounding box: "xmin,ymin,zmin,xmax,ymax,zmax"


class SfmSettings(BaseModel):
    matcher_type: str = "sequential_matcher"
    single_camera: bool = True
    quality: str = "high"
    enable_dense: bool = False


class TrainSettings(BaseModel):
    max_steps: int = 7000
    # Main branch fields
    use_scaffold: bool = True
    voxel_size: float = 0.001
    denoise_strength: str = "off"  # off | light | medium | aggressive
    sh_degree: int = 0  # 0-3, higher = view-dependent color
    enable_depth: bool = False
    depth_weight: float = 0.1
    temporal_mode: str = "static"  # "static" | "4d"
    temporal_smoothness: float = 0.01
    resume: bool = False
    # Our branch fields
    strategy: str = "default"
    two_phase: bool = True
    phase1_steps: Optional[int] = None  # auto-derived if None
    phase2_steps: Optional[int] = None
    densify_grad_thresh: Optional[float] = None


class NovelViewSettings(BaseModel):
    model: str = "zero123pp"  # zero123pp | wonder3d | era3d | sd_inpaint
    num_refs: int = 4         # number of reference frames to generate from
    output_size: int = 800    # output image size (square)


class RefineSettings(BaseModel):
    refine_steps: int = 3000  # additional training iterations for refinement
    alpha_low: float = 0.5   # visibility transfer: low-confidence threshold
    alpha_high: float = 0.8  # visibility transfer: high-confidence threshold
    diffusion_inpaint: bool = False  # run SD inpainting for truly unseen regions
    novel_view_model: str = "zero123pp"  # zero123pp | wonder3d | era3d | sd_inpaint
    num_novel_views: int = 8
    novel_view_weight: float = 0.3
    diffusion_steps: int = 20
    diffusion_guidance: float = 3.0


class SceneStatsResponse(BaseModel):
    num_points: int = 0
    num_cameras: int = 0
    num_images: int = 0
    bbox_min: list[float] = [0.0, 0.0, 0.0]
    bbox_max: list[float] = [0.0, 0.0, 0.0]
    centroid: list[float] = [0.0, 0.0, 0.0]
    scene_radius: float = 0.0
    mean_point_density: float = 0.0
    camera_baseline: float = 0.0


class SceneConfigResponse(BaseModel):
    max_steps: int = 7000
    phase1_steps: int = 5600
    phase2_steps: int = 1400
    densify_grad_thresh: float = 0.0002
    sh_degree: int = 3
    scene_complexity: str = "medium"
    reasoning: list[str] = []
    stats: Optional[SceneStatsResponse] = None


class CleanupSettings(BaseModel):
    sor_k: int = 50
    sor_std: float = 2.0
    sparse_min_neighbors: int = 3
    large_splat_percentile: float = 99.0
    opacity_threshold: float = 0.05
    bg_std_multiplier: float = 3.0


class HolefillSettings(BaseModel):
    grid_resolution: int = 64            # occupancy grid resolution (32/64/128)
    min_hole_size: int = 2               # min voxels for a hole cluster
    max_hole_size: int = 500             # max voxels for a hole cluster
    fill_density: float = 1.0            # fill density multiplier (0.5-2.0)


class AugmentSettings(BaseModel):
    num_views_per_frame: int = 2
    angle_range: float = 15.0
    max_source_frames: int = 10


class PortraitSettings(BaseModel):
    stride: int = 2                      # pixel stride for downsampling Gaussians
    focal_multiplier: float = 0.8        # focal_length = image_width * this
    num_novel_views: int = 6             # synthetic novel views to generate
    include_background: bool = False     # include background at lower opacity
    depth_model: str = "small"           # "small" (~100MB) or "base" (~400MB)


class SystemDepStatus(BaseModel):
    name: str
    installed: bool
    version: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None


class SystemStatus(BaseModel):
    # nvidia-smi driver detection
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_vram_mb: Optional[int] = None
    # PyTorch CUDA -- this is what actually matters for training
    torch_cuda_available: bool = False
    torch_cuda_version: Optional[str] = None
    ffmpeg: SystemDepStatus
    colmap: SystemDepStatus
    python_deps: SystemDepStatus


SAMPLE_VIDEOS = [
    SampleVideo(
        id="hong-kong-night",
        title="Hong Kong at Night (Aerial)",
        url="https://www.pexels.com/download/video/3129671/",
        thumbnail="https://images.pexels.com/videos/3129671/free-video-3129671.jpg?auto=compress&cs=tinysrgb&w=400",
        duration="0:30",
    ),
    SampleVideo(
        id="mountain-building",
        title="Building on Mountain",
        url="https://www.pexels.com/download/video/4571563/",
        thumbnail="https://images.pexels.com/videos/4571563/pexels-photo-4571563.jpeg?auto=compress&cs=tinysrgb&w=400",
        duration="0:15",
    ),
    SampleVideo(
        id="city-panoramic",
        title="Panoramic View of a City",
        url="https://www.pexels.com/download/video/3573921/",
        thumbnail="https://images.pexels.com/videos/3573921/free-video-3573921.jpg?auto=compress&cs=tinysrgb&w=400",
        duration="0:20",
    ),
]
