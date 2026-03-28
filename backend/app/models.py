from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class PipelineStep(str, Enum):
    CREATED = "created"
    EXTRACTING_FRAMES = "extracting_frames"
    FRAMES_READY = "frames_ready"
    RUNNING_SFM = "running_sfm"
    SFM_READY = "sfm_ready"
    TRAINING = "training"
    TRAINING_COMPLETE = "training_complete"
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


class ProjectDetail(ProjectSummary):
    video_filename: Optional[str] = None
    frame_count: int = 0
    sfm_points: int = 0
    training_iterations: int = 0
    has_output: bool = False


class SampleVideo(BaseModel):
    id: str
    title: str
    url: str
    thumbnail: str
    duration: str


class ExtractSettings(BaseModel):
    fps: float = 2.0


class SfmSettings(BaseModel):
    matcher_type: str = "sequential_matcher"
    single_camera: bool = True
    quality: str = "high"


class TrainSettings(BaseModel):
    max_steps: int = 7000
    strategy: str = "default"


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
    # PyTorch CUDA — this is what actually matters for training
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
