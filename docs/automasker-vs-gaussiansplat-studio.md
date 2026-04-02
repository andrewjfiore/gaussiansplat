# AutoMasker + 360 Gaussian: Summary and Comparison to This Tool

## What the described program does

Based on your transcript, the external workflow consists of two tightly-related parts:

1. **AutoMasker**
   - Uses **SAM 2** + **GroundingDINO** style keyword-driven detection.
   - Creates masks from image folders using text keywords (multi-keyword prompt syntax).
   - Supports multiple export modes:
     - binary mask output,
     - transparent cutout,
     - combined overlay previews,
     - plus inverted variants.
   - Includes controls for precision/sensitivity, mask grow/expand, and feathering.
   - Ships as a standalone executable with local model files and can be used from CLI.

2. **360 Gaussian workflow (v1.3 + sphere-SfM updates)**
   - Accepts equirectangular video/images and batch media.
   - Adds sharp-frame extraction logic.
   - Adds sphere-SfM / cubemap-oriented alignment options.
   - Adds auto step scaling for training based on image count.
   - Can chain extraction → masking → alignment → training in one run.
   - Supports different external trainers/alignment paths (including open-source and paid options).

## How that compares to this repository (GaussianSplat Studio)

### Similarities

- **End-to-end pipeline orchestration** exists here too:
  - extract frames,
  - SfM alignment,
  - training,
  - optional refinement and mesh extraction.  
  (See pipeline router endpoints and background task orchestration.)
- **Multi-video awareness** is present (auto matcher/camera behavior for multi-video projects).
- **Frame quality filtering** is present (blur-based filtering after extraction).
- **Resumable / async operations** with WebSocket logs and status updates are built in.

### Key differences

- **Masking integration**
  - Your described AutoMasker pipeline centers masking as a first-class configurable feature.
  - This repo currently does **not** expose a first-class SAM2/GroundingDINO masking stage in the main backend pipeline endpoints.
- **Sphere/cubemap-specific UX**
  - Your transcript emphasizes sphere-SfM/cubemap conversion and 360-specific mask transforms.
  - This repo currently focuses on generic frame extraction + COLMAP + Gaussian training flow; 360-specialized sphere/cubemap transforms are not explicit as a dedicated stage in the pipeline API.
- **Tool packaging**
  - Your transcript describes a desktop `.exe` workflow with bundled models and GUI controls.
  - This repo is a **web app + backend service** architecture (FastAPI + Next.js), not a standalone desktop executable.

## Practical mapping between the two

- If you use AutoMasker externally, the natural handoff into this tool is:
  1. run extraction/masking externally,
  2. feed resulting frames (or preprocessed assets) into project workflow,
  3. continue with SfM/train/refine stages here.
- The closest in-repo equivalents today are:
  - **sharp-frame selection** (new): optional denser sampling + sharpest-per-window down-selection during extraction,
  - **blur filtering** (quality prefilter),
  - **SfM quality gates** (registered-image and reprojection checks),
  - **refine pipeline** (visibility transfer + optional inpainting pass),
  not direct keyword mask generation.

## Bottom line

Your described stack is stronger on **automated semantic masking for 360 capture cleanup** and desktop convenience; this repository is stronger as an **open, server-style orchestration backend** for frame→SfM→training/refinement pipelines, with robust async execution and status plumbing.
