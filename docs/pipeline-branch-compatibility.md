# Rendering Pipeline Branch Compatibility

This document summarizes whether branching decisions in the pipeline can be combined safely based on current backend logic.

## ASCII branch map

```text
                                +--------------------+
                                | project created    |
                                +---------+----------+
                                          |
                                          v
                           +--------------+--------------+
                           | extract-frames               |
                           | requires: video uploaded     |
                           +--------------+--------------+
                                          |
                         +----------------+-------------------------------+
                         |                                                |
                         v                                                v
              sharp_frame_selection=true                       sharp_frame_selection=false
         (dense sample + sharpest-per-bucket)                    (standard sampling)
                         |                                                |
                         +----------------+-------------------------------+
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 v
             filter_blur = true                  filter_blur = false
             (quality filtering)                 (skip filter)
                         \                                 /
                          +--------------+---------------+
                                         |
                                         v
                                  frames_ready
                                         |
                                         v
                               +---------+----------+
                               | sfm                |
                               +---------+----------+
                                         |
                  +----------------------+----------------------+
                  |                                             |
                  v                                             v
      video_count > 1                                 video_count <= 1
      force single_camera=false                       keep requested settings
      sequential->exhaustive matcher                  matcher unchanged
                  \                                             /
                   +----------------------+--------------------+
                                          |
                               SfM quality gate passed?
                                 /                    \
                               yes                    no
                                |                     |
                                v                     v
                            sfm_ready               failed
                                |
                                v
                      +---------+----------+
                      | train              |
                      +---------+----------+
                                |
       +------------------------+------------------------+
       |                                                 |
       v                                                 v
 enable_depth=true                               enable_depth=false
 run depth estimation if needed                  skip depth branch
 (failure is warning, continues)                 directly train
       \                                                 /
        +--------------------------+--------------------+
                                   |
                                   v
                          training_complete
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
        extract-mesh                               refine
    (requires training_complete)            (requires training_complete)
              |                                         |
              v                           +-------------+-------------+
                                       stage 1 + stage 2 (required)
                                                     |
                                    diffusion_inpaint enabled?
                                          /                 \
                                        yes                 no
                                         |                   |
                                         v                   v
                            stage 3 inpaint -> (if success) stage 4
                                         \                   /
                                          +--------+--------+
                                                   |
                                                   v
                                          training_complete
```

## Compatibility notes

- **Compatible combinations**
  - `extract-frames` with/without sharp-frame selection.
  - `extract-frames` with/without blur filtering.
  - `sfm` single-video and multi-video branches (multi-video auto-tunes matcher/camera mode).
  - `train` with or without depth (depth failure degrades to warning and continues training).
  - `refine` with or without diffusion inpaint.

- **Intended exclusivity**
  - Only one project task can run at a time via `task_runner.is_running(project_id)` gate in precondition checks.
  - Mesh extraction now has an explicit duplicate-run guard (`project_id + "_mesh"`) to prevent repeated mesh jobs running concurrently.

- **Behavioral caveat (important)**
  - `extract-mesh` uses a separate runner key (`project_id + "_mesh"`), so it is logically outside the main step-state pipeline. This is okay for post-training jobs, but clients should treat mesh extraction as sidecar work, not a step transition.
