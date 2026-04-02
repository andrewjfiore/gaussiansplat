# Dear Claude — Handoff Notes (What’s Still Missing)

Hi Claude 👋

This repo has partial improvements for pipeline robustness and frame quality, but it is **not done** relative to the requested “adopt AutoMasker-like features” direction.

## What was already added

- WebSocket cleanup improvements (backend + frontend reconnect handling).
- Mesh duplicate-run guard (`/extract-mesh` now rejects concurrent mesh jobs).
- Optional sharp-frame extraction path:
  - API fields: `sharp_frame_selection`, `sharp_window`
  - Dense extract then keep sharpest-per-bucket via Laplacian score.

## What is still missing (high priority)

1. **Native masking stage in pipeline (major gap)**
   - No first-class `mask` step exists in pipeline state machine.
   - No API endpoint like `/pipeline/mask` with settings (keywords/mode/invert/feather/expand).
   - No integration between extracted frames and masks for downstream SfM/train stages.
   - No persistent DB fields for mask status/artifacts.

2. **AutoMasker-style model integration**
   - No SAM2/GroundingDINO orchestration in backend services.
   - No model discovery/validation checks for local model folder.
   - No fallback/error UX when models are missing.

3. **CLI bridge support**
   - If external AutoMasker executable is expected, there is no wrapper service:
     - no command builder,
     - no stdout parser,
     - no progress routing to WebSocket logs.

4. **Frontend controls for masking**
   - No UI for:
     - keyword prompt entry,
     - output mode selection (mask/transparent/combined),
     - invert options,
     - precision, grow/expand, feather settings.
   - No per-project masking status/preview in pipeline pages.

5. **Sharp-frame extraction UX + guardrails**
   - Backend supports it, but frontend does not expose controls yet.
   - Missing parameter validation:
     - `sharp_window` upper/lower bound clamp,
     - safety cap for `sample_fps` to avoid huge extraction load.
   - Missing tests for:
     - multi-video prefix renumbering correctness,
     - manifest continuity after sharp selection,
     - behavior when bucket size is larger than frame count.

6. **State-machine clarity**
   - Pipeline enum has no dedicated step values for masking/refinement sub-states.
   - Mesh still runs as sidecar key (`project_id + "_mesh"`), not unified project step.
   - Health endpoint stale-detection currently tracks main active steps only.

## Suggested implementation order

1. Add `MASKING` / `MASKS_READY` pipeline steps and DB fields.
2. Add backend mask router endpoint + service abstraction (`internal model` or `external exe` adapter).
3. Add frontend masking controls + status/preview.
4. Add end-to-end tests for extract→mask→sfm.
5. Add guardrails for sharp mode + tests.

## “Definition of done” for this handoff

- A user can run: upload → extract (optional sharp) → mask (keyword-based) → sfm → train
  without manual file juggling.
- Mask artifacts are versioned per project and visible in UI.
- Pipeline status/health accurately reports active/failed masking and sidecar tasks.
- Tests cover happy path + missing model/tool failures.

Thanks — this should give you a clean starting point to finish the missing product pieces quickly.
