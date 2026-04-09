"use client";
import type { PipelineStep } from "@/lib/types";
import { Check, Loader2 } from "lucide-react";

const VIDEO_STEPS = [
  { key: "frames", label: "Frames", doneAt: "frames_ready" },
  { key: "sfm", label: "SfM", doneAt: "sfm_ready" },
  { key: "train", label: "Train", doneAt: "training_complete" },
  { key: "view", label: "View", doneAt: "training_complete" },
] as const;

const PORTRAIT_STEPS = [
  { key: "upload", label: "Upload" },
  { key: "process", label: "Process" },
  { key: "view", label: "View" },
] as const;

const STEP_PROGRESS: Record<string, number> = {
  created: 0,
  extracting_frames: 0,
  frames_ready: 1,
  running_sfm: 1,
  sfm_ready: 2,
  training: 2,
  training_complete: 3,
  failed: -1,
};

const PORTRAIT_STEP_PROGRESS: Record<string, number> = {
  created: 0,
  portrait_processing: 1,
  training_complete: 2,
  failed: -1,
};

export function StepIndicator({ currentStep, isPortrait = false }: { currentStep: PipelineStep; isPortrait?: boolean }) {
  if (isPortrait) {
    const progress = PORTRAIT_STEP_PROGRESS[currentStep] ?? 0;
    const isRunning = currentStep === "portrait_processing";

    return (
      <div className="flex items-center gap-2 py-4">
        {PORTRAIT_STEPS.map((s, i) => {
          const done = progress > i;
          const active = progress === i;
          const running = active && isRunning;

          return (
            <div key={s.key} className="flex items-center gap-2">
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium border-2 transition-colors ${
                  done
                    ? "bg-green-500 border-green-500 text-white"
                    : active
                    ? "border-violet-500 text-violet-500 bg-violet-50"
                    : "border-gray-300 text-gray-400"
                }`}
              >
                {done ? (
                  <Check className="w-4 h-4" />
                ) : running ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  i + 1
                )}
              </div>
              <span
                className={`text-sm ${
                  done ? "text-green-600" : active ? "text-violet-600 font-medium" : "text-gray-400"
                }`}
              >
                {s.label}
              </span>
              {i < PORTRAIT_STEPS.length - 1 && (
                <div className={`w-8 h-0.5 ${done ? "bg-green-500" : "bg-gray-200"}`} />
              )}
            </div>
          );
        })}
      </div>
    );
  }

  const progress = STEP_PROGRESS[currentStep] ?? 0;
  const isRunning = ["extracting_frames", "running_sfm", "training"].includes(currentStep);

  return (
    <div className="flex items-center gap-2 py-4">
      {VIDEO_STEPS.map((s, i) => {
        const done = progress > i;
        const active = progress === i;
        const running = active && isRunning;

        return (
          <div key={s.key} className="flex items-center gap-2">
            <div
              className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium border-2 transition-colors ${
                done
                  ? "bg-green-500 border-green-500 text-white"
                  : active
                  ? "border-blue-500 text-blue-500 bg-blue-50"
                  : "border-gray-300 text-gray-400"
              }`}
            >
              {done ? (
                <Check className="w-4 h-4" />
              ) : running ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                i + 1
              )}
            </div>
            <span
              className={`text-sm ${
                done ? "text-green-600" : active ? "text-blue-600 font-medium" : "text-gray-400"
              }`}
            >
              {s.label}
            </span>
            {i < VIDEO_STEPS.length - 1 && (
              <div className={`w-8 h-0.5 ${done ? "bg-green-500" : "bg-gray-200"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
