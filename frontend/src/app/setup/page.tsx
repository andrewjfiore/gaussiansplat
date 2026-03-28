"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { SystemStatus } from "@/lib/types";
import type { InstallProgress } from "@/lib/api";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Download,
  Loader2,
  Cpu,
  Film,
  Box,
  Sparkles,
  ArrowRight,
  PartyPopper,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Install state (same pattern as settings page)                     */
/* ------------------------------------------------------------------ */

interface InstallState {
  active: boolean;
  phase: string;
  label: string;
  percent: number;
  downloaded_mb: number;
  total_mb: number;
  error?: string;
}

const INITIAL_INSTALL: InstallState = {
  active: false,
  phase: "",
  label: "",
  percent: 0,
  downloaded_mb: 0,
  total_mb: 0,
};

/* ------------------------------------------------------------------ */
/*  Step indicator                                                    */
/* ------------------------------------------------------------------ */

const STEP_NAMES = ["Welcome", "System Check", "Ready"];

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="flex items-center justify-center gap-3 mb-8">
      {STEP_NAMES.map((name, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div key={name} className="flex items-center gap-3">
            {i > 0 && (
              <div
                className={`w-8 h-px ${
                  done ? "bg-blue-500" : "bg-gray-600"
                }`}
              />
            )}
            <div className="flex items-center gap-2">
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold border-2 transition-colors ${
                  done
                    ? "bg-blue-600 border-blue-600 text-white"
                    : active
                    ? "border-blue-500 text-blue-400"
                    : "border-gray-600 text-gray-500"
                }`}
              >
                {done ? <CheckCircle className="w-4 h-4" /> : i + 1}
              </div>
              <span
                className={`text-xs hidden sm:inline ${
                  active ? "text-white" : "text-gray-500"
                }`}
              >
                {name}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Dep card used in step 2                                           */
/* ------------------------------------------------------------------ */

interface DepCardProps {
  icon: React.ReactNode;
  title: string;
  status: "ok" | "warning" | "missing";
  detail: string;
  canInstall?: boolean;
  installing?: InstallState;
  onInstall?: () => void;
}

function DepCard({
  icon,
  title,
  status,
  detail,
  canInstall,
  installing,
  onInstall,
}: DepCardProps) {
  const isInstalling = installing?.active;

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-gray-400">{icon}</div>
          <div>
            <div className="text-sm font-medium text-white">{title}</div>
            <div
              className={`text-xs mt-0.5 ${
                status === "ok"
                  ? "text-gray-400"
                  : status === "warning"
                  ? "text-yellow-400"
                  : "text-red-400"
              }`}
            >
              {installing?.error || detail}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {status === "ok" && (
            <CheckCircle className="w-5 h-5 text-green-400" />
          )}
          {status === "warning" && !isInstalling && (
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
          )}
          {status === "missing" && !isInstalling && (
            <>
              <XCircle className="w-5 h-5 text-red-400" />
              {canInstall && onInstall && (
                <button
                  onClick={onInstall}
                  className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-3 py-1.5 rounded-lg flex items-center gap-1 transition"
                >
                  <Download className="w-3 h-3" /> Install
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {isInstalling && (
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span className="flex items-center gap-1.5">
              <Loader2 className="w-3 h-3 animate-spin" />
              {installing.phase === "downloading"
                ? `Downloading ${installing.label}...`
                : installing.phase === "extracting"
                ? `Extracting ${installing.label}...`
                : `Setting up ${installing.label}...`}
            </span>
            {installing.phase === "downloading" && (
              <span>
                {installing.downloaded_mb} / {installing.total_mb} MB (
                {installing.percent}%)
              </span>
            )}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                installing.phase === "extracting"
                  ? "bg-yellow-500 animate-pulse"
                  : "bg-blue-500"
              }`}
              style={{
                width:
                  installing.phase === "extracting"
                    ? "100%"
                    : `${installing.percent}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Just-completed message */}
      {installing &&
        !installing.active &&
        installing.phase === "complete" && (
          <div className="mt-2 text-xs text-green-400 flex items-center gap-1">
            <CheckCircle className="w-3 h-3" /> Installed successfully
          </div>
        )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main wizard                                                       */
/* ------------------------------------------------------------------ */

export default function SetupPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);

  // Step 2 state
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [installs, setInstalls] = useState<Record<string, InstallState>>({});

  /* --- helpers --- */

  const refresh = async () => {
    setChecking(true);
    try {
      const s = await api.systemStatus();
      setStatus(s);
    } finally {
      setChecking(false);
    }
  };

  const handleInstall = async (dep: string) => {
    setInstalls((prev) => ({
      ...prev,
      [dep]: { ...INITIAL_INSTALL, active: true, label: dep, phase: "starting" },
    }));

    try {
      await api.installDepStream(dep, (event: InstallProgress) => {
        setInstalls((prev) => ({
          ...prev,
          [dep]: {
            active: event.phase !== "complete" && event.phase !== "error",
            phase: event.phase,
            label: event.label || dep,
            percent: event.percent || 0,
            downloaded_mb: event.downloaded_mb || 0,
            total_mb: event.total_mb || 0,
            error: event.phase === "error" ? event.message : undefined,
          },
        }));
      });
      refresh();
    } catch (err: any) {
      setInstalls((prev) => ({
        ...prev,
        [dep]: {
          ...INITIAL_INSTALL,
          active: false,
          phase: "error",
          label: dep,
          error: err.message,
        },
      }));
    }
  };

  // Auto-run check when entering step 2
  useEffect(() => {
    if (step === 1 && !status) {
      refresh();
    }
  }, [step]);

  /* --- derived state for step 2 --- */

  const ffmpegOk =
    status?.ffmpeg.installed || installs.ffmpeg?.phase === "complete";
  const colmapOk =
    status?.colmap.installed || installs.colmap?.phase === "complete";
  const anyInstalling = Object.values(installs).some((s) => s.active);
  const canContinue = ffmpegOk && colmapOk && !anyInstalling;

  /* --- build dep info for cards --- */

  function gpuStatus(): { status: "ok" | "warning"; detail: string } {
    if (status?.cuda_available) {
      return {
        status: "ok",
        detail: `${status.gpu_name || "GPU"} \u2014 CUDA ${status.cuda_version || ""}`,
      };
    }
    return {
      status: "warning",
      detail: "No CUDA GPU detected. Training will be unavailable.",
    };
  }

  function depStatus(
    dep: { installed: boolean; version?: string | null; error?: string | null } | undefined
  ): { status: "ok" | "missing"; detail: string } {
    if (!dep) return { status: "missing", detail: "Unknown" };
    if (dep.installed) return { status: "ok", detail: dep.version || "Installed" };
    return { status: "missing", detail: dep.error || "Not installed" };
  }

  function pytorchStatus(): { status: "ok" | "warning" | "missing"; detail: string } {
    if (!status) return { status: "missing", detail: "Checking..." };
    if (!status.python_deps.installed) {
      return { status: "missing", detail: status.python_deps.error || "Not installed" };
    }
    if (status.cuda_available) {
      return { status: "ok", detail: `${status.python_deps.version || "Installed"} (CUDA)` };
    }
    return {
      status: "warning",
      detail: `${status.python_deps.version || "Installed"} (CPU only \u2014 training will be slow)`,
    };
  }

  /* --- installed summary for step 3 --- */

  const installedItems: string[] = [];
  if (status?.ffmpeg.installed || installs.ffmpeg?.phase === "complete")
    installedItems.push("FFmpeg");
  if (status?.colmap.installed || installs.colmap?.phase === "complete")
    installedItems.push("COLMAP");
  if (status?.python_deps.installed) installedItems.push("PyTorch");
  if (status?.cuda_available) installedItems.push("CUDA");

  /* ---------------------------------------------------------------- */
  /*  Render                                                          */
  /* ---------------------------------------------------------------- */

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-xl">
        <StepIndicator current={step} />

        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 sm:p-8">
          {/* ====== STEP 1: Welcome ====== */}
          {step === 0 && (
            <div className="text-center space-y-6">
              <div className="flex justify-center">
                <Sparkles className="w-12 h-12 text-blue-400" />
              </div>
              <h1 className="text-2xl font-bold text-white">
                Welcome to GaussianSplat Studio
              </h1>
              <p className="text-gray-400 text-sm leading-relaxed max-w-md mx-auto">
                Turn ordinary videos into stunning 3D Gaussian Splat scenes.
                This wizard will check your system for the required tools and
                help you install anything that&apos;s missing.
              </p>
              <button
                onClick={() => setStep(1)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition flex items-center gap-2 mx-auto"
              >
                Let&apos;s check your system
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* ====== STEP 2: System Check ====== */}
          {step === 1 && (
            <div className="space-y-5">
              <div className="text-center mb-2">
                <h1 className="text-xl font-bold text-white">System Check</h1>
                <p className="text-gray-400 text-xs mt-1">
                  Checking for required dependencies
                </p>
              </div>

              {checking && !status ? (
                <div className="flex items-center justify-center gap-2 py-8 text-gray-400 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Scanning your system...
                </div>
              ) : status ? (
                <div className="space-y-3">
                  {/* GPU / CUDA */}
                  <DepCard
                    icon={<Cpu className="w-5 h-5" />}
                    title="GPU / CUDA"
                    {...gpuStatus()}
                  />

                  {/* FFmpeg */}
                  <DepCard
                    icon={<Film className="w-5 h-5" />}
                    title="FFmpeg"
                    {...depStatus(status.ffmpeg)}
                    canInstall={!status.ffmpeg.installed}
                    installing={installs.ffmpeg}
                    onInstall={() => handleInstall("ffmpeg")}
                  />

                  {/* COLMAP */}
                  <DepCard
                    icon={<Box className="w-5 h-5" />}
                    title="COLMAP"
                    {...depStatus(status.colmap)}
                    canInstall={!status.colmap.installed}
                    installing={installs.colmap}
                    onInstall={() => handleInstall("colmap")}
                  />

                  {/* PyTorch */}
                  <DepCard
                    icon={<Sparkles className="w-5 h-5" />}
                    title="PyTorch"
                    {...pytorchStatus()}
                  />

                  {/* Continue button */}
                  <div className="pt-3 flex items-center justify-between">
                    <button
                      onClick={() => setStep(0)}
                      className="text-gray-400 hover:text-white text-sm transition"
                    >
                      Back
                    </button>
                    <button
                      onClick={() => setStep(2)}
                      disabled={!canContinue}
                      className={`px-6 py-2.5 rounded-lg font-medium transition flex items-center gap-2 ${
                        canContinue
                          ? "bg-blue-600 hover:bg-blue-700 text-white"
                          : "bg-gray-700 text-gray-500 cursor-not-allowed"
                      }`}
                    >
                      Continue
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  {!canContinue && !anyInstalling && (
                    <p className="text-xs text-gray-500 text-center">
                      Install FFmpeg and COLMAP to continue. GPU/CUDA warnings
                      won&apos;t block setup.
                    </p>
                  )}
                </div>
              ) : null}
            </div>
          )}

          {/* ====== STEP 3: Ready ====== */}
          {step === 2 && (
            <div className="text-center space-y-6">
              <div className="flex justify-center">
                <PartyPopper className="w-12 h-12 text-yellow-400" />
              </div>
              <h1 className="text-2xl font-bold text-white">
                You&apos;re all set!
              </h1>
              <p className="text-gray-400 text-sm leading-relaxed max-w-md mx-auto">
                Your system is ready to create Gaussian Splat scenes.
              </p>

              {installedItems.length > 0 && (
                <div className="bg-gray-700/50 rounded-lg p-4 text-left text-sm space-y-1.5 max-w-xs mx-auto">
                  {installedItems.map((item) => (
                    <div
                      key={item}
                      className="flex items-center gap-2 text-green-400"
                    >
                      <CheckCircle className="w-4 h-4 shrink-0" />
                      <span className="text-gray-300">{item}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="space-y-3 pt-2">
                <button
                  onClick={() => router.push("/new")}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition flex items-center gap-2 mx-auto"
                >
                  Create Your First Project
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => router.push("/")}
                  className="text-gray-400 hover:text-white text-sm transition block mx-auto"
                >
                  Go to Dashboard
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
