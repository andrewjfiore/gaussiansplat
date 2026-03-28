"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import type { SystemStatus } from "@/lib/types";
import type { InstallProgress } from "@/lib/api";
import {
  CheckCircle,
  XCircle,
  Download,
  Loader2,
  Cpu,
  HardDrive,
  ChevronDown,
  ChevronRight,
  Copy,
  RefreshCw,
  FileText,
} from "lucide-react";

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

export default function SettingsPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [installs, setInstalls] = useState<Record<string, InstallState>>({});

  // Log viewer state
  const [logsOpen, setLogsOpen] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logTotal, setLogTotal] = useState(0);
  const [logsLoading, setLogsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const autoRefreshRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const refresh = () => {
    api.systemStatus().then(setStatus).finally(() => setLoading(false));
  };

  const fetchLogs = useCallback(async () => {
    setLogsLoading(true);
    try {
      const data = await api.getSystemLogs(100);
      setLogLines(data.lines);
      setLogTotal(data.total_lines);
    } catch {
      setLogLines(["Failed to load logs."]);
    } finally {
      setLogsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, []);

  // Fetch logs when section is opened
  useEffect(() => {
    if (logsOpen) {
      fetchLogs();
    }
  }, [logsOpen, fetchLogs]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logLines]);

  // Auto-refresh timer
  useEffect(() => {
    if (autoRefresh && logsOpen) {
      autoRefreshRef.current = setInterval(fetchLogs, 5000);
    }
    return () => {
      if (autoRefreshRef.current) {
        clearInterval(autoRefreshRef.current);
        autoRefreshRef.current = null;
      }
    };
  }, [autoRefresh, logsOpen, fetchLogs]);

  const handleCopyLogs = async () => {
    const text = logLines.join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for insecure contexts
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadLog = () => {
    const url = api.downloadLogsUrl();
    const a = document.createElement("a");
    a.href = url;
    a.download = "gaussiansplat.log";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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

  if (loading)
    return (
      <div className="text-gray-400 text-center py-12">Checking system...</div>
    );

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">System Settings</h1>

      {/* GPU Info */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h2 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4" /> GPU & CUDA
        </h2>
        {status?.cuda_available ? (
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">GPU:</span>
              <span className="text-white">
                {status.gpu_name || "Unknown"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">VRAM:</span>
              <span className="text-white">
                {status.gpu_vram_mb
                  ? `${status.gpu_vram_mb} MB`
                  : "Unknown"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Driver:</span>
              <span className="text-white">
                {status.cuda_version || "Unknown"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">NVIDIA Driver:</span>
              <span className="text-green-400 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Detected
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">PyTorch CUDA:</span>
              {(status as any).torch_cuda_available ? (
                <span className="text-green-400 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" /> Ready
                  {(status as any).torch_cuda_version && (
                    <span className="text-gray-500 ml-1">
                      (CUDA {(status as any).torch_cuda_version})
                    </span>
                  )}
                </span>
              ) : (
                <span className="text-yellow-400 flex items-center gap-1">
                  <XCircle className="w-3 h-3" /> Not available — training will fail
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="text-yellow-400 text-sm flex items-center gap-2">
            <XCircle className="w-4 h-4" /> No NVIDIA GPU / CUDA detected.
            Training will not work.
          </div>
        )}
      </div>

      {/* Dependencies */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 divide-y divide-gray-700">
        <h2 className="text-sm font-medium text-gray-300 px-4 py-3 flex items-center gap-2">
          <HardDrive className="w-4 h-4" /> Dependencies
        </h2>

        {status &&
          [status.ffmpeg, status.colmap, status.python_deps].map((dep) => {
            const installState = installs[dep.name];
            const isInstalling = installState?.active;

            return (
              <div key={dep.name} className="px-4 py-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-medium text-white capitalize">
                      {dep.name.replace("_", " ")}
                    </div>
                    {dep.installed ? (
                      <div className="text-xs text-gray-400 mt-0.5">
                        {dep.version || "Installed"}
                      </div>
                    ) : installState?.error ? (
                      <div className="text-xs text-red-400 mt-0.5">
                        {installState.error}
                      </div>
                    ) : (
                      <div className="text-xs text-red-400 mt-0.5">
                        {dep.error}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {dep.installed ? (
                      <span className="text-green-400">
                        <CheckCircle className="w-5 h-5" />
                      </span>
                    ) : (
                      <>
                        {!isInstalling && (
                          <span className="text-red-400">
                            <XCircle className="w-5 h-5" />
                          </span>
                        )}
                        {(dep.name === "ffmpeg" || dep.name === "colmap") &&
                          !isInstalling && (
                            <button
                              onClick={() => handleInstall(dep.name)}
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
                        {installState.phase === "downloading"
                          ? `Downloading ${installState.label}...`
                          : installState.phase === "extracting"
                          ? `Extracting ${installState.label}...`
                          : `Setting up ${installState.label}...`}
                      </span>
                      {installState.phase === "downloading" && (
                        <span>
                          {installState.downloaded_mb} / {installState.total_mb}{" "}
                          MB ({installState.percent}%)
                        </span>
                      )}
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          installState.phase === "extracting"
                            ? "bg-yellow-500 animate-pulse"
                            : "bg-blue-500"
                        }`}
                        style={{
                          width:
                            installState.phase === "extracting"
                              ? "100%"
                              : `${installState.percent}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Completed install (just finished) */}
                {installState &&
                  !installState.active &&
                  installState.phase === "complete" && (
                    <div className="mt-2 text-xs text-green-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Installed
                      successfully
                    </div>
                  )}
              </div>
            );
          })}
      </div>

      {/* View Logs */}
      <div className="bg-gray-800 rounded-lg border border-gray-700">
        <button
          onClick={() => setLogsOpen(!logsOpen)}
          className="w-full px-4 py-3 flex items-center justify-between text-sm font-medium text-gray-300 hover:text-white transition"
        >
          <span className="flex items-center gap-2">
            <FileText className="w-4 h-4" /> Application Logs
          </span>
          {logsOpen ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>

        {logsOpen && (
          <div className="border-t border-gray-700">
            {/* Log toolbar */}
            <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-700">
              <button
                onClick={fetchLogs}
                disabled={logsLoading}
                className="text-xs px-2.5 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center gap-1 transition disabled:opacity-50"
              >
                <RefreshCw
                  className={`w-3 h-3 ${logsLoading ? "animate-spin" : ""}`}
                />
                Refresh
              </button>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`text-xs px-2.5 py-1.5 rounded flex items-center gap-1 transition ${
                  autoRefresh
                    ? "bg-blue-600 hover:bg-blue-700 text-white"
                    : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                }`}
              >
                <RefreshCw className={`w-3 h-3 ${autoRefresh ? "animate-spin" : ""}`} />
                Auto
              </button>
              <button
                onClick={handleCopyLogs}
                className="text-xs px-2.5 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center gap-1 transition"
              >
                <Copy className="w-3 h-3" />
                {copied ? "Copied!" : "Copy Logs"}
              </button>
              <button
                onClick={handleDownloadLog}
                className="text-xs px-2.5 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 flex items-center gap-1 transition"
              >
                <Download className="w-3 h-3" />
                Download Full Log
              </button>
              <span className="ml-auto text-xs text-gray-500">
                {logTotal > 0
                  ? `Showing last ${logLines.length} of ${logTotal} lines`
                  : "No logs yet"}
              </span>
            </div>

            {/* Log content */}
            <div className="max-h-80 overflow-y-auto font-mono text-xs leading-relaxed">
              {logLines.length > 0 ? (
                <pre className="px-4 py-3 text-gray-300 whitespace-pre-wrap break-all">
                  {logLines.map((line, i) => {
                    let color = "text-gray-300";
                    if (line.includes("[ERROR]") || line.includes("[CRITICAL]"))
                      color = "text-red-400";
                    else if (line.includes("[WARNING]"))
                      color = "text-yellow-400";
                    else if (line.includes("[DEBUG]"))
                      color = "text-gray-500";
                    return (
                      <span key={i} className={`block ${color}`}>
                        {line}
                      </span>
                    );
                  })}
                  <div ref={logEndRef} />
                </pre>
              ) : (
                <div className="px-4 py-6 text-gray-500 text-center">
                  No log entries found.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="text-xs text-gray-500 text-center">
        GaussianSplat Studio checks for dependencies automatically. Click
        Install to auto-download missing tools.
      </div>
    </div>
  );
}
