"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import type { WsMessage } from "@/lib/types";

export function useWebSocket(projectId: string | null) {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState<{
    step: string;
    substep: string;
    percent: number;
  } | null>(null);
  const [status, setStatus] = useState<{
    step: string;
    state: string;
    error?: string | null;
  } | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setMetrics([]);
    setProgress(null);
    setStatus(null);
  }, []);

  useEffect(() => {
    if (!projectId) return;

    let disposed = false;

    function connect() {
      if (disposed) return;

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      // Connect directly to backend — Next.js rewrites don't proxy WebSockets
      const backendHost = process.env.NEXT_PUBLIC_API_URL
        ? new URL(process.env.NEXT_PUBLIC_API_URL).host
        : `${window.location.hostname}:8000`;
      const url = `${protocol}//${backendHost}/ws/projects/${projectId}/logs`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (evt) => {
        try {
          const msg: WsMessage = JSON.parse(evt.data);
          switch (msg.type) {
            case "log":
              if (msg.line) setLogs((prev) => [...prev.slice(-500), msg.line!]);
              break;
            case "progress":
              setProgress({
                step: msg.step || "",
                substep: msg.substep || "",
                percent: msg.percent || 0,
              });
              break;
            case "status":
              setStatus({
                step: msg.step || "",
                state: msg.state || "",
                error: msg.error,
              });
              break;
            case "metric":
              setMetrics((prev) => [...prev.slice(-200), msg]);
              break;
          }
        } catch {}
      };

      ws.onclose = () => {
        if (!disposed) {
          reconnectTimer.current = setTimeout(connect, 2000);
        }
      };
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [projectId]);

  return { logs, progress, status, metrics, clearLogs };
}
