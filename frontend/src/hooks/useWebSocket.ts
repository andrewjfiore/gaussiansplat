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

  const clearLogs = useCallback(() => {
    setLogs([]);
    setMetrics([]);
    setProgress(null);
    setStatus(null);
  }, []);

  useEffect(() => {
    if (!projectId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/projects/${projectId}/logs`;
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
      // Auto-reconnect after 2s
      setTimeout(() => {
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
      }, 2000);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [projectId]);

  return { logs, progress, status, metrics, clearLogs };
}
