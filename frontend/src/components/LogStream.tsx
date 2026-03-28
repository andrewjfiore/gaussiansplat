"use client";
import { useEffect, useRef } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

export function LogStream({ logs }: { logs: string[] }) {
  const [expanded, setExpanded] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden bg-gray-950">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 bg-gray-800 text-gray-300 text-sm hover:bg-gray-700"
      >
        <span>Logs ({logs.length} lines)</span>
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {expanded && (
        <div className="max-h-64 overflow-y-auto p-3 font-mono text-xs text-green-400 space-y-0.5">
          {logs.length === 0 && <div className="text-gray-500">Waiting for output...</div>}
          {logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">
              {line}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
