"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { CheckCircle, XCircle, AlertTriangle, Info, X } from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type ToastType = "success" | "error" | "warning" | "info";

interface ToastOptions {
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastEntry extends ToastOptions {
  id: number;
  removing?: boolean;
}

interface ToastContextValue {
  toast: (opts: ToastOptions) => void;
}

/* ------------------------------------------------------------------ */
/*  Styling helpers                                                    */
/* ------------------------------------------------------------------ */

const ICON_MAP: Record<ToastType, typeof CheckCircle> = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

const COLOR_MAP: Record<ToastType, { border: string; icon: string; bg: string }> = {
  success: {
    border: "border-green-500/40",
    icon: "text-green-400",
    bg: "bg-green-500/10",
  },
  error: {
    border: "border-red-500/40",
    icon: "text-red-400",
    bg: "bg-red-500/10",
  },
  warning: {
    border: "border-yellow-500/40",
    icon: "text-yellow-400",
    bg: "bg-yellow-500/10",
  },
  info: {
    border: "border-blue-500/40",
    icon: "text-blue-400",
    bg: "bg-blue-500/10",
  },
};

const MAX_TOASTS = 5;
const DEFAULT_DURATION = 5000;

/* ------------------------------------------------------------------ */
/*  Context                                                            */
/* ------------------------------------------------------------------ */

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within a <ToastProvider>");
  return ctx;
}

/* ------------------------------------------------------------------ */
/*  Provider                                                           */
/* ------------------------------------------------------------------ */

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastEntry[]>([]);
  const nextId = useRef(0);

  const dismiss = useCallback((id: number) => {
    // Mark as removing to trigger the slide-out animation
    setToasts((prev) => prev.map((t) => (t.id === id ? { ...t, removing: true } : t)));
    // Remove from DOM after the animation completes
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 300);
  }, []);

  const toast = useCallback(
    (opts: ToastOptions) => {
      const id = nextId.current++;
      setToasts((prev) => {
        const next = [...prev, { ...opts, id }];
        // If over the limit, mark the oldest for removal
        if (next.length > MAX_TOASTS) {
          const oldest = next[0];
          setTimeout(() => dismiss(oldest.id), 0);
        }
        return next;
      });

      // Auto-dismiss
      const duration = opts.duration ?? DEFAULT_DURATION;
      if (duration > 0) {
        setTimeout(() => dismiss(id), duration);
      }
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}

      {/* Toast container – bottom-right, stacks upward */}
      <div
        aria-live="polite"
        className="fixed bottom-4 right-4 z-50 flex flex-col-reverse gap-2 pointer-events-none"
      >
        {toasts.map((t) => (
          <ToastCard key={t.id} entry={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/* ------------------------------------------------------------------ */
/*  Individual toast card                                              */
/* ------------------------------------------------------------------ */

function ToastCard({ entry, onDismiss }: { entry: ToastEntry; onDismiss: () => void }) {
  const { type, title, message, removing } = entry;
  const colors = COLOR_MAP[type];
  const Icon = ICON_MAP[type];

  // Trigger the enter animation after mount
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const isShowing = visible && !removing;

  return (
    <div
      role="alert"
      className={`
        pointer-events-auto w-80 rounded-lg border ${colors.border} ${colors.bg}
        bg-gray-900 shadow-lg backdrop-blur
        transition-all duration-300 ease-in-out
        ${isShowing ? "translate-x-0 opacity-100" : "translate-x-full opacity-0"}
      `}
    >
      <div className="flex items-start gap-3 p-4">
        <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${colors.icon}`} />

        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-gray-100">{title}</p>
          {message && <p className="mt-1 text-sm text-gray-400">{message}</p>}
        </div>

        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded p-1 text-gray-500 hover:bg-gray-700 hover:text-gray-300 transition-colors"
          aria-label="Dismiss notification"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
