"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Loader2, Undo2, Trash2, MousePointer2 } from "lucide-react";

interface Point {
  x: number;
  y: number;
  label: number; // 1=foreground, 0=background
}

interface Props {
  projectId: string;
  frameUrl: string;
  frameName: string;
  onApply: (points: number[][], labels: number[], refFrame: string) => void;
  onClose: () => void;
}

export function MaskPointSelector({ projectId, frameUrl, frameName, onApply, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);

  // Load the frame image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded(true);
    };
    img.src = frameUrl;
  }, [frameUrl]);

  // Redraw canvas when image or points change
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);

    // Draw mask preview overlay
    if (previewSrc) {
      const overlayImg = new Image();
      overlayImg.onload = () => {
        ctx.drawImage(overlayImg, 0, 0);
        drawPoints(ctx);
      };
      overlayImg.src = previewSrc;
    } else {
      drawPoints(ctx);
    }
  }, [previewSrc, points]);

  const drawPoints = (ctx: CanvasRenderingContext2D) => {
    for (const pt of points) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = pt.label === 1 ? "rgba(34, 197, 94, 0.8)" : "rgba(239, 68, 68, 0.8)";
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Cross for background points
      if (pt.label === 0) {
        ctx.beginPath();
        ctx.moveTo(pt.x - 5, pt.y - 5);
        ctx.lineTo(pt.x + 5, pt.y + 5);
        ctx.moveTo(pt.x + 5, pt.y - 5);
        ctx.lineTo(pt.x - 5, pt.y + 5);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  };

  useEffect(() => { if (imgLoaded) redraw(); }, [imgLoaded, redraw]);

  // Fetch preview after points change
  useEffect(() => {
    if (points.length === 0) {
      setPreviewSrc(null);
      return;
    }

    const timeout = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api.maskPreview(projectId, {
          frame: frameName,
          points: points.map((p) => [p.x, p.y]),
          labels: points.map((p) => p.label),
        });
        setPreviewSrc(`data:image/png;base64,${res.mask_b64}`);
      } catch (err) {
        console.error("Mask preview failed:", err);
      }
      setLoading(false);
    }, 300); // Debounce

    return () => clearTimeout(timeout);
  }, [points, projectId, frameName]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Left-click = foreground (1), right-click handled separately
    setPoints((prev) => [...prev, { x, y, label: 1 }]);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setPoints((prev) => [...prev, { x, y, label: 0 }]);
  };

  const handleUndo = () => {
    setPoints((prev) => prev.slice(0, -1));
  };

  const handleClear = () => {
    setPoints([]);
    setPreviewSrc(null);
  };

  const handleApply = () => {
    if (points.length === 0) return;
    onApply(
      points.map((p) => [p.x, p.y]),
      points.map((p) => p.label),
      frameName,
    );
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-600 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-gray-300">
          <MousePointer2 className="w-4 h-4" />
          <span>
            Click to select objects to mask.{" "}
            <span className="text-green-400">Left-click</span> = include,{" "}
            <span className="text-red-400">Right-click</span> = exclude
          </span>
          {loading && <Loader2 className="w-4 h-4 animate-spin text-purple-400" />}
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-sm"
        >
          Close
        </button>
      </div>

      <div className="relative overflow-auto max-h-[500px] rounded border border-gray-700">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onContextMenu={handleContextMenu}
          className="w-full cursor-crosshair"
          style={{ imageRendering: "auto" }}
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={handleUndo}
          disabled={points.length === 0}
          className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white px-3 py-1.5 rounded text-sm flex items-center gap-1.5 transition"
        >
          <Undo2 className="w-3.5 h-3.5" /> Undo
        </button>
        <button
          onClick={handleClear}
          disabled={points.length === 0}
          className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white px-3 py-1.5 rounded text-sm flex items-center gap-1.5 transition"
        >
          <Trash2 className="w-3.5 h-3.5" /> Clear
        </button>
        <span className="text-xs text-gray-500 ml-2">
          {points.filter((p) => p.label === 1).length} foreground,{" "}
          {points.filter((p) => p.label === 0).length} background points
        </span>
        <div className="flex-1" />
        <button
          onClick={handleApply}
          disabled={points.length === 0 || loading}
          className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white px-4 py-1.5 rounded text-sm font-medium transition"
        >
          Apply to All Frames
        </button>
      </div>
    </div>
  );
}
