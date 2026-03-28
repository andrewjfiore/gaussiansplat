"use client";
import type { FrameInfo } from "@/lib/types";

export function FrameGrid({ frames }: { frames: FrameInfo[] }) {
  if (frames.length === 0) {
    return <div className="text-gray-400 text-center py-8">No frames extracted yet</div>;
  }

  return (
    <div>
      <div className="text-sm text-gray-400 mb-2">{frames.length} frames</div>
      <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
        {frames.map((f) => (
          <div key={f.name} className="aspect-video bg-gray-800 rounded overflow-hidden">
            <img
              src={f.url}
              alt={f.name}
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
