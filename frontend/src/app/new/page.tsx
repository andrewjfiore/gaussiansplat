"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Upload, Download, Loader2 } from "lucide-react";

const SAMPLES = [
  {
    id: "hong-kong-night",
    title: "Hong Kong at Night (Aerial)",
    thumbnail:
      "https://images.pexels.com/videos/3129671/free-video-3129671.jpg?auto=compress&cs=tinysrgb&w=400",
    duration: "0:30",
  },
  {
    id: "mountain-building",
    title: "Building on Mountain",
    thumbnail:
      "https://images.pexels.com/videos/4571563/pexels-photo-4571563.jpeg?auto=compress&cs=tinysrgb&w=400",
    duration: "0:15",
  },
  {
    id: "city-panoramic",
    title: "Panoramic View of a City",
    thumbnail:
      "https://images.pexels.com/videos/3573921/free-video-3573921.jpg?auto=compress&cs=tinysrgb&w=400",
    duration: "0:20",
  },
];

export default function NewProjectPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [tab, setTab] = useState<"upload" | "sample">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Project name is required");
      return;
    }
    if (tab === "upload" && !file) {
      setError("Please select a video file");
      return;
    }
    if (tab === "sample" && !selectedSample) {
      setError("Please select a sample video");
      return;
    }

    setCreating(true);
    setError(null);
    setUploadPercent(null);

    try {
      setStatusText("Creating project...");
      const project = await api.createProject(name);

      if (tab === "upload" && file) {
        setStatusText("Uploading video...");
        await api.uploadVideo(project.id, file, (pct) => {
          setUploadPercent(pct);
        });
      } else if (tab === "sample" && selectedSample) {
        setStatusText("Downloading sample video...");
        await api.downloadSample(project.id, selectedSample);
      }

      router.push(`/project/${project.id}/frames`);
    } catch (err: any) {
      setError(err.message);
      setCreating(false);
      setUploadPercent(null);
      setStatusText(null);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">New Project</h1>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Project Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Gaussian Splat"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
          />
        </div>

        <div>
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setTab("upload")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                tab === "upload"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-white"
              }`}
            >
              <Upload className="w-4 h-4 inline mr-2" /> Upload Video
            </button>
            <button
              onClick={() => setTab("sample")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                tab === "sample"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-white"
              }`}
            >
              <Download className="w-4 h-4 inline mr-2" /> Sample Videos
            </button>
          </div>

          {tab === "upload" && (
            <div
              className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-gray-500 transition cursor-pointer"
              onClick={() =>
                document.getElementById("file-input")?.click()
              }
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) setFile(f);
              }}
            >
              <input
                id="file-input"
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) =>
                  setFile(e.target.files?.[0] || null)
                }
              />
              {file ? (
                <div>
                  <p className="text-white font-medium">{file.name}</p>
                  <p className="text-gray-400 text-sm mt-1">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
              ) : (
                <div>
                  <Upload className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                  <p className="text-gray-400">
                    Drag & drop a video or click to browse
                  </p>
                  <p className="text-gray-500 text-sm mt-1">
                    .mp4, .mov, .avi supported
                  </p>
                </div>
              )}
            </div>
          )}

          {tab === "sample" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {SAMPLES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedSample(s.id)}
                  className={`rounded-lg overflow-hidden border-2 transition text-left ${
                    selectedSample === s.id
                      ? "border-blue-500"
                      : "border-gray-700 hover:border-gray-500"
                  }`}
                >
                  <div className="aspect-video bg-gray-800">
                    <img
                      src={s.thumbnail}
                      alt={s.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="p-2">
                    <p className="text-sm font-medium truncate">{s.title}</p>
                    <p className="text-xs text-gray-400">{s.duration}</p>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {error && (
          <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-lg p-3">
            {error}
          </div>
        )}

        <button
          onClick={handleCreate}
          disabled={creating}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-400 text-white py-3 rounded-lg font-medium transition flex items-center justify-center gap-2"
        >
          {creating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />{" "}
              {statusText || "Creating..."}
            </>
          ) : (
            "Create Project"
          )}
        </button>

        {/* Upload progress bar */}
        {uploadPercent !== null && (
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Uploading video...</span>
              <span>{uploadPercent}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-500 h-full rounded-full transition-all duration-300"
                style={{ width: `${uploadPercent}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
