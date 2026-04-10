"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Upload, Download, Loader2, Camera, Video } from "lucide-react";

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
  const [mode, setMode] = useState<"video" | "portrait">("video");
  const [tab, setTab] = useState<"upload" | "sample">("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [portraitFile, setPortraitFile] = useState<File | null>(null);
  const [portraitPreview, setPortraitPreview] = useState<string | null>(null);
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
    if (mode === "video" && tab === "upload" && files.length === 0) {
      setError("Please select at least one video file");
      return;
    }
    if (mode === "video" && tab === "sample" && !selectedSample) {
      setError("Please select a sample video");
      return;
    }
    if (mode === "portrait" && !portraitFile) {
      setError("Please select a portrait image");
      return;
    }

    setCreating(true);
    setError(null);
    setUploadPercent(null);

    try {
      setStatusText("Creating project...");
      const project = await api.createProject(name);

      if (mode === "portrait" && portraitFile) {
        setStatusText("Uploading portrait...");
        await api.uploadPortrait(project.id, portraitFile, (pct) => {
          setUploadPercent(pct);
        });
        router.push(`/project/${project.id}/portrait`);
        return;
      }

      if (tab === "upload" && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          setStatusText(`Uploading video ${i + 1}/${files.length}...`);
          await api.uploadVideo(project.id, files[i], (pct) => {
            setUploadPercent(pct);
          });
        }
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

        {/* Mode toggle: Video vs Portrait */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Input Mode
          </label>
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setMode("video")}
              className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition border-2 flex items-center justify-center gap-2 ${
                mode === "video"
                  ? "bg-blue-600/10 border-blue-500 text-blue-400"
                  : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600 hover:text-white"
              }`}
            >
              <Video className="w-4 h-4" />
              Video Mode
            </button>
            <button
              onClick={() => setMode("portrait")}
              className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition border-2 flex items-center justify-center gap-2 ${
                mode === "portrait"
                  ? "bg-violet-600/10 border-violet-500 text-violet-400"
                  : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600 hover:text-white"
              }`}
            >
              <Camera className="w-4 h-4" />
              Portrait Mode
            </button>
          </div>
        </div>

        <div>
          {mode === "video" && (
          <>
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
            <div>
              <div
                className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-gray-500 transition cursor-pointer"
                onClick={() =>
                  document.getElementById("file-input")?.click()
                }
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const dropped = Array.from(e.dataTransfer.files).filter(
                    (f) => f.type.startsWith("video/")
                  );
                  if (dropped.length > 0)
                    setFiles((prev) => [...prev, ...dropped]);
                }}
              >
                <input
                  id="file-input"
                  type="file"
                  accept="video/*"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const selected = Array.from(e.target.files || []);
                    if (selected.length > 0)
                      setFiles((prev) => [...prev, ...selected]);
                  }}
                />
                {files.length > 0 ? (
                  <div>
                    <p className="text-white font-medium">
                      {files.length} video{files.length > 1 ? "s" : ""} selected
                    </p>
                    <p className="text-gray-500 text-sm mt-1">
                      Click or drop to add more
                    </p>
                  </div>
                ) : (
                  <div>
                    <Upload className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">
                      Drag & drop video(s) or click to browse
                    </p>
                    <p className="text-gray-500 text-sm mt-1">
                      .mp4, .mov, .avi — multiple videos supported for multi-cam
                    </p>
                  </div>
                )}
              </div>
              {files.length > 0 && (
                <div className="mt-3 space-y-1">
                  {files.map((f, i) => (
                    <div
                      key={`${f.name}-${i}`}
                      className="flex items-center justify-between bg-gray-800 rounded px-3 py-1.5 text-sm"
                    >
                      <span className="text-white truncate mr-2">{f.name}</span>
                      <div className="flex items-center gap-2 text-gray-400 text-xs">
                        <span>{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                        <button
                          onClick={() =>
                            setFiles((prev) => prev.filter((_, j) => j !== i))
                          }
                          className="text-red-400 hover:text-red-300"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
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
          </>
          )}

          {mode === "portrait" && (
            <div>
              <div
                className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center hover:border-violet-500/50 transition cursor-pointer"
                onClick={() =>
                  document.getElementById("portrait-input")?.click()
                }
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const dropped = Array.from(e.dataTransfer.files).find((f) =>
                    f.type.startsWith("image/")
                  );
                  if (dropped) {
                    setPortraitFile(dropped);
                    setPortraitPreview(URL.createObjectURL(dropped));
                  }
                }}
              >
                <input
                  id="portrait-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp"
                  className="hidden"
                  onChange={(e) => {
                    const selected = e.target.files?.[0];
                    if (selected) {
                      setPortraitFile(selected);
                      setPortraitPreview(URL.createObjectURL(selected));
                    }
                  }}
                />
                {portraitPreview ? (
                  <div className="space-y-2">
                    <img
                      src={portraitPreview}
                      alt="Portrait preview"
                      className="max-h-48 mx-auto rounded-lg object-contain"
                    />
                    <p className="text-white font-medium">
                      {portraitFile?.name}
                    </p>
                    <p className="text-gray-500 text-sm">
                      {portraitFile
                        ? `${(portraitFile.size / 1024 / 1024).toFixed(1)} MB`
                        : ""}
                      {" -- "}Click or drop to change
                    </p>
                  </div>
                ) : (
                  <div>
                    <Camera className="w-10 h-10 text-violet-500/60 mx-auto mb-3" />
                    <p className="text-gray-400">
                      Drag & drop a portrait photo or click to browse
                    </p>
                    <p className="text-gray-500 text-sm mt-1">
                      .jpg, .png, .webp -- single photo to 3D
                    </p>
                  </div>
                )}
              </div>
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
          className={`w-full disabled:bg-gray-700 disabled:text-gray-400 text-white py-3 rounded-lg font-medium transition flex items-center justify-center gap-2 ${
            mode === "portrait"
              ? "bg-violet-600 hover:bg-violet-700"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {creating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />{" "}
              {statusText || "Creating..."}
            </>
          ) : mode === "portrait" ? (
            "Create Portrait Project"
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
