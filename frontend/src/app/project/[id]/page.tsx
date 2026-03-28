"use client";
import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function ProjectPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  useEffect(() => {
    api.getProject(id).then((p) => {
      switch (p.step) {
        case "training_complete":
          router.replace(`/project/${id}/view`);
          break;
        case "sfm_ready":
        case "training":
          router.replace(`/project/${id}/train`);
          break;
        case "frames_ready":
        case "running_sfm":
          router.replace(`/project/${id}/sfm`);
          break;
        default:
          router.replace(`/project/${id}/frames`);
      }
    });
  }, [id, router]);

  return <div className="text-gray-400 text-center py-12">Redirecting...</div>;
}
