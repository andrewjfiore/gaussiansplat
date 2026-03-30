/**
 * renderers/index.ts — Auto-detect best available renderer.
 *
 * WebGPU + no VR → Mkkellogg (WebGL) with native gsplat support.
 * WebXR VR mode is only supported through Mkkellogg (Three.js WebGLRenderer).
 *
 * Currently ships only the Mkkellogg renderer. The Visionary (WebGPU)
 * renderer can be added here when visionary-lab supports WebXR.
 */

import { MkkelloggRenderer, SplatRenderer } from "./mkkellogg";

export type { SplatRenderer };

export async function createRenderer(
  url: string,
  container: HTMLElement
): Promise<SplatRenderer & { type: "webgpu" | "webgl" }> {
  // Visionary (WebGPU) renderer placeholder — add when visionary-lab ships
  // const gpuAvailable = typeof navigator !== "undefined" &&
  //   navigator.gpu !== undefined &&
  //   (await navigator.gpu.requestAdapter()) !== null;
  // const vrRequested = new URLSearchParams(location.search).has("vr");
  // if (gpuAvailable && !vrRequested) {
  //   const { VisionaryRenderer } = await import("./visionary");
  //   return new VisionaryRenderer(url, container);
  // }

  return new MkkelloggRenderer(container);
}
