/**
 * renderers/index.ts — Auto-detect best available renderer.
 *
 * Primary: PlayCanvas (WebGPU + GPU sorting, automatic WebGL2 fallback)
 * Fallback: Mkkellogg (Three.js WebGL, if PlayCanvas fails to init)
 */

import type { SplatRenderer, RendererStats } from "./mkkellogg";

export type { SplatRenderer, RendererStats };

export async function createRenderer(
  url: string,
  container: HTMLElement
): Promise<SplatRenderer & { type: "webgpu" | "webgl" }> {
  // Try PlayCanvas first (WebGPU + GPU-driven sorting)
  try {
    const { PlayCanvasRenderer } = await import("./playcanvas");
    return new PlayCanvasRenderer(container);
  } catch (e) {
    console.warn("PlayCanvas renderer failed to init, falling back to mkkellogg:", e);
  }

  // Fallback to mkkellogg (Three.js WebGL)
  const { MkkelloggRenderer } = await import("./mkkellogg");
  return new MkkelloggRenderer(container);
}
