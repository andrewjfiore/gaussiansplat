/**
 * playcanvas.ts — PlayCanvas WebGPU/WebGL2 renderer for Gaussian Splats.
 *
 * Uses PlayCanvas engine's native GSplatComponent with GPU-driven radix sort.
 * Supports WASD + pointer-lock FPS controls, orbit camera, and WebXR VR.
 */

import type { SplatRenderer, RendererStats } from "./mkkellogg";

// PlayCanvas is imported dynamically to avoid SSR issues in Next.js
let pc: typeof import("playcanvas");

export class PlayCanvasRenderer implements SplatRenderer {
  readonly type = "webgpu" as const;

  private app: any = null;
  private canvas: HTMLCanvasElement;
  private camera: any = null;
  private splatEntity: any = null;
  private keys: Record<string, boolean> = {};
  private fpsMode = false;
  private _lastFrameTime = performance.now();
  private _fps = 0;
  private _fpsFrames = 0;
  private _fpsAccum = 0;
  private _lightEntity: any = null;
  private _disposed = false;

  constructor(private container: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.canvas.style.display = "block";
    container.appendChild(this.canvas);

    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);
  }

  private _onKeyDown = (e: KeyboardEvent) => { this.keys[e.code] = true; };
  private _onKeyUp   = (e: KeyboardEvent) => { this.keys[e.code] = false; };

  async load(url: string): Promise<void> {
    pc = await import("playcanvas");

    // Create graphics device — try WebGPU first, fall back to WebGL2
    const device = await pc.createGraphicsDevice(this.canvas, {
      deviceTypes: ["webgpu", "webgl2"],
      antialias: false,
    });
    device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

    const createOptions = new pc.AppOptions();
    createOptions.graphicsDevice = device;
    createOptions.mouse = new pc.Mouse(this.canvas);
    createOptions.touch = new pc.TouchDevice(this.canvas);
    createOptions.componentSystems = [
      pc.RenderComponentSystem,
      pc.CameraComponentSystem,
      pc.LightComponentSystem,
      pc.ScriptComponentSystem,
      pc.GSplatComponentSystem,
    ];
    createOptions.resourceHandlers = [
      pc.TextureHandler,
      pc.ContainerHandler,
      pc.ScriptHandler,
      pc.GSplatHandler,
    ];

    this.app = new pc.AppBase(this.canvas);
    this.app.init(createOptions);
    this.app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
    this.app.setCanvasResolution(pc.RESOLUTION_AUTO);

    // Resize handling
    const resize = () => this.app?.resizeCanvas();
    window.addEventListener("resize", resize);
    (this as any)._onResize = resize;

    // Load the splat asset
    const asset = await new Promise<any>((resolve, reject) => {
      // Determine filename for format detection
      let filename = url;
      if (!url.includes(".ply") && !url.includes(".sog") && !url.includes(".splat")) {
        filename = "scene.ply"; // default hint for API URLs without extension
      }
      this.app.assets.loadFromUrlAndFilename(url, filename, "gsplat",
        (err: any, loadedAsset: any) => {
          if (err) reject(err);
          else resolve(loadedAsset);
        }
      );
    });

    // Create splat entity
    this.splatEntity = new pc.Entity("splat");
    this.splatEntity.addComponent("gsplat", {
      asset: asset,
      unified: true,
    });
    this.splatEntity.setLocalEulerAngles(180, 0, 0);
    this.app.root.addChild(this.splatEntity);

    // Create camera
    this.camera = new pc.Entity("camera");
    this.camera.addComponent("camera", {
      clearColor: new pc.Color(0, 0, 0),
      fov: 75,
      farClip: 10000,
      nearClip: 0.01,
    });
    this.camera.setLocalPosition(0, 1.6, 3);
    this.app.root.addChild(this.camera);

    // Create ambient + directional light
    this._lightEntity = new pc.Entity("light");
    this._lightEntity.addComponent("light", {
      type: "directional",
      color: new pc.Color(1, 1, 1),
      intensity: 1,
      castShadows: true,
      shadowResolution: 2048,
    });
    this._lightEntity.setEulerAngles(55, 0, 20);
    this.app.root.addChild(this._lightEntity);

    // Start the app
    this.app.start();

    // Wait a frame for bounds to be available, then center camera
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    this._centerCamera();

    // Register per-frame update for WASD movement + FPS tracking
    this.app.on("update", this._onUpdate, this);

    // Click to lock pointer for FPS mode
    this.canvas.addEventListener("click", this._onCanvasClick);
  }

  private _onCanvasClick = () => {
    if (this.fpsMode && !document.pointerLockElement) {
      this.canvas.requestPointerLock();
    }
  };

  private _centerCamera() {
    if (!this.splatEntity?.gsplat) return;
    try {
      const aabb = this.splatEntity.gsplat.customAabb;
      if (aabb) {
        const center = aabb.center;
        const size = aabb.halfExtents.length() * 2;
        this.camera.setLocalPosition(
          center.x,
          center.y + size * 0.3,
          center.z + size * 0.5,
        );
        this.camera.camera.farClip = size * 10;
      }
    } catch { /* best effort */ }
  }

  private _onUpdate = (dt: number) => {
    if (this._disposed) return;

    // FPS tracking
    const now = performance.now();
    this._fpsAccum += now - this._lastFrameTime;
    this._lastFrameTime = now;
    this._fpsFrames++;
    if (this._fpsAccum >= 1000) {
      this._fps = Math.round((this._fpsFrames * 1000) / this._fpsAccum);
      this._fpsFrames = 0;
      this._fpsAccum = 0;
    }

    // WASD movement in FPS mode
    if (this.fpsMode && document.pointerLockElement === this.canvas) {
      const speed = 2.0 * dt;
      const forward = this.camera.forward;
      const right = this.camera.right;
      const pos = this.camera.getLocalPosition();

      if (this.keys["KeyW"] || this.keys["ArrowUp"]) {
        pos.x += forward.x * speed;
        pos.y += forward.y * speed;
        pos.z += forward.z * speed;
      }
      if (this.keys["KeyS"] || this.keys["ArrowDown"]) {
        pos.x -= forward.x * speed;
        pos.y -= forward.y * speed;
        pos.z -= forward.z * speed;
      }
      if (this.keys["KeyA"] || this.keys["ArrowLeft"]) {
        pos.x -= right.x * speed;
        pos.z -= right.z * speed;
      }
      if (this.keys["KeyD"] || this.keys["ArrowRight"]) {
        pos.x += right.x * speed;
        pos.z += right.z * speed;
      }
      if (this.keys["Space"]) pos.y += speed;
      if (this.keys["ShiftLeft"]) pos.y -= speed;

      this.camera.setLocalPosition(pos.x, pos.y, pos.z);
    }
  };

  setFPSControls(enabled: boolean) {
    this.fpsMode = enabled;
    if (enabled) {
      // Enable pointer lock for mouse look
      this.canvas.requestPointerLock();
      // Add mouse move handler for looking around
      document.addEventListener("mousemove", this._onMouseMove);
    } else {
      document.exitPointerLock();
      document.removeEventListener("mousemove", this._onMouseMove);
    }
  }

  private _onMouseMove = (e: MouseEvent) => {
    if (!this.fpsMode || document.pointerLockElement !== this.canvas) return;
    const sensitivity = 0.002;
    const euler = this.camera.getLocalEulerAngles();
    euler.x -= e.movementY * sensitivity * 57.3; // rad to deg
    euler.y -= e.movementX * sensitivity * 57.3;
    euler.x = Math.max(-89, Math.min(89, euler.x));
    this.camera.setLocalEulerAngles(euler.x, euler.y, 0);
  };

  enterVR() {
    if (!this.app) return;
    try {
      if (this.app.xr.supported) {
        this.app.xr.start(this.camera.camera, "immersive-vr", "local-floor");
      }
    } catch (e) {
      console.warn("WebXR not available:", e);
    }
  }

  getCamera(): any {
    return this.camera;
  }

  setCameraTransform(position: number[], lookAt: number[]): void {
    if (!this.camera) return;
    this.camera.setLocalPosition(position[0], position[1], position[2]);
    this.camera.lookAt(lookAt[0], lookAt[1], lookAt[2]);
  }

  getCanvas(): HTMLCanvasElement | null {
    return this.canvas;
  }

  getStats(): RendererStats {
    let splatCount = 0;
    try {
      const resource = this.splatEntity?.gsplat?.resource;
      if (resource?.splatData) {
        splatCount = resource.splatData.numSplats ?? 0;
      }
    } catch { /* ignore */ }

    const deviceType = this.app?.graphicsDevice?.deviceType ?? "unknown";
    return {
      fps: this._fps,
      splatCount,
      drawCalls: this.app?.graphicsDevice?.renderPassCount ?? 0,
      rendererType: `PlayCanvas (${deviceType === "webgpu" ? "WebGPU" : "WebGL2"})`,
    };
  }

  resetCamera(): void {
    this._centerCamera();
  }

  setBackgroundColor(hex: string): void {
    if (!this.camera?.camera) return;
    // Parse hex to RGB
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    this.camera.camera.clearColor = new pc.Color(r, g, b, 1);
  }

  setLighting(direction: [number, number, number], intensity: number, ambient: number): void {
    if (!this._lightEntity) return;
    this._lightEntity.light.intensity = intensity;
    // Position directional light based on direction vector
    this._lightEntity.setEulerAngles(
      Math.asin(direction[1]) * 57.3,
      Math.atan2(direction[0], direction[2]) * 57.3,
      0,
    );
    // Set scene ambient
    if (this.app?.scene) {
      this.app.scene.ambientLight = new pc.Color(ambient, ambient, ambient);
    }
  }

  dispose() {
    this._disposed = true;
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    document.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("click", this._onCanvasClick);
    if ((this as any)._onResize) {
      window.removeEventListener("resize", (this as any)._onResize);
    }
    try {
      this.app?.destroy();
    } catch { /* */ }
    if (this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
  }
}
