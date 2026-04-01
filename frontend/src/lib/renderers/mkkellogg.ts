/**
 * mkkellogg.ts — Three.js/WebGL renderer using @mkkellogg/gaussian-splats-3d Viewer
 * in selfDrivenMode: false with custom camera + WASD controls.
 *
 * Supports WASD + PointerLock first-person controls and WebXR VR mode.
 */

import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";

export interface RendererStats {
  fps: number;
  splatCount: number;
  drawCalls: number;
  rendererType: string;
}

export interface SplatRenderer {
  type: "webgpu" | "webgl";
  load(url: string, format?: number): Promise<void>;
  setFPSControls(enabled: boolean): void;
  enterVR(): void;
  dispose(): void;
  getCamera(): THREE.Camera | null;
  setCameraTransform(position: number[], lookAt: number[]): void;
  getCanvas(): HTMLCanvasElement | null;
  getStats(): RendererStats;
  resetCamera(): void;
  setBackgroundColor(hex: string): void;
  setLighting(direction: [number, number, number], intensity: number, ambient: number): void;
}

export class MkkelloggRenderer implements SplatRenderer {
  readonly type = "webgl" as const;

  private renderer: THREE.WebGLRenderer;
  private camera: THREE.PerspectiveCamera;
  private controls: PointerLockControls;
  private keys: Record<string, boolean> = {};
  private vrBtn: HTMLElement | null = null;
  private viewer: any = null;
  private _lastFrameTime = performance.now();
  private _fps = 0;
  private _fpsFrames = 0;
  private _fpsAccum = 0;
  private _dirLight: THREE.DirectionalLight | null = null;
  private _ambLight: THREE.AmbientLight | null = null;

  constructor(private container: HTMLElement) {
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.01,
      10000
    );
    this.camera.position.set(0, 1.6, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.controls = new PointerLockControls(this.camera, this.renderer.domElement);
    this.renderer.domElement.addEventListener("click", () => this.controls.lock());

    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);

    const onResize = () => {
      this.camera.aspect = container.clientWidth / container.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);
    (this as any)._onResize = onResize;
  }

  private _onKeyDown = (e: KeyboardEvent) => { this.keys[e.code] = true; };
  private _onKeyUp   = (e: KeyboardEvent) => { this.keys[e.code] = false; };

  private _startLoop() {
    const speed = 0.05;
    this.renderer.setAnimationLoop(() => {
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

      // WASD movement
      if (this.controls.isLocked) {
        if (this.keys["KeyW"] || this.keys["ArrowUp"])    this.controls.moveForward(speed);
        if (this.keys["KeyS"] || this.keys["ArrowDown"])  this.controls.moveForward(-speed);
        if (this.keys["KeyA"] || this.keys["ArrowLeft"])  this.controls.moveRight(-speed);
        if (this.keys["KeyD"] || this.keys["ArrowRight"]) this.controls.moveRight(speed);
        if (this.keys["Space"])     this.camera.position.y += speed;
        if (this.keys["ShiftLeft"]) this.camera.position.y -= speed;
      }

      // Viewer update + render (selfDrivenMode: false)
      if (this.viewer) {
        this.viewer.update();
        this.viewer.render();
      }
    });
  }

  async load(url: string, format?: number): Promise<void> {
    const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d") as any;

    // Use Viewer with selfDrivenMode: false — we drive update/render manually
    this.viewer = new GaussianSplats3D.Viewer({
      selfDrivenMode: false,
      renderer: this.renderer,
      camera: this.camera,
      rootElement: this.container,
      useBuiltInControls: false,
      gpuAcceleratedSort: true,
      sharedMemoryForWorkers: false,
      dynamicScene: false,
      antialiased: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
      logLevel: GaussianSplats3D.LogLevel.None,
      sphericalHarmonicsDegree: 2,
    });

    const sceneOpts: any = {
      progressiveLoad: true,
    };
    if (format !== undefined) {
      sceneOpts.format = format;
    }

    await this.viewer.addSplatScene(url, sceneOpts);
    this.viewer.start();

    // Center camera on the loaded splat
    try {
      const mesh = this.viewer.splatMesh;
      if (mesh) {
        mesh.geometry?.computeBoundingBox?.();
        const box = mesh.geometry?.boundingBox ?? mesh.boundingBox;
        if (box) {
          const center = new THREE.Vector3();
          box.getCenter(center);
          const size = new THREE.Vector3();
          box.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z);
          this.camera.position.copy(center);
          this.camera.position.z += maxDim * 0.5;
          this.camera.position.y += maxDim * 0.2;
          this.camera.far = maxDim * 10;
          this.camera.updateProjectionMatrix();
        }
      }
    } catch { /* best effort */ }

    // Start render loop after scene is loaded
    this._startLoop();
  }

  setFPSControls(enabled: boolean) {
    if (enabled) {
      this.controls.lock();
    } else {
      this.controls.unlock();
    }
  }

  enterVR() {
    if (!this.vrBtn) {
      this.vrBtn = VRButton.createButton(this.renderer);
      this.vrBtn.style.position = "absolute";
      this.vrBtn.style.bottom = "20px";
      this.vrBtn.style.right = "20px";
      this.container.appendChild(this.vrBtn);
    }
    (this.vrBtn as HTMLButtonElement).click();
  }

  getCamera(): THREE.Camera | null {
    return this.camera;
  }

  setCameraTransform(position: number[], lookAt: number[]): void {
    this.camera.position.set(position[0], position[1], position[2]);
    this.camera.lookAt(lookAt[0], lookAt[1], lookAt[2]);
  }

  getCanvas(): HTMLCanvasElement | null {
    return this.renderer.domElement;
  }

  getStats(): RendererStats {
    const info = this.renderer.info;
    let splatCount = 0;
    try {
      const mesh = this.viewer?.splatMesh;
      if (mesh?.getSplatCount) {
        splatCount = mesh.getSplatCount();
      }
    } catch { /* ignore */ }
    return {
      fps: this._fps,
      splatCount,
      drawCalls: info.render?.calls ?? 0,
      rendererType: "WebGL (mkkellogg)",
    };
  }

  resetCamera(): void {
    this.camera.position.set(0, 1.6, 3);
    this.camera.lookAt(0, 0, 0);
  }

  setBackgroundColor(hex: string): void {
    try {
      this.renderer.setClearColor(new THREE.Color(hex), 1);
    } catch { /* best effort */ }
  }

  setLighting(direction: [number, number, number], intensity: number, ambient: number): void {
    // Add lights to the viewer's scene (accessed through splatMesh parent)
    const scene = this.viewer?.splatMesh?.parent;
    if (!scene) return;

    if (!this._dirLight) {
      this._dirLight = new THREE.DirectionalLight(0xffffff, intensity);
      this._ambLight = new THREE.AmbientLight(0xffffff, ambient);
      scene.add(this._dirLight);
      scene.add(this._ambLight);
    }
    this._dirLight.position.set(direction[0] * 100, direction[1] * 100, direction[2] * 100);
    this._dirLight.intensity = intensity;
    if (this._ambLight) this._ambLight.intensity = ambient;
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    if ((this as any)._onResize) {
      window.removeEventListener("resize", (this as any)._onResize);
    }
    if (this.vrBtn?.parentNode) {
      this.vrBtn.parentNode.removeChild(this.vrBtn);
    }
    this.controls.dispose();
    try { this.viewer?.dispose(); } catch { /* */ }
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
