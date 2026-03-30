/**
 * mkkellogg.ts — Three.js/WebGL renderer wrapper using @mkkellogg/gaussian-splats-3d.
 *
 * Supports WASD + PointerLock first-person controls and WebXR VR mode.
 * Used as the fallback renderer when WebGPU is unavailable, or as the
 * primary renderer when VR is requested (Visionary does not support WebXR).
 */

import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";

export interface SplatRenderer {
  type: "webgpu" | "webgl";
  load(url: string): Promise<void>;
  setFPSControls(enabled: boolean): void;
  enterVR(): void;
  dispose(): void;
}

export class MkkelloggRenderer implements SplatRenderer {
  readonly type = "webgl" as const;

  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: PointerLockControls;
  private keys: Record<string, boolean> = {};
  private animFrameId: number | null = null;
  private vrBtn: HTMLElement | null = null;
  private splat: any = null;

  constructor(private container: HTMLElement) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.01,
      1000
    );
    this.camera.position.set(0, 1.6, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.controls = new PointerLockControls(this.camera, this.renderer.domElement);
    this.renderer.domElement.addEventListener("click", () => this.controls.lock());

    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);

    this._startLoop();

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
      if (this.controls.isLocked) {
        if (this.keys["KeyW"] || this.keys["ArrowUp"])    this.controls.moveForward(speed);
        if (this.keys["KeyS"] || this.keys["ArrowDown"])  this.controls.moveForward(-speed);
        if (this.keys["KeyA"] || this.keys["ArrowLeft"])  this.controls.moveRight(-speed);
        if (this.keys["KeyD"] || this.keys["ArrowRight"]) this.controls.moveRight(speed);
        if (this.keys["Space"])     this.camera.position.y += speed;
        if (this.keys["ShiftLeft"]) this.camera.position.y -= speed;
      }
      this.renderer.render(this.scene, this.camera);
    });
  }

  async load(url: string): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mod = await import("@mkkellogg/gaussian-splats-3d") as any;
    const GaussianSplatMesh = mod.GaussianSplatMesh ?? mod.default?.GaussianSplatMesh;
    const WebXRMode = mod.WebXRMode ?? mod.default?.WebXRMode;
    this.splat = new GaussianSplatMesh(this.renderer, this.scene, this.camera, {
      webXRMode: WebXRMode?.VR ?? 1,
    });
    await this.splat.addSplatScene(url, { progressiveLoad: true });
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
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
