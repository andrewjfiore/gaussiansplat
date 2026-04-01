/**
 * cameraPath.ts — Camera path animation with Catmull-Rom spline interpolation.
 *
 * Defines keyframes (position + lookAt + fov + timestamp) and provides
 * smooth interpolation for cinematic flythrough animations.
 */

import * as THREE from "three";

export interface CameraKeyframe {
  position: [number, number, number];
  lookAt: [number, number, number];
  fov: number;
  /** Normalized time in [0, 1] within the path duration. */
  t: number;
}

export class CameraPath {
  keyframes: CameraKeyframe[] = [];

  addKeyframe(kf: CameraKeyframe) {
    this.keyframes.push(kf);
    // Keep sorted by t
    this.keyframes.sort((a, b) => a.t - b.t);
  }

  removeKeyframe(index: number) {
    this.keyframes.splice(index, 1);
  }

  get length() {
    return this.keyframes.length;
  }

  /**
   * Interpolate camera state at normalized time t in [0, 1].
   * Uses Catmull-Rom spline for position and lookAt.
   */
  interpolate(t: number): { position: THREE.Vector3; lookAt: THREE.Vector3; fov: number } | null {
    if (this.keyframes.length < 2) return null;
    t = Math.max(0, Math.min(1, t));

    const posPoints = this.keyframes.map(
      (kf) => new THREE.Vector3(kf.position[0], kf.position[1], kf.position[2])
    );
    const lookAtPoints = this.keyframes.map(
      (kf) => new THREE.Vector3(kf.lookAt[0], kf.lookAt[1], kf.lookAt[2])
    );

    const posCurve = new THREE.CatmullRomCurve3(posPoints);
    const lookAtCurve = new THREE.CatmullRomCurve3(lookAtPoints);

    // Remap t based on keyframe timestamps
    const remapped = this._remapTime(t);

    const position = posCurve.getPoint(remapped);
    const lookAt = lookAtCurve.getPoint(remapped);

    // Lerp FOV
    const fov = this._lerpFov(remapped);

    return { position, lookAt, fov };
  }

  private _remapTime(t: number): number {
    if (this.keyframes.length < 2) return t;
    // Map global t to the keyframe t-space
    const kfs = this.keyframes;
    const totalT = kfs[kfs.length - 1].t - kfs[0].t;
    if (totalT <= 0) return 0;
    return (t * totalT + kfs[0].t - kfs[0].t) / totalT;
  }

  private _lerpFov(t: number): number {
    const kfs = this.keyframes;
    if (kfs.length === 0) return 75;
    if (kfs.length === 1) return kfs[0].fov;

    // Find the two surrounding keyframes
    const scaledIdx = t * (kfs.length - 1);
    const lo = Math.floor(scaledIdx);
    const hi = Math.min(lo + 1, kfs.length - 1);
    const frac = scaledIdx - lo;
    return kfs[lo].fov * (1 - frac) + kfs[hi].fov * frac;
  }

  toJSON(): CameraKeyframe[] {
    return [...this.keyframes];
  }

  static fromJSON(data: CameraKeyframe[]): CameraPath {
    const path = new CameraPath();
    path.keyframes = [...data];
    return path;
  }
}
