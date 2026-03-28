declare module "@mkkellogg/gaussian-splats-3d" {
  export class Viewer {
    constructor(options?: {
      cameraUp?: number[];
      initialCameraPosition?: number[];
      initialCameraLookAt?: number[];
      rootElement?: HTMLElement;
      sharedMemoryForWorkers?: boolean;
      [key: string]: any;
    });
    addSplatScene(url: string, options?: {
      showLoadingUI?: boolean;
      progressiveLoad?: boolean;
      [key: string]: any;
    }): Promise<void>;
    start(): void;
    dispose(): void;
  }
}
