// Minimal WebXR surface the game touches; the bundled TypeScript lib has no WebXR types.
interface XRRigidTransform { readonly matrix: Float32Array; }
interface XRView { readonly eye: "none" | "left" | "right"; readonly transform: XRRigidTransform; readonly projectionMatrix: Float32Array; }
interface XRViewerPose { readonly transform: XRRigidTransform; readonly views: readonly XRView[]; }
interface XRPose { readonly transform: XRRigidTransform; }
interface XRSpace {}
interface XRFrame { getViewerPose(space: XRSpace): XRViewerPose | null; getPose(space: XRSpace, base: XRSpace): XRPose | null; }
interface XRInputSource { readonly handedness: "none" | "left" | "right"; readonly targetRaySpace: XRSpace; readonly gripSpace?: XRSpace; readonly gamepad?: Gamepad & { hapticActuators?: readonly { pulse(value: number, duration: number): Promise<boolean> }[] }; }
interface XRSession {
  readonly inputSources: readonly XRInputSource[];
  readonly visibilityState: string;
  requestReferenceSpace(type: string): Promise<XRSpace>;
  requestAnimationFrame(callback: (time: number, frame: XRFrame) => void): number;
  updateRenderState(state: { baseLayer?: XRWebGLLayer; layers?: readonly unknown[] }): void;
}
interface XRSystem { isSessionSupported(mode: string): Promise<boolean>; requestSession(mode: string, options?: { requiredFeatures?: string[]; optionalFeatures?: string[] }): Promise<XRSession>; }
interface Navigator { readonly xr?: XRSystem; }
interface XRWebGLLayer { readonly framebuffer: WebGLFramebuffer | null; readonly framebufferWidth: number; readonly framebufferHeight: number; getViewport(view: XRView): { x: number; y: number; width: number; height: number } | null; }
declare const XRWebGLLayer: new (session: XRSession, gl: WebGL2RenderingContext, options?: { framebufferScaleFactor?: number }) => XRWebGLLayer;
interface WebGL2RenderingContext { makeXRCompatible(): Promise<void>; }
