/// <reference path="./webxr.d.ts" />
/** Stereo presentation for a Vapour WebGPU host on a WebXR session.
 *
 * Uses the WebXR/WebGPU binding when the browser exposes it, otherwise an
 * XRWebGLLayer fed by blitting the WebGPU canvas. Both go through the engine's
 * `XrPresentation` in multiview mode: both eyes side by side in one submission
 * (this needs the patched runtime: see engine-patches/README.md). */
import { XrPresentation, canvasSubImageCopier, type WebGpuRenderHost } from "@vapour/engine";

export type Presenter = { present: XrPresentation<unknown, XRView>["present"] };

export async function createPresentation(session: XRSession, host: WebGpuRenderHost, canvas: HTMLCanvasElement, scale: number): Promise<Presenter> {
  const XRGPUBinding = (globalThis as { XRGPUBinding?: new (s: XRSession, d: GPUDevice) => never }).XRGPUBinding;
  if (XRGPUBinding !== undefined) {
    const adapter = await navigator.gpu.requestAdapter({ xrCompatible: true } as GPURequestAdapterOptions);
    const device = await adapter!.requestDevice();
    const binding = new XRGPUBinding(session, device);
    const p = new XrPresentation<unknown, XRView>({ host, binding, session, copy: canvasSubImageCopier(device, canvas), scaleFactor: scale });
    return { present: (views, rigFrame, frame, t) => p.present(views, rigFrame, frame, t) };
  }
  const gl = document.createElement("canvas").getContext("webgl2", { xrCompatible: true, alpha: false, antialias: false } as WebGLContextAttributes) as WebGL2RenderingContext;
  await gl.makeXRCompatible();
  const layer = new XRWebGLLayer(session, gl, { framebufferScaleFactor: scale });
  session.updateRenderState({ baseLayer: layer });
  const blit = canvasBlitter(gl);
  // a fake binding that accepts "texture-array" so XrPresentation renders both eyes in one submission; the copier does the WebGL blit
  const binding = {
    createProjectionLayer: () => layer,
    getViewSubImage: (_l: unknown, view: XRView) => { const v = layer.getViewport(view)!; return { colorTexture: { width: layer.framebufferWidth, height: layer.framebufferHeight }, viewport: { x: v.x, y: v.y, width: v.width, height: v.height }, imageIndex: 0 }; },
    getPreferredColorFormat: () => "rgba8unorm" as const,
  };
  const p = new XrPresentation<unknown, XRView>({ host, binding, session: { updateRenderState() {} }, copy: ({ sourceRect, destinationRect }) => blit(layer.framebuffer, canvas, sourceRect, destinationRect) });
  return { present: (views, rigFrame, frame, t) => { const r = p.present(views, rigFrame, frame, t); gl.flush(); return r; } };
}

function canvasBlitter(gl: WebGL2RenderingContext) {
  const compile = (type: number, src: string) => { const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram()!;
  // highp: mediump UVs are 16-bit on the Quest's GPU and quantize a 2700px-wide texture into visible blocks
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, "#version 300 es\nprecision highp float;in vec2 p;uniform vec4 rect;out vec2 uv;void main(){vec2 t=(p+1.0)*0.5;uv=vec2(rect.x+t.x*rect.z,rect.y+(1.0-t.y)*rect.w);gl_Position=vec4(p,0.0,1.0);}"));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, "#version 300 es\nprecision highp float;uniform sampler2D img;in vec2 uv;out vec4 c;void main(){c=texture(img,uv);}"));
  gl.linkProgram(prog);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k!, v!);
  const rectLoc = gl.getUniformLocation(prog, "rect");
  return (framebuffer: WebGLFramebuffer | null, source: HTMLCanvasElement, src: { x: number; y: number; width: number; height: number }, dst: { x: number; y: number; width: number; height: number }) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    // one canvas upload per frame (both eyes are side by side in it); the first eye's copy starts at x = 0
    if (src.x === 0) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.uniform4f(rectLoc, src.x / source.width, src.y / source.height, src.width / source.width, src.height / source.height);
    gl.viewport(dst.x, dst.y, dst.width, dst.height);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}

/** Yaw (radians, about +Y) of a column-major transform whose forward is -Z. */
export function yawOf(mat: ArrayLike<number>): number { return Math.atan2(mat[8]!, mat[10]!); }
