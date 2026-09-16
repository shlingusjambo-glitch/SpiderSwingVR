/** Skinned GLB avatars: Vapour's importer drops skins, so joints, weights and
 * inverse binds are read straight from the GLB here, and the avatar is posed by
 * world-space joint matrices (bind pose × root, with two-bone IK for limbs). */
import { importGltf, solveTwoBoneIk, uploadGltfAsset, type DrawOptions, type FrameBuilder, type WebGpuRenderHost } from "@vapour/engine";
import { compose, invert, multiplyMat4, transformPoint, type Mat4, type Vec3 } from "@vapour/math";

export interface SkinnedModel {
  readonly draws: readonly { mesh: string; options: DrawOptions }[];
  /** Bind-pose joint matrices in model space (inverse of the inverse binds). */
  readonly bind: readonly Mat4[];
  readonly ibm: readonly Mat4[];
  /** Joint index by name. */
  readonly joint: Record<string, number>;
  readonly parent: readonly number[];
}

export async function loadSkinnedGlb(host: WebGpuRenderHost, url: URL, id: string): Promise<SkinnedModel> {
  const bytes = await (await fetch(url)).arrayBuffer();
  const view = new DataView(bytes), jsonLength = view.getUint32(12, true), json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, jsonLength))), binOffset = 20 + jsonLength + 8;
  const accessor = (index: number): Float32Array | Uint16Array | Uint8Array => {
    const a = json.accessors[index], bv = json.bufferViews[a.bufferView], offset = binOffset + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const count = a.count * ({ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 } as Record<string, number>)[a.type]!;
    return a.componentType === 5126 ? new Float32Array(bytes, offset, count) : a.componentType === 5123 ? new Uint16Array(bytes, offset, count) : new Uint8Array(bytes, offset, count);
  };
  const model = await importGltf(bytes, { sourceName: url.pathname });
  const asset = await uploadGltfAsset(host, model, { id, maxTextureSize: 1024 });
  const draws: { mesh: string; options: DrawOptions }[] = [];
  for (const [meshIndex, mesh] of (json.meshes as { primitives: { attributes: Record<string, number> }[] }[]).entries()) for (const [p, prim] of mesh.primitives.entries()) {
    const index = model.meshPrimitives[meshIndex]![p]!, original = asset.meshIds[index]!, skinned = `${id}:skinned:${index}`;
    if (prim.attributes["JOINTS_0"] === undefined) { draws.push({ mesh: original, options: asset.draws.find((d) => d.mesh === original)?.options ?? {} }); continue; }
    // a fresh id: replacing a mesh in place keeps its unskinned vertex layout
    host.uploadMesh(skinned, { ...model.primitives[index]!.mesh, boneIndices: Uint32Array.from(accessor(prim.attributes["JOINTS_0"]!)), boneWeights: Float32Array.from(accessor(prim.attributes["WEIGHTS_0"]!)) });
    host.releaseMesh(original);
    draws.push({ mesh: skinned, options: asset.draws.find((d) => d.mesh === original)?.options ?? {} });
  }
  const skin = json.skins[0] as { joints: number[]; inverseBindMatrices: number }, flat = accessor(skin.inverseBindMatrices) as Float32Array;
  const ibm: Mat4[] = skin.joints.map((_, i) => Array.from(flat.subarray(i * 16, i * 16 + 16)));
  const joint: Record<string, number> = {}, parent: number[] = skin.joints.map(() => -1);
  skin.joints.forEach((node, i) => { const n = json.nodes[node] as { name: string; children?: number[] }; joint[n.name] = i; for (const c of n.children ?? []) { const ci = skin.joints.indexOf(c); if (ci >= 0) parent[ci] = i; } });
  return { draws, bind: ibm.map((mat) => invert(mat)), ibm, joint, parent };
}

const Q_ID = [0, 0, 0, 1] as const;
const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** World joint matrices for the bind pose under `root` (model space → world). */
export function bindPose(model: SkinnedModel, root: Mat4): Mat4[] { return model.bind.map((b) => multiplyMat4(root, b)); }

/** Bends the chain shoulder → elbow → wrist so the wrist lands on `target`; descendants of the elbow follow. */
export function reachFor(model: SkinnedModel, world: Mat4[], shoulder: number, elbow: number, wrist: number, target: Vec3, pole: Vec3): void {
  const r = translation(world[shoulder]!), mid = translation(world[elbow]!), tip = translation(world[wrist]!);
  let ik;
  try { ik = solveTwoBoneIk({ root: r, mid, tip, target, pole }); } catch { return; }
  const about = (p: Vec3, q: readonly number[]): Mat4 => multiplyMat4(compose(p, q, [1, 1, 1]), compose([-p[0], -p[1], -p[2]], Q_ID, [1, 1, 1]));
  const dRoot = about(r, ik.rootRotation);
  const dMid = multiplyMat4(about(transformPoint(dRoot, mid), ik.midRotation), dRoot);
  for (let j = 0; j < world.length; j += 1) {
    if (j === shoulder) world[j] = multiplyMat4(dRoot, world[j]!);
    else if (j === elbow || isDescendant(j, elbow, model.parent)) world[j] = multiplyMat4(dMid, world[j]!);
  }
}

/** Collapses a joint (and everything it drives) to a point — the way to hide a first-person head. */
export function collapse(world: Mat4[], joint: number, at: Vec3): void { world[joint] = compose(at, Q_ID, [0.001, 0.001, 0.001]); }

export function drawSkinned(fb: FrameBuilder, model: SkinnedModel, world: readonly Mat4[], tint?: readonly [number, number, number, number]): void {
  const palette = new Float32Array(world.length * 16); // per draw: the frame builder keeps a reference until it packs
  for (let j = 0; j < world.length; j += 1) palette.set(multiplyMat4(world[j]!, model.ibm[j]!), j * 16);
  for (const d of model.draws) fb.draw(d.mesh, IDENTITY, { ...d.options, bones: palette, ...(tint ? { color: tint } : {}) });
}

export function isDescendant(j: number, ancestor: number, parent: readonly number[]): boolean { for (let p = parent[j]!; p >= 0; p = parent[p]!) if (p === ancestor) return true; return false; }
export function translation(mat: Mat4): Vec3 { return [mat[12]!, mat[13]!, mat[14]!]; }
