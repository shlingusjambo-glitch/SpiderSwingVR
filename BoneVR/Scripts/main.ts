import { AudioEngine, FrameBuilder, GrabController, WebGpuRenderHost, XrRig, composeMatrix, createPhysicsWorld3d, importGltf, uploadGltfAsset, type GltfAsset, type PhysicsBodySnapshot3d, type PhysicsCommand3d, type PhysicsWorld3dBridge, type XrFrameSnapshot } from "@vapour/engine";
import { compose, invert, multiplyMat4, multiplyQuat, quatFromAxisAngle, quatFromEuler, rotateVector, transformPoint, type Mat4, type Quat, type Vec3 } from "@vapour/math";
import { createPresentation, yawOf } from "../../shared/xr-present.js";
import { bindPose, collapse, drawSkinned, loadSkinnedGlb, reachFor, type SkinnedModel } from "../../shared/skinned-glb.js";

// ---------- tuning knobs ----------
const XR_SCALE = 0.7;          // ponytail: eye-buffer scale for Quest 2
const WALK_SPEED = 3.5;
const JUMP = 5;
const GRAVITY = 9.81;
const GRAB_REACH = 0.22;       // metres from the palm a prop can be grabbed
const BULLET_SPEED = 45;
const FORD_SCALE = 0.0254;     // the model is in inches
const FORD_YAW = -Math.PI / 2; // ponytail: model facing; adjust if Ford faces the wrong way
const FORD_UPRIGHT: Quat = quatFromAxisAngle([0, 0, 1], Math.PI / 2); // the FBX came in lying on its side: its +X is up
const GUN_LENGTH = 0.9;        // the KMP-60 model's barrel runs along +Z

// ---------- world ----------
const canvas = document.querySelector<HTMLCanvasElement>("#vapour-game")!;
const moduleUrl = new URL("vapour_runtime.js", document.baseURI).href;
const host = new WebGpuRenderHost(canvas, { moduleUrl });
host.setPostProcess({ exposure: 1, toneMapping: "aces", antiAliasing: "none" });
const fb = new FrameBuilder(512);
const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const Q_ID: Quat = [0, 0, 0, 1];
let physics: PhysicsWorld3dBridge, ford: SkinnedModel, gun: GltfAsset;

// the level: axis-aligned boxes, drawn as cubes and inserted as static bodies
type Color = [number, number, number, number];
const level: { pos: Vec3; size: Vec3; color: Color }[] = [
  { pos: [0, -0.5, 0], size: [40, 1, 40], color: [0.45, 0.45, 0.48, 1] },
  { pos: [0, 3, -20], size: [40, 8, 1], color: [0.7, 0.68, 0.62, 1] }, { pos: [0, 3, 20], size: [40, 8, 1], color: [0.7, 0.68, 0.62, 1] },
  { pos: [-20, 3, 0], size: [1, 8, 40], color: [0.7, 0.68, 0.62, 1] }, { pos: [20, 3, 0], size: [1, 8, 40], color: [0.7, 0.68, 0.62, 1] },
  { pos: [6, 0.5, -6], size: [4, 1, 4], color: [0.85, 0.5, 0.2, 1] }, { pos: [8, 1.5, -8], size: [4, 1, 4], color: [0.85, 0.5, 0.2, 1] }, { pos: [10, 2.5, -10], size: [4, 1, 4], color: [0.85, 0.5, 0.2, 1] },
  { pos: [-8, 1, 6], size: [6, 2, 3], color: [0.3, 0.55, 0.85, 1] }, { pos: [-8, 2.5, 8.5], size: [6, 1, 2], color: [0.3, 0.55, 0.85, 1] },
  { pos: [0, 2, -12], size: [12, 0.3, 2], color: [0.9, 0.9, 0.9, 1] }, { pos: [-6, 1, -12], size: [0.4, 2, 2], color: [0.9, 0.9, 0.9, 1] }, { pos: [6, 1, -12], size: [0.4, 2, 2], color: [0.9, 0.9, 0.9, 1] },
  { pos: [0, 0.45, -1.5], size: [0.8, 0.9, 0.5], color: [0.35, 0.3, 0.28, 1] }, // the gun's table
  { pos: [12, 1.5, 12], size: [1, 3, 1], color: [0.6, 0.2, 0.2, 1] }, { pos: [-12, 1.5, -12], size: [1, 3, 1], color: [0.6, 0.2, 0.2, 1] }, { pos: [-14, 1.5, 12], size: [1, 3, 1], color: [0.6, 0.2, 0.2, 1] },
];
// props: dynamic bodies you can grab and throw
interface Prop { id: number; shape: { type: "box"; halfExtents: Vec3 } | { type: "sphere"; radius: number } | { type: "cylinder"; halfHeight: number; radius: number }; color: Color; }
const props: Prop[] = [];
const GUN_ID = 200;
const bullets: { id: number; born: number }[] = [];
let nextBullet = 300;
const poses = new Map<number, PhysicsBodySnapshot3d>();

function buildWorld(): void {
  level.forEach((b, i) => physics.insert({ id: 1 + i, bodyType: "static", position: b.pos, rotation: Q_ID, shape: { type: "box", halfExtents: [b.size[0] / 2, b.size[1] / 2, b.size[2] / 2] }, friction: 0.9 }));
  let id = 100;
  const add = (shape: Prop["shape"], color: Color, pos: Vec3, density = 1): void => { props.push({ id, shape, color }); physics.insert({ id: id++, bodyType: "dynamic", position: pos, rotation: Q_ID, shape, density, friction: 0.8, linearDamping: 0.05, angularDamping: 0.2 }); };
  for (let i = 0; i < 8; i += 1) add({ type: "box", halfExtents: [0.2, 0.2, 0.2] }, [0.75, 0.55, 0.3, 1], [-3 + (i % 4) * 0.6, 0.3 + Math.floor(i / 4) * 0.5, -3]);
  for (let i = 0; i < 3; i += 1) add({ type: "box", halfExtents: [0.45, 0.45, 0.45] }, [0.55, 0.4, 0.25, 1], [3 + i * 1.2, 0.5, 2], 0.6);
  for (let i = 0; i < 5; i += 1) add({ type: "sphere", radius: 0.18 }, [0.9, 0.2, 0.2, 1], [-2 + i * 0.5, 0.3, 3], 0.4);
  for (let i = 0; i < 4; i += 1) add({ type: "cylinder", halfHeight: 0.42, radius: 0.28 }, [0.25, 0.5, 0.3, 1], [4, 0.5, -3 + i * 0.8], 0.7);
  add({ type: "sphere", radius: 0.5 }, [0.2, 0.3, 0.8, 1], [-5, 1, -5], 0.15); // beach ball
  physics.insert({ id: GUN_ID, bodyType: "dynamic", position: [0, 1.05, -1.5], rotation: quatFromAxisAngle([0, 1, 0], Math.PI / 2), shape: { type: "box", halfExtents: [0.04, 0.1, GUN_LENGTH / 2] }, density: 2, friction: 0.9, angularDamping: 0.5 });
  physics.flush(); physics.step(1 / 60);
}

// ---------- player ----------
interface Hand { pos: Vec3; rot: Quat; prev: Vec3; held?: number; holdRot: Quat; trigger: boolean; }
const player = { pos: [0, 0, 3] as Vec3, vy: 0, grounded: true };
const hands: Record<"left" | "right", Hand> = { left: { pos: [-0.3, 1.2, 2.5], rot: Q_ID, prev: [-0.3, 1.2, 2.5], holdRot: Q_ID, trigger: false }, right: { pos: [0.3, 1.2, 2.5], rot: Q_ID, prev: [0.3, 1.2, 2.5], holdRot: Q_ID, trigger: false } };
const grab = new GrabController();
let pulse: (hand: "left" | "right", strength: number, ms: number) => void = () => {};
(globalThis as { bone?: unknown }).bone = { player, hands, get physics() { return physics; }, poses }; // console debugging

const audioContext = new AudioContext();
const audio = new AudioEngine(audioContext);
{
  const buffer = audioContext.createBuffer(1, Math.floor(audioContext.sampleRate * 0.3), audioContext.sampleRate), data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) { const t = i / audioContext.sampleRate; data[i] = Math.exp(-t * 18) * (Math.random() * 2 - 1) * (t < 0.01 ? 1 : 0.6) + Math.exp(-t * 40) * Math.sin(t * 90 * Math.PI * 2) * 0.8; }
  audio.registerClip("shot", buffer);
}

function applyCommands(commands: readonly PhysicsCommand3d[]): void {
  for (const c of commands) {
    if (c.type === "setLinearVelocity") physics.setLinearVelocity(c.id, c.velocity);
    else if (c.type === "setAngularVelocity") physics.setAngularVelocity(c.id, c.velocity);
    else if (c.type === "applyImpulse") physics.applyImpulse(c.id, c.impulse);
  }
}

function updateHand(hand: Hand, which: "left" | "right", gripHeld: boolean, triggerHeld: boolean, now: number): void {
  if (gripHeld && hand.held === undefined) {
    // the closest dynamic body within reach of the palm
    const candidates = physics.overlapShape(hand.pos, Q_ID, { type: "sphere", radius: GRAB_REACH }).filter((id) => poses.has(id) && !grab.isHeld(id));
    let best: number | undefined, bestD = Infinity;
    for (const id of candidates) { const d = dist(poses.get(id)!.position as Vec3, hand.pos); if (d < bestD) { bestD = d; best = id; } }
    if (best !== undefined) {
      const pose = poses.get(best)!;
      hand.held = best;
      // the gun snaps to aim along the controller; anything else keeps the orientation it was grabbed with
      hand.holdRot = best === GUN_ID ? quatFromAxisAngle([0, 1, 0], Math.PI) : multiplyQuat(invertQuat(hand.rot), pose.rotation as Quat);
      grab.grab(best, hand.pos, { stiffness: 50, damping: 10, maxSpeed: 25, breakDistance: 1.2, angularStiffness: 40 });
      pulse(which, 0.4, 30);
    }
  } else if (!gripHeld && hand.held !== undefined) { grab.release(hand.held); delete hand.held; }
  if (hand.held !== undefined) grab.moveHold(hand.held, hand.pos, multiplyQuat(hand.rot, hand.holdRot));
  // fire on the trigger's rising edge while holding the gun
  if (hand.held === GUN_ID && triggerHeld && !hand.trigger) fire(now, which);
  hand.trigger = triggerHeld;
}

function fire(now: number, which: "left" | "right"): void {
  const g = poses.get(GUN_ID);
  if (g === undefined) return;
  const fwd = rotateVector(g.rotation, [0, 0, 1]), muzzle = add(g.position as Vec3, fwd, GUN_LENGTH / 2 + 0.05);
  const id = nextBullet++;
  physics.insert({ id, bodyType: "dynamic", position: muzzle, rotation: Q_ID, shape: { type: "sphere", radius: 0.03 }, density: 200, linearVelocity: scaleV(fwd, BULLET_SPEED), continuousCollisionDetection: true, gravityScale: 0.3 });
  physics.applyImpulse(GUN_ID, scaleV(fwd, -0.05)); // recoil (the gun body weighs ~60 g)
  bullets.push({ id, born: now });
  pulse(which, 1, 60);
  try { audio.play("shot", { pitch: 0.95 + Math.random() * 0.1, volume: 0.7 }); } catch { /* not unlocked yet */ }
}

/** Moves the player capsule with the stick, gravity and jumps; steps the physics world. */
function simulate(dt: number, now: number, stick: [number, number], yaw: number, jump: boolean): void {
  const fwd = rotateVector(quatFromAxisAngle([0, 1, 0], yaw), [0, 0, -1]), right: Vec3 = [-fwd[2], 0, fwd[0]];
  if (jump && player.grounded) { player.vy = JUMP; player.grounded = false; }
  player.vy -= GRAVITY * dt;
  const desired: Vec3 = [(right[0] * stick[0] - fwd[0] * stick[1]) * WALK_SPEED * dt, player.vy * dt, (right[2] * stick[0] - fwd[2] * stick[1]) * WALK_SPEED * dt];
  const centre: Vec3 = [player.pos[0], player.pos[1] + 0.95, player.pos[2]];
  const move = physics.moveCharacter(dt, centre, Q_ID, desired, { shape: { type: "capsule", halfHeight: 0.55, radius: 0.35 }, offset: 0.02, slide: true, snapToGround: 0.2, autostepMaxHeight: 0.45, autostepMinWidth: 0.2, maxSlopeClimbDegrees: 50 });
  player.pos = add(player.pos, move.translation as Vec3, 1);
  player.grounded = move.grounded;
  if (move.grounded && player.vy < 0) player.vy = 0;
  if (player.pos[1] < -20) { player.pos = [0, 0, 3]; player.vy = 0; }
  // held props chase the hands, bullets expire, then the world steps
  applyCommands(grab.update(poses, dt).commands);
  for (const b of [...bullets]) if (now - b.born > 2.5) { physics.remove(b.id); bullets.splice(bullets.indexOf(b), 1); }
  const step = physics.step(dt);
  poses.clear();
  for (const body of step.bodies) poses.set(body.id, body);
}

// ---------- drawing ----------
const m = new Float32Array(16);
function drawWorld(head: Vec3, headRot: Quat, firstPerson: boolean): void {
  fb.setEnvironment({ clearColor: [0.06, 0.07, 0.1, 1], fog: { mode: "exponential", color: [0.08, 0.09, 0.12], density: 0.012 }, sky: { mode: "procedural", zenithColor: [0.03, 0.04, 0.08], horizonColor: [0.25, 0.22, 0.3], groundColor: [0.05, 0.05, 0.06], horizonCurve: 3 } });
  fb.lights.addHemisphere([0.9, 0.9, 1], [0.35, 0.3, 0.3], 1.5); // no directional light: it would turn on cascaded shadows
  fb.lights.addPoint([0, 6, 0], [1, 0.85, 0.7], 60, 30); fb.lights.addPoint([10, 5, -10], [0.5, 0.8, 1], 40, 25); fb.lights.addPoint([-10, 5, 8], [1, 0.5, 0.4], 40, 25);
  for (const b of level) fb.draw("builtin:cube", composeMatrix(m, b.pos, Q_ID, b.size), { color: b.color, roughness: 0.85 });
  for (const p of props) {
    const pose = poses.get(p.id); if (pose === undefined) continue;
    const s = p.shape;
    if (s.type === "box") fb.draw("builtin:cube", composeMatrix(m, pose.position, pose.rotation, [s.halfExtents[0] * 2, s.halfExtents[1] * 2, s.halfExtents[2] * 2]), { color: p.color, roughness: 0.7 });
    else if (s.type === "sphere") fb.draw("builtin:sphere", composeMatrix(m, pose.position, pose.rotation, [s.radius * 2, s.radius * 2, s.radius * 2]), { color: p.color, roughness: 0.4 });
    else fb.draw("builtin:cylinder", composeMatrix(m, pose.position, pose.rotation, [s.radius * 2, s.halfHeight * 2, s.radius * 2]), { color: p.color, roughness: 0.6, metallic: 0.4 });
  }
  for (const b of bullets) { const pose = poses.get(b.id); if (pose) fb.draw("builtin:sphere", composeMatrix(m, pose.position, Q_ID, [0.06, 0.06, 0.06]), { color: [1, 0.8, 0.3, 1], emissive: 2 }); }
  const g = poses.get(GUN_ID);
  if (g !== undefined) { const root = compose(g.position, g.rotation, [1, 1, 1]); for (const d of gun.draws) if (!d.name.includes("8x40mm")) fb.draw(d.mesh, multiplyMat4(root, d.matrix), d.options); }
  drawFord(head, headRot, firstPerson);
}

/** Ford: head on the HMD, arms reaching for the controllers, the rest in bind pose hanging below. */
function drawFord(head: Vec3, headRot: Quat, firstPerson: boolean): void {
  const j = (name: string) => ford.joint[`ValveBiped.${name}`]!;
  const yaw = quatFromAxisAngle([0, 1, 0], yawOfQuat(headRot));
  const modelRot = multiplyQuat(multiplyQuat(yaw, quatFromAxisAngle([0, 1, 0], FORD_YAW)), FORD_UPRIGHT);
  const root = multiplyMat4(compose(head, modelRot, [FORD_SCALE, FORD_SCALE, FORD_SCALE]), invert(ford.bind[j("Bip01_Head1")]!)); // head joint lands on the HMD
  const world = bindPose(ford, root);
  const back = rotateVector(yaw, [0, 0, 1]);
  reachFor(ford, world, j("Bip01_R_UpperArm"), j("Bip01_R_Forearm"), j("Bip01_R_Hand"), hands.right.pos, add(add(head, back, 0.5), rotateVector(yaw, [0.5, -0.6, 0]), 1));
  reachFor(ford, world, j("Bip01_L_UpperArm"), j("Bip01_L_Forearm"), j("Bip01_L_Hand"), hands.left.pos, add(add(head, back, 0.5), rotateVector(yaw, [-0.5, -0.6, 0]), 1));
  if (firstPerson) for (const name of ["Bip01_Head1", "Bip01_Neck1", "forward", "blender_implicit"]) if (ford.joint[`ValveBiped.${name}`] !== undefined) collapse(world, j(name), head); // hide the head, hair and face
  drawSkinned(fb, ford, world);
}
function yawOfQuat(q: Quat): number { const f = rotateVector(q, [0, 0, -1]); return Math.atan2(-f[0], -f[2]); }

// ---------- boot ----------
const rig = new XrRig({ referenceSpace: "local-floor" });
let rigYaw = 0;
const overlay = document.createElement("div");
overlay.style.cssText = "position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font:18px system-ui;color:#fff;background:rgba(0,0,0,.6);text-align:center;padding:24px";
overlay.innerHTML = `<h1 style="margin:0">🦴 BoneVR</h1><p>A physics sandbox. <b>Grip</b> grabs anything near your hand — crates, balls, barrels, the KMP-60.<br>Holding the gun, <b>trigger</b> fires. Let go mid-swing to throw.<br>Left stick walks, right stick snap-turns, <b>A</b> jumps.</p><button id="enter" style="font:22px system-ui;padding:14px 32px;border-radius:12px;border:0;background:#c9a227;color:#000" disabled>Loading…</button><p id="note" style="opacity:.75;font-size:14px">Ford (Bonelab) model and Signalis KMP-60 from Sketchfab. Fan project, not affiliated with Stress Level Zero or rose-engine.</p>`;
document.body.append(overlay);
const note = overlay.querySelector<HTMLElement>("#note")!, enter = overlay.querySelector<HTMLButtonElement>("#enter")!;

(async () => {
  await host.initialize();
  physics = await createPhysicsWorld3d({ moduleUrl });
  buildWorld();
  const gunModel = await importGltf(await (await fetch(new URL("assets/Models/kmp60.glb", document.baseURI))).arrayBuffer(), { sourceName: "kmp60.glb" });
  [ford, gun] = await Promise.all([loadSkinnedGlb(host, new URL("assets/Models/ford.glb", document.baseURI), "ford"), uploadGltfAsset(host, gunModel, { id: "gun", maxTextureSize: 1024 })]);
  const xrOk = await navigator.xr?.isSessionSupported("immersive-vr").catch(() => false);
  enter.disabled = false;
  enter.textContent = xrOk ? "Enter VR" : "Play on desktop";
  if (!xrOk) note.textContent += " No VR headset found. Desktop: mouse look, WASD, E grabs with the right hand, left mouse fires, space jumps, hold T to see Ford.";
  enter.onclick = () => { overlay.remove(); void audio.unlock(); void (xrOk ? startXr() : startDesktop()); };
})().catch(showError);

function showError(error: unknown): void {
  const pre = document.createElement("pre");
  pre.style.cssText = "position:fixed;inset:16px;color:#fff;background:#300;white-space:pre-wrap;padding:12px;z-index:9";
  pre.textContent = `BoneVR failed\n\n${error instanceof Error ? error.stack ?? error.message : String(error)}`;
  document.body.append(pre);
}

async function startXr(): Promise<void> {
  const session = await navigator.xr!.requestSession("immersive-vr", { requiredFeatures: ["local-floor"] });
  const space = await session.requestReferenceSpace("local-floor");
  const presentation = await createPresentation(session, host, canvas, XR_SCALE);
  pulse = (hand, strength, ms) => { for (const src of session.inputSources) if (src.handedness === hand) void src.gamepad?.hapticActuators?.[0]?.pulse(strength, ms); };
  let last = 0, turnArmed = true, jumpArmed = true;
  session.requestAnimationFrame(function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);
    const viewer = frame.getViewerPose(space);
    if (viewer === null) return;
    const dt = Math.min(0.05, Math.max(0.001, last === 0 ? 1 / 72 : (time - last) / 1000)); last = time;
    const snapshot: XrFrameSnapshot = { time, viewerTransform: new Float32Array(viewer.transform.matrix), views: viewer.views.map((v) => ({ eye: v.eye, transform: new Float32Array(v.transform.matrix), projection: new Float32Array(v.projectionMatrix) })), inputs: [], visibilityState: session.visibilityState };
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    const R = rig.rigMatrix(snapshot) as Mat4, Ryaw = quatFromAxisAngle([0, 1, 0], rigYaw);
    const hmd = Array.from(viewer.transform.matrix);
    const stick: [number, number] = [0, 0];
    let jump = false;
    for (const src of session.inputSources) {
      if (src.handedness !== "left" && src.handedness !== "right") continue;
      const hand = hands[src.handedness], grip = src.gripSpace && frame.getPose(src.gripSpace, space);
      if (grip) {
        const mat = Array.from(grip.transform.matrix);
        hand.prev = hand.pos; hand.pos = transformPoint(R, [mat[12]!, mat[13]!, mat[14]!]);
        hand.rot = multiplyQuat(Ryaw, quatOf(mat));
      }
      const pad = src.gamepad;
      if (!pad) continue;
      updateHand(hand, src.handedness, (pad.buttons[1]?.value ?? 0) > 0.5, (pad.buttons[0]?.value ?? 0) > 0.5, time / 1000);
      const ax = pad.axes[2] ?? 0, ay = pad.axes[3] ?? 0;
      if (src.handedness === "left") { stick[0] = Math.abs(ax) > 0.15 ? ax : 0; stick[1] = Math.abs(ay) > 0.15 ? ay : 0; }
      else {
        if (Math.abs(ax) > 0.6 && turnArmed) { rigYaw -= Math.sign(ax) * Math.PI / 6; turnArmed = false; } else if (Math.abs(ax) < 0.3) turnArmed = true;
        const a = pad.buttons[4]?.pressed ?? false; jump = a && jumpArmed; jumpArmed = !a;
      }
    }
    simulate(dt, time / 1000, stick, rigYaw + yawOf(hmd), jump);
    fb.reset();
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    const R2 = rig.rigMatrix(snapshot) as Mat4;
    drawWorld(transformPoint(R2, [hmd[12]!, hmd[13]!, hmd[14]!]), multiplyQuat(Ryaw, quatOf(hmd)), true);
    presentation.present(viewer.views, rig.resolve(snapshot), fb.pack(), time);
  });
}

// ---------- desktop fallback ----------
async function startDesktop(): Promise<void> {
  const view = { yaw: 0, pitch: 0 }; const keys = new Set<string>(); const mouse = [false, false];
  (globalThis as { bone?: { view?: unknown } }).bone!.view = view;
  canvas.onclick = () => void canvas.requestPointerLock();
  addEventListener("mousemove", (e) => { if (document.pointerLockElement === canvas) { view.yaw -= e.movementX * 0.0025; view.pitch = Math.max(-1.4, Math.min(1.4, view.pitch - e.movementY * 0.0025)); } });
  addEventListener("mousedown", (e) => { mouse[e.button === 0 ? 0 : 1] = true; });
  addEventListener("mouseup", (e) => { mouse[e.button === 0 ? 0 : 1] = false; });
  addEventListener("keydown", (e) => keys.add(e.code)); addEventListener("keyup", (e) => keys.delete(e.code));
  let last = performance.now();
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    const dt = Math.min(0.05, Math.max(0.001, (now - last) / 1000)); last = now;
    view.yaw += ((keys.has("ArrowLeft") ? 1 : 0) - (keys.has("ArrowRight") ? 1 : 0)) * 1.5 * dt; view.pitch += ((keys.has("ArrowUp") ? 1 : 0) - (keys.has("ArrowDown") ? 1 : 0)) * 1.5 * dt;
    const q = quatFromEuler([view.pitch * 180 / Math.PI, view.yaw * 180 / Math.PI, 0]);
    const head: Vec3 = [player.pos[0], player.pos[1] + 1.6, player.pos[2]];
    const fwd = rotateVector(q, [0, 0, -1]), right = rotateVector(q, [1, 0, 0]);
    for (const [i, hand] of [hands.left, hands.right].entries()) {
      hand.prev = hand.pos; hand.pos = add(add(add(head, right, i ? 0.25 : -0.25), fwd, 0.45), [0, -0.25, 0], 1); hand.rot = q;
      updateHand(hand, i ? "right" : "left", i ? keys.has("KeyE") : keys.has("KeyQ"), mouse[0]!, now / 1000);
    }
    simulate(dt, now / 1000, [(keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0), (keys.has("KeyS") ? 1 : 0) - (keys.has("KeyW") ? 1 : 0)], view.yaw, keys.has("Space"));
    fb.reset();
    const third = keys.has("KeyT");
    drawWorld(third ? add(head, fwd, 2) : [player.pos[0], player.pos[1] + 1.6, player.pos[2]], q, !third);
    fb.setCamera(composeMatrix(new Float32Array(16), [player.pos[0], player.pos[1] + 1.6, player.pos[2]], q, [1, 1, 1]), 80, 0.05, 300);
    host.submitFrame(fb.pack());
    host.renderFrame(now);
  };
  requestAnimationFrame(loop);
}

// ---------- tiny helpers ----------
function add(a: Vec3, b: Vec3, s: number): Vec3 { return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]; }
function scaleV(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function dist(a: Vec3, b: Vec3): number { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function invertQuat(q: Quat): Quat { return [-q[0], -q[1], -q[2], q[3]]; }
/** Rotation of a column-major rigid transform (unit scale). */
function quatOf(mat: number[]): Quat {
  const m00 = mat[0]!, m01 = mat[4]!, m02 = mat[8]!, m10 = mat[1]!, m11 = mat[5]!, m12 = mat[9]!, m20 = mat[2]!, m21 = mat[6]!, m22 = mat[10]!;
  const t = m00 + m11 + m22;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; return [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4]; }
  if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; return [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; return [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s]; }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2; return [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
}
