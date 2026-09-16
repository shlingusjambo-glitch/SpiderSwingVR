import { AudioEngine, FrameBuilder, WebGpuRenderHost, XrArmLocomotionSolver, XrHandCollisionResolver, XrRig, composeMatrix, createPhysicsWorld3d, importGltf, uploadGltfAsset, type DrawOptions, type PhysicsWorld3dBridge, type XrFrameSnapshot, type XrHandContact } from "@vapour/engine";
import { compose, invert, multiplyMat4, quatFromAxisAngle, quatFromEuler, quatLookRotation, rotateVector, transformDirection, transformPoint, type Mat4, type Vec3 } from "@vapour/math";
import { createPresentation, yawOf } from "../../shared/xr-present.js";
import { bindPose, collapse, drawSkinned, loadSkinnedGlb, reachFor, translation, type SkinnedModel } from "../../shared/skinned-glb.js";

// ---------- tuning knobs ----------
const GRAVITY = 12;            // m/s²
const MAX_SPEED = 40;
const WEB_RANGE = 60;          // metres a web can reach
const SWING_BOOST = 5;         // m/s² forward assist while hanging
const REEL_RATE = 3;           // m/s the rope shortens per second while the trigger is squeezed
const REEL_ACCEL = 18;         // m/s² pull toward the anchor while the rope is longer than its target
const RELEASE_BOOST = 1.1;
const ZIP = 7;                 // m/s kick toward the anchor when a web lands
const AIR_CONTROL = 5;
const HAND_RADIUS = 0.07;
const BODY_RADIUS = 0.3;       // the body is a sphere around the head, like Gorilla Tag
const XR_SCALE = 0.7;          // ponytail: eye-buffer scale; Quest 2 could not hold 72 Hz at 1
const SPAWN: Vec3 = [0, 3, 0];   // forest floor
const TAG_RANGE = 1.1;

// ---------- world ----------
const canvas = document.querySelector<HTMLCanvasElement>("#vapour-game")!;
const moduleUrl = new URL("vapour_runtime.js", document.baseURI).href;
const host = new WebGpuRenderHost(canvas, { moduleUrl });
host.setPostProcess({ exposure: 1, toneMapping: "aces", antiAliasing: "none" });
const fb = new FrameBuilder(1024);
let physics: PhysicsWorld3dBridge;
const mapDraws: { mesh: string; options: DrawOptions }[] = [];
let monke: SkinnedModel | undefined;
const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

async function loadMap(): Promise<void> {
  const model = await importGltf(await (await fetch(new URL("assets/Models/map.glb", document.baseURI))).arrayBuffer(), { sourceName: "map.glb" });
  const asset = await uploadGltfAsset(host, model, { id: "map", maxTextureSize: 1024 });
  // merge the 688 primitives by material into one mesh each (one draw call per material on Quest), and one static trimesh for physics
  const groups = new Map<string, { options: DrawOptions; positions: number[]; normals: number[]; uvs: number[]; indices: number[] }>();
  const physVerts: [number, number, number][] = [], physTris: [number, number, number][] = [];
  for (const draw of asset.draws) {
    const src = model.primitives[asset.meshIds.indexOf(draw.mesh)]!.mesh;
    const key = JSON.stringify(draw.options);
    let g = groups.get(key);
    if (g === undefined) { g = { options: draw.options, positions: [], normals: [], uvs: [], indices: [] }; groups.set(key, g); }
    const base = g.positions.length / 3, physBase = physVerts.length;
    for (let i = 0; i < src.positions.length; i += 3) {
      const p = transformPoint(draw.matrix, [src.positions[i]!, src.positions[i + 1]!, src.positions[i + 2]!]);
      g.positions.push(p[0], p[1], p[2]); physVerts.push(p);
      const n: Vec3 = src.normals ? transformDirection(draw.matrix, [src.normals[i]!, src.normals[i + 1]!, src.normals[i + 2]!]) : [0, 1, 0];
      const l = Math.hypot(n[0], n[1], n[2]) || 1;
      g.normals.push(n[0] / l, n[1] / l, n[2] / l);
      g.uvs.push(src.uvs?.[(i / 3) * 2] ?? 0, src.uvs?.[(i / 3) * 2 + 1] ?? 0);
    }
    for (let i = 0; i < src.indices.length; i += 3) {
      g.indices.push(base + src.indices[i]!, base + src.indices[i + 1]!, base + src.indices[i + 2]!);
      physTris.push([physBase + src.indices[i]!, physBase + src.indices[i + 1]!, physBase + src.indices[i + 2]!]);
    }
    host.releaseMesh(draw.mesh);
  }
  let n = 0;
  for (const g of groups.values()) {
    const id = `map:merged:${n++}`;
    host.uploadMesh(id, { positions: Float32Array.from(g.positions), normals: Float32Array.from(g.normals), uvs: Float32Array.from(g.uvs), indices: Uint32Array.from(g.indices) });
    mapDraws.push({ mesh: id, options: g.options });
  }
  physics.insert({ id: 1, bodyType: "static", position: [0, 0, 0], rotation: [0, 0, 0, 1], shape: { type: "triangleMesh", vertices: physVerts, indices: physTris }, friction: 1 });
  physics.flush();
  physics.step(1 / 60); // queries only see colliders after a step
}

// ---------- player ----------
interface Hand { pos: Vec3; prev: Vec3; aim: Vec3; anchor?: Vec3; rope: number; target: number; squeeze: number; }
const player = { pos: [0, 0, 0] as Vec3, vel: [0, 0, 0] as Vec3, grounded: true, tags: 0, hitFlash: 0 };
const hands: Record<"left" | "right", Hand> = { left: { pos: [-0.3, 1.2, -0.3], prev: [-0.3, 1.2, -0.3], aim: [0, 0, -1], rope: 0, target: 0, squeeze: 0 }, right: { pos: [0.3, 1.2, -0.3], prev: [0.3, 1.2, -0.3], aim: [0, 0, -1], rope: 0, target: 0, squeeze: 0 } };
let pulse: (hand: "left" | "right", strength: number, ms: number) => void = () => {};
(globalThis as { spider?: unknown }).spider = { player, hands, get physics() { return physics; }, get bots() { return bots; } }; // console debugging

const audioContext = new AudioContext();
const audio = new AudioEngine(audioContext);
{
  const buffer = audioContext.createBuffer(1, Math.floor(audioContext.sampleRate * 0.22), audioContext.sampleRate), data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) { const t = i / audioContext.sampleRate; data[i] = Math.exp(-t * 28) * ((Math.random() * 2 - 1) * 0.5 + Math.sin(t * (1400 - t * 5000) * Math.PI * 2) * 0.6); }
  audio.registerClip("thwip", buffer);
}

function castWeb(origin: Vec3, dir: Vec3): Vec3 | undefined {
  // aim assist: the pointed ray, then rays bent upward and sideways; the first anchor well above the player wins
  let fallback: Vec3 | undefined;
  const side: Vec3 = [-dir[2], 0, dir[0]];
  for (const up of [0, 0.15, 0.35, 0.6, 0.9]) for (const lat of [0, 0.3, -0.3]) {
    const d = norm([dir[0] + side[0] * lat, dir[1] + up, dir[2] + side[2] * lat]);
    const hit = physics.raycast(origin, d, WEB_RANGE);
    if (hit === undefined) continue;
    const p: Vec3 = [hit.point[0], hit.point[1], hit.point[2]];
    if (p[1] > origin[1] + 4) return p;
    fallback ??= p;
  }
  return fallback;
}

function updateWeb(hand: Hand, which: "left" | "right", held: boolean, head: Vec3): void {
  if (held && hand.anchor === undefined) {
    const hit = castWeb(hand.pos, hand.aim);
    if (hit !== undefined) {
      const toHit = sub(hit, head), d = len(toHit);
      hand.anchor = hit; hand.rope = d; hand.target = player.grounded ? Math.min(d, Math.max(hit[1] - head[1] - 1, 2)) : d * 0.9; player.grounded = false;
      addTo(player.vel, toHit, ZIP / d); player.vel[1] += ZIP * 0.5;
      pulse(which, 0.6, 40); try { audio.play("thwip", { pitch: 0.9 + Math.random() * 0.25, volume: 0.6 }); } catch { /* not unlocked yet */ }
    }
  } else if (!held && hand.anchor !== undefined) {
    delete hand.anchor;
    if (!player.grounded) player.vel = clampLen(scaleV(player.vel, RELEASE_BOOST), MAX_SPEED);
    pulse(which, 0.25, 20);
  }
}

/** One step of the body: `armMove` is a hand-driven translation (Gorilla Tag pushing) when hands are touching, else free flight with gravity, ropes and air control. */
function simulate(dt: number, head: Vec3, armMove: Vec3 | undefined, launch: Vec3 | undefined, stick: [number, number], yaw: number): void {
  let desired: Vec3;
  if (armMove !== undefined) {
    desired = armMove; player.vel = scaleV(armMove, 1 / dt); player.grounded = true;
  } else {
    if (launch !== undefined) { player.vel = launch; player.grounded = false; }
    const v = player.vel;
    v[1] -= GRAVITY * dt;
    const fwd = rotateVector(quatFromAxisAngle([0, 1, 0], yaw), [0, 0, -1]), right: Vec3 = [-fwd[2], 0, fwd[0]];
    addTo(v, right, stick[0] * AIR_CONTROL * dt); addTo(v, fwd, -stick[1] * AIR_CONTROL * dt);
    desired = [0, 0, 0];
    for (const hand of [hands.left, hands.right]) {
      if (hand.anchor === undefined) continue;
      const toAnchor = sub(hand.anchor, head), d = len(toAnchor), n = scaleV(toAnchor, 1 / d);
      hand.target = Math.max(2, hand.target - REEL_RATE * hand.squeeze * dt);
      if (hand.rope > hand.target) addTo(v, n, REEL_ACCEL * dt);
      hand.rope = Math.max(hand.target, Math.min(hand.rope, d));
      if (d > hand.rope) {
        addTo(desired, n, d - hand.rope);
        const outward = -dot(v, n);
        if (outward > 0) addTo(v, n, outward);
        const speed = len(v);
        if (speed > 1) addTo(v, v, (SWING_BOOST * dt) / speed);
      }
      player.grounded = false;
    }
    player.vel = clampLen(player.vel, MAX_SPEED);
    addTo(desired, player.vel, dt);
  }
  const move = physics.moveCharacter(dt, head, [0, 0, 0, 1], desired, { shape: { type: "sphere", radius: BODY_RADIUS }, offset: 0.02, slide: true, snapToGround: null, autostepMaxHeight: null });
  const moved: Vec3 = [move.translation[0], move.translation[1], move.translation[2]];
  addTo(player.pos, moved, 1);
  if (armMove === undefined) {
    // whatever the map stopped becomes lost velocity
    const blocked = sub(desired, moved);
    if (len(blocked) > 1e-4 && move.collisions.length > 0) { const n = norm(blocked), into = dot(player.vel, n); if (into > 0) addTo(player.vel, n, -into); }
    if (move.grounded) { player.grounded = true; if (player.vel[1] < 0) player.vel[1] = 0; player.vel[0] *= 1 - Math.min(1, 8 * dt); player.vel[2] *= 1 - Math.min(1, 8 * dt); }
  }
  player.hitFlash = Math.max(0, player.hitFlash - dt);
  if (player.pos[1] < -60) { player.pos = [...SPAWN]; player.vel = [0, 0, 0]; } // fell off the world
}

// ---------- tag bots ----------
interface Bot { pos: Vec3; vy: number; yaw: number; it: boolean; cooldown: number; hop: number; }
const bots: Bot[] = ([[4, 3, -6], [-5, 3, 4], [6, 3, 5]] as Vec3[]).map((p, i) => ({ pos: p, vy: 0, yaw: 0, it: i === 0, cooldown: 0, hop: Math.random() * 3 }));
function updateBots(dt: number, head: Vec3): void {
  for (const b of bots) {
    b.cooldown = Math.max(0, b.cooldown - dt); b.hop += dt;
    const to = sub(head, b.pos); to[1] = 0; const d = len(to);
    const chase = b.it && b.cooldown === 0 ? Math.min(1, d / 3) : -0.4; // it: chase; not it: keep a little distance
    if (d > 0.5) b.yaw = Math.atan2(-to[0], -to[2]);
    const step = scaleV(norm(to), chase * 4.2 * dt);
    b.vy -= GRAVITY * dt; step[1] = b.vy * dt;
    const move = physics.moveCharacter(dt, b.pos, [0, 0, 0, 1], step, { shape: { type: "sphere", radius: 0.35 }, offset: 0.02, slide: true, snapToGround: 0.3, autostepMaxHeight: 0.6, autostepMinWidth: 0.2 });
    addTo(b.pos, [move.translation[0], move.translation[1], move.translation[2]], 1);
    if (move.grounded) b.vy = Math.sin(b.hop * 6) > 0.98 ? 3.5 : 0; // little hops while it runs
    if (b.pos[1] < -60) { b.pos = [...SPAWN]; b.vy = 0; }
    if (b.it && b.cooldown === 0 && len(sub(head, b.pos)) < TAG_RANGE) { b.cooldown = 4; player.tags += 1; player.hitFlash = 0.6; pulse("left", 1, 150); pulse("right", 1, 150); }
  }
}

// ---------- drawing ----------
const m = new Float32Array(16);
function drawWorld(head: Vec3, headYaw: number, firstPerson: boolean): void {
  fb.setEnvironment({ clearColor: [0.55, 0.75, 0.95, 1], fog: { mode: "exponential", color: [0.7, 0.8, 0.92], density: 0.006 }, sky: { mode: "procedural", zenithColor: [0.2, 0.45, 0.95], horizonColor: [0.75, 0.85, 0.95], groundColor: [0.3, 0.3, 0.32], horizonCurve: 2 } });
  fb.lights.addHemisphere([0.9, 0.95, 1], [0.35, 0.3, 0.25], 1.6); // no directional light: it would turn on cascaded shadows
  for (const d of mapDraws) fb.draw(d.mesh, IDENTITY, d.options);
  for (const b of bots) drawMonke(compose([b.pos[0], b.pos[1] + 0.1, b.pos[2]], quatFromAxisAngle([0, 1, 0], b.yaw + Math.PI), [1, 1, 1]), undefined, b.it ? [1, 0.45, 0.2, 1] : [1, 1, 1, 1]);
  const tint: [number, number, number, number] = player.hitFlash > 0 ? [1, 0.2, 0.2, 1] : [1, 1, 1, 1];
  if (!(globalThis as { spider?: { noSelf?: boolean } }).spider?.noSelf) drawMonke(compose(head, quatFromAxisAngle([0, 1, 0], headYaw + Math.PI), [1, 1, 1]), { left: hands.left, right: hands.right, hideHead: firstPerson }, tint);
  for (const hand of [hands.left, hands.right]) {
    if (hand.anchor === undefined) continue;
    const d = sub(hand.anchor, hand.pos), L = len(d);
    fb.draw("builtin:cube", composeMatrix(m, [hand.pos[0] + d[0] / 2, hand.pos[1] + d[1] / 2, hand.pos[2] + d[2] / 2], quatLookRotation(d), [0.025, 0.025, L]), { color: [1, 1, 1, 1], emissive: 0.4, roughness: 1 });
  }
}

/** Draws the gorilla with its head joint at `root`. With `arms`, the arms reach for the hands by two-bone IK. */
function drawMonke(root: Mat4, arms: { left: Hand; right: Hand; hideHead: boolean } | undefined, tint: [number, number, number, number]): void {
  if (monke === undefined) return;
  const j = (name: string) => monke!.joint[name]!;
  const world = bindPose(monke, multiplyMat4(root, invert(monke.bind[j("Head_0")]!))); // model space -> world, with the head joint landing on root
  if (arms !== undefined) {
    reachFor(monke, world, j("Upper Arm R_6"), j("Lower Arm R_5"), j("Wrist R_4"), arms.right.pos, transformPoint(root, [-0.5, -0.4, 0.6])); // elbows out, down and back (model faces +Z)
    reachFor(monke, world, j("Upper Arm L_12"), j("Lower Arm L_11"), j("Wrist L_10"), arms.left.pos, transformPoint(root, [0.5, -0.4, 0.6]));
    if (arms.hideHead) for (const name of ["Head_0", "Main_13"]) collapse(world, j(name), translation(root)); // first person: collapse head and body, keep the arms
  }
  drawSkinned(fb, monke, world, tint);
}

// ---------- boot ----------
const rig = new XrRig({ referenceSpace: "local-floor" });
let rigYaw = 0;
const overlay = document.createElement("div");
overlay.style.cssText = "position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font:18px system-ui;color:#fff;background:rgba(0,0,0,.55);text-align:center;padding:24px";
overlay.innerHTML = `<h1 style="margin:0">🦍 Monke Swing</h1><p>Push off the ground and walls with your hands to move, Gorilla Tag style.<br>Point at something and hold <b>grip</b> or <b>trigger</b> to fire a web and swing.<br>Left stick steers in the air, right stick snap-turns.<br>The orange monke is <b>it</b> — don't get tagged.</p><button id="enter" style="font:22px system-ui;padding:14px 32px;border-radius:12px;border:0;background:#d1202a;color:#fff" disabled>Loading…</button><p id="note" style="opacity:.75;font-size:14px">Map "Gorilla Tag Map" by familycucui4, player model by S3ntaGT (Sketchfab, CC-BY-4.0). Fan game, not affiliated with Another Axiom.</p>`;
document.body.append(overlay);
const note = overlay.querySelector<HTMLElement>("#note")!, enter = overlay.querySelector<HTMLButtonElement>("#enter")!;

(async () => {
  await host.initialize();
  physics = await createPhysicsWorld3d({ moduleUrl });
  [, monke] = await Promise.all([loadMap(), loadSkinnedGlb(host, new URL("assets/Models/monke.glb", document.baseURI), "monke")]);
  player.pos = [...SPAWN];
  const xrOk = await navigator.xr?.isSessionSupported("immersive-vr").catch(() => false);
  enter.disabled = false;
  enter.textContent = xrOk ? "Enter VR" : "Play on desktop";
  if (!xrOk) note.textContent += " No VR headset found. Desktop: mouse look, WASD, left/right mouse buttons fire webs, space jumps.";
  enter.onclick = () => { overlay.remove(); void audio.unlock(); void (xrOk ? startXr() : startDesktop()); };
})().catch(showError);

function showError(error: unknown): void {
  const pre = document.createElement("pre");
  pre.style.cssText = "position:fixed;inset:16px;color:#fff;background:#300;white-space:pre-wrap;padding:12px;z-index:9";
  pre.textContent = `Monke Swing failed\n\n${error instanceof Error ? error.stack ?? error.message : String(error)}`;
  document.body.append(pre);
}

async function startXr(): Promise<void> {
  const session = await navigator.xr!.requestSession("immersive-vr", { requiredFeatures: ["local-floor"] });
  const space = await session.requestReferenceSpace("local-floor");
  const presentation = await createPresentation(session, host, canvas, XR_SCALE);
  pulse = (hand, strength, ms) => { for (const src of session.inputSources) if (src.handedness === hand) void src.gamepad?.hapticActuators?.[0]?.pulse(strength, ms); };
  const contacts = new XrHandCollisionResolver(physics, { defaultSingleHandSlip: 0.01, defaultBracedSlip: 0.03 });
  const arms = new XrArmLocomotionSolver({ gravity: 0, maxSpeed: 9, maxLaunchSpeed: 8, launchMultiplier: 1.2 });
  let last = 0, turnArmed = true;
  session.requestAnimationFrame(function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);
    const viewer = frame.getViewerPose(space);
    if (viewer === null) return;
    const dt = Math.min(0.05, Math.max(0.001, last === 0 ? 1 / 72 : (time - last) / 1000)); last = time;
    const snapshot: XrFrameSnapshot = { time, viewerTransform: new Float32Array(viewer.transform.matrix), views: viewer.views.map((v) => ({ eye: v.eye, transform: new Float32Array(v.transform.matrix), projection: new Float32Array(v.projectionMatrix) })), inputs: [], visibilityState: session.visibilityState };
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    const R = rig.rigMatrix(snapshot) as Mat4, Ryaw = quatFromAxisAngle([0, 1, 0], rigYaw);
    const hmd: Vec3 = [viewer.transform.matrix[12]!, viewer.transform.matrix[13]!, viewer.transform.matrix[14]!];
    const head = transformPoint(R, hmd);
    const stick: [number, number] = [0, 0];
    const tracked: { hand: "left" | "right"; position: Vec3; tracked: boolean }[] = [];
    const touching: Partial<Record<"left" | "right", XrHandContact>> = {};
    for (const src of session.inputSources) {
      if (src.handedness !== "left" && src.handedness !== "right") continue;
      const hand = hands[src.handedness], grip = src.gripSpace && frame.getPose(src.gripSpace, space), aim = frame.getPose(src.targetRaySpace, space);
      if (grip) {
        const local: Vec3 = [grip.transform.matrix[12]!, grip.transform.matrix[13]!, grip.transform.matrix[14]!];
        hand.prev = hand.pos; hand.pos = transformPoint(R, local);
        tracked.push({ hand: src.handedness, position: local, tracked: true });
        const contact = contacts.resolve(hand.prev, sub(hand.pos, hand.prev), HAND_RADIUS, true);
        if (contact !== undefined && hand.anchor === undefined) { touching[src.handedness] = contact; hand.pos = [contact.constrainedPosition[0], contact.constrainedPosition[1], contact.constrainedPosition[2]]; }
      }
      if (aim) hand.aim = transformDirection(multiplyMat4(R, Array.from(aim.transform.matrix)), [0, 0, -1]);
      const pad = src.gamepad;
      if (!pad) continue;
      const trigger = pad.buttons[0]?.value ?? 0, gripBtn = pad.buttons[1]?.value ?? 0;
      hand.squeeze = trigger;
      updateWeb(hand, src.handedness, trigger > 0.3 || gripBtn > 0.3, head);
      const ax = pad.axes[2] ?? 0, ay = pad.axes[3] ?? 0;
      if (src.handedness === "left") { stick[0] = Math.abs(ax) > 0.15 ? ax : 0; stick[1] = Math.abs(ay) > 0.15 ? ay : 0; }
      else if (Math.abs(ax) > 0.6 && turnArmed) { rigYaw -= Math.sign(ax) * Math.PI / 6; turnArmed = false; } else if (Math.abs(ax) < 0.3) turnArmed = true;
    }
    const arm = arms.sample({ deltaSeconds: dt, headPosition: hmd, hands: tracked, contacts: touching });
    const swinging = hands.left.anchor !== undefined || hands.right.anchor !== undefined;
    const armMove = arm.supportedHands.length > 0 && !swinging ? rotateVector(Ryaw, arm.translation) : undefined;
    const launch = arm.launchVelocity.some((c) => c !== 0) ? rotateVector(Ryaw, arm.launchVelocity) : undefined;
    const yaw = rigYaw + yawOf(viewer.transform.matrix);
    simulate(dt, head, armMove, launch, stick, yaw);
    updateBots(dt, head);
    fb.reset();
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    drawWorld(transformPoint(rig.rigMatrix(snapshot) as Mat4, hmd), yaw, true);
    presentation.present(viewer.views, rig.resolve(snapshot), fb.pack(), time);
  });
}

// ---------- desktop fallback: mouse look, WASD, mouse buttons fire webs ----------
async function startDesktop(): Promise<void> {
  const view = { yaw: 0, pitch: 0 }; const keys = new Set<string>(); const mouse = [false, false];
  (globalThis as { spider?: { view?: unknown } }).spider!.view = view;
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
    const { yaw, pitch } = view;
    const q = quatFromEuler([pitch * 180 / Math.PI, yaw * 180 / Math.PI, 0]);
    const head: Vec3 = [player.pos[0], player.pos[1] + 1.2, player.pos[2]];
    const fwd = rotateVector(q, [0, 0, -1]), right = rotateVector(q, [1, 0, 0]);
    for (const [i, hand] of [hands.left, hands.right].entries()) {
      hand.pos = [head[0] + right[0] * (i ? 0.3 : -0.3) + fwd[0] * 0.5, head[1] - 0.15, head[2] + right[2] * (i ? 0.3 : -0.3) + fwd[2] * 0.5];
      hand.aim = fwd; hand.squeeze = keys.has("ShiftLeft") ? 1 : 0;
      updateWeb(hand, i ? "right" : "left", mouse[i]!, head);
    }
    const stick: [number, number] = [(keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0), (keys.has("KeyS") ? 1 : 0) - (keys.has("KeyW") ? 1 : 0)];
    // on the ground WASD walks (stands in for pushing with the hands); in the air it steers
    const walking = player.grounded && (stick[0] !== 0 || stick[1] !== 0) && !keys.has("Space");
    const f = rotateVector(quatFromAxisAngle([0, 1, 0], yaw), [0, 0, -1]), r: Vec3 = [-f[2], 0, f[0]];
    const armMove: Vec3 | undefined = walking ? [(r[0] * stick[0] - f[0] * stick[1]) * 4 * dt, -0.05 * dt, (r[2] * stick[0] - f[2] * stick[1]) * 4 * dt] : undefined;
    const launch: Vec3 | undefined = keys.has("Space") && player.grounded ? [player.vel[0], 6, player.vel[2]] : undefined;
    simulate(dt, head, armMove, launch, walking ? [0, 0] : stick, yaw);
    updateBots(dt, head);
    fb.reset();
    drawWorld([player.pos[0], player.pos[1] + 1.2, player.pos[2]], yaw, !keys.has("KeyT")); // hold T to see your own monke
    fb.setCamera(composeMatrix(new Float32Array(16), [player.pos[0], player.pos[1] + 1.2, player.pos[2]], q, [1, 1, 1]), 80, 0.05, 600);
    host.submitFrame(fb.pack());
    host.renderFrame(now);
  };
  requestAnimationFrame(loop);
}

// ---------- tiny vec helpers ----------
function addTo(target: Vec3, source: Vec3, s: number): void { target[0] += source[0] * s; target[1] += source[1] * s; target[2] += source[2] * s; }
function sub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scaleV(a: Vec3, s: number): Vec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function len(a: Vec3): number { return Math.hypot(a[0], a[1], a[2]); }
function norm(a: Vec3): Vec3 { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
function clampLen(a: Vec3, max: number): Vec3 { const l = len(a); return l > max ? scaleV(a, max / l) : a; }
