import { AudioEngine, FrameBuilder, WebGpuRenderHost, XrPresentation, XrRig, canvasSubImageCopier, composeMatrix, type XrFrameSnapshot } from "@vapour/engine";
import { aabb, multiplyMat4, quatFromAxisAngle, quatFromEuler, quatLookRotation, ray, rayIntersectsAabb, rotateVector, transformDirection, transformPoint, type Aabb, type Mat4, type Vec3 } from "@vapour/math";

// ---------- tuning knobs ----------
const GRAVITY = 16;            // m/s²; heavier than Earth so swings feel snappy
const MAX_SPEED = 55;
const WEB_RANGE = 90;          // metres a web can reach
const SWING_BOOST = 6;         // m/s² forward assist while hanging (Spider-Man 2 style)
const REEL_RATE = 3.5;         // m/s the rope shortens per second while the trigger is squeezed
const RELEASE_BOOST = 1.1;     // velocity multiplier on let-go
const AIR_CONTROL = 7;         // m/s² thumbstick steering in the air
const JUMP = 9;
const ZIP = 9;                 // m/s kick toward the anchor when a web lands, so a fresh web launches you
const REEL_ACCEL = 22;         // m/s² pull toward the anchor while the rope is longer than its target
const XR_SCALE = 1;            // ponytail: eye-buffer scale; drop toward 0.8 if the Quest 2 misses 72 Hz

// ---------- city ----------
interface Building { box: Aabb; color: [number, number, number, number]; }
const rand = mulberry32(1962);
const BLOCK = 44, STREET = 14, GRID = 8, HALF = (GRID * BLOCK) / 2;
const buildings: Building[] = [];
const palette: [number, number, number, number][] = [[0.55, 0.52, 0.5, 1], [0.35, 0.4, 0.5, 1], [0.6, 0.45, 0.35, 1], [0.3, 0.32, 0.38, 1], [0.7, 0.66, 0.6, 1], [0.25, 0.45, 0.55, 1]];
for (let gx = 0; gx < GRID; gx += 1) for (let gz = 0; gz < GRID; gz += 1) {
  const x0 = -HALF + gx * BLOCK + STREET / 2, z0 = -HALF + gz * BLOCK + STREET / 2, inner = BLOCK - STREET;
  const split = rand() < 0.5;
  for (let i = 0; i < (split ? 2 : 1); i += 1) {
    const w = split ? inner / 2 - 2 : inner, d = inner;
    const centerDist = Math.hypot(gx - GRID / 2 + 0.5, gz - GRID / 2 + 0.5);
    const h = 14 + rand() * (90 - centerDist * 12);
    const x = x0 + (split ? i * (inner / 2 + 2) : 0), z = z0;
    buildings.push({ box: aabb([x, 0, z], [x + w, h, z + d]), color: palette[Math.floor(rand() * palette.length)]! });
  }
}

// ---------- npcs ----------
interface Npc { pos: Vec3; dir: Vec3; color: [number, number, number, number]; turnIn: number; hop: number; phase: number; }
const npcs: Npc[] = [];
for (let i = 0; i < 160; i += 1) {
  const line = -HALF + Math.floor(rand() * (GRID + 1)) * BLOCK, along = -HALF + rand() * GRID * BLOCK, side = rand() < 0.5 ? -5.5 : 5.5;
  const onX = rand() < 0.5;
  npcs.push({ pos: onX ? [along, 0, line + side] : [line + side, 0, along], dir: onX ? [rand() < 0.5 ? 1 : -1, 0, 0] : [0, 0, rand() < 0.5 ? 1 : -1], color: [rand(), rand(), rand(), 1], turnIn: 3 + rand() * 8, hop: 0, phase: rand() * 7 });
}

// ---------- tokens ----------
const tokens: { pos: Vec3; taken: boolean }[] = [];
for (let i = 0; i < 40; i += 1) {
  const b = buildings[Math.floor(rand() * buildings.length)]!;
  tokens.push({ pos: [(b.box.min[0] + b.box.max[0]) / 2 + (rand() - 0.5) * 20, b.box.max[1] + 4 + rand() * 12, (b.box.min[2] + b.box.max[2]) / 2 + (rand() - 0.5) * 20], taken: false });
}

// ---------- player ----------
interface Hand { pos: Vec3; aim: Vec3; anchor?: Vec3; rope: number; target: number; squeeze: number; }
const player = { pos: [0, 0, 0] as Vec3, vel: [0, 0, 0] as Vec3, grounded: true, score: 0 };
const hands: Record<"left" | "right", Hand> = { left: { pos: [-0.3, 1.2, -0.3], aim: [0, 0, -1], rope: 0, target: 0, squeeze: 0 }, right: { pos: [0.3, 1.2, -0.3], aim: [0, 0, -1], rope: 0, target: 0, squeeze: 0 } };
let pulse: (hand: "left" | "right", strength: number, ms: number) => void = () => {};

// ---------- sounds: synthesized once, so there are no audio files to host ----------
const audioContext = new AudioContext();
const audio = new AudioEngine(audioContext);
function synth(id: string, seconds: number, sample: (t: number, noise: number) => number): void {
  const buffer = audioContext.createBuffer(1, Math.floor(audioContext.sampleRate * seconds), audioContext.sampleRate), data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) data[i] = sample(i / audioContext.sampleRate, Math.random() * 2 - 1);
  audio.registerClip(id, buffer);
}
synth("thwip", 0.22, (t, n) => Math.exp(-t * 28) * (n * 0.5 + Math.sin(t * (1400 - t * 5000) * Math.PI * 2) * 0.6));
synth("release", 0.12, (t, n) => Math.exp(-t * 45) * n * 0.5);
synth("token", 0.5, (t) => Math.exp(-t * 7) * (Math.sin(t * 1320 * Math.PI * 2) + Math.sin(t * 1980 * Math.PI * 2) * 0.5) * 0.35);
let windLow = 0;
synth("wind", 2, (_t, n) => (windLow += (n - windLow) * 0.08) * 2.5); // one-pole low-pass noise, seamless enough as a loop
let wind: ReturnType<AudioEngine["play"]> | undefined;
function sfx(id: string, pitch = 1, volume = 0.6): void { try { audio.play(id, { pitch, volume }); } catch { /* audio not unlocked yet */ } }
(globalThis as { spider?: unknown }).spider = { player, hands }; // console debugging

function bodyCentre(head: Vec3): Vec3 { return [head[0], head[1] - 0.3, head[2]]; }

function castWeb(origin: Vec3, dir: Vec3): Vec3 | undefined {
  // aim assist: the pointed ray, then rays bent further and further upward; the first anchor
  // well above the player wins (a high anchor is what makes a pendulum), else the nearest hit
  let fallback: Vec3 | undefined;
  const side: Vec3 = [-dir[2], 0, dir[0]];
  const fan = [0, 0.15, 0.35, 0.6, 0.9, 1.3].flatMap((up) => [0, 0.35, -0.35].map((lat): Vec3 => [dir[0] + side[0] * lat, dir[1] + up, dir[2] + side[2] * lat]));
  for (const d of fan) {
    const r = ray(origin, d);
    let best: Vec3 | undefined, bestT = WEB_RANGE;
    for (const b of buildings) {
      const t = rayIntersectsAabb(r, b.box);
      if (t !== undefined && t < bestT) { bestT = t; best = [r.origin[0] + r.direction[0] * t, r.origin[1] + r.direction[1] * t, r.origin[2] + r.direction[2] * t]; }
    }
    if (best === undefined) continue;
    if (best[1] > origin[1] + 6) return best;
    fallback ??= best;
  }
  return fallback;
}

function updateHand(hand: Hand, which: "left" | "right", held: boolean, head: Vec3): void {
  if (held && hand.anchor === undefined) {
    const hit = castWeb(hand.pos, hand.aim);
    if (hit !== undefined) {
      const c = bodyCentre(head), toHit = sub(hit, c), d = len(toHit);
      // from the ground the rope reels in short enough to swing clear of the street; in the air it just tightens a little
      hand.anchor = hit; hand.rope = d; hand.target = player.grounded ? Math.min(d, Math.max(hit[1] - c[1] - 1.5, 3)) : d * 0.9; player.grounded = false;
      addTo(player.vel, toHit, ZIP / d); player.vel[1] += ZIP * 0.5;
      pulse(which, 0.6, 40); sfx("thwip", 0.9 + Math.random() * 0.25);
    }
  } else if (!held && hand.anchor !== undefined) {
    delete hand.anchor;
    if (!player.grounded) player.vel = clampLen(scaleV(player.vel, RELEASE_BOOST), MAX_SPEED);
    pulse(which, 0.25, 20); sfx("release", 1, 0.3);
  }
}

function simulate(dt: number, head: Vec3, stick: [number, number], yaw: number): void {
  const v = player.vel;
  v[1] -= GRAVITY * dt;
  // thumbstick air control, relative to where the player is facing
  const fwd = rotateVector(quatFromAxisAngle([0, 1, 0], yaw), [0, 0, -1]), right: Vec3 = [-fwd[2], 0, fwd[0]];
  addTo(v, right, stick[0] * AIR_CONTROL * dt); addTo(v, fwd, -stick[1] * AIR_CONTROL * dt);
  // ropes: keep the body inside each rope's sphere and kill outward velocity
  for (const hand of [hands.left, hands.right]) {
    if (hand.anchor === undefined) continue;
    const c = bodyCentre(head), toAnchor = sub(hand.anchor, c), d = len(toAnchor), n = scaleV(toAnchor, 1 / d);
    hand.target = Math.max(2, hand.target - REEL_RATE * hand.squeeze * dt);
    if (hand.rope > hand.target) addTo(v, n, REEL_ACCEL * dt); // being reeled in
    hand.rope = Math.max(hand.target, Math.min(hand.rope, d)); // the rope follows you in, never pays out
    if (d > hand.rope) {
      addTo(player.pos, n, d - hand.rope);
      const outward = -dot(v, n);
      if (outward > 0) addTo(v, n, outward);
      // swing assist: push along the tangent the player is already travelling
      const speed = len(v);
      if (speed > 1) addTo(v, v, (SWING_BOOST * dt) / speed);
    }
    player.grounded = false;
  }
  player.vel = clampLen(v, MAX_SPEED);
  addTo(player.pos, player.vel, dt);
  // ground and roofs: a capsule-ish point test against every box, push out along the shallowest axis
  player.grounded = false;
  if (player.pos[1] < 0) { player.pos[1] = 0; if (player.vel[1] < 0) player.vel[1] = 0; player.grounded = true; }
  const p = player.pos, r = 0.45;
  for (const b of buildings) {
    const { min, max } = b.box;
    if (p[0] + r < min[0] || p[0] - r > max[0] || p[2] + r < min[2] || p[2] - r > max[2] || p[1] + 1.6 < min[1] || p[1] > max[1]) continue;
    const pushes: [number, number][] = [[max[1] - p[1], 1], [min[0] - (p[0] + r), 0], [max[0] - (p[0] - r), 0], [min[2] - (p[2] + r), 2], [max[2] - (p[2] - r), 2]];
    const [amount, axis] = pushes.reduce((a, c) => (Math.abs(c[0]) < Math.abs(a[0]) ? c : a));
    if (axis === 1) { p[1] += amount; player.grounded = true; if (player.vel[1] < 0) player.vel[1] = 0; }
    else if (axis === 0) { p[0] += amount; player.vel[0] *= -0.2; } else { p[2] += amount; player.vel[2] *= -0.2; }
  }
  if (player.grounded) { player.vel[0] *= 1 - Math.min(1, 6 * dt); player.vel[2] *= 1 - Math.min(1, 6 * dt); }
  // tokens
  for (const t of tokens) if (!t.taken && dist(t.pos, head) < 1.6) { t.taken = true; player.score += 1; pulse("left", 1, 80); pulse("right", 1, 80); sfx("token"); }
  // wind rises with airspeed
  const speed = len(player.vel);
  wind?.setVolume(Math.min(1, Math.max(0, (speed - 4) / 30)) * 0.9); wind?.setPitch(0.7 + speed / 50);
  // npcs wander the sidewalks and hop when Spidey flies past
  const fast = len(player.vel) > 12;
  for (const n of npcs) {
    n.turnIn -= dt;
    if (n.turnIn <= 0) { n.turnIn = 3 + rand() * 8; n.dir = rand() < 0.5 ? [-n.dir[0], 0, -n.dir[2]] : [n.dir[2], 0, -n.dir[0]]; }
    addTo(n.pos, n.dir, 1.4 * dt);
    if (Math.abs(n.pos[0]) > HALF || Math.abs(n.pos[2]) > HALF) n.dir = [-n.dir[0], 0, -n.dir[2]];
    if (fast && n.hop <= 0 && dist(n.pos, player.pos) < 10) n.hop = 0.6;
    if (n.hop > 0) n.hop -= dt;
  }
}

// ---------- drawing ----------
const m = new Float32Array(16);
const Q_ID: [number, number, number, number] = [0, 0, 0, 1];
function drawWorld(fb: FrameBuilder, time: number): void {
  fb.setEnvironment({ clearColor: [0.55, 0.75, 0.95, 1], fog: { mode: "exponential", color: [0.7, 0.8, 0.92], density: 0.0035 }, sky: { mode: "procedural", zenithColor: [0.2, 0.45, 0.95], horizonColor: [0.75, 0.85, 0.95], groundColor: [0.3, 0.3, 0.32], horizonCurve: 2 } });
  fb.lights.addAmbient([0.55, 0.65, 0.8], 0.9);
  fb.lights.addDirectional([-0.4, -0.8, -0.3], [1, 0.95, 0.85], 3.2);
  fb.draw("builtin:cube", composeMatrix(m, [0, -0.5, 0], Q_ID, [HALF * 2 + 200, 1, HALF * 2 + 200]), { color: [0.42, 0.42, 0.4, 1], roughness: 1 });
  for (let i = 0; i <= GRID; i += 1) {
    const line = -HALF + i * BLOCK;
    fb.draw("builtin:cube", composeMatrix(m, [0, 0.02, line], Q_ID, [HALF * 2, 0.04, STREET - 3]), { color: [0.15, 0.15, 0.16, 1], roughness: 1 });
    fb.draw("builtin:cube", composeMatrix(m, [line, 0.02, 0], Q_ID, [STREET - 3, 0.04, HALF * 2]), { color: [0.15, 0.15, 0.16, 1], roughness: 1 });
  }
  for (const b of buildings) {
    const { min, max } = b.box;
    fb.draw("builtin:cube", composeMatrix(m, [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2], Q_ID, [max[0] - min[0], max[1] - min[1], max[2] - min[2]]), { color: b.color, roughness: 0.7 });
    // roof cap for a little silhouette variety
    fb.draw("builtin:cube", composeMatrix(m, [(min[0] + max[0]) / 2, max[1] + 0.6, (min[2] + max[2]) / 2], Q_ID, [(max[0] - min[0]) * 0.6, 1.2, (max[2] - min[2]) * 0.6]), { color: [0.2, 0.2, 0.22, 1], roughness: 0.9 });
  }
  for (const n of npcs) {
    const bob = Math.abs(Math.sin(time * 9 + n.phase)) * 0.05 + (n.hop > 0 ? Math.sin((0.6 - n.hop) / 0.6 * Math.PI) * 0.8 : 0);
    const q = quatLookRotation(n.dir);
    fb.draw("builtin:capsule", composeMatrix(m, [n.pos[0], 0.75 + bob, n.pos[2]], q, [0.9, 1.5, 0.9]), { color: n.color, roughness: 0.8 });
    fb.draw("builtin:sphere", composeMatrix(m, [n.pos[0], 1.65 + bob, n.pos[2]], q, [0.32, 0.32, 0.32]), { color: [0.9, 0.75, 0.6, 1], roughness: 0.8 });
  }
  for (const t of tokens) if (!t.taken) fb.draw("builtin:cube", composeMatrix(m, t.pos, quatFromEuler([45, time * 120, 0]), [0.6, 0.6, 0.6]), { color: [1, 0.85, 0.2, 1], emissive: 1.5, metallic: 0.8, roughness: 0.3 });
  // hands, webs and the score stack on the left wrist
  for (const [which, hand] of Object.entries(hands) as ["left" | "right", Hand][]) {
    fb.draw("builtin:sphere", composeMatrix(m, hand.pos, quatLookRotation(hand.aim), [0.09, 0.09, 0.14]), { color: [0.85, 0.1, 0.12, 1], roughness: 0.6 });
    if (hand.anchor !== undefined) {
      const d = sub(hand.anchor, hand.pos), L = len(d);
      fb.draw("builtin:cube", composeMatrix(m, [hand.pos[0] + d[0] / 2, hand.pos[1] + d[1] / 2, hand.pos[2] + d[2] / 2], quatLookRotation(d), [0.03, 0.03, L]), { color: [1, 1, 1, 1], emissive: 0.4, roughness: 1 });
    }
    if (which === "left") for (let i = 0; i < player.score; i += 1) fb.draw("builtin:cube", composeMatrix(m, [hand.pos[0], hand.pos[1] + 0.12 + i * 0.03, hand.pos[2]], Q_ID, [0.05, 0.02, 0.05]), { color: [1, 0.85, 0.2, 1], emissive: 1 });
  }
}

// ---------- boot ----------
const canvas = document.querySelector<HTMLCanvasElement>("#vapour-game")!;
const host = new WebGpuRenderHost(canvas, { moduleUrl: new URL("vapour_runtime.js", document.baseURI).href });
host.setPostProcess({ exposure: 1, toneMapping: "aces", antiAliasing: "none" });
const fb = new FrameBuilder(1024);
const rig = new XrRig({ referenceSpace: "local-floor" });
let rigYaw = 0;

const overlay = document.createElement("div");
overlay.style.cssText = "position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font:18px system-ui;color:#fff;background:rgba(0,0,0,.55);text-align:center;padding:24px";
overlay.innerHTML = `<h1 style="margin:0">🕷️ Spider Swing</h1><p>Point a controller at a building and hold <b>grip</b> or <b>trigger</b> to fire a web.<br>Squeeze the trigger to reel in. Let go at the bottom of the swing to launch.<br>Left stick steers in the air, right stick snap-turns, <b>A</b> jumps.<br>Collect the gold tokens.</p><button id="enter" style="font:22px system-ui;padding:14px 32px;border-radius:12px;border:0;background:#d1202a;color:#fff">Enter VR</button><p id="note" style="opacity:.75;font-size:14px"></p>`;
document.body.append(overlay);
const note = overlay.querySelector<HTMLElement>("#note")!, enter = overlay.querySelector<HTMLButtonElement>("#enter")!;

host.initialize().then(async () => {
  const xrOk = await navigator.xr?.isSessionSupported("immersive-vr").catch(() => false);
  if (!xrOk) { enter.textContent = "Play on desktop"; note.textContent = "No VR headset found. Desktop: mouse look, WASD steer, left/right mouse buttons fire webs, space jumps."; }
  enter.onclick = () => { overlay.remove(); void audio.unlock().then(() => { wind = audio.play("wind", { loop: true, volume: 0 }); }); void (xrOk ? startXr() : startDesktop()); };
}).catch(showError);

function showError(error: unknown): void {
  const pre = document.createElement("pre");
  pre.style.cssText = "position:fixed;inset:16px;color:#fff;background:#300;white-space:pre-wrap;padding:12px;z-index:9";
  pre.textContent = `Spider Swing failed\n\n${error instanceof Error ? error.stack ?? error.message : String(error)}`;
  document.body.append(pre);
}

async function startXr(): Promise<void> {
  const session = await navigator.xr!.requestSession("immersive-vr", { requiredFeatures: ["local-floor"] });
  const space = await session.requestReferenceSpace("local-floor");
  const presentation = await createPresentation(session);
  pulse = (hand, strength, ms) => { for (const src of session.inputSources) if (src.handedness === hand) void src.gamepad?.hapticActuators?.[0]?.pulse(strength, ms); };
  let last = 0, turnArmed = true, jumpArmed = true;
  session.requestAnimationFrame(function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);
    const viewer = frame.getViewerPose(space);
    if (viewer === null) return;
    const dt = Math.min(0.05, last === 0 ? 1 / 72 : (time - last) / 1000); last = time;
    const snapshot: XrFrameSnapshot = { time, viewerTransform: new Float32Array(viewer.transform.matrix), views: viewer.views.map((v) => ({ eye: v.eye, transform: new Float32Array(v.transform.matrix), projection: new Float32Array(v.projectionMatrix) })), inputs: [], visibilityState: session.visibilityState };
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    const R = rig.rigMatrix(snapshot) as Mat4;
    const head = transformPoint(R, [viewer.transform.matrix[12]!, viewer.transform.matrix[13]!, viewer.transform.matrix[14]!]);
    const stick: [number, number] = [0, 0];
    for (const src of session.inputSources) {
      if (src.handedness !== "left" && src.handedness !== "right") continue;
      const hand = hands[src.handedness], grip = src.gripSpace && frame.getPose(src.gripSpace, space), aim = frame.getPose(src.targetRaySpace, space);
      if (grip) hand.pos = transformPoint(R, [grip.transform.matrix[12]!, grip.transform.matrix[13]!, grip.transform.matrix[14]!]);
      if (aim) hand.aim = transformDirection(multiplyMat4(R, Array.from(aim.transform.matrix)), [0, 0, -1]);
      const pad = src.gamepad;
      if (!pad) continue;
      const trigger = pad.buttons[0]?.value ?? 0, gripBtn = pad.buttons[1]?.value ?? 0;
      hand.squeeze = trigger;
      updateHand(hand, src.handedness, trigger > 0.3 || gripBtn > 0.3, head);
      const ax = pad.axes[2] ?? 0, ay = pad.axes[3] ?? 0;
      if (src.handedness === "left") { stick[0] = Math.abs(ax) > 0.15 ? ax : 0; stick[1] = Math.abs(ay) > 0.15 ? ay : 0; }
      else {
        if (Math.abs(ax) > 0.6 && turnArmed) { rigYaw -= Math.sign(ax) * Math.PI / 6; turnArmed = false; } else if (Math.abs(ax) < 0.3) turnArmed = true;
        const a = pad.buttons[4]?.pressed ?? false;
        if (a && jumpArmed && player.grounded) { player.vel[1] = JUMP; player.grounded = false; } jumpArmed = !a;
      }
    }
    simulate(dt, head, stick, rigYaw + headYaw(viewer.transform.matrix));
    fb.reset();
    drawWorld(fb, time / 1000);
    rig.setPose(player.pos, quatFromAxisAngle([0, 1, 0], rigYaw));
    presentation.present(viewer.views, rig.resolve(snapshot), fb.pack(), time);
  });
}

function headYaw(mat: ArrayLike<number>): number { return Math.atan2(mat[8]!, mat[10]!); }

/** WebXR/WebGPU binding when Meta Browser exposes it, otherwise an XRWebGLLayer fed by blitting the WebGPU canvas. */
async function createPresentation(session: XRSession): Promise<{ present: XrPresentation<unknown, XRView>["present"] }> {
  const XRGPUBinding = (globalThis as { XRGPUBinding?: new (s: XRSession, d: GPUDevice) => never }).XRGPUBinding;
  if (XRGPUBinding !== undefined) {
    const adapter = await navigator.gpu.requestAdapter({ xrCompatible: true } as GPURequestAdapterOptions);
    const device = await adapter!.requestDevice();
    const binding = new XRGPUBinding(session, device);
    // two-pass: the renderer clears the whole canvas per layer, so a single side-by-side submission loses the first eye
    const p = new XrPresentation<unknown, XRView>({ host, binding, session, copy: canvasSubImageCopier(device, canvas), scaleFactor: XR_SCALE, multiview: false });
    return { present: (views, rigFrame, frame, t) => p.present(views, rigFrame, frame, t) };
  }
  const gl = document.createElement("canvas").getContext("webgl2", { xrCompatible: true, alpha: false, antialias: false } as WebGLContextAttributes) as WebGL2RenderingContext;
  await gl.makeXRCompatible();
  const layer = new XRWebGLLayer(session, gl, { framebufferScaleFactor: XR_SCALE });
  session.updateRenderState({ baseLayer: layer });
  const blit = canvasBlitter(gl);
  // a fake binding so XrPresentation drives the eyes for us; the copier does the WebGL blit
  const binding = {
    createProjectionLayer: () => layer,
    getViewSubImage: (_l: unknown, view: XRView) => { const v = layer.getViewport(view)!; return { colorTexture: { width: layer.framebufferWidth, height: layer.framebufferHeight }, viewport: { x: v.x, y: v.y, width: v.width, height: v.height }, imageIndex: 0 }; },
    getPreferredColorFormat: () => "rgba8unorm" as const,
  };
  const p = new XrPresentation<unknown, XRView>({ host, binding, session: { updateRenderState() {} }, copy: ({ sourceRect, destinationRect }) => blit(layer.framebuffer, canvas, sourceRect, destinationRect), multiview: false });
  return { present: (views, rigFrame, frame, t) => { const r = p.present(views, rigFrame, frame, t); gl.flush(); return r; } };
}

function canvasBlitter(gl: WebGL2RenderingContext) {
  const compile = (type: number, src: string) => { const s = gl.createShader(type)!; gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, "#version 300 es\nprecision highp float;in vec2 p;out vec2 uv;void main(){vec2 t=(p+1.0)*0.5;uv=vec2(t.x,1.0-t.y);gl_Position=vec4(p,0.0,1.0);}"));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, "#version 300 es\nprecision highp float;uniform sampler2D img;in vec2 uv;out vec4 c;void main(){c=texture(img,uv);}"));
  gl.linkProgram(prog);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k!, v!);
  return (framebuffer: WebGLFramebuffer | null, source: HTMLCanvasElement, src: { x: number; y: number; width: number; height: number }, dst: { x: number; y: number; width: number; height: number }) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    // upload only this eye's rectangle of the side-by-side canvas (WebGL2 sub-rectangle upload), mediump would quantize the UVs
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, source.width); gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, src.x); gl.pixelStorei(gl.UNPACK_SKIP_ROWS, src.y);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, src.width, src.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.viewport(dst.x, dst.y, dst.width, dst.height);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}

// ---------- desktop fallback: mouse look, WASD, mouse buttons fire webs ----------
async function startDesktop(): Promise<void> {
  let yaw = 0, pitch = 0; const keys = new Set<string>(); const mouse = [false, false];
  canvas.onclick = () => void canvas.requestPointerLock();
  addEventListener("mousemove", (e) => { if (document.pointerLockElement === canvas) { yaw -= e.movementX * 0.0025; pitch = Math.max(-1.4, Math.min(1.4, pitch - e.movementY * 0.0025)); } });
  addEventListener("mousedown", (e) => { mouse[e.button === 0 ? 0 : 1] = true; });
  addEventListener("mouseup", (e) => { mouse[e.button === 0 ? 0 : 1] = false; });
  addEventListener("keydown", (e) => keys.add(e.code)); addEventListener("keyup", (e) => keys.delete(e.code));
  let last = performance.now();
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const q = quatFromEuler([pitch * 180 / Math.PI, yaw * 180 / Math.PI, 0]);
    const head: Vec3 = [player.pos[0], player.pos[1] + 1.6, player.pos[2]];
    const fwd = rotateVector(q, [0, 0, -1]), right = rotateVector(q, [1, 0, 0]);
    for (const [i, hand] of [hands.left, hands.right].entries()) {
      hand.pos = [head[0] + right[0] * (i ? 0.3 : -0.3) + fwd[0] * 0.5, head[1] - 0.3, head[2] + right[2] * (i ? 0.3 : -0.3) + fwd[2] * 0.5];
      hand.aim = fwd; hand.squeeze = keys.has("ShiftLeft") ? 1 : 0;
      updateHand(hand, i ? "right" : "left", mouse[i]!, head);
    }
    if (keys.has("Space") && player.grounded) { player.vel[1] = JUMP; player.grounded = false; }
    simulate(dt, head, [(keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0), (keys.has("KeyS") ? 1 : 0) - (keys.has("KeyW") ? 1 : 0)], yaw);
    fb.reset();
    drawWorld(fb, now / 1000);
    fb.setCamera(composeMatrix(new Float32Array(16), [player.pos[0], player.pos[1] + 1.6, player.pos[2]], q, [1, 1, 1]), 80, 0.1, 1500);
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
function dist(a: Vec3, b: Vec3): number { return len(sub(a, b)); }
function clampLen(a: Vec3, max: number): Vec3 { const l = len(a); return l > max ? scaleV(a, max / l) : a; }
function mulberry32(seed: number): () => number { let a = seed; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
