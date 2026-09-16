import type { MeshUpload } from "../render/host/index.js";

export interface FontGlyph {
  /** Unicode grapheme represented by this glyph. */
  readonly character: string;
  /** Atlas rectangle in pixels, with a top-left origin. */
  readonly rect: readonly [number, number, number, number];
  /** Pen advance in font-design pixels. */
  readonly advance: number;
  /** Offset from the pen to the glyph's top-left corner. */
  readonly bearing: readonly [number, number];
}

export interface FontAtlas {
  readonly id: string;
  readonly material: string;
  readonly textureSize: readonly [number, number];
  readonly designSize: number;
  readonly lineHeight: number;
  readonly glyphs: readonly FontGlyph[];
  /** Optional pair adjustments keyed as `left\0right`, in design pixels. */
  readonly kerning?: Readonly<Record<string, number>>;
  readonly fallbackCharacter?: string;
}

export type TextAlignment = "left" | "center" | "right";

export interface TextLayoutOptions {
  readonly fontSize: number;
  readonly maxWidth?: number;
  readonly lineSpacing?: number;
  readonly alignment?: TextAlignment;
  readonly color?: readonly [number, number, number, number];
}

export interface TextLayoutResult {
  readonly mesh: MeshUpload;
  readonly material: string;
  readonly width: number;
  readonly height: number;
  readonly lineCount: number;
  readonly glyphCount: number;
}

interface PlacedGlyph { glyph: FontGlyph; x: number; line: number; color: readonly [number, number, number, number] }

/** Unicode-aware, renderer-independent atlas text layout. The result is an XY
 * mesh accepted by the same upload path as sprites and procedural geometry. */
export function layoutText2D(font: FontAtlas, text: string, options: TextLayoutOptions): TextLayoutResult {
  const glyphs = validateFont(font);
  const size = positive(options.fontSize, "Text fontSize");
  const scale = size / font.designSize;
  const maxWidth = options.maxWidth === undefined ? Number.POSITIVE_INFINITY : positive(options.maxWidth, "Text maxWidth");
  const spacing = options.lineSpacing ?? 1;
  positive(spacing, "Text lineSpacing");
  const color = options.color ?? [1, 1, 1, 1];
  if (color.length !== 4 || color.some((value) => !Number.isFinite(value))) throw new Error("Text color must contain four finite values.");
  const graphemes = segment(text);
  const lines: PlacedGlyph[][] = [[]];
  const widths: number[] = [0];
  let previous: string | undefined;
  for (const character of graphemes) {
    if (character === "\n") { lines.push([]); widths.push(0); previous = undefined; continue; }
    const glyph = glyphs.get(character) ?? (font.fallbackCharacter === undefined ? undefined : glyphs.get(font.fallbackCharacter));
    if (glyph === undefined) throw new Error(`Font '${font.id}' has no glyph for ${JSON.stringify(character)} and no usable fallbackCharacter.`);
    const line = lines.length - 1;
    const kern = previous === undefined ? 0 : (font.kerning?.[`${previous}\0${character}`] ?? 0) * scale;
    const advance = glyph.advance * scale;
    if (widths[line]! > 0 && widths[line]! + kern + advance > maxWidth) {
      lines.push([]); widths.push(0); previous = undefined;
    }
    const target = lines.length - 1;
    const adjustedKern = previous === undefined ? 0 : kern;
    const x = widths[target]! + adjustedKern;
    lines[target]!.push({ glyph, x, line: target, color });
    widths[target] = x + advance;
    previous = character;
  }
  const width = widths.reduce((largest, value) => Math.max(largest, value), 0);
  const lineHeight = font.lineHeight * scale * spacing;
  const positions: number[] = []; const normals: number[] = []; const uvs: number[] = []; const colors: number[] = []; const indices: number[] = [];
  for (let line = 0; line < lines.length; line += 1) {
    const offset = alignmentOffset(options.alignment ?? "left", width, widths[line]!);
    for (const placed of lines[line]!) appendGlyph(font, placed, scale, offset, -line * lineHeight, positions, normals, uvs, colors, indices);
  }
  return Object.freeze({
    mesh: Object.freeze({ positions: new Float32Array(positions), normals: new Float32Array(normals), uvs: new Float32Array(uvs), colors: new Float32Array(colors), indices: new Uint32Array(indices) }),
    material: font.material,
    width,
    height: lines.length * lineHeight,
    lineCount: lines.length,
    glyphCount: indices.length / 6,
  });
}

function appendGlyph(font: FontAtlas, placed: PlacedGlyph, scale: number, offsetX: number, baselineY: number, positions: number[], normals: number[], uvs: number[], colors: number[], indices: number[]): void {
  const [rx, ry, rw, rh] = placed.glyph.rect;
  if (rw === 0 || rh === 0) return;
  const x0 = offsetX + placed.x + placed.glyph.bearing[0] * scale;
  const y0 = baselineY + placed.glyph.bearing[1] * scale;
  const x1 = x0 + rw * scale; const y1 = y0 - rh * scale;
  const u0 = rx / font.textureSize[0]; const u1 = (rx + rw) / font.textureSize[0];
  const v0 = ry / font.textureSize[1]; const v1 = (ry + rh) / font.textureSize[1];
  const base = positions.length / 3;
  positions.push(x0, y1, 0, x1, y1, 0, x1, y0, 0, x0, y0, 0);
  normals.push(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1);
  uvs.push(u0, v1, u1, v1, u1, v0, u0, v0);
  for (let index = 0; index < 4; index += 1) colors.push(...placed.color);
  indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

// Validated glyph tables are cached per atlas: text is laid out every frame and validation allocated a
// grapheme segmenter per glyph on every call (a 40-label HUD cost ~60 ms per frame).
const validatedFonts = new WeakMap<FontAtlas, Map<string, FontGlyph>>();
function validateFont(font: FontAtlas): Map<string, FontGlyph> {
  const cached = validatedFonts.get(font); if (cached !== undefined) return cached;
  const table = validateFontUncached(font); validatedFonts.set(font, table); return table;
}
function validateFontUncached(font: FontAtlas): Map<string, FontGlyph> {
  if (!font.id.trim() || !font.material.trim()) throw new Error("Font atlas ID and material must not be empty.");
  const [tw, th] = font.textureSize; positive(tw, `Font '${font.id}' texture width`); positive(th, `Font '${font.id}' texture height`);
  positive(font.designSize, `Font '${font.id}' designSize`); positive(font.lineHeight, `Font '${font.id}' lineHeight`);
  const result = new Map<string, FontGlyph>();
  for (const glyph of font.glyphs) {
    if (!glyph.character || segment(glyph.character).length !== 1) throw new Error(`Font '${font.id}' glyph characters must each be one grapheme.`);
    if (result.has(glyph.character)) throw new Error(`Font '${font.id}' declares duplicate glyph ${JSON.stringify(glyph.character)}.`);
    const [x, y, width, height] = glyph.rect;
    if ([x, y, width, height, glyph.advance, ...glyph.bearing].some((value) => !Number.isFinite(value)) || x < 0 || y < 0 || width < 0 || height < 0 || x + width > tw || y + height > th || glyph.advance < 0) throw new Error(`Font '${font.id}' glyph ${JSON.stringify(glyph.character)} has invalid metrics or an out-of-bounds rect.`);
    result.set(glyph.character, glyph);
  }
  return result;
}

function alignmentOffset(alignment: TextAlignment, width: number, lineWidth: number): number { return alignment === "center" ? (width - lineWidth) / 2 : alignment === "right" ? width - lineWidth : 0; }
function positive(value: number, label: string): number { if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be finite and positive.`); return value; }
const graphemeSegmenter = Intl.Segmenter === undefined ? undefined : new Intl.Segmenter(undefined, { granularity: "grapheme" });
function segment(value: string): string[] {
  if (graphemeSegmenter === undefined) return Array.from(value);
  return [...graphemeSegmenter.segment(value)].map((part) => part.segment);
}
