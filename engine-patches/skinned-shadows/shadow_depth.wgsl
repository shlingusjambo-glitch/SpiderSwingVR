// Scalars keep this final lane at 16 bytes. A vec3 after `far` would align to
// the next lane and require 96 bytes, while Rust uploads an 80-byte uniform.
struct ShadowPass {
  view_projection: mat4x4<f32>, far: f32,
  padding_0: f32, padding_1: f32, padding_2: f32,
}
@group(0) @binding(0) var<uniform> shadow: ShadowPass;
// Shared with standard_pbr.wgsl: skinned casters pose their vertices here too.
@group(0) @binding(1) var<storage, read> bone_matrices: array<mat4x4<f32>>;

// Keep this layout identical to standard_pbr.wgsl. The depth pass only needs
// the UV controls, base-colour texture and sampler, but sharing the material
// bind group avoids a second resource graph for alpha-tested casters.
struct MaterialUniforms {
  base_color: vec4<f32>, emission: vec3<f32>, metallic: f32,
  uv_scale: vec2<f32>, uv_offset: vec2<f32>,
  roughness: f32, normal_scale: f32, occlusion_strength: f32, coverage_mode: f32,
  atlas_size: vec2<f32>, atlas_inset: f32, atlas_origin_top_left: f32,
}
@group(1) @binding(0) var<uniform> material: MaterialUniforms;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emission_texture: texture_2d<f32>;
@group(1) @binding(6) var material_sampler: sampler;
struct Input {
  @location(0) position: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(5) texture_layer: f32,
  @location(6) model_0: vec4<f32>, @location(7) model_1: vec4<f32>,
  @location(8) model_2: vec4<f32>, @location(9) model_3: vec4<f32>,
  @location(10) instance_base_color: vec4<f32>,
  @location(11) instance_material: vec4<f32>,
  @location(12) skin: vec4<f32>,
  @location(13) bone_indices: vec4<u32>,
  @location(14) bone_weights: vec4<f32>,
}
struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) tint_alpha: f32,
  @location(2) alpha_cutoff: f32,
  @location(3) @interpolate(flat) texture_layer: f32,
}
@vertex fn vertex_main(input: Input) -> Output {
  let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
  var output: Output;
  var local_position = vec4<f32>(input.position, 1.0);
  let total_weight = input.bone_weights.x + input.bone_weights.y + input.bone_weights.z + input.bone_weights.w;
  if input.skin.x >= 0.0 && total_weight > 0.0001 {
    let base = u32(input.skin.x);
    var skin_matrix = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    skin_matrix = skin_matrix + bone_matrices[base + input.bone_indices.x] * (input.bone_weights.x / total_weight);
    skin_matrix = skin_matrix + bone_matrices[base + input.bone_indices.y] * (input.bone_weights.y / total_weight);
    skin_matrix = skin_matrix + bone_matrices[base + input.bone_indices.z] * (input.bone_weights.z / total_weight);
    skin_matrix = skin_matrix + bone_matrices[base + input.bone_indices.w] * (input.bone_weights.w / total_weight);
    local_position = skin_matrix * local_position;
  }
  output.position = shadow.view_projection * model * local_position;
  output.uv = input.uv * material.uv_scale + material.uv_offset;
  output.tint_alpha = material.base_color.a * input.instance_base_color.a;
  output.alpha_cutoff = clamp(input.instance_material.w, 0.0, 1.0);
  output.texture_layer = input.texture_layer;
  return output;
}
fn surface_uv(local_uv: vec2<f32>, texture_layer: f32) -> vec2<f32> {
  if material.atlas_size.x < 1.0 || material.atlas_size.y < 1.0 { return local_uv; }
  let cell = vec2<f32>(max(0.0, floor(texture_layer + 0.5)) % material.atlas_size.x,
    floor(max(0.0, floor(texture_layer + 0.5)) / material.atlas_size.x));
  let inset = clamp(material.atlas_inset, 0.0, 0.4999);
  let repeated = clamp(fract(local_uv), vec2(inset), vec2(1.0 - inset));
  if material.atlas_origin_top_left > 0.5 {
    return vec2((cell.x + repeated.x) / material.atlas_size.x,
      (cell.y + 1.0 - repeated.y) / material.atlas_size.y);
  }
  return vec2((cell.x + repeated.x) / material.atlas_size.x,
    1.0 - (cell.y + repeated.y) / material.atlas_size.y);
}
@fragment fn fragment_main(input: Output) {
  // A zero cutoff is the opaque fast-path semantically: every valid alpha
  // survives. MASK instances use the same anti-aliased/SDF coverage policy as
  // the colour pass before writing depth.
  let alpha = textureSample(base_color_texture, material_sampler,
    surface_uv(input.uv, input.texture_layer)).a;
  let width = max(fwidth(alpha), 0.0001);
  let coverage = select(alpha, smoothstep(0.5 - width, 0.5 + width, alpha),
    material.coverage_mode > 0.5);
  if coverage * input.tint_alpha < input.alpha_cutoff { discard; }
}
