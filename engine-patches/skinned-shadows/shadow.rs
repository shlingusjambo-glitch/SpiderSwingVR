//! Directional cascaded shadow-map resources and stable cascade fitting.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use super::{
    InstanceData, ViewState,
    lighting::{GpuLight, LIGHT_KIND_DIRECTIONAL, LIGHT_KIND_POINT, LIGHT_KIND_SPOT},
    mesh::VertexLayout,
    pipeline::{DEPTH_FORMAT, uniform_buffer},
};
use crate::mesh::MeshVertex;

pub(crate) const SHADOW_CASCADE_COUNT: usize = 4;
pub(crate) const SHADOW_MAP_SIZE: u32 = 2048;

/// Atlas budget for point and spot shadows: 12 slots. A point light needs 6
/// consecutive slots (one per cube face); a spot light needs 1. That's room
/// for two full point lights, or one point light plus six spot lights, or
/// twelve spot lights, in any mix — [`assign_shadow_slots`] documents how a
/// scene that asks for more than fits gets prioritised.
pub(crate) const MAX_SHADOW_SLOTS: usize = 12;
/// Resolution of each point/spot shadow face. Smaller than the directional
/// cascades because there are up to twelve of them alive at once.
pub(crate) const POINT_SPOT_SHADOW_SIZE: u32 = 512;

/// One point or spot light that was granted atlas slots this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ShadowAssignment {
    pub(crate) light_index: usize,
    pub(crate) base_slot: usize,
    pub(crate) slot_count: usize,
}

/// Decides which point/spot lights render into the shared atlas this frame.
///
/// Eviction policy: every shadow-casting point/spot light is scored by
/// `intensity / distance_to_camera^2` (closer, brighter lights win). Point
/// lights need 6 consecutive slots, spot lights need 1. Candidates are
/// visited highest-priority first and packed greedily into the remaining
/// budget; a light that no longer fits is skipped entirely (rendered lit but
/// unshadowed) rather than fragmenting slots or preempting a light already
/// placed — so the assignment for a given frame is deterministic and cheap to
/// compute every frame with no persistent state.
pub(crate) fn assign_shadow_slots(
    lights: &[GpuLight],
    camera: Vec3,
    budget: usize,
) -> Vec<ShadowAssignment> {
    let mut candidates: Vec<(usize, f32, usize)> = lights
        .iter()
        .enumerate()
        .filter_map(|(index, light)| {
            let kind = light.vector_kind[3];
            let slots = if (kind - LIGHT_KIND_POINT).abs() < 0.1 {
                6
            } else if (kind - LIGHT_KIND_SPOT).abs() < 0.1 {
                1
            } else {
                return None;
            };
            if light.color_intensity[3] <= 0.0 {
                return None;
            }
            let position = Vec3::from_slice(&light.vector_kind[..3]);
            let distance = position.distance(camera).max(1.0);
            let priority = light.color_intensity[3] / (distance * distance);
            Some((index, priority, slots))
        })
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut used = 0usize;
    let mut assignments = Vec::new();
    for (light_index, _priority, slot_count) in candidates {
        if used + slot_count > budget {
            continue;
        }
        assignments.push(ShadowAssignment {
            light_index,
            base_slot: used,
            slot_count,
        });
        used += slot_count;
    }
    assignments
}

/// The six cube-face view matrices for a point light at `position`, in the
/// standard cubemap face order (+X, -X, +Y, -Y, +Z, -Z) with the up vectors
/// that convention expects.
pub(crate) fn cube_face_view_matrices(position: Vec3) -> [Mat4; 6] {
    let directions: [(Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ];
    directions.map(|(forward, up)| Mat4::look_at_rh(position, position + forward, up))
}

/// A 90-degree symmetric perspective projection, matched to one cube face.
pub(crate) fn point_shadow_projection(near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far.max(near + 0.01))
}

/// The single view-projection matrix a spot light's shadow slot renders with.
pub(crate) fn spot_shadow_matrix(light: &GpuLight) -> Mat4 {
    let position = Vec3::from_slice(&light.vector_kind[..3]);
    let direction = Vec3::from_slice(&light.secondary_range[..3]).normalize_or(Vec3::NEG_Y);
    let range = light.secondary_range[3].max(0.1);
    let cos_outer = light.cone[1].clamp(-0.999, 0.999);
    let outer_angle = cos_outer.acos().max(0.01);
    let up = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(position, position + direction, up);
    let projection = Mat4::perspective_rh((outer_angle * 2.0).min(3.0), 1.0, 0.05, range);
    projection * view
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ShadowPassUniform {
    pub(crate) view_projection: [[f32; 4]; 4],
    pub(crate) far: f32,
    pub(crate) _padding: [f32; 3],
}

pub(crate) struct ShadowGpu {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) _texture: wgpu::Texture,
    pub(crate) sampling_view: wgpu::TextureView,
    pub(crate) layer_views: Vec<wgpu::TextureView>,
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) cascade_buffers: Vec<wgpu::Buffer>,
    pub(crate) cascade_bind_groups: Vec<wgpu::BindGroup>,
    /// Point/spot atlas: [`MAX_SHADOW_SLOTS`] depth layers sharing the same
    /// pass uniform layout as the directional cascades, so the existing
    /// shadow pipeline draws into either without a second pipeline.
    pub(crate) _point_spot_texture: wgpu::Texture,
    pub(crate) point_spot_sampling_view: wgpu::TextureView,
    pub(crate) point_spot_layer_views: Vec<wgpu::TextureView>,
    pub(crate) point_spot_buffers: Vec<wgpu::Buffer>,
    pub(crate) point_spot_bind_groups: Vec<wgpu::BindGroup>,
}

impl ShadowGpu {
    pub(crate) fn new(device: &wgpu::Device, bone_buffer: &wgpu::Buffer) -> Self {
        // Binding 1 is the same bone palette the colour pass reads, so skinned
        // casters shadow their posed shape instead of their bind pose.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vapour shadow-pass layout"),
            entries: &[
                super::uniform_layout_entry(0, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vapour directional cascade maps"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: SHADOW_CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sampling_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Vapour shadow sampling view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..SHADOW_CASCADE_COUNT)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Vapour shadow layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Vapour shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut cascade_buffers = Vec::new();
        let mut cascade_bind_groups = Vec::new();
        for index in 0..SHADOW_CASCADE_COUNT {
            let buffer = uniform_buffer::<ShadowPassUniform>(
                device,
                &format!("Vapour shadow cascade {index}"),
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Vapour shadow cascade bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bone_buffer.as_entire_binding() },
                ],
            });
            cascade_buffers.push(buffer);
            cascade_bind_groups.push(bind_group);
        }

        let point_spot_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vapour point/spot shadow atlas"),
            size: wgpu::Extent3d {
                width: POINT_SPOT_SHADOW_SIZE,
                height: POINT_SPOT_SHADOW_SIZE,
                depth_or_array_layers: MAX_SHADOW_SLOTS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let point_spot_sampling_view =
            point_spot_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Vapour point/spot shadow sampling view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let point_spot_layer_views = (0..MAX_SHADOW_SLOTS)
            .map(|slot| {
                point_spot_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Vapour point/spot shadow slot"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: slot as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let mut point_spot_buffers = Vec::new();
        let mut point_spot_bind_groups = Vec::new();
        for slot in 0..MAX_SHADOW_SLOTS {
            let buffer =
                uniform_buffer::<ShadowPassUniform>(device, &format!("Vapour shadow slot {slot}"));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Vapour point/spot shadow slot bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bone_buffer.as_entire_binding() },
                ],
            });
            point_spot_buffers.push(buffer);
            point_spot_bind_groups.push(bind_group);
        }

        Self {
            layout,
            _texture: texture,
            sampling_view,
            layer_views,
            sampler,
            cascade_buffers,
            cascade_bind_groups,
            _point_spot_texture: point_spot_texture,
            point_spot_sampling_view,
            point_spot_layer_views,
            point_spot_buffers,
            point_spot_bind_groups,
        }
    }
}

pub(crate) fn create_shadow_pipeline(
    device: &wgpu::Device,
    shadow_layout: &wgpu::BindGroupLayout,
    material_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Vapour shadow depth shader"),
        source: wgpu::ShaderSource::Wgsl(crate::SHADOW_DEPTH_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Vapour shadow pipeline layout"),
        bind_group_layouts: &[Some(shadow_layout), Some(material_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Vapour directional shadow pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &[
                Some(<MeshVertex as VertexLayout>::layout()),
                Some(InstanceData::layout()),
                Some(crate::mesh::SkinVertex::layout()),
            ],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        // The fragment stage samples base-colour alpha so foliage, fences and
        // other MASK materials cast silhouettes instead of solid quads.
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment_main"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    })
}

pub(crate) fn shadow_cascades(
    view: &ViewState,
    aspect: f32,
) -> [ShadowPassUniform; SHADOW_CASCADE_COUNT] {
    let direction = view
        .lights
        .iter()
        .find(|light| (light.vector_kind[3] - LIGHT_KIND_DIRECTIONAL).abs() < 0.1)
        .map(|light| Vec3::from_slice(&light.vector_kind[..3]))
        .unwrap_or(Vec3::new(-0.4, -1.0, -0.35))
        .normalize();
    let near = view.near.max(0.01);
    let far = view.far.min(200.0).max(near + 0.01);
    let mut splits = [near; SHADOW_CASCADE_COUNT + 1];
    for (index, split) in splits.iter_mut().enumerate().skip(1) {
        let t = index as f32 / SHADOW_CASCADE_COUNT as f32;
        *split = (near * (far / near).powf(t)) * 0.75 + (near + (far - near) * t) * 0.25;
    }
    let position = view.camera_world.transform_point3(Vec3::ZERO);
    let forward = view.camera_world.transform_vector3(Vec3::NEG_Z).normalize();
    std::array::from_fn(|index| {
        let slice_near = splits[index];
        let slice_far = splits[index + 1];
        let middle = (slice_near + slice_far) * 0.5;
        let half_height = if view.projection == crate::CameraProjection::Perspective {
            (view.field_of_view_degrees.to_radians() * 0.5).tan() * slice_far
        } else {
            view.orthographic_size * 0.5
        };
        let half_width = half_height * aspect;
        let half_depth = (slice_far - slice_near) * 0.5;
        let radius =
            (half_width * half_width + half_height * half_height + half_depth * half_depth)
                .sqrt()
                .max(0.01);
        let center = position + forward * middle;
        let texel = radius * 2.0 / SHADOW_MAP_SIZE as f32;
        let light_up = if direction.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let basis = Mat4::look_at_rh(Vec3::ZERO, direction, light_up);
        let light_center = basis.transform_point3(center);
        let snapped = Vec3::new(
            (light_center.x / texel).round() * texel,
            (light_center.y / texel).round() * texel,
            light_center.z,
        );
        let target = basis.inverse().transform_point3(snapped);
        let eye = target - direction * (radius + 50.0);
        let matrix =
            Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.01, radius * 2.0 + 50.0)
                * Mat4::look_at_rh(eye, target, light_up);
        ShadowPassUniform {
            view_projection: matrix.to_cols_array_2d(),
            far: slice_far,
            _padding: [0.0; 3],
        }
    })
}

pub(crate) fn has_directional_shadow(view: &ViewState) -> bool {
    view.lights.iter().any(|light| {
        (light.vector_kind[3] - LIGHT_KIND_DIRECTIONAL).abs() < 0.1
            && light.color_intensity[3] > 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_uniform_matches_wgsl_buffer_layout() {
        let module = naga::front::wgsl::parse_str(crate::SHADOW_DEPTH_SHADER).unwrap();
        let (_, uniform) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("ShadowPass"))
            .expect("shadow shader must declare its pass uniform");
        let naga::TypeInner::Struct { members, span } = &uniform.inner else {
            panic!("ShadowPass must be a uniform struct");
        };
        assert_eq!(
            *span as usize,
            std::mem::size_of::<ShadowPassUniform>(),
            "WGSL uniform span must fit the Rust buffer bound at draw time"
        );
        assert_eq!(members[0].offset, 0);
        assert_eq!(
            members[1].offset as usize,
            std::mem::offset_of!(ShadowPassUniform, far)
        );
    }

    #[test]
    fn cascades_are_ordered_and_end_at_the_shadow_distance() {
        let cascades = shadow_cascades(&ViewState::default(), 16.0 / 9.0);
        assert!(cascades.windows(2).all(|pair| pair[0].far < pair[1].far));
        assert!((cascades[3].far - 200.0).abs() < 0.001);
        assert!(
            cascades
                .iter()
                .flat_map(|cascade| cascade.view_projection)
                .flatten()
                .all(f32::is_finite)
        );
    }

    #[test]
    fn sub_texel_camera_motion_keeps_the_light_projection_stable() {
        let view = ViewState::default();
        let first = shadow_cascades(&view, 1.0);
        let texel_scale = 0.00001;
        let mut moved = view.clone();
        moved.camera_world.w_axis.x += texel_scale;
        let second = shadow_cascades(&moved, 1.0);
        let delta = first[0].view_projection[3][0] - second[0].view_projection[3][0];
        assert!(delta.abs() < 0.0001, "cascade projection moved by {delta}");
    }

    #[test]
    fn each_cube_face_view_matrix_maps_its_own_direction_straight_ahead() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let matrices = cube_face_view_matrices(position);
        let directions = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        for (face, direction) in directions.into_iter().enumerate() {
            // A point 10 units out along the face's own direction must land
            // on the camera-space -Z axis (straight ahead), centred in x/y,
            // for every face — and only that face.
            let probe = position + direction * 10.0;
            let view_space = matrices[face].transform_point3(probe);
            assert!(
                view_space.x.abs() < 1e-4 && view_space.y.abs() < 1e-4,
                "face {face} did not centre its own probe point: {view_space:?}"
            );
            assert!(
                view_space.z < -9.999,
                "face {face} did not put its probe on -Z: {view_space:?}"
            );
        }
    }

    #[test]
    fn point_shadow_projection_is_a_symmetric_ninety_degree_frustum() {
        let projection = point_shadow_projection(0.1, 25.0);
        // A point on the +45-degree edge of a 90-degree FOV frustum sits
        // exactly on the clip-space boundary (x == w after projection).
        let edge = projection * glam::Vec4::new(1.0, 0.0, -1.0, 1.0);
        assert!((edge.x - edge.w).abs() < 1e-4, "edge={edge:?}");
    }

    #[test]
    fn shadow_slots_are_granted_to_the_highest_priority_lights_first() {
        let bright_near = GpuLight {
            vector_kind: [0.0, 0.0, 0.0, LIGHT_KIND_POINT],
            color_intensity: [1.0, 1.0, 1.0, 10.0],
            secondary_range: [0.0, 0.0, 0.0, 20.0],
            cone: [0.0; 4],
        };
        let dim_far = GpuLight {
            vector_kind: [50.0, 0.0, 0.0, LIGHT_KIND_POINT],
            color_intensity: [1.0, 1.0, 1.0, 1.0],
            secondary_range: [0.0, 0.0, 0.0, 20.0],
            cone: [0.0; 4],
        };
        let spot = GpuLight {
            vector_kind: [0.0, 0.0, 5.0, LIGHT_KIND_SPOT],
            color_intensity: [1.0, 1.0, 1.0, 5.0],
            secondary_range: [0.0, -1.0, 0.0, 20.0],
            cone: [0.9, 0.8, 0.0, 0.0],
        };
        let lights = [bright_near, dim_far, spot];
        // Budget for one point light (6) plus the spot (1); the second point
        // light (needs 6 more, only 5 left) must be skipped, not fragmented.
        let assignments = assign_shadow_slots(&lights, Vec3::ZERO, 7);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].light_index, 0);
        assert_eq!(assignments[0].base_slot, 0);
        assert_eq!(assignments[0].slot_count, 6);
        assert_eq!(assignments[1].light_index, 2);
        assert_eq!(assignments[1].base_slot, 6);
        assert_eq!(assignments[1].slot_count, 1);
    }

    #[test]
    fn shadow_slots_never_exceed_the_budget() {
        // Twelve point lights (72 slots of demand) against the real
        // MAX_SHADOW_SLOTS budget must still pack to exactly two full lights.
        let lights: Vec<GpuLight> = (0..12)
            .map(|i| GpuLight {
                vector_kind: [i as f32 * 5.0, 0.0, 0.0, LIGHT_KIND_POINT],
                color_intensity: [1.0, 1.0, 1.0, 1.0],
                secondary_range: [0.0, 0.0, 0.0, 20.0],
                cone: [0.0; 4],
            })
            .collect();
        let assignments = assign_shadow_slots(&lights, Vec3::ZERO, MAX_SHADOW_SLOTS);
        let used: usize = assignments.iter().map(|a| a.slot_count).sum();
        assert!(used <= MAX_SHADOW_SLOTS);
        assert_eq!(assignments.len(), 2);
    }

    #[test]
    fn rigs_without_a_live_directional_light_disable_shadow_sampling() {
        let mut view = ViewState::default();
        view.lights.clear();
        assert!(!has_directional_shadow(&view));
        view.lights
            .push(super::super::lighting::GpuLight::directional(
                Vec3::NEG_Y,
                Vec3::ONE,
                0.0,
            ));
        assert!(!has_directional_shadow(&view));
    }
}
