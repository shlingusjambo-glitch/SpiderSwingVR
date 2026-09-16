//! The WebGPU surface renderer.
//!
//! This module owns frame orchestration: it validates an incoming frame, sorts
//! and batches submissions, and drives the compiled render graph. The pieces it
//! orchestrates live in sibling modules — meshes, materials, lighting, view
//! state, per-frame types, errors, and pipeline construction — so that adding a
//! pass does not mean growing one file without bound.

mod culling;
mod decal;
mod environment;
mod error;
mod frame;
mod lighting;
mod material;
mod mesh;
mod particles;
mod pipeline;
mod postprocess;
mod render_target;
mod shadow;
mod texture_mips;
mod view;

pub(crate) use culling::*;
pub use culling::{frustum_planes, sphere_inside, world_bounds};
pub(crate) use decal::*;
pub use decal::{DECAL_STRIDE, MAX_DECALS, decal_atlas_uv, decal_local_uv};
pub(crate) use environment::*;
pub use environment::{
    BRDF_LUT_SIZE, IRRADIANCE_FACE_SIZE, MAX_ENVIRONMENT_FACE_SIZE, PREFILTER_MIN_FACE_SIZE,
    SKY_CAPTURE_FACE_SIZE, cube_face_direction, cube_face_uv, mip_roughness, prefiltered_mip_count,
};
pub use error::*;
pub use frame::*;
pub use lighting::*;
pub use material::*;
pub(crate) use mesh::*;
pub(crate) use particles::*;
pub use particles::{
    GpuParticle, MAX_PARTICLE_CAPACITY, PARTICLE_LUT_SAMPLES, PARTICLE_STRIDE, ParticleBlend,
    ParticleCounters, ParticleError, ParticleParams, ParticleShapeKind, ParticleSystemDescriptor,
    ParticleSystemStats, seed_state, spawn_particle, step_cpu,
};
pub use pipeline::MAX_BONE_MATRICES;
pub(crate) use pipeline::*;
pub use postprocess::{AntiAliasing, DebugView, PostProcessSettings, ToneMapping};
pub(crate) use postprocess::{HDR_FORMAT, HdrTarget, PostProcessGpu};
pub(crate) use render_target::*;
pub use render_target::{RenderTargetSize, detect_render_target_cycle, target_extent};
pub(crate) use shadow::*;
pub(crate) use view::*;

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use lighting::{LIGHT_KIND_POINT, LIGHT_KIND_SPOT};

/// One layer's shadow data for the frame just being encoded: the directional
/// cascades plus the point/spot atlas slots `update_uniforms` filled in.
struct LayerShadowData {
    /// The layer's camera view-projection, which frustum culling shares.
    view_projection: Mat4,
    cascades: [ShadowPassUniform; SHADOW_CASCADE_COUNT],
    point_spot: [ShadowPassUniform; MAX_SHADOW_SLOTS],
    point_spot_used: usize,
}

use crate::{
    CameraProjection, CompiledRenderGraph, GpuContext, MAX_RENDER_INSTANCES, PassKind, RenderWorld,
    RenderWorldError, ShaderLibrary,
    clock::Stopwatch,
    mesh::{BuiltinMesh, MeshData},
};

pub struct SurfaceRenderer<'surface> {
    pub context: GpuContext,
    surface: wgpu::Surface<'surface>,
    config: wgpu::SurfaceConfiguration,
    pipelines: HashMap<PipelineKind, wgpu::RenderPipeline>,
    frame_layout: wgpu::BindGroupLayout,
    frame_uniforms: Vec<FrameGpu>,
    shadows: ShadowGpu,
    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    material_layout: wgpu::BindGroupLayout,
    default_textures: Vec<TextureGpu>,
    textures: HashMap<String, TextureGpu>,
    materials: MaterialRegistry,
    meshes: MeshRegistry,
    layers: Vec<RenderLayerState>,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    culling: CullState,
    particles: ParticlesGpu,
    particle_systems: HashMap<String, ParticleSystemGpu>,
    prefer_gpu_particles: bool,
    /// Bone matrices for this frame's skinned instances; fixed capacity, see
    /// [`MAX_BONE_MATRICES`].
    bone_buffer: wgpu::Buffer,
    environment: EnvironmentGpu,
    decals: DecalGpu,
    render_targets: HashMap<String, RenderTargetGpu>,
    instance_count: u32,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    frame_graph: CompiledRenderGraph,
    hdr_target: HdrTarget,
    post_process: PostProcessGpu,
    post_process_settings: PostProcessSettings,
}

impl<'surface> SurfaceRenderer<'surface> {
    pub fn new(
        context: GpuContext,
        surface: wgpu::Surface<'surface>,
        width: u32,
        height: u32,
    ) -> Result<Self, SurfaceRendererError> {
        let width = width.max(1);
        let height = height.max(1);
        let mut config = surface
            .get_default_config(&context.adapter, width, height)
            .ok_or(SurfaceRendererError::UnsupportedSurface)?;
        let capabilities = surface.get_capabilities(&context.adapter);
        if let Some(format) = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
        {
            config.format = format;
        }
        config.present_mode = if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };
        surface.configure(&context.device, &config);

        let shader_library = ShaderLibrary::standard()?;
        let shader = shader_library.create_module(&context.device, "vapour.standard-pbr")?;
        let sky_shader = shader_library.create_module(&context.device, "vapour.procedural-sky")?;
        let bone_buffer = create_bone_buffer(&context.device);
        let shadows = ShadowGpu::new(&context.device, &bone_buffer);
        let frame_layout = create_frame_layout(&context.device);
        let material_layout = create_material_layout(&context.device);
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Vapour standard PBR pipeline layout"),
                    bind_group_layouts: &[Some(&frame_layout), Some(&material_layout)],
                    immediate_size: 0,
                });
        let pipelines =
            create_standard_pipelines(&context.device, &pipeline_layout, &shader, HDR_FORMAT);
        let shadow_pipeline =
            create_shadow_pipeline(&context.device, &shadows.layout, &material_layout);
        let sky_pipeline = create_sky_pipeline(&context.device, &frame_layout, &sky_shader);
        let decal_shader = shader_library.create_module(&context.device, "vapour.decal")?;
        let decals = DecalGpu::new(&context.device, &frame_layout, &decal_shader, HDR_FORMAT);
        let cull_shader = shader_library.create_module(&context.device, "vapour.cull-instances")?;
        let culling = CullState::new(&context.device, &cull_shader);
        let particles = ParticlesGpu::new(
            &context.device,
            &frame_layout,
            &shader_library.create_module(&context.device, "vapour.particles-sim")?,
            &shader_library.create_module(&context.device, "vapour.particles")?,
        );

        let environment = EnvironmentGpu::new(&context.device, &context.queue);
        let frame_uniforms = vec![create_frame_gpu(
            &context.device,
            &frame_layout,
            &shadows,
            &bone_buffer,
            &environment,
            0,
        )];

        let default_textures = DEFAULT_MATERIAL_TEXTURES
            .iter()
            .map(|(label, color, srgb)| solid_texture(&context, label, *color, *srgb))
            .collect::<Vec<_>>();
        let textures = HashMap::new();
        let default_definition =
            normalize_material("builtin:default", &StandardMaterialDescriptor::default())?;
        let default_material = create_material_gpu(
            &context,
            &material_layout,
            &default_textures,
            &textures,
            "builtin:default",
            default_definition,
        )?;
        let mut materials = MaterialRegistry::default();
        let default_slot = materials.insert("builtin:default", default_material);
        debug_assert_eq!(default_slot, 0);

        // Built-in meshes are generated and uploaded once at startup. They are
        // small, and holding them resident keeps switching primitives in the
        // editor instant.
        let mut meshes = MeshRegistry::default();
        for builtin in BuiltinMesh::ALL {
            meshes.insert(
                builtin.id(),
                MeshGpu::upload(&context.device, builtin.id(), &builtin.build()),
            );
        }
        let instance_capacity = 1;
        let instance_buffer = create_instance_buffer(&context.device, instance_capacity);
        let (depth_texture, depth_view) = create_depth(&context.device, width, height);
        let frame_graph = create_frame_graph()?;
        let hdr_target = HdrTarget::new(&context.device, width, height);
        let post_process = PostProcessGpu::new(&context.device, config.format);

        Ok(Self {
            context,
            surface,
            config,
            pipelines,
            frame_layout,
            frame_uniforms,
            shadows,
            shadow_pipeline,
            sky_pipeline,
            material_layout,
            default_textures,
            textures,
            materials,
            meshes,
            layers: Vec::new(),
            instance_buffer,
            instance_capacity,
            culling,
            particles,
            particle_systems: HashMap::new(),
            prefer_gpu_particles: true,
            instance_count: 0,
            bone_buffer,
            environment,
            decals,
            render_targets: HashMap::new(),
            depth_texture,
            depth_view,
            frame_graph,
            hdr_target,
            post_process,
            post_process_settings: PostProcessSettings::default(),
        })
    }

    /// Prefers the compute culling path (the default) or forces the CPU
    /// fallback. The GPU path also needs `RenderCapabilities::gpu_culling`;
    /// `FrameStats::gpu_culling` reports which one actually ran.
    pub fn set_gpu_culling(&mut self, enabled: bool) {
        self.culling.prefer_gpu = enabled;
    }

    #[must_use]
    pub fn gpu_culling_preferred(&self) -> bool {
        self.culling.prefer_gpu
    }

    /// Whether systems created from now on simulate on the GPU (the default)
    /// or on the CPU fallback. Existing systems keep the path they were
    /// created with; `ParticleSystemStats::gpu` reports it.
    pub fn set_gpu_particles(&mut self, enabled: bool) {
        self.prefer_gpu_particles = enabled;
    }

    /// Creates a particle system, replacing any with the same identifier.
    pub fn create_particle_system(
        &mut self,
        id: &str,
        descriptor: ParticleSystemDescriptor,
    ) -> Result<(), SurfaceRendererError> {
        descriptor.validate()?;
        let mut system = ParticleSystemGpu::new(
            &self.context.device,
            &self.particles,
            descriptor,
            self.prefer_gpu_particles,
        );
        system.reset(&self.context.queue);
        self.particle_systems.insert(id.to_owned(), system);
        Ok(())
    }

    /// Spawns `spawn` particles at `origin` and advances the system by
    /// `dt` seconds. On the GPU path this submits the compute work now.
    pub fn update_particle_system(
        &mut self,
        id: &str,
        dt: f32,
        spawn: u32,
        origin: [f32; 3],
    ) -> Result<(), SurfaceRendererError> {
        if !dt.is_finite() || dt < 0.0 || !origin.iter().all(|v| v.is_finite()) {
            return Err(ParticleError::Invalid(
                "update needs a finite, non-negative delta and a finite origin",
            )
            .into());
        }
        let system = self
            .particle_systems
            .get_mut(id)
            .ok_or_else(|| ParticleError::Missing(id.to_owned()))?;
        system.step(
            &self.context.device,
            &self.context.queue,
            &self.particles,
            dt,
            spawn,
            origin,
        );
        Ok(())
    }

    /// Removes every particle and restarts the system's generator.
    pub fn clear_particle_system(&mut self, id: &str) -> Result<(), SurfaceRendererError> {
        let system = self
            .particle_systems
            .get_mut(id)
            .ok_or_else(|| ParticleError::Missing(id.to_owned()))?;
        system.reset(&self.context.queue);
        Ok(())
    }

    pub fn release_particle_system(&mut self, id: &str) -> bool {
        self.particle_systems.remove(id).is_some()
    }

    pub fn particle_system_stats(
        &self,
        id: &str,
    ) -> Result<ParticleSystemStats, SurfaceRendererError> {
        Ok(self
            .particle_systems
            .get(id)
            .ok_or_else(|| ParticleError::Missing(id.to_owned()))?
            .stats())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.context.device, &self.config);
        (self.depth_texture, self.depth_view) = create_depth(&self.context.device, width, height);
        self.hdr_target = HdrTarget::new(&self.context.device, width, height);
        let ids: Vec<String> = self.render_targets.keys().cloned().collect();
        for id in ids {
            let target = self.render_targets.get_mut(&id).expect("listed key");
            if target.resize(&self.context.device, &id, width, height) {
                let texture = target.as_texture();
                // Cannot fail: the materials already bound this identifier.
                let _ = self.replace_texture(&id, texture);
            }
        }
    }

    /// Creates or resizes a render target that layers can render into and
    /// materials can sample under `id`. Surface-scaled targets reallocate on
    /// resize and rebuild every material bound to them.
    pub fn create_render_target(
        &mut self,
        id: &str,
        size: RenderTargetSize,
    ) -> Result<(), SurfaceRendererError> {
        validate_asset_id("texture", id)?;
        if self.textures.contains_key(id) && !self.render_targets.contains_key(id) {
            return Err(SurfaceRendererError::InvalidRenderTarget {
                id: id.to_owned(),
                reason: "an uploaded texture already uses this identifier",
            });
        }
        validate_target_size(
            id,
            size,
            self.context.device.limits().max_texture_dimension_2d,
        )?;
        let target = RenderTargetGpu::new(
            &self.context.device,
            id,
            size,
            self.config.width,
            self.config.height,
        );
        let texture = target.as_texture();
        self.render_targets.insert(id.to_owned(), target);
        self.replace_texture(id, texture)
    }

    /// Releases a render target. Fails while a material still samples it.
    pub fn release_render_target(&mut self, id: &str) -> Result<bool, SurfaceRendererError> {
        if !self.render_targets.contains_key(id) {
            return Ok(false);
        }
        self.release_texture(id)?;
        Ok(self.render_targets.remove(id).is_some())
    }

    #[must_use]
    pub fn render_target_extent(&self, id: &str) -> Option<(u32, u32)> {
        self.render_targets
            .get(id)
            .map(|target| (target.width, target.height))
    }

    pub fn set_post_process(
        &mut self,
        settings: PostProcessSettings,
    ) -> Result<(), SurfaceRendererError> {
        self.post_process_settings = settings
            .validate()
            .map_err(|reason| SurfaceRendererError::InvalidPostProcess { reason })?;
        Ok(())
    }

    #[must_use]
    pub fn post_process(&self) -> PostProcessSettings {
        self.post_process_settings
    }

    /// Uploads or replaces a runtime mesh under `id`.
    ///
    /// Replacing keeps the identifier's slot, so instances already referencing
    /// it show the new geometry on the next frame — the path a remeshed voxel
    /// chunk takes. Identifiers beginning with `builtin:` are reserved so game
    /// content cannot shadow a primitive the editor offers.
    pub fn upload_mesh(&mut self, id: &str, data: &MeshData) -> Result<(), SurfaceRendererError> {
        if id.trim().is_empty() {
            return Err(SurfaceRendererError::InvalidMeshId {
                id: id.to_owned(),
                reason: "identifier is empty",
            });
        }
        if id.starts_with("builtin:") && BuiltinMesh::ALL.iter().all(|mesh| mesh.id() != id) {
            return Err(SurfaceRendererError::InvalidMeshId {
                id: id.to_owned(),
                reason: "the 'builtin:' prefix is reserved for engine primitives",
            });
        }
        data.validate()?;
        let mesh = MeshGpu::upload(&self.context.device, id, data);
        self.meshes.insert(id, mesh);
        Ok(())
    }

    /// Releases a runtime mesh. Returns whether it existed. Built-in primitives
    /// cannot be released; the editor always needs them.
    pub fn release_mesh(&mut self, id: &str) -> Result<bool, SurfaceRendererError> {
        if BuiltinMesh::ALL.iter().any(|mesh| mesh.id() == id) {
            return Err(SurfaceRendererError::InvalidMeshId {
                id: id.to_owned(),
                reason: "built-in primitives cannot be released",
            });
        }
        Ok(self.meshes.remove(id))
    }

    #[must_use]
    pub fn mesh_ids(&self) -> Vec<&str> {
        self.meshes.ids()
    }

    #[must_use]
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Uploads or replaces an RGBA8 texture. Replacing automatically rebuilds
    /// every referencing material bind group, so atlas hot reload is atomic
    /// from the next submitted frame.
    pub fn upload_texture_rgba8(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        pixels: &[u8],
        srgb: bool,
    ) -> Result<(), SurfaceRendererError> {
        self.upload_texture_rgba8_with_mipmaps(id, width, height, pixels, srgb, false)
    }

    /// Optionally builds a complete mip chain once at upload time. Keep this
    /// disabled for grid atlases, whose cells need separately authored mipmaps.
    pub fn upload_texture_rgba8_with_mipmaps(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        pixels: &[u8],
        srgb: bool,
        mipmaps: bool,
    ) -> Result<(), SurfaceRendererError> {
        validate_asset_id("texture", id)?;
        if width == 0 || height == 0 {
            return Err(SurfaceRendererError::InvalidTextureId {
                id: id.to_owned(),
                reason: "dimensions must be positive",
            });
        }
        let max_dimension = self.context.device.limits().max_texture_dimension_2d;
        if width > max_dimension || height > max_dimension {
            return Err(SurfaceRendererError::InvalidTextureId {
                id: id.to_owned(),
                reason: "dimensions exceed the device maxTextureDimension2D limit",
            });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|area| area.checked_mul(4))
            .ok_or_else(|| SurfaceRendererError::InvalidTextureId {
                id: id.to_owned(),
                reason: "RGBA8 dimensions exceed the addressable upload size",
            })?;
        if pixels.len() != expected {
            return Err(SurfaceRendererError::TextureDataLength {
                id: id.to_owned(),
                expected,
                actual: pixels.len(),
            });
        }
        if self.render_targets.contains_key(id) {
            return Err(SurfaceRendererError::InvalidTextureId {
                id: id.to_owned(),
                reason: "a render target already uses this identifier",
            });
        }
        let texture = texture_rgba8(&self.context, id, width, height, pixels, srgb, mipmaps);
        self.replace_texture(id, texture)
    }

    /// Installs `texture` under `id` and rebuilds every material bound to
    /// that identifier, so a replacement is atomic from the next frame.
    fn replace_texture(
        &mut self,
        id: &str,
        texture: TextureGpu,
    ) -> Result<(), SurfaceRendererError> {
        self.textures.insert(id.to_owned(), texture);

        let definitions: Vec<(usize, String, MaterialDefinition)> = self
            .materials
            .slots
            .iter()
            .filter_map(|(material_id, slot)| {
                let material = self.materials.get(*slot)?;
                material
                    .definition
                    .textures
                    .iter()
                    .any(|texture_id| texture_id.as_deref() == Some(id))
                    .then(|| (*slot, material_id.clone(), material.definition.clone()))
            })
            .collect();
        let replacements: Result<Vec<_>, _> = definitions
            .into_iter()
            .map(|(slot, material_id, definition)| {
                create_material_gpu(
                    &self.context,
                    &self.material_layout,
                    &self.default_textures,
                    &self.textures,
                    &material_id,
                    definition,
                )
                .map(|material| (slot, material))
            })
            .collect();
        for (slot, material) in replacements? {
            self.materials.materials[slot] = Some(material);
        }
        Ok(())
    }

    pub fn release_texture(&mut self, id: &str) -> Result<bool, SurfaceRendererError> {
        for (material_id, slot) in &self.materials.slots {
            let Some(material) = self.materials.get(*slot) else {
                continue;
            };
            if material
                .definition
                .textures
                .iter()
                .any(|texture_id| texture_id.as_deref() == Some(id))
            {
                return Err(SurfaceRendererError::TextureInUse {
                    texture: id.to_owned(),
                    material: material_id.clone(),
                });
            }
        }
        Ok(self.textures.remove(id).is_some())
    }

    pub fn define_material(
        &mut self,
        id: &str,
        descriptor: &StandardMaterialDescriptor,
    ) -> Result<(), SurfaceRendererError> {
        validate_asset_id("material", id)?;
        let definition = normalize_material(id, descriptor)?;
        let material = create_material_gpu(
            &self.context,
            &self.material_layout,
            &self.default_textures,
            &self.textures,
            id,
            definition,
        )?;
        self.materials.insert(id, material);
        Ok(())
    }

    pub fn release_material(&mut self, id: &str) -> Result<bool, SurfaceRendererError> {
        if id == "builtin:default" {
            return Err(SurfaceRendererError::InvalidMaterialId {
                id: id.to_owned(),
                reason: "the default material cannot be released",
            });
        }
        Ok(self.materials.remove(id))
    }

    /// Replaces the environment map with six RGBA8 faces (+X, -X, +Y, -Y,
    /// +Z, -Z, each `face_size²` texels) and convolves it for image-based
    /// lighting. `intensity` scales the map's radiance. The map lights every
    /// layer until it is replaced or cleared.
    pub fn upload_environment_cubemap(
        &mut self,
        face_size: u32,
        faces: &[u8],
        srgb: bool,
        intensity: f32,
    ) -> Result<(), SurfaceRendererError> {
        if face_size == 0 || face_size > MAX_ENVIRONMENT_FACE_SIZE {
            return Err(SurfaceRendererError::InvalidEnvironmentMap {
                reason: format!(
                    "face size must be in 1..={MAX_ENVIRONMENT_FACE_SIZE}, received {face_size}"
                ),
            });
        }
        let expected = (face_size as usize) * (face_size as usize) * 4 * 6;
        if faces.len() != expected {
            return Err(SurfaceRendererError::InvalidEnvironmentMap {
                reason: format!(
                    "six {face_size}x{face_size} RGBA8 faces need {expected} bytes, received {}",
                    faces.len()
                ),
            });
        }
        validate_environment_intensity(intensity)?;
        let source = EnvironmentGpu::source_from_rgba8(
            &self.context.device,
            &self.context.queue,
            face_size,
            faces,
            srgb,
        );
        self.environment.convolve(
            &self.context.device,
            &self.context.queue,
            &source,
            intensity,
        );
        self.rebuild_frame_bind_groups();
        Ok(())
    }

    /// Captures the first layer's procedural sky and light rig into an
    /// environment map and convolves it, so scene lighting matches the sky
    /// the camera sees. Requires a submitted frame whose first layer has a
    /// procedural sky. This is a deliberate, explicit call, not automatic:
    /// convolution costs several dozen small render passes, so a day/night
    /// cycle should recapture at a rate it chooses (every few seconds is
    /// plenty) rather than every frame.
    pub fn set_environment_from_sky(&mut self, intensity: f32) -> Result<(), SurfaceRendererError> {
        validate_environment_intensity(intensity)?;
        let Some(layer) = self.layers.first() else {
            return Err(SurfaceRendererError::InvalidEnvironmentMap {
                reason: "no frame has been submitted to capture a sky from".into(),
            });
        };
        if layer.view.sky_zenith_mode[3] != 1.0 {
            return Err(SurfaceRendererError::InvalidEnvironmentMap {
                reason: "the first layer has no procedural sky to capture".into(),
            });
        }
        let aspect = self.layer_aspect(layer);
        let (mut frame, _) = self.layer_uniforms(0, layer, 0.0, aspect);
        let mip_count = full_mip_count(SKY_CAPTURE_FACE_SIZE);
        let source = CubemapGpu::new(
            &self.context.device,
            "Vapour sky capture",
            SKY_CAPTURE_FACE_SIZE,
            mip_count,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        // The sky shader reconstructs a ray as `inverse_view_projection *
        // (clip_xy, 1, 1)` minus the camera position, so a matrix whose
        // rotation is the face basis and whose translation is the camera
        // writes each face in sampling orientation.
        let camera = Vec3::from_slice(&frame.camera_position_exposure[..3]);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Vapour sky capture"),
                });
        // One uniform buffer and bind group per face keeps every face in one
        // submission; they are dropped once the queue owns the commands.
        let mut face_bind_groups = Vec::with_capacity(6);
        for face in 0..6 {
            let basis = cube_face_ray_basis(face);
            let inverse = Mat4::from_cols(
                basis.x_axis.extend(0.0),
                basis.y_axis.extend(0.0),
                basis.z_axis.extend(0.0),
                camera.extend(1.0),
            );
            frame.inverse_view_projection = inverse.to_cols_array_2d();
            let buffer =
                self.context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vapour sky capture uniforms"),
                        contents: bytemuck::bytes_of(&frame),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            face_bind_groups.push(frame_bind_group_with_buffer(
                &self.context.device,
                &self.frame_layout,
                &buffer,
                &self.shadows,
                &self.bone_buffer,
                &self.environment,
            ));
        }
        for (face, bind_group) in face_bind_groups.iter().enumerate() {
            for mip in 0..mip_count {
                let view = source.face_view(face as u32, mip);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Vapour sky capture face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.sky_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.context.queue.submit([encoder.finish()]);
        self.environment.convolve(
            &self.context.device,
            &self.context.queue,
            &source,
            intensity,
        );
        self.rebuild_frame_bind_groups();
        Ok(())
    }

    /// Replaces this frame's decals. `packed` holds [`DECAL_STRIDE`] floats
    /// per decal (see `decal.rs`); `atlas` names an uploaded texture every
    /// decal samples through its region, or `None` for tint-only decals.
    /// Decals project onto the first layer's opaque depth only; overlay
    /// layers are never decaled. The set persists until replaced, so a
    /// static scene submits once.
    pub fn set_decals(
        &mut self,
        atlas: Option<&str>,
        packed: &[f32],
    ) -> Result<(), SurfaceRendererError> {
        if let Some(atlas) = atlas
            && !self.textures.contains_key(atlas)
        {
            return Err(SurfaceRendererError::UnknownTexture {
                material: "decals".into(),
                texture: atlas.to_owned(),
            });
        }
        let decals = decode_decals(packed)?;
        self.decals
            .upload(&self.context.device, &self.context.queue, &decals);
        self.decals.atlas = atlas.map(str::to_owned);
        Ok(())
    }

    #[must_use]
    pub fn decal_count(&self) -> u32 {
        self.decals.count
    }

    /// Drops the environment map; ambient lighting returns to the analytic
    /// sky gradient.
    pub fn clear_environment(&mut self) {
        self.environment.clear();
    }

    #[must_use]
    pub fn environment_live(&self) -> bool {
        self.environment.live
    }

    /// The frame bind groups reference the environment textures, so a new
    /// environment means new bind groups. Buffers are rewritten every frame.
    fn rebuild_frame_bind_groups(&mut self) {
        for index in 0..self.frame_uniforms.len() {
            self.frame_uniforms[index] = create_frame_gpu(
                &self.context.device,
                &self.frame_layout,
                &self.shadows,
                &self.bone_buffer,
                &self.environment,
                index,
            );
        }
    }

    /// Submits an authored scene. This is the editor's path: readable JSON,
    /// convenient, and fine at document-editing rates.
    ///
    /// Games use [`set_render_frame`](Self::set_render_frame) instead, which
    /// takes the same information as packed arrays.
    pub fn set_render_world(
        &mut self,
        render_world: RenderWorld,
    ) -> Result<(), SurfaceRendererError> {
        render_world.validate()?;

        let mut view = ViewState::default();
        let default_view = render_world.camera.is_none();
        if let Some(camera) = &render_world.camera {
            view.camera_world = Mat4::from_cols_array(&camera.world_matrix);
            view.projection = camera.projection;
            view.field_of_view_degrees = camera.field_of_view;
            view.orthographic_size = camera.orthographic_size;
            view.near = camera.near;
            view.far = camera.far;
        }
        if let Some(light) = &render_world.directional_light {
            let world = Mat4::from_cols_array(&light.world_matrix);
            view.set_single_directional(
                world.transform_vector3(Vec3::NEG_Z),
                Vec3::from_array(light.color),
                light.intensity,
            );
        }

        let mut submission = Vec::with_capacity(render_world.instances.len());
        for instance in &render_world.instances {
            let material_id = if instance.material == "builtin:standard" {
                "builtin:default"
            } else {
                instance.material.as_str()
            };
            let material = self.materials.slot(material_id).ok_or_else(|| {
                SurfaceRendererError::UnknownMaterial {
                    id: material_id.to_owned(),
                }
            })?;
            let pipeline = self
                .materials
                .get(material)
                .expect("a registered material slot must be live")
                .definition
                .pipeline;
            submission.push(InstanceSubmission {
                slot: self.resolve_mesh(&instance.mesh, &instance.entity_id)?,
                material,
                pipeline,
                model: Mat4::from_cols_array(&instance.model_matrix),
                base_color: instance.base_color,
                material_params: instance.material_params,
                // The editor's JSON scene submission predates skinning; a
                // skinned character authored there renders in its rest pose.
                bone_offset: -1.0,
            });
        }
        self.commit(view, default_view, submission);
        Ok(())
    }

    /// Submits a frame as packed arrays, the path a running game uses.
    ///
    /// Geometry-heavy scenes make JSON untenable: a voxel world with a thousand
    /// visible chunks would serialize hundreds of kilobytes every frame just to
    /// restate matrices. Here each array is copied once, and `mesh_table` is
    /// the only text — one entry per distinct mesh, not per instance.
    ///
    /// * `mesh_indices` — one entry per instance, indexing `mesh_table`
    /// * `matrices` — 16 floats per instance, column-major
    /// * `base_colors` — 4 floats per instance; empty means opaque white
    /// * `material_params` — 4 floats per instance; empty means the default
    ///   dielectric
    /// * `camera` — 16 matrix floats, projection mode, projection size, near, far
    /// * `light` — 3 direction floats, 3 colour floats, then intensity
    pub fn set_render_frame(&mut self, frame: RenderFrame<'_>) -> Result<(), SurfaceRendererError> {
        let RenderFrame {
            mesh_table,
            mesh_indices,
            material_table,
            material_indices,
            matrices,
            base_colors,
            material_params,
            camera,
            light,
            environment,
        } = frame;
        let count = mesh_indices.len();
        expect_len("matrices", matrices.len(), count * 16)?;
        expect_len("materialIndices", material_indices.len(), count)?;
        if !base_colors.is_empty() {
            expect_len("baseColors", base_colors.len(), count * 4)?;
        }
        if !material_params.is_empty() {
            expect_len("materialParams", material_params.len(), count * 4)?;
        }
        if count > MAX_RENDER_INSTANCES {
            return Err(RenderWorldError::TooManyInstances {
                actual: count,
                maximum: MAX_RENDER_INSTANCES,
            }
            .into());
        }

        // Resolve the small table once instead of per instance.
        let mut slots = Vec::with_capacity(mesh_table.len());
        for id in mesh_table {
            slots.push(self.resolve_mesh(id, "frame")?);
        }
        let mut material_slots = Vec::with_capacity(material_table.len());
        for id in material_table {
            material_slots.push(
                self.materials
                    .slot(id)
                    .ok_or_else(|| SurfaceRendererError::UnknownMaterial { id: id.clone() })?,
            );
        }

        let mut view = ViewState::default();
        let default_view = camera.is_empty();
        if !camera.is_empty() {
            expect_len("camera", camera.len(), 20)?;
            finite("camera", camera)?;
            let mut world = [0.0_f32; 16];
            world.copy_from_slice(&camera[..16]);
            view.camera_world = Mat4::from_cols_array(&world);
            view.projection = decode_projection(camera[16])?;
            if view.projection == CameraProjection::Perspective {
                view.field_of_view_degrees = camera[17].clamp(1.0, 179.0);
            } else {
                view.orthographic_size = camera[17].max(1e-4);
            }
            view.near = camera[18].max(1e-4);
            view.far = camera[19].max(view.near * 1.001);
        }
        if let Some(rig) = decode_light_rig(light, "light")? {
            view.lights = rig;
        }
        if let Some(environment) = decode_environment(environment, "environment")? {
            view.clear_color = environment.clear_color;
            view.fog_color_mode = environment.fog_color_mode;
            view.fog_params = environment.fog_params;
            view.sky_zenith_mode = environment.sky_zenith_mode;
            view.sky_horizon_curve = environment.sky_horizon_curve;
            view.sky_ground_sun_size = environment.sky_ground_sun_size;
        }

        let mut submission = Vec::with_capacity(count);
        for index in 0..count {
            let table_index = mesh_indices[index] as usize;
            let slot =
                *slots
                    .get(table_index)
                    .ok_or(SurfaceRendererError::MeshIndexOutOfRange {
                        instance: index,
                        index: table_index,
                        table: mesh_table.len(),
                    })?;
            let material_index = material_indices[index] as usize;
            let material = *material_slots.get(material_index).ok_or(
                SurfaceRendererError::MaterialIndexOutOfRange {
                    instance: index,
                    index: material_index,
                    table: material_table.len(),
                },
            )?;
            let mut model = [0.0_f32; 16];
            model.copy_from_slice(&matrices[index * 16..index * 16 + 16]);
            if !model.iter().all(|value| value.is_finite()) {
                return Err(RenderWorldError::NonFinite {
                    field: format!("matrices[{index}]"),
                }
                .into());
            }
            submission.push(InstanceSubmission {
                slot,
                material,
                pipeline: self
                    .materials
                    .get(material)
                    .expect("resolved material slots stay occupied during submission")
                    .definition
                    .pipeline,
                model: Mat4::from_cols_array(&model),
                base_color: read4(base_colors, index, [1.0, 1.0, 1.0, 1.0]),
                material_params: read4(material_params, index, [0.0, 0.6, 0.0, 0.01]),
                bone_offset: -1.0,
            });
        }
        self.commit(view, default_view, submission);
        Ok(())
    }

    /// Atomically replaces the live frame with several camera layers. The
    /// first layer clears the presentation target; later layers may preserve
    /// colour and optionally clear depth for view models, gizmos, or overlays.
    pub fn set_render_layers(
        &mut self,
        frame: RenderLayers<'_>,
    ) -> Result<(), SurfaceRendererError> {
        let RenderLayers {
            mesh_table,
            mesh_indices,
            material_table,
            material_indices,
            matrices,
            base_colors,
            material_params,
            layer_instance_counts,
            layer_flags,
            cameras,
            lights,
            layer_light_counts,
            environments,
            bone_offsets,
            bone_matrices,
            layer_targets,
            viewports,
            projections,
        } = frame;
        let layer_count = layer_instance_counts.len();
        if layer_count == 0 {
            return Err(SurfaceRendererError::InvalidRenderLayer {
                layer: 0,
                reason: "at least one layer is required",
            });
        }
        expect_len("layerFlags", layer_flags.len(), layer_count)?;
        expect_len("cameras", cameras.len(), layer_count * 20)?;
        expect_len(
            "environments",
            environments.len(),
            layer_count * ENVIRONMENT_STRIDE,
        )?;
        if !layer_light_counts.is_empty() {
            expect_len("layerLightCounts", layer_light_counts.len(), layer_count)?;
        }
        if !layer_targets.is_empty() {
            expect_len("layerTargets", layer_targets.len(), layer_count)?;
        }
        if !viewports.is_empty() {
            expect_len("viewports", viewports.len(), layer_count * 4)?;
            finite("viewports", viewports)?;
        }
        if !projections.is_empty() {
            expect_len("projections", projections.len(), layer_count * 16)?;
            finite("projections", projections)?;
        }
        let total_lights: usize = layer_light_counts.iter().map(|value| *value as usize).sum();
        expect_len("lights", lights.len(), total_lights * LIGHT_STRIDE)?;
        if layer_flags[0] & (RENDER_LAYER_CLEAR_COLOR | RENDER_LAYER_CLEAR_DEPTH)
            != RENDER_LAYER_CLEAR_COLOR | RENDER_LAYER_CLEAR_DEPTH
        {
            return Err(SurfaceRendererError::InvalidRenderLayer {
                layer: 0,
                reason: "the first layer must clear colour and depth",
            });
        }
        let count: usize = layer_instance_counts
            .iter()
            .map(|value| *value as usize)
            .sum();
        expect_len("meshIndices", mesh_indices.len(), count)?;
        expect_len("materialIndices", material_indices.len(), count)?;
        expect_len("matrices", matrices.len(), count * 16)?;
        expect_len("baseColors", base_colors.len(), count * 4)?;
        expect_len("materialParams", material_params.len(), count * 4)?;
        if count > MAX_RENDER_INSTANCES {
            return Err(RenderWorldError::TooManyInstances {
                actual: count,
                maximum: MAX_RENDER_INSTANCES,
            }
            .into());
        }
        if !bone_offsets.is_empty() {
            expect_len("boneOffsets", bone_offsets.len(), count)?;
        }
        if !bone_matrices.len().is_multiple_of(16) {
            return Err(SurfaceRendererError::FrameArrayLength {
                field: "boneMatrices",
                actual: bone_matrices.len(),
                expected: bone_matrices.len().next_multiple_of(16),
            });
        }
        let bone_count = bone_matrices.len() / 16;
        if bone_count > MAX_BONE_MATRICES {
            return Err(SurfaceRendererError::TooManyBones {
                field: "boneMatrices",
                actual: bone_count,
                limit: MAX_BONE_MATRICES,
            });
        }
        if !bone_matrices.is_empty() {
            self.context.queue.write_buffer(
                &self.bone_buffer,
                0,
                bytemuck::cast_slice(bone_matrices),
            );
        }

        let mut slots = Vec::with_capacity(mesh_table.len());
        for id in mesh_table {
            slots.push(self.resolve_mesh(id, "layered frame")?);
        }
        let mut material_slots = Vec::with_capacity(material_table.len());
        for id in material_table {
            material_slots.push(
                self.materials
                    .slot(id)
                    .ok_or_else(|| SurfaceRendererError::UnknownMaterial { id: id.clone() })?,
            );
        }

        let mut layers = Vec::with_capacity(layer_count);
        let mut instance_start = 0;
        let mut light_offset = 0;
        // Every layer's destination and the targets its materials sample, for
        // the cycle check once all layers are decoded.
        let mut target_graph: Vec<(Option<String>, Vec<String>)> = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            let flags = layer_flags[layer_index];
            let target = match layer_targets.get(layer_index).map(String::as_str) {
                None | Some("") => None,
                Some(id) => {
                    if !self.render_targets.contains_key(id) {
                        return Err(SurfaceRendererError::UnknownRenderTarget {
                            layer: layer_index,
                            id: id.to_owned(),
                        });
                    }
                    Some(id.to_owned())
                }
            };
            let viewport = match viewports.get(layer_index * 4..layer_index * 4 + 4) {
                Some(rect) => {
                    let rect = [rect[0], rect[1], rect[2], rect[3]];
                    if rect.iter().any(|value| !(0.0..=1.0).contains(value))
                        || rect[2] <= 0.0
                        || rect[3] <= 0.0
                        || rect[0] + rect[2] > 1.0001
                        || rect[1] + rect[3] > 1.0001
                    {
                        return Err(SurfaceRendererError::InvalidRenderLayer {
                            layer: layer_index,
                            reason: "viewport must be a non-empty x, y, width, height rectangle inside [0, 1]",
                        });
                    }
                    rect
                }
                None => FULL_VIEWPORT,
            };
            if flags
                & !(RENDER_LAYER_CLEAR_COLOR
                    | RENDER_LAYER_CLEAR_DEPTH
                    | RENDER_LAYER_HAS_CAMERA
                    | RENDER_LAYER_HAS_LIGHT
                    | RENDER_LAYER_HAS_ENVIRONMENT
                    | RENDER_LAYER_HAS_PROJECTION)
                != 0
            {
                return Err(SurfaceRendererError::InvalidRenderLayer {
                    layer: layer_index,
                    reason: "flags contain unknown bits",
                });
            }
            if flags & RENDER_LAYER_HAS_PROJECTION != 0 && projections.is_empty() {
                return Err(SurfaceRendererError::InvalidRenderLayer {
                    layer: layer_index,
                    reason: "flags claim a projection matrix but none was supplied",
                });
            }
            let mut view = ViewState::default();
            let default_view = flags & RENDER_LAYER_HAS_CAMERA == 0;
            if !default_view {
                let camera = &cameras[layer_index * 20..layer_index * 20 + 20];
                finite("layer camera", camera)?;
                let mut world = [0.0_f32; 16];
                world.copy_from_slice(&camera[..16]);
                view.camera_world = Mat4::from_cols_array(&world);
                view.projection = decode_projection(camera[16])?;
                if view.projection == CameraProjection::Perspective {
                    view.field_of_view_degrees = camera[17].clamp(1.0, 179.0);
                } else {
                    view.orthographic_size = camera[17].max(1e-4);
                }
                view.near = camera[18].max(1e-4);
                view.far = camera[19].max(view.near * 1.001);
            }
            if flags & RENDER_LAYER_HAS_PROJECTION != 0 {
                let mut matrix = [0.0_f32; 16];
                matrix.copy_from_slice(&projections[layer_index * 16..layer_index * 16 + 16]);
                let projection = Mat4::from_cols_array(&matrix);
                if projection.determinant() == 0.0 {
                    return Err(SurfaceRendererError::InvalidRenderLayer {
                        layer: layer_index,
                        reason: "projection matrix is singular",
                    });
                }
                view.projection_override = Some(projection);
            }
            if flags & RENDER_LAYER_HAS_LIGHT != 0 {
                let count = layer_light_counts
                    .get(layer_index)
                    .copied()
                    .unwrap_or_default() as usize;
                let start = light_offset * LIGHT_STRIDE;
                let slice = &lights[start..start + count * LIGHT_STRIDE];
                if let Some(rig) = decode_light_rig(slice, "layer light")? {
                    view.lights = rig;
                }
            }
            light_offset += layer_light_counts
                .get(layer_index)
                .copied()
                .unwrap_or_default() as usize;
            if flags & RENDER_LAYER_HAS_ENVIRONMENT != 0 {
                let start = layer_index * ENVIRONMENT_STRIDE;
                if let Some(environment) = decode_environment(
                    &environments[start..start + ENVIRONMENT_STRIDE],
                    "layer environment",
                )? {
                    view.clear_color = environment.clear_color;
                    view.fog_color_mode = environment.fog_color_mode;
                    view.fog_params = environment.fog_params;
                    view.sky_zenith_mode = environment.sky_zenith_mode;
                    view.sky_horizon_curve = environment.sky_horizon_curve;
                    view.sky_ground_sun_size = environment.sky_ground_sun_size;
                }
            }

            let instance_end = instance_start + layer_instance_counts[layer_index] as usize;
            let mut instances = Vec::with_capacity(instance_end - instance_start);
            let mut sampled_targets: Vec<String> = Vec::new();
            for index in instance_start..instance_end {
                let mesh_index = mesh_indices[index] as usize;
                let slot =
                    *slots
                        .get(mesh_index)
                        .ok_or(SurfaceRendererError::MeshIndexOutOfRange {
                            instance: index,
                            index: mesh_index,
                            table: mesh_table.len(),
                        })?;
                let material_index = material_indices[index] as usize;
                let material = *material_slots.get(material_index).ok_or(
                    SurfaceRendererError::MaterialIndexOutOfRange {
                        instance: index,
                        index: material_index,
                        table: material_table.len(),
                    },
                )?;
                let mut model = [0.0_f32; 16];
                model.copy_from_slice(&matrices[index * 16..index * 16 + 16]);
                if !model.iter().all(|value| value.is_finite()) {
                    return Err(RenderWorldError::NonFinite {
                        field: format!("matrices[{index}]"),
                    }
                    .into());
                }
                let definition = &self
                    .materials
                    .get(material)
                    .expect("resolved material slots stay occupied during submission")
                    .definition;
                if !self.render_targets.is_empty() {
                    for texture in definition.textures.iter().flatten() {
                        if self.render_targets.contains_key(texture)
                            && !sampled_targets.contains(texture)
                        {
                            sampled_targets.push(texture.clone());
                        }
                    }
                }
                instances.push(InstanceSubmission {
                    slot,
                    material,
                    pipeline: definition.pipeline,
                    model: Mat4::from_cols_array(&model),
                    base_color: read4(base_colors, index, [1.0; 4]),
                    material_params: read4(material_params, index, [0.0, 0.6, 0.0, 0.01]),
                    bone_offset: bone_offsets.get(index).copied().unwrap_or(-1.0),
                });
            }
            target_graph.push((target.clone(), sampled_targets));
            layers.push(LayerSubmission {
                view,
                default_view,
                clear_color: flags & RENDER_LAYER_CLEAR_COLOR != 0,
                clear_depth: flags & RENDER_LAYER_CLEAR_DEPTH != 0,
                instances,
                target,
                viewport,
            });
            instance_start = instance_end;
        }
        let sampled_refs: Vec<Vec<&str>> = target_graph
            .iter()
            .map(|(_, sampled)| sampled.iter().map(String::as_str).collect())
            .collect();
        let graph: Vec<(Option<&str>, &[&str])> = target_graph
            .iter()
            .zip(&sampled_refs)
            .map(|((target, _), sampled)| (target.as_deref(), sampled.as_slice()))
            .collect();
        detect_render_target_cycle(&graph)?;
        self.commit_layers(layers);
        Ok(())
    }

    fn resolve_mesh(&self, id: &str, entity: &str) -> Result<usize, SurfaceRendererError> {
        self.meshes
            .slot(id)
            .ok_or_else(|| SurfaceRendererError::UnknownMesh {
                id: id.to_owned(),
                entity: entity.to_owned(),
                available: self.meshes.ids().join(", "),
            })
    }

    /// Sorts opaque instances for batching and blended instances back-to-front,
    /// then packs and uploads them. Both submission paths end here, so ordering
    /// and batching behaviour is identical for the editor and a game.
    fn commit(&mut self, view: ViewState, default_view: bool, submission: Vec<InstanceSubmission>) {
        self.commit_layers(vec![LayerSubmission {
            view,
            default_view,
            clear_color: true,
            clear_depth: true,
            instances: submission,
            target: None,
            viewport: FULL_VIEWPORT,
        }]);
    }

    fn commit_layers(&mut self, mut submissions: Vec<LayerSubmission>) {
        let total_instances: usize = submissions.iter().map(|layer| layer.instances.len()).sum();
        let mut instances = Vec::with_capacity(total_instances);
        let mut cull_inputs = Vec::with_capacity(total_instances);
        let mut args: Vec<DrawArgs> = Vec::new();
        let mut layers = Vec::with_capacity(submissions.len());
        for layer in &mut submissions {
            sort_submissions(
                &mut layer.instances,
                layer.view.camera_world.transform_point3(Vec3::ZERO),
            );
            let first_instance = instances.len() as u32;
            let first_batch = args.len() as u32;
            let mut batches: Vec<DrawBatch> = Vec::new();
            for entry in layer.instances.drain(..) {
                match batches.last_mut() {
                    Some(batch)
                        if batch.mesh == entry.slot
                            && batch.material == entry.material
                            && batch.pipeline == entry.pipeline =>
                    {
                        batch.instance_count += 1;
                    }
                    _ => batches.push(DrawBatch {
                        mesh: entry.slot,
                        material: entry.material,
                        pipeline: entry.pipeline,
                        first_instance: instances.len() as u32,
                        instance_count: 1,
                    }),
                }
                // Blended batches draw back-to-front, so they pass through
                // culling uncompacted; opaque batches are compacted.
                let pass_through = entry.pipeline.blended();
                let (center, radius) = world_bounds(
                    entry.model,
                    self.meshes
                        .get(entry.slot)
                        .map_or(0.0, |mesh| mesh.bounding_radius),
                );
                cull_inputs.push(CullInput {
                    center_radius: [center.x, center.y, center.z, radius],
                    batch_flags: [
                        first_batch + batches.len() as u32 - 1,
                        if pass_through { CULL_PASS_THROUGH } else { 0 },
                        0,
                        0,
                    ],
                });
                instances.push(InstanceData {
                    model: entry.model.to_cols_array_2d(),
                    base_color: entry.base_color,
                    material_params: entry.material_params,
                    skin: [entry.bone_offset, 0.0, 0.0, 0.0],
                });
            }
            args.extend(batches.iter().map(|batch| {
                DrawArgs {
                    index_count: self
                        .meshes
                        .get(batch.mesh)
                        .map_or(0, |mesh| mesh.index_count),
                    instance_count: if batch.pipeline.blended() {
                        batch.instance_count
                    } else {
                        0
                    },
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: batch.first_instance,
                }
            }));
            layers.push(RenderLayerState {
                first_instance,
                instance_count: instances.len() as u32 - first_instance,
                first_batch,
                view: layer.view.clone(),
                default_view: layer.default_view,
                clear_color: layer.clear_color,
                clear_depth: layer.clear_depth,
                batches,
                target: layer.target.take(),
                viewport: layer.viewport,
            });
        }

        if instances.len() > self.instance_capacity {
            self.instance_capacity = instances.len().next_power_of_two();
            self.instance_buffer =
                create_instance_buffer(&self.context.device, self.instance_capacity);
        }
        if !instances.is_empty() {
            self.context.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances),
            );
        }
        self.instance_count = instances.len() as u32;
        self.culling.upload(
            &self.context.device,
            &self.context.queue,
            cull_inputs,
            instances,
            args,
        );
        while self.frame_uniforms.len() < layers.len() {
            let index = self.frame_uniforms.len();
            self.frame_uniforms.push(create_frame_gpu(
                &self.context.device,
                &self.frame_layout,
                &self.shadows,
                &self.bone_buffer,
                &self.environment,
                index,
            ));
        }
        self.layers = layers;
    }

    pub fn render(&mut self, time_seconds: f32) -> Result<FrameStats, SurfaceRendererError> {
        let started = Stopwatch::start();
        let Some(output) = self.acquire_surface_texture()? else {
            return Ok(FrameStats {
                draw_calls: 0,
                triangles: 0,
                cpu_encode_ms: started.elapsed_ms(),
                gpu_culling: false,
                culled_instances: Some(0),
            });
        };
        let shadow_data = self.update_uniforms(time_seconds);
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Vapour frame encoder"),
                });
        self.post_process
            .update(&self.context.queue, self.post_process_settings);
        let post_bind_group = self
            .post_process
            .bind_group(&self.context.device, &self.hdr_target.view);
        // Frustum-cull every layer up front: the material pass below reads
        // the compacted buffer, the shadow passes keep the full set since
        // they see the scene from the light.
        let jobs: Vec<CullJob> = self
            .layers
            .iter()
            .zip(&shadow_data)
            .map(|(layer, data)| CullJob {
                planes: frustum_planes(data.view_projection),
                first: layer.first_instance,
                count: layer.instance_count,
            })
            .collect();
        let cull = self.culling.cull(
            &self.context.device,
            &self.context.queue,
            &mut encoder,
            &self.instance_buffer,
            &jobs,
            self.context.capabilities.gpu_culling,
        );
        let mut draw_calls = 0;
        for pass in self.frame_graph.passes() {
            match (pass.name.as_str(), pass.kind) {
                ("directional-shadow", PassKind::Render) => {
                    if let (Some(shadow_layer), Some(layer)) =
                        (shadow_data.first(), self.layers.first())
                    {
                        for (cascade_index, cascade) in shadow_layer.cascades.iter().enumerate() {
                            self.context.queue.write_buffer(
                                &self.shadows.cascade_buffers[cascade_index],
                                0,
                                bytemuck::bytes_of(cascade),
                            );
                            let mut shadow_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Vapour directional cascade depth pass"),
                                    color_attachments: &[],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &self.shadows.layer_views[cascade_index],
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                            shadow_pass.set_pipeline(&self.shadow_pipeline);
                            shadow_pass.set_bind_group(
                                0,
                                &self.shadows.cascade_bind_groups[cascade_index],
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                            for batch in &layer.batches {
                                if batch.pipeline.blended() {
                                    continue;
                                }
                                let Some(mesh) = self.meshes.get(batch.mesh) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_vertex_buffer(2, mesh.skin_buffer.slice(..));
                                let Some(material) = self.materials.get(batch.material) else {
                                    continue;
                                };
                                shadow_pass.set_bind_group(1, &material.bind_group, &[]);
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                let first = batch.first_instance;
                                shadow_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    first..first + batch.instance_count,
                                );
                                draw_calls += 1;
                            }
                        }
                    }
                }
                ("point-spot-shadow", PassKind::Render) => {
                    if let (Some(shadow_layer), Some(layer)) =
                        (shadow_data.first(), self.layers.first())
                    {
                        for slot in 0..shadow_layer.point_spot_used {
                            self.context.queue.write_buffer(
                                &self.shadows.point_spot_buffers[slot],
                                0,
                                bytemuck::bytes_of(&shadow_layer.point_spot[slot]),
                            );
                            let mut shadow_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Vapour point/spot shadow slot pass"),
                                    color_attachments: &[],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &self.shadows.point_spot_layer_views[slot],
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                            shadow_pass.set_pipeline(&self.shadow_pipeline);
                            shadow_pass.set_bind_group(
                                0,
                                &self.shadows.point_spot_bind_groups[slot],
                                &[],
                            );
                            shadow_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                            for batch in &layer.batches {
                                if batch.pipeline.blended() {
                                    continue;
                                }
                                let Some(mesh) = self.meshes.get(batch.mesh) else {
                                    continue;
                                };
                                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                shadow_pass.set_vertex_buffer(2, mesh.skin_buffer.slice(..));
                                let Some(material) = self.materials.get(batch.material) else {
                                    continue;
                                };
                                shadow_pass.set_bind_group(1, &material.bind_group, &[]);
                                shadow_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                let first = batch.first_instance;
                                shadow_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    first..first + batch.instance_count,
                                );
                                draw_calls += 1;
                            }
                        }
                    }
                }
                ("standard-pbr", PassKind::Render) => {
                    for (layer_index, layer) in self.layers.iter().enumerate() {
                        // A load-op clear ignores the scissor, so a viewport layer that is not
                        // the first into its destination would wipe the layers beside it (the
                        // other eye of a side-by-side XR frame). Those layers load instead; the
                        // first layer's clear already covered their rectangle.
                        let shares_destination = layer.viewport != FULL_VIEWPORT
                            && self.layers[..layer_index]
                                .iter()
                                .any(|earlier| earlier.target == layer.target);
                        let color_load = if layer.clear_color && !shares_destination {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: layer.view.clear_color[0] as f64,
                                g: layer.view.clear_color[1] as f64,
                                b: layer.view.clear_color[2] as f64,
                                a: layer.view.clear_color[3] as f64,
                            })
                        } else {
                            wgpu::LoadOp::Load
                        };
                        let depth_load = if layer.clear_depth && !shares_destination {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        };
                        let (color_view, depth_view, extent) = self.layer_destination(layer);
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Vapour layered standard PBR pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: color_view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: color_load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: depth_load,
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        if layer.viewport != FULL_VIEWPORT {
                            let (x, y, w, h) = viewport_pixels(layer.viewport, extent);
                            render_pass
                                .set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                            render_pass.set_scissor_rect(x, y, w, h);
                        }
                        render_pass.set_bind_group(
                            0,
                            &self.frame_uniforms[layer_index].bind_group,
                            &[],
                        );
                        if layer.clear_color && layer.view.sky_zenith_mode[3] == 1.0 {
                            render_pass.set_pipeline(&self.sky_pipeline);
                            render_pass.draw(0..3, 0..1);
                            draw_calls += 1;
                        }
                        render_pass.set_vertex_buffer(1, self.culling.compacted.slice(..));
                        for (batch_offset, batch) in layer.batches.iter().enumerate() {
                            let batch_index = layer.first_batch as usize + batch_offset;
                            let Some(mesh) = self.meshes.get(batch.mesh) else {
                                continue;
                            };
                            let Some(material) = self.materials.get(batch.material) else {
                                continue;
                            };
                            let args = self.culling.args[batch_index];
                            if !cull.gpu && args.instance_count == 0 {
                                continue;
                            }
                            let pipeline = self
                                .pipelines
                                .get(&batch.pipeline)
                                .expect("every material pipeline variant is created at startup");
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(1, &material.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_vertex_buffer(2, mesh.skin_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            if cull.gpu {
                                render_pass.draw_indexed_indirect(
                                    &self.culling.args_buffer,
                                    args_offset(batch_index),
                                );
                            } else {
                                let first = args.first_instance;
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    first..first + args.instance_count,
                                );
                            }
                            draw_calls += 1;
                        }
                    }
                }
                ("decals", PassKind::Render) => {
                    let atlas_view = match &self.decals.atlas {
                        Some(id) => self.textures.get(id).map(|texture| &texture.view),
                        None => Some(&self.default_textures[0].view),
                    };
                    let cube = self
                        .meshes
                        .slot(BuiltinMesh::Cube.id())
                        .and_then(|slot| self.meshes.get(slot));
                    // Decals read the surface depth, so they apply only when
                    // the world layer presents to the surface.
                    let world_on_surface = self
                        .layers
                        .first()
                        .is_some_and(|layer| layer.target.is_none());
                    if let (Some(atlas_view), Some(cube), true) =
                        (atlas_view, cube, self.decals.count > 0 && world_on_surface)
                    {
                        let bind_group = self.decals.bind_group(
                            &self.context.device,
                            &self.depth_view,
                            atlas_view,
                        );
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Vapour decal pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &self.hdr_target.view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        render_pass.set_pipeline(&self.decals.pipeline);
                        render_pass.set_bind_group(0, &self.frame_uniforms[0].bind_group, &[]);
                        render_pass.set_bind_group(1, &bind_group, &[]);
                        render_pass.set_vertex_buffer(0, cube.vertex_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, self.decals.instance_buffer.slice(..));
                        render_pass.set_index_buffer(
                            cube.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..cube.index_count, 0, 0..self.decals.count);
                        draw_calls += 1;
                    }
                }
                ("particles", PassKind::Render) => {
                    // Particles depth-test against the surface depth, so
                    // like decals they draw only when the world layer
                    // presents to the surface.
                    let world_on_surface = self
                        .layers
                        .first()
                        .is_some_and(|layer| layer.target.is_none());
                    if !self.particle_systems.is_empty() && world_on_surface {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Vapour particle pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &self.hdr_target.view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: &self.depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                        render_pass.set_pipeline(&self.particles.draw_pipeline);
                        render_pass.set_bind_group(0, &self.frame_uniforms[0].bind_group, &[]);
                        for system in self.particle_systems.values() {
                            if system.draw(&mut render_pass) {
                                draw_calls += 1;
                            }
                        }
                    }
                }
                ("tone-map", PassKind::Render) => {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Vapour HDR tone-map pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    render_pass.set_pipeline(&self.post_process.pipeline);
                    render_pass.set_bind_group(0, &post_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                    draw_calls += 1;
                }
                _ => unreachable!("compiled frame graph contains an unknown pass"),
            }
        }
        self.context.queue.submit([encoder.finish()]);
        self.context.queue.present(output);
        Ok(FrameStats {
            draw_calls,
            triangles: self.triangle_count(),
            cpu_encode_ms: started.elapsed_ms(),
            gpu_culling: cull.gpu,
            culled_instances: cull.culled,
        })
    }

    /// Triangles submitted this frame, summed across every batch.
    fn triangle_count(&self) -> u32 {
        self.layers
            .iter()
            .flat_map(|layer| &layer.batches)
            .filter_map(|batch| {
                Some(self.meshes.get(batch.mesh)?.index_count / 3 * batch.instance_count)
            })
            .sum()
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.config.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.config.height
    }

    /// Where a layer draws: its render target's colour and depth, or the
    /// surface's HDR target and depth, plus that destination's texel size.
    fn layer_destination(
        &self,
        layer: &RenderLayerState,
    ) -> (&wgpu::TextureView, &wgpu::TextureView, (u32, u32)) {
        match layer
            .target
            .as_deref()
            .and_then(|id| self.render_targets.get(id))
        {
            Some(target) => (
                &target.color_view,
                &target.depth_view,
                (target.width, target.height),
            ),
            None => (
                &self.hdr_target.view,
                &self.depth_view,
                (self.config.width, self.config.height),
            ),
        }
    }

    /// The aspect ratio a layer's projection uses: its viewport rectangle on
    /// its destination, not the surface.
    fn layer_aspect(&self, layer: &RenderLayerState) -> f32 {
        let (_, _, extent) = self.layer_destination(layer);
        let (_, _, width, height) = viewport_pixels(layer.viewport, extent);
        width as f32 / height.max(1) as f32
    }

    fn update_uniforms(&self, time_seconds: f32) -> Vec<LayerShadowData> {
        let mut shadow_sets = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let aspect = self.layer_aspect(layer);
            let (frame, shadow_data) = self.layer_uniforms(index, layer, time_seconds, aspect);
            self.context.queue.write_buffer(
                &self.frame_uniforms[index].buffer,
                0,
                bytemuck::bytes_of(&frame),
            );
            shadow_sets.push(shadow_data);
        }
        shadow_sets
    }

    /// The frame uniforms and shadow-pass data for one layer.
    fn layer_uniforms(
        &self,
        index: usize,
        layer: &RenderLayerState,
        time_seconds: f32,
        aspect: f32,
    ) -> (FrameUniforms, LayerShadowData) {
        {
            let view = &layer.view;
            let camera = view.camera_world.transform_point3(Vec3::ZERO);
            let (view_matrix, projection) = if layer.default_view {
                // With no authored camera, frame the origin so an empty scene still
                // shows something recognisable rather than a black screen.
                (
                    Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y),
                    projection_matrix(view, aspect),
                )
            } else {
                let forward = view
                    .camera_world
                    .transform_vector3(Vec3::NEG_Z)
                    .normalize_or(Vec3::NEG_Z);
                let up = view
                    .camera_world
                    .transform_vector3(Vec3::Y)
                    .normalize_or(Vec3::Y);
                (
                    Mat4::look_to_rh(camera, forward, up),
                    projection_matrix(view, aspect),
                )
            };
            let live = view.lights.len().min(MAX_LIGHTS);
            let cascades = shadow_cascades(view, aspect);
            let shadow_count = if index == 0 && has_directional_shadow(view) {
                SHADOW_CASCADE_COUNT as f32
            } else {
                0.0
            };
            let camera_forward = view
                .camera_world
                .transform_vector3(Vec3::NEG_Z)
                .normalize_or(Vec3::NEG_Z);
            let mut lights = [GpuLight::ambient(Vec3::ZERO, 0.0); MAX_LIGHTS];
            lights[..live].copy_from_slice(&view.lights[..live]);
            // A point/spot light's shadow slot travels in its own record (see
            // `FrameUniforms::point_spot_matrices`), so every candidate starts
            // "no slot" and only the ones `assign_shadow_slots` grants get one.
            for light in lights.iter_mut().take(live) {
                let kind = light.vector_kind[3];
                if (kind - LIGHT_KIND_POINT).abs() < 0.1 || (kind - LIGHT_KIND_SPOT).abs() < 0.1 {
                    light.cone[2] = -1.0;
                }
            }
            let assignments = if index == 0 {
                assign_shadow_slots(&lights[..live], camera, MAX_SHADOW_SLOTS)
            } else {
                Vec::new()
            };
            let mut point_spot = [ShadowPassUniform {
                view_projection: Mat4::IDENTITY.to_cols_array_2d(),
                far: 1.0,
                _padding: [0.0; 3],
            }; MAX_SHADOW_SLOTS];
            let mut point_spot_used = 0usize;
            for assignment in &assignments {
                let light = &lights[assignment.light_index];
                let range = light.secondary_range[3].max(0.1);
                if assignment.slot_count == 6 {
                    let position = Vec3::from_slice(&light.vector_kind[..3]);
                    let projection = point_shadow_projection(0.05, range);
                    for (face, view_matrix) in
                        cube_face_view_matrices(position).into_iter().enumerate()
                    {
                        point_spot[assignment.base_slot + face] = ShadowPassUniform {
                            view_projection: (projection * view_matrix).to_cols_array_2d(),
                            far: range,
                            _padding: [0.0; 3],
                        };
                    }
                } else {
                    point_spot[assignment.base_slot] = ShadowPassUniform {
                        view_projection: spot_shadow_matrix(light).to_cols_array_2d(),
                        far: range,
                        _padding: [0.0; 3],
                    };
                }
                lights[assignment.light_index].cone[2] = assignment.base_slot as f32;
                point_spot_used = point_spot_used.max(assignment.base_slot + assignment.slot_count);
            }
            let view_projection = projection * view_matrix;
            let frame = FrameUniforms {
                view_projection: view_projection.to_cols_array_2d(),
                inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
                camera_position_exposure: [camera.x, camera.y, camera.z, 1.0],
                camera_forward: [camera_forward.x, camera_forward.y, camera_forward.z, 0.0],
                light_count_time: [live as f32, time_seconds, 0.0, 0.0],
                fog_color_mode: view.fog_color_mode,
                fog_params: view.fog_params,
                sky_zenith_mode: view.sky_zenith_mode,
                sky_horizon_curve: view.sky_horizon_curve,
                sky_ground_sun_size: view.sky_ground_sun_size,
                lights,
                shadow_matrices: cascades.map(|cascade| cascade.view_projection),
                shadow_splits: cascades.map(|cascade| cascade.far),
                shadow_params: [shadow_count, SHADOW_MAP_SIZE as f32, 0.0015, 0.0],
                point_spot_matrices: point_spot.map(|slot| slot.view_projection),
                point_spot_params: [
                    point_spot_used as f32,
                    POINT_SPOT_SHADOW_SIZE as f32,
                    0.002,
                    0.0,
                ],
                ibl_params: self.environment.params(),
            };
            (
                frame,
                LayerShadowData {
                    view_projection,
                    cascades,
                    point_spot,
                    point_spot_used,
                },
            )
        }
    }

    fn acquire_surface_texture(
        &mut self,
    ) -> Result<Option<wgpu::SurfaceTexture>, SurfaceRendererError> {
        use wgpu::CurrentSurfaceTexture;
        match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(texture)
            | CurrentSurfaceTexture::Suboptimal(texture) => Ok(Some(texture)),
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => Ok(None),
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.context.device, &self.config);
                match self.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(texture)
                    | CurrentSurfaceTexture::Suboptimal(texture) => Ok(Some(texture)),
                    CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => Ok(None),
                    CurrentSurfaceTexture::Outdated => Err(SurfaceRendererError::Surface(
                        "surface remained outdated after reconfiguration".into(),
                    )),
                    CurrentSurfaceTexture::Lost => Err(SurfaceRendererError::Surface(
                        "surface remained lost after reconfiguration".into(),
                    )),
                    CurrentSurfaceTexture::Validation => Err(SurfaceRendererError::Surface(
                        "surface acquisition triggered a WebGPU validation error".into(),
                    )),
                }
            }
            CurrentSurfaceTexture::Validation => Err(SurfaceRendererError::Surface(
                "surface acquisition triggered a WebGPU validation error".into(),
            )),
        }
    }
}

/// A normalized viewport rectangle in whole texels of `extent`, clamped so
/// rounding can never push it past the destination's edge and never leaves
/// it empty.
pub(crate) fn viewport_pixels(viewport: [f32; 4], extent: (u32, u32)) -> (u32, u32, u32, u32) {
    let (width, height) = (extent.0.max(1), extent.1.max(1));
    let x = ((viewport[0] * width as f32).round() as u32).min(width - 1);
    let y = ((viewport[1] * height as f32).round() as u32).min(height - 1);
    let w = ((viewport[2] * width as f32).round() as u32).clamp(1, width - x);
    let h = ((viewport[3] * height as f32).round() as u32).clamp(1, height - y);
    (x, y, w, h)
}

fn validate_environment_intensity(intensity: f32) -> Result<(), SurfaceRendererError> {
    if !intensity.is_finite() || !(0.0..=64.0).contains(&intensity) {
        return Err(SurfaceRendererError::InvalidEnvironmentMap {
            reason: format!("intensity must be finite and in [0, 64], received {intensity}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshVertex;

    #[test]
    fn camera_projection_modes_are_strict_and_actionable() {
        assert_eq!(
            decode_projection(0.0).unwrap(),
            CameraProjection::Perspective
        );
        assert_eq!(
            decode_projection(1.0).unwrap(),
            CameraProjection::Orthographic
        );
        assert!(matches!(
            decode_projection(2.0),
            Err(SurfaceRendererError::InvalidCameraProjection { value: 2.0 })
        ));
    }

    #[test]
    fn orthographic_projection_does_not_change_scale_with_depth() {
        let view = ViewState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 10.0,
            near: 0.1,
            far: 100.0,
            ..ViewState::default()
        };
        let projection = projection_matrix(&view, 2.0);
        let near = projection.project_point3(Vec3::new(3.0, 2.0, -1.0));
        let far = projection.project_point3(Vec3::new(3.0, 2.0, -50.0));
        assert!((near.x - far.x).abs() < f32::EPSILON);
        assert!((near.y - far.y).abs() < f32::EPSILON);
    }

    /// Mirrors the grouping in `set_render_world` without needing a GPU, so
    /// batching is verified even where no adapter exists.
    fn batches_for(slots: &[usize]) -> Vec<DrawBatch> {
        let entries: Vec<_> = slots.iter().map(|slot| (*slot, 0)).collect();
        batches_for_meshes_and_materials(&entries)
    }

    fn batches_for_meshes_and_materials(entries: &[(usize, usize)]) -> Vec<DrawBatch> {
        let mut ordered = entries.to_vec();
        ordered.sort_unstable_by_key(|(mesh, material)| (*material, *mesh));
        let mut batches: Vec<DrawBatch> = Vec::new();
        for (index, (mesh, material)) in ordered.iter().enumerate() {
            match batches.last_mut() {
                Some(batch) if batch.mesh == *mesh && batch.material == *material => {
                    batch.instance_count += 1;
                }
                _ => batches.push(DrawBatch {
                    mesh: *mesh,
                    material: *material,
                    pipeline: PipelineKind::OpaqueCulled,
                    first_instance: index as u32,
                    instance_count: 1,
                }),
            }
        }
        batches
    }

    #[test]
    fn instances_sharing_a_mesh_collapse_into_one_draw() {
        let batches = batches_for(&[0, 0, 0, 0]);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instance_count, 4);
        assert_eq!(batches[0].first_instance, 0);
    }

    #[test]
    fn interleaved_meshes_still_produce_one_draw_each() {
        // Scene order must not affect the draw count; that is the whole point
        // of sorting before batching.
        let batches = batches_for(&[2, 0, 2, 1, 0, 2]);
        assert_eq!(batches.len(), 3, "one batch per distinct mesh");
        let total: u32 = batches.iter().map(|batch| batch.instance_count).sum();
        assert_eq!(total, 6);
        // Batches must tile the instance buffer with no gaps or overlap.
        let mut cursor = 0;
        for batch in &batches {
            assert_eq!(batch.first_instance, cursor);
            cursor += batch.instance_count;
        }
    }

    #[test]
    fn an_empty_scene_issues_no_draws() {
        assert!(batches_for(&[]).is_empty());
    }

    #[test]
    fn one_mesh_with_two_materials_forms_two_batches() {
        let batches = batches_for_meshes_and_materials(&[(3, 1), (3, 2), (3, 1)]);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].instance_count, 2);
        assert_eq!(batches[1].instance_count, 1);
    }

    #[test]
    fn blended_submissions_follow_opaque_and_sort_back_to_front() {
        let entry = |slot: usize, pipeline: PipelineKind, z: f32| InstanceSubmission {
            slot,
            material: slot,
            pipeline,
            model: Mat4::from_translation(Vec3::new(0.0, 0.0, z)),
            base_color: [1.0; 4],
            material_params: [0.0; 4],
            bone_offset: -1.0,
        };
        let mut submissions = vec![
            entry(1, PipelineKind::BlendCulledNoDepthWrite, 2.0),
            entry(2, PipelineKind::OpaqueCulled, 9.0),
            entry(3, PipelineKind::BlendDoubleSidedNoDepthWrite, 12.0),
            entry(4, PipelineKind::BlendCulledDepthWrite, 5.0),
        ];
        sort_submissions(&mut submissions, Vec3::ZERO);
        assert_eq!(
            submissions
                .iter()
                .map(|entry| entry.slot)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 1]
        );
    }

    #[test]
    fn material_descriptors_normalize_defaults_and_reject_bad_values() {
        let material = normalize_material("test", &StandardMaterialDescriptor::default()).unwrap();
        assert_eq!(material.uniforms.base_color, [1.0; 4]);
        assert_eq!(material.filter, wgpu::FilterMode::Linear);
        let sdf = normalize_material(
            "font",
            &StandardMaterialDescriptor {
                coverage_mode: Some("sdf".into()),
                alpha_mode: Some("blend".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sdf.uniforms.roughness_normal_occlusion_cutoff[3], 1.0);
        assert!(matches!(
            normalize_material(
                "bad-coverage",
                &StandardMaterialDescriptor {
                    coverage_mode: Some("msdf".into()),
                    ..Default::default()
                }
            ),
            Err(SurfaceRendererError::InvalidMaterialId { .. })
        ));
        let invalid = StandardMaterialDescriptor {
            normal_scale: Some(f32::NAN),
            ..Default::default()
        };
        assert!(matches!(
            normalize_material("bad", &invalid),
            Err(SurfaceRendererError::InvalidMaterialId { .. })
        ));
        let atlas = normalize_material(
            "atlas",
            &StandardMaterialDescriptor {
                atlas: Some(TextureAtlasDescriptor {
                    columns: 16,
                    rows: 96,
                    inset: Some(0.006),
                    origin: Some("top-left".into()),
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            atlas.uniforms.atlas_grid_inset_origin,
            [16.0, 96.0, 0.006, 1.0]
        );
        let transparent = normalize_material(
            "glass",
            &StandardMaterialDescriptor {
                alpha_mode: Some("blend".into()),
                double_sided: Some(true),
                depth_write: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            transparent.pipeline,
            PipelineKind::BlendDoubleSidedNoDepthWrite
        );
        assert!(
            normalize_material(
                "bad-atlas",
                &StandardMaterialDescriptor {
                    atlas: Some(TextureAtlasDescriptor {
                        columns: 0,
                        rows: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .is_err()
        );
    }

    /// An XR eye's asymmetric frustum replaces the derived projection
    /// wherever the renderer projects: the frame's view-projection and the
    /// culling planes both come from `projection_matrix`.
    #[test]
    fn an_explicit_projection_overrides_the_derived_one() {
        let mut view = ViewState::default();
        let derived = projection_matrix(&view, 1.5);
        // A left eye: frustum shifted so the right edge is wider than the left.
        let eye = Mat4::perspective_rh(80f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::from_translation(Vec3::new(0.15, 0.0, 0.0));
        view.projection_override = Some(eye);
        assert_eq!(projection_matrix(&view, 1.5), eye);
        assert_ne!(projection_matrix(&view, 1.5), derived);
        let planes = frustum_planes(eye);
        // The shift moves the frustum: a point just inside the derived
        // left edge at z = -10 is outside the shifted frustum's left edge.
        let derived_planes = frustum_planes(derived);
        let probe = Vec3::new(-8.6, 0.0, -10.0);
        assert!(sphere_inside(&derived_planes, probe, 0.0));
        assert!(!sphere_inside(&planes, probe, 0.0));
    }

    #[test]
    fn viewports_resolve_to_whole_texels_inside_the_destination() {
        assert_eq!(viewport_pixels(FULL_VIEWPORT, (640, 480)), (0, 0, 640, 480));
        // Side-by-side stereo halves tile the surface exactly.
        assert_eq!(
            viewport_pixels([0.0, 0.0, 0.5, 1.0], (641, 480)),
            (0, 0, 321, 480)
        );
        assert_eq!(
            viewport_pixels([0.5, 0.0, 0.5, 1.0], (641, 480)),
            (321, 0, 320, 480)
        );
        // A rectangle that rounds past the edge is clamped, never empty.
        assert_eq!(
            viewport_pixels([0.999, 0.999, 0.001, 0.001], (10, 10)),
            (9, 9, 1, 1)
        );
        assert_eq!(
            viewport_pixels([0.25, 0.25, 0.5, 0.5], (0, 0)),
            (0, 0, 1, 1)
        );
    }

    #[test]
    fn frame_arrays_are_length_checked_before_reaching_the_gpu() {
        // These come straight from a game's typed arrays. A short matrix array
        // would otherwise read past its end into whatever follows in memory.
        assert!(matches!(
            expect_len("matrices", 31, 32),
            Err(SurfaceRendererError::FrameArrayLength {
                field: "matrices",
                actual: 31,
                expected: 32
            })
        ));
        assert!(expect_len("matrices", 32, 32).is_ok());
    }

    #[test]
    fn frame_arrays_reject_non_finite_values() {
        assert!(finite("camera", &[1.0, 2.0, 3.0]).is_ok());
        assert!(matches!(
            finite("camera", &[1.0, f32::NAN]),
            Err(SurfaceRendererError::RenderWorld(
                RenderWorldError::NonFinite { .. }
            ))
        ));
    }

    #[test]
    fn environments_validate_fog_modes_and_ranges_without_a_gpu() {
        let linear = [
            0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0, 10.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let decoded = decode_environment(&linear, "environment").unwrap().unwrap();
        assert_eq!(decoded.clear_color, [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(decoded.fog_color_mode[3], 1.0);
        assert_eq!(decoded.fog_params[..2], [10.0, 200.0]);

        let mut invalid_range = linear;
        invalid_range[9] = 5.0;
        assert!(matches!(
            decode_environment(&invalid_range, "environment"),
            Err(SurfaceRendererError::InvalidEnvironment { .. })
        ));
        let mut invalid_mode = linear;
        invalid_mode[7] = 9.0;
        assert!(matches!(
            decode_environment(&invalid_mode, "environment"),
            Err(SurfaceRendererError::InvalidEnvironment { .. })
        ));

        let mut sky = linear;
        sky[12..16].copy_from_slice(&[0.05, 0.2, 1.4, 1.0]);
        sky[16..20].copy_from_slice(&[0.9, 0.45, 0.18, 0.7]);
        sky[20..24].copy_from_slice(&[0.03, 0.025, 0.02, 0.00465]);
        let decoded = decode_environment(&sky, "environment").unwrap().unwrap();
        assert_eq!(decoded.sky_zenith_mode, [0.05, 0.2, 1.4, 1.0]);
        let mut invalid_sun = sky;
        invalid_sun[23] = 0.0;
        assert!(decode_environment(&invalid_sun, "environment").is_err());
    }

    #[test]
    fn omitted_per_instance_arrays_fall_back_instead_of_reading_out_of_bounds() {
        // Colours and material parameters are optional in a frame submission.
        assert_eq!(read4(&[], 7, [1.0, 1.0, 1.0, 1.0]), [1.0, 1.0, 1.0, 1.0]);
        let colors = [0.0, 0.5, 1.0, 1.0, 0.25, 0.25, 0.25, 0.5];
        assert_eq!(read4(&colors, 1, [1.0; 4]), [0.25, 0.25, 0.25, 0.5]);
        // A stray NaN degrades to the fallback for that channel rather than
        // poisoning the whole frame.
        let poisoned = [f32::NAN, 0.5, 1.0, 1.0];
        assert_eq!(
            read4(&poisoned, 0, [1.0, 1.0, 1.0, 1.0]),
            [1.0, 0.5, 1.0, 1.0]
        );
    }

    #[test]
    fn instance_data_matches_the_declared_vertex_attributes() {
        // Seven vec4 attributes: 4x4 model, colour, material, and the skin
        // lane (bone offset + 3 reserved floats).
        assert_eq!(
            std::mem::size_of::<InstanceData>(),
            7 * 4 * std::mem::size_of::<f32>()
        );
        assert_eq!(InstanceData::ATTRIBUTES.len(), 7);
    }

    #[test]
    fn the_registry_recycles_slots_and_replaces_in_place() {
        // Streaming a voxel world uploads and releases meshes continuously, so
        // the table must not grow without bound, and remeshing a chunk must
        // keep its slot so instances already referencing it stay valid.
        let mut registry = MeshRegistry::default();
        assert_eq!(registry.slot("chunk:0"), None);

        registry.slots.insert("chunk:0".into(), 0);
        registry.meshes.push(None);
        registry.slots.insert("chunk:1".into(), 1);
        registry.meshes.push(None);
        assert_eq!(registry.len(), 2);

        assert!(registry.remove("chunk:0"));
        assert!(!registry.remove("chunk:0"));
        assert_eq!(registry.free, vec![0]);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.ids(), vec!["chunk:1"]);
    }

    #[test]
    fn vertex_and_instance_attributes_do_not_collide() {
        // Both buffers feed one shader, so their shader locations must be
        // disjoint or pipeline creation fails at runtime.
        let vertex: Vec<u32> = <MeshVertex as VertexLayout>::ATTRIBUTES
            .iter()
            .map(|attribute| attribute.shader_location)
            .collect();
        let instance: Vec<u32> = InstanceData::ATTRIBUTES
            .iter()
            .map(|attribute| attribute.shader_location)
            .collect();
        for location in &vertex {
            assert!(
                !instance.contains(location),
                "location {location} is used twice"
            );
        }
        assert_eq!(vertex, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(instance, vec![6, 7, 8, 9, 10, 11, 12]);
        // Locations 13/14 carry the per-vertex SkinVertex buffer (see
        // `surface::mesh::SkinVertex`).
        let skin: Vec<u32> = crate::mesh::SkinVertex::ATTRIBUTES
            .iter()
            .map(|attribute| attribute.shader_location)
            .collect();
        assert_eq!(skin, vec![13, 14]);
        for location in &skin {
            assert!(!vertex.contains(location) && !instance.contains(location));
        }
        // WebGPU guarantees exactly 16 vertex attributes; the three buffers
        // together must never exceed it or pipeline creation fails on every
        // browser.
        assert!(vertex.len() + instance.len() + skin.len() <= 16);
        assert!(vertex.iter().chain(&instance).chain(&skin).all(|l| *l < 16));
    }

    /// Builds one packed light record in the wire layout the shader expects.
    fn light_record(
        kind: f32,
        vector: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        secondary: [f32; 3],
        range: f32,
    ) -> Vec<f32> {
        vec![
            kind,
            vector[0],
            vector[1],
            vector[2],
            color[0],
            color[1],
            color[2],
            intensity,
            secondary[0],
            secondary[1],
            secondary[2],
            range,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    #[test]
    fn an_empty_light_submission_keeps_the_default_rig() {
        assert!(decode_light_rig(&[], "light").unwrap().is_none());
        // The default must light a scene from more than one direction, or a
        // surface facing away from the key light renders as a black hole.
        let rig = default_light_rig();
        assert!(rig.len() >= 2);
        assert!(
            rig.iter()
                .any(|light| light.vector_kind[3] == LIGHT_KIND_DIRECTIONAL)
        );
        assert!(
            rig.iter()
                .any(|light| light.vector_kind[3] == LIGHT_KIND_HEMISPHERE)
        );
    }

    #[test]
    fn the_legacy_seven_float_light_still_describes_one_directional_light() {
        // Games written against the original single-light API must keep
        // working unchanged after the rig landed.
        let rig = decode_light_rig(&[0.0, -1.0, 0.0, 1.0, 0.9, 0.8, 3.0], "light")
            .unwrap()
            .expect("a rig");
        let directional = rig
            .iter()
            .find(|light| light.vector_kind[3] == LIGHT_KIND_DIRECTIONAL)
            .expect("a directional light");
        assert_eq!(directional.vector_kind[..3], [0.0, -1.0, 0.0]);
        assert_eq!(directional.color_intensity, [1.0, 0.9, 0.8, 3.0]);
        // A bare directional light gets a little fill so unlit faces read as
        // shaded rather than as holes.
        assert!(
            rig.iter()
                .any(|light| light.vector_kind[3] == LIGHT_KIND_AMBIENT)
        );
    }

    #[test]
    fn every_light_kind_survives_the_round_trip() {
        let mut packed = Vec::new();
        packed.extend(light_record(
            0.0,
            [0.0; 3],
            [0.2, 0.3, 0.4],
            1.0,
            [0.0; 3],
            0.0,
        ));
        packed.extend(light_record(
            1.0,
            [0.0, -1.0, 0.0],
            [1.0; 3],
            2.0,
            [0.0; 3],
            0.0,
        ));
        packed.extend(light_record(
            2.0,
            [4.0, 5.0, 6.0],
            [1.0, 0.5, 0.0],
            3.0,
            [0.0; 3],
            12.0,
        ));
        packed.extend(light_record(
            3.0,
            [0.0; 3],
            [0.4, 0.6, 0.9],
            0.5,
            [0.2, 0.2, 0.1],
            0.0,
        ));
        packed.extend(light_record(
            4.0,
            [1.0, 2.0, 3.0],
            [1.0; 3],
            4.0,
            [0.0, -1.0, 0.0],
            20.0,
        ));

        let rig = decode_light_rig(&packed, "light").unwrap().expect("a rig");
        assert_eq!(rig.len(), 5);
        assert_eq!(rig[0].vector_kind[3], LIGHT_KIND_AMBIENT);
        assert_eq!(rig[1].vector_kind[3], LIGHT_KIND_DIRECTIONAL);

        let point = rig[2];
        assert_eq!(point.vector_kind[3], LIGHT_KIND_POINT);
        // A point light's position must survive verbatim; normalizing it, as a
        // direction would be, would move every torch to the world origin.
        assert_eq!(point.vector_kind[..3], [4.0, 5.0, 6.0]);
        assert_eq!(point.secondary_range[3], 12.0);

        let hemisphere = rig[3];
        assert_eq!(hemisphere.vector_kind[3], LIGHT_KIND_HEMISPHERE);
        assert_eq!(hemisphere.color_intensity[..3], [0.4, 0.6, 0.9]);
        assert_eq!(hemisphere.secondary_range[..3], [0.2, 0.2, 0.1]);

        let spot = rig[4];
        assert_eq!(spot.vector_kind[3], LIGHT_KIND_SPOT);
        assert_eq!(spot.vector_kind[..3], [1.0, 2.0, 3.0]);
        assert_eq!(spot.secondary_range[..3], [0.0, -1.0, 0.0]);
    }

    #[test]
    fn a_spot_cone_is_ordered_so_the_falloff_span_stays_positive() {
        // An inner cosine below the outer one would divide by a negative span
        // and light the world inside out.
        let mut record = light_record(4.0, [0.0; 3], [1.0; 3], 1.0, [0.0, -1.0, 0.0], 10.0);
        record[12] = 0.2;
        record[13] = 0.9;
        let rig = decode_light_rig(&record, "light").unwrap().expect("a rig");
        assert!(rig[0].cone[0] >= rig[0].cone[1]);
    }

    #[test]
    fn a_partial_light_record_is_rejected_rather_than_read_out_of_bounds() {
        let error = decode_light_rig(&[1.0; 20], "light").unwrap_err();
        assert!(matches!(
            error,
            SurfaceRendererError::FrameArrayLength { field: "light", .. }
        ));
    }

    #[test]
    fn a_rig_larger_than_the_shader_array_is_refused() {
        // Silently dropping the surplus would lose lighting without saying so.
        let packed = vec![0.0_f32; (MAX_LIGHTS + 1) * LIGHT_STRIDE];
        let error = decode_light_rig(&packed, "light").unwrap_err();
        assert!(matches!(
            error,
            SurfaceRendererError::TooManyLights {
                limit: MAX_LIGHTS,
                ..
            }
        ));
    }

    #[test]
    fn a_light_rig_fits_the_shader_uniform_block() {
        // The uniform array is fixed size; if the Rust struct and the WGSL
        // block disagree the GPU reads whatever follows in the buffer.
        assert_eq!(size_of::<GpuLight>(), 64);
        assert_eq!(
            size_of::<FrameUniforms>(),
            64 + 64
                + 16
                + 16
                + 16
                + 16
                + 16
                + 16
                + 16
                + 16
                + MAX_LIGHTS * 64
                + SHADOW_CASCADE_COUNT * 64
                + 16
                + 16
                + MAX_SHADOW_SLOTS * 64
                + 16
                + 16
        );
    }

    #[test]
    fn a_non_finite_light_is_rejected() {
        let mut record = light_record(1.0, [0.0, -1.0, 0.0], [1.0; 3], 1.0, [0.0; 3], 0.0);
        record[5] = f32::NAN;
        assert!(decode_light_rig(&record, "light").is_err());
    }
}
