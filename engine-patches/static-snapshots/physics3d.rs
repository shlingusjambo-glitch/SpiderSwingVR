use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, Sender, channel},
};

use rapier3d::na::Unit;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BodyType3d {
    Static,
    Dynamic,
    KinematicPosition,
    KinematicVelocity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ColliderShape3d {
    Box {
        half_extents: [f32; 3],
    },
    Sphere {
        radius: f32,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    Cylinder {
        half_height: f32,
        radius: f32,
    },
    /// Infinite plane through the body's origin, facing `normal`.
    Plane {
        normal: [f32; 3],
    },
    ConvexMesh {
        points: Vec<[f32; 3]>,
    },
    TriangleMesh {
        vertices: Vec<[f32; 3]>,
        indices: Vec<[u32; 3]>,
    },
    /// Regular X/Z grid. `rows * columns` heights are stored row-major.
    Heightfield {
        rows: u32,
        columns: u32,
        heights: Vec<f32>,
        scale: [f32; 3],
    },
    /// Several child shapes attached to one rigid body. Each transform is
    /// local to the body and compound shapes may be nested.
    Compound {
        children: Vec<CompoundColliderChild3d>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompoundColliderChild3d {
    pub position: [f32; 3],
    /// Local quaternion in browser-friendly `[x, y, z, w]` order.
    pub rotation: [f32; 4],
    pub shape: ColliderShape3d,
}

/// How two colliders' friction or restitution coefficients are reconciled.
///
/// Contacts involve two surfaces, so a single coefficient per collider is not
/// enough to say what happens when rubber meets ice. `Min` is the rule that
/// makes "slippery wins" work, `Max` the one that makes "bouncy wins" work, and
/// `Average` is the default because it is the least surprising.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CombineRule {
    #[default]
    Average,
    Min,
    Max,
    Multiply,
}

impl From<CombineRule> for CoefficientCombineRule {
    fn from(value: CombineRule) -> Self {
        match value {
            CombineRule::Average => Self::Average,
            CombineRule::Min => Self::Min,
            CombineRule::Max => Self::Max,
            CombineRule::Multiply => Self::Multiply,
        }
    }
}

/// A named surface: friction, restitution, and how each combines with another
/// collider's.
///
/// Naming it is the point. "ice" defined once and referenced by every icy
/// collider is one place to retune when skating feels wrong, and it survives a
/// scene round trip as a short string instead of four duplicated numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsMaterial {
    pub name: String,
    pub friction: f32,
    pub restitution: f32,
    #[serde(default)]
    pub friction_combine: CombineRule,
    #[serde(default)]
    pub restitution_combine: CombineRule,
}

/// Explicit mass, centre of mass, and inertia, replacing the density-derived
/// values.
///
/// Offsetting the centre of mass is what makes a weighted object behave like
/// one: a bottom-heavy buoy rights itself, a top-heavy crate tips. Density
/// alone always puts the centre of mass at the shape's centroid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MassProperties3d {
    pub mass: f32,
    /// Centre of mass in the body's own local space.
    pub center_of_mass: [f32; 3],
    /// Principal moments of inertia. Derived from `mass` and the shape's
    /// extents when omitted, which is what most callers want.
    #[serde(default)]
    pub principal_inertia: Option<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsBodyDescriptor3d {
    pub id: u64,
    pub body_type: BodyType3d,
    pub position: [f32; 3],
    /// Quaternion in browser-friendly `[x, y, z, w]` order.
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub continuous_collision_detection: bool,
    pub shape: ColliderShape3d,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
    pub sensor: bool,
    /// One or more of 32 collision-layer bits.
    pub layer: u32,
    /// Layers this collider accepts.
    pub mask: u32,
    /// A material registered with [`PhysicsWorld3d::define_material`]. It
    /// replaces `friction` and `restitution` and supplies the combine rules.
    #[serde(default)]
    pub material: Option<String>,
    /// Combine rules used when no material is named.
    #[serde(default)]
    pub friction_combine: CombineRule,
    #[serde(default)]
    pub restitution_combine: CombineRule,
    /// Overrides the density-derived mass, centre of mass, and inertia.
    #[serde(default)]
    pub mass_properties: Option<MassProperties3d>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsBodySnapshot3d {
    pub id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub sleeping: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PhysicsEventKind {
    CollisionEnter,
    CollisionExit,
    TriggerEnter,
    TriggerExit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsEvent3d {
    pub kind: PhysicsEventKind,
    pub body_a: u64,
    pub body_b: u64,
}

/// One solver contact, in world space. Debug rendering's raw material.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContactPoint3d {
    pub body_a: u64,
    pub body_b: u64,
    pub point: [f32; 3],
    /// Points out of `body_a`.
    pub normal: [f32; 3],
    /// Negative while the shapes overlap.
    pub distance: f32,
}

/// What the solver is actually carrying this step.
///
/// Body counts split by type and by sleep state are the two numbers that
/// explain a physics frame: a world with 4000 bodies of which 40 are awake
/// costs nothing, and the same world with all 4000 awake is the hitch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsStats3d {
    pub bodies: usize,
    pub dynamic: usize,
    pub kinematic: usize,
    pub fixed: usize,
    pub awake: usize,
    pub sleeping: usize,
    pub joints: usize,
    pub materials: usize,
    /// Collider pairs the narrow phase is tracking.
    pub contact_pairs: usize,
    pub manifolds: usize,
    pub contact_points: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RaycastHit3d {
    pub body_id: u64,
    pub distance: f32,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShapeCastHit3d {
    pub body_id: u64,
    pub distance: f32,
    /// World-space position of the swept shape when impact occurs.
    pub position: [f32; 3],
    /// World-space normal pointing out of the hit collider.
    pub normal: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterControllerDescriptor3d {
    pub shape: ColliderShape3d,
    pub offset: f32,
    pub slide: bool,
    pub autostep_max_height: Option<f32>,
    pub autostep_min_width: f32,
    pub autostep_dynamic_bodies: bool,
    pub max_slope_climb_degrees: f32,
    pub min_slope_slide_degrees: f32,
    pub snap_to_ground: Option<f32>,
    pub layer: u32,
    pub mask: u32,
    pub include_sensors: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterMovement3d {
    pub translation: [f32; 3],
    pub grounded: bool,
    pub sliding_down_slope: bool,
    pub collisions: Vec<u64>,
}

/// A constraint between two bodies.
///
/// The set is deliberately small and named for what games build with them: a
/// hinge is a door and an elbow, a slider is a piston and a drawer, a ball is a
/// shoulder and a tow hitch, a spring is suspension and a rope bridge. Exposing
/// Rapier's full generic joint would be more expressive and much harder to use
/// correctly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum JointDescriptor3d {
    /// Welds two bodies rigidly. Used for breakable joins and compound props.
    Fixed {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
    },
    /// One rotational degree of freedom about `axis`.
    Hinge {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        /// Angular limits in radians, as `[min, max]`.
        limits: Option<[f32; 2]>,
        /// Target angular velocity and maximum force, as `[velocity, force]`.
        motor: Option<[f32; 2]>,
    },
    /// One translational degree of freedom along `axis`.
    Slider {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        axis: [f32; 3],
        /// Travel limits along the axis, as `[min, max]`.
        limits: Option<[f32; 2]>,
        motor: Option<[f32; 2]>,
    },
    /// Three rotational degrees of freedom: a shoulder, a hip, a tow hitch.
    Ball {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
    },
    /// A damped spring holding the anchors at `rest_length`.
    Spring {
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JointSpec3d {
    pub id: u64,
    pub body_a: u64,
    pub body_b: u64,
    pub joint: JointDescriptor3d,
    /// Whether the two connected bodies collide with each other. Off by
    /// default: a hinge whose halves collide jitters at the joint.
    #[serde(default)]
    pub collide_connected: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum PhysicsError3d {
    #[error("physics body ID must be non-zero")]
    ZeroId,
    #[error("physics body '{0}' already exists")]
    DuplicateBody(u64),
    #[error("unknown physics body '{0}'")]
    UnknownBody(u64),
    #[error("{0} must contain only finite values")]
    NonFinite(&'static str),
    #[error("{0} must be finite and non-negative")]
    NonNegative(&'static str),
    #[error("{0} must be finite and positive")]
    Positive(&'static str),
    #[error("physics body rotation quaternion cannot have zero length")]
    ZeroQuaternion,
    #[error("physics collision layer must contain at least one bit")]
    EmptyLayer,
    #[error("physics timestep must be finite and positive")]
    InvalidTimestep,
    #[error("ray direction cannot have zero length")]
    ZeroRayDirection,
    #[error("invalid collider shape: {0}")]
    InvalidShape(String),
    #[error("physics joint '{0}' already exists")]
    DuplicateJoint(u64),
    #[error("unknown physics joint '{0}'")]
    UnknownJoint(u64),
    #[error("a joint cannot connect body '{0}' to itself")]
    SelfJoint(u64),
    #[error("joint axis cannot have zero length")]
    ZeroJointAxis,
    #[error("joint limits must be ordered as [min, max]")]
    InvalidJointLimits,
    #[error("physics material name cannot be empty")]
    EmptyMaterialName,
    #[error("unknown physics material '{0}'")]
    UnknownMaterial(String),
}

/// Rust-owned general 3D rigid-body world. Rapier handles never cross the
/// public boundary; stable IDs survive arena recycling and serialize cleanly.
pub struct PhysicsWorld3d {
    inner: PhysicsWorld,
    bodies: HashMap<u64, (RigidBodyHandle, ColliderHandle)>,
    collider_ids: HashMap<ColliderHandle, u64>,
    joints: HashMap<u64, ImpulseJointHandle>,
    materials: HashMap<String, PhysicsMaterial>,
    collision_send: Sender<CollisionEvent>,
    collision_recv: Receiver<CollisionEvent>,
    force_send: Sender<ContactForceEvent>,
    force_recv: Receiver<ContactForceEvent>,
}

impl Default for PhysicsWorld3d {
    fn default() -> Self {
        Self::new([0.0, -9.81, 0.0])
    }
}

impl PhysicsWorld3d {
    #[must_use]
    pub fn new(gravity: [f32; 3]) -> Self {
        let (collision_send, collision_recv) = channel();
        let (force_send, force_recv) = channel();
        let mut inner = PhysicsWorld::new();
        inner.gravity = Vector::new(gravity[0], gravity[1], gravity[2]);
        Self {
            inner,
            bodies: HashMap::new(),
            collider_ids: HashMap::new(),
            joints: HashMap::new(),
            materials: HashMap::new(),
            collision_send,
            collision_recv,
            force_send,
            force_recv,
        }
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> Result<(), PhysicsError3d> {
        finite3(gravity, "physics gravity")?;
        self.inner.gravity = Vector::from(gravity);
        Ok(())
    }
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Registers a named surface, replacing any material of the same name.
    ///
    /// Redefinition is deliberate: retuning "ice" at runtime is how a game
    /// makes a road icy in the rain. It affects colliders assigned the material
    /// from then on, not colliders already built from it — call
    /// [`Self::set_material`] for those.
    pub fn define_material(&mut self, material: PhysicsMaterial) -> Result<(), PhysicsError3d> {
        if material.name.trim().is_empty() {
            return Err(PhysicsError3d::EmptyMaterialName);
        }
        non_negative(material.friction, "material friction")?;
        non_negative(material.restitution, "material restitution")?;
        self.materials.insert(material.name.clone(), material);
        Ok(())
    }

    #[must_use]
    pub fn material(&self, name: &str) -> Option<&PhysicsMaterial> {
        self.materials.get(name)
    }

    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Re-surfaces an existing collider. Wet asphalt, an iced-over ramp.
    pub fn set_material(&mut self, id: u64, name: &str) -> Result<(), PhysicsError3d> {
        let material = self
            .materials
            .get(name)
            .ok_or_else(|| PhysicsError3d::UnknownMaterial(name.to_owned()))?
            .clone();
        let (body, handle) = *self
            .bodies
            .get(&id)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        let collider = self
            .inner
            .colliders
            .get_mut(handle)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        collider.set_friction(material.friction);
        collider.set_restitution(material.restitution);
        collider.set_friction_combine_rule(material.friction_combine.into());
        collider.set_restitution_combine_rule(material.restitution_combine.into());
        // A sleeping body would keep sliding on its old friction until
        // something else disturbed it, which reads as the change not working.
        if let Some(body) = self.inner.bodies.get_mut(body) {
            body.wake_up(true);
        }
        Ok(())
    }

    pub fn insert(&mut self, descriptor: PhysicsBodyDescriptor3d) -> Result<(), PhysicsError3d> {
        validate_descriptor(&descriptor)?;
        if self.bodies.contains_key(&descriptor.id) {
            return Err(PhysicsError3d::DuplicateBody(descriptor.id));
        }
        let surface = match &descriptor.material {
            Some(name) => self
                .materials
                .get(name)
                .ok_or_else(|| PhysicsError3d::UnknownMaterial(name.clone()))?
                .clone(),
            None => PhysicsMaterial {
                name: String::new(),
                friction: descriptor.friction,
                restitution: descriptor.restitution,
                friction_combine: descriptor.friction_combine,
                restitution_combine: descriptor.restitution_combine,
            },
        };
        let rotation = quaternion(descriptor.rotation)?;
        let position = Pose::from_parts(descriptor.position.into(), rotation);
        let mut body = match descriptor.body_type {
            BodyType3d::Static => RigidBodyBuilder::fixed(),
            BodyType3d::Dynamic => RigidBodyBuilder::dynamic(),
            BodyType3d::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
            BodyType3d::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
        };
        body = body
            .pose(position)
            .linvel(descriptor.linear_velocity.into())
            .angvel(descriptor.angular_velocity.into())
            .gravity_scale(descriptor.gravity_scale)
            .linear_damping(descriptor.linear_damping)
            .angular_damping(descriptor.angular_damping)
            .ccd_enabled(descriptor.continuous_collision_detection)
            .user_data(u128::from(descriptor.id));
        let groups = InteractionGroups::new(
            Group::from_bits_truncate(descriptor.layer),
            Group::from_bits_truncate(descriptor.mask),
            InteractionTestMode::And,
        );
        let mut collider = collider_builder(&descriptor.shape)?
            .density(descriptor.density)
            .friction(surface.friction)
            .restitution(surface.restitution)
            .friction_combine_rule(surface.friction_combine.into())
            .restitution_combine_rule(surface.restitution_combine.into())
            .sensor(descriptor.sensor)
            .collision_groups(groups)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .user_data(u128::from(descriptor.id));
        if let Some(mass) = descriptor.mass_properties {
            collider = collider.mass_properties(mass_properties(&descriptor.shape, mass)?);
        }
        let (body_handle, collider_handle) = self.inner.insert(body, collider);
        self.bodies
            .insert(descriptor.id, (body_handle, collider_handle));
        self.collider_ids.insert(collider_handle, descriptor.id);
        Ok(())
    }

    pub fn remove(&mut self, id: u64) -> Result<(), PhysicsError3d> {
        let (body, collider) = self
            .bodies
            .remove(&id)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        self.collider_ids.remove(&collider);
        // A joint referencing a removed body would dangle; Rapier drops the
        // constraint itself, so the id map is all that needs cleaning up.
        let orphaned: Vec<u64> = self
            .joints
            .iter()
            .filter(|(_, handle)| {
                self.inner
                    .impulse_joints
                    .get(**handle)
                    .is_none_or(|joint| joint.body1() == body || joint.body2() == body)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in orphaned {
            self.joints.remove(&id);
        }
        self.inner.remove_body(body);
        Ok(())
    }

    pub fn add_force(&mut self, id: u64, force: [f32; 3]) -> Result<(), PhysicsError3d> {
        finite3(force, "physics force")?;
        self.body_mut(id)?.add_force(force.into(), true);
        Ok(())
    }
    pub fn add_force_at_point(
        &mut self,
        id: u64,
        force: [f32; 3],
        point: [f32; 3],
    ) -> Result<(), PhysicsError3d> {
        finite3(force, "physics force")?;
        finite3(point, "physics force point")?;
        self.body_mut(id)?
            .add_force_at_point(force.into(), point.into(), true);
        Ok(())
    }
    pub fn add_torque(&mut self, id: u64, torque: [f32; 3]) -> Result<(), PhysicsError3d> {
        finite3(torque, "physics torque")?;
        self.body_mut(id)?.add_torque(torque.into(), true);
        Ok(())
    }
    pub fn apply_impulse(&mut self, id: u64, impulse: [f32; 3]) -> Result<(), PhysicsError3d> {
        finite3(impulse, "physics impulse")?;
        self.body_mut(id)?.apply_impulse(impulse.into(), true);
        Ok(())
    }
    pub fn apply_torque_impulse(
        &mut self,
        id: u64,
        torque: [f32; 3],
    ) -> Result<(), PhysicsError3d> {
        finite3(torque, "physics torque impulse")?;
        self.body_mut(id)?.apply_torque_impulse(torque.into(), true);
        Ok(())
    }
    pub fn set_linear_velocity(
        &mut self,
        id: u64,
        velocity: [f32; 3],
    ) -> Result<(), PhysicsError3d> {
        finite3(velocity, "physics linear velocity")?;
        self.body_mut(id)?.set_linvel(velocity.into(), true);
        Ok(())
    }
    pub fn set_angular_velocity(
        &mut self,
        id: u64,
        velocity: [f32; 3],
    ) -> Result<(), PhysicsError3d> {
        finite3(velocity, "physics angular velocity")?;
        self.body_mut(id)?.set_angvel(velocity.into(), true);
        Ok(())
    }
    pub fn set_kinematic_target(
        &mut self,
        id: u64,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> Result<(), PhysicsError3d> {
        finite3(position, "kinematic target position")?;
        let pose = Pose::from_parts(position.into(), quaternion(rotation)?);
        self.body_mut(id)?.set_next_kinematic_position(pose);
        Ok(())
    }

    #[must_use]
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Connects two bodies with a constraint.
    ///
    /// Anchors are given in each body's own local space, which is what makes a
    /// joint description independent of where the bodies happen to be when it
    /// is created — a door hinge is "the left edge of the door" regardless of
    /// which way the door is currently swinging.
    pub fn add_joint(&mut self, spec: JointSpec3d) -> Result<(), PhysicsError3d> {
        if spec.id == 0 {
            return Err(PhysicsError3d::ZeroId);
        }
        if self.joints.contains_key(&spec.id) {
            return Err(PhysicsError3d::DuplicateJoint(spec.id));
        }
        if spec.body_a == spec.body_b {
            return Err(PhysicsError3d::SelfJoint(spec.body_a));
        }
        let (handle_a, _) = *self
            .bodies
            .get(&spec.body_a)
            .ok_or(PhysicsError3d::UnknownBody(spec.body_a))?;
        let (handle_b, _) = *self
            .bodies
            .get(&spec.body_b)
            .ok_or(PhysicsError3d::UnknownBody(spec.body_b))?;

        let joint = build_joint(&spec.joint, spec.collide_connected)?;
        let handle = self
            .inner
            .impulse_joints
            .insert(handle_a, handle_b, joint, true);
        self.joints.insert(spec.id, handle);
        Ok(())
    }

    pub fn remove_joint(&mut self, id: u64) -> Result<(), PhysicsError3d> {
        let handle = self
            .joints
            .remove(&id)
            .ok_or(PhysicsError3d::UnknownJoint(id))?;
        // Waking the bodies matters: a sleeping body whose joint was removed
        // would otherwise stay frozen in place until something else disturbs it.
        self.inner.impulse_joints.remove(handle, true);
        Ok(())
    }

    /// Updates a motor's target velocity and maximum force.
    pub fn set_joint_motor(
        &mut self,
        id: u64,
        target_velocity: f32,
        max_force: f32,
    ) -> Result<(), PhysicsError3d> {
        if !target_velocity.is_finite() {
            return Err(PhysicsError3d::NonFinite("joint motor velocity"));
        }
        if !max_force.is_finite() || max_force < 0.0 {
            return Err(PhysicsError3d::NonNegative("joint motor force"));
        }
        let handle = *self
            .joints
            .get(&id)
            .ok_or(PhysicsError3d::UnknownJoint(id))?;
        let joint = self
            .inner
            .impulse_joints
            .get_mut(handle, true)
            .ok_or(PhysicsError3d::UnknownJoint(id))?;
        for axis in [
            JointAxis::AngX,
            JointAxis::AngY,
            JointAxis::AngZ,
            JointAxis::LinX,
            JointAxis::LinY,
            JointAxis::LinZ,
        ] {
            if joint.data.motor_axes.contains(axis.into()) {
                joint
                    .data
                    .set_motor_velocity(axis, target_velocity, max_force);
            }
        }
        Ok(())
    }

    pub fn step(&mut self, timestep: f32) -> Result<Vec<PhysicsEvent3d>, PhysicsError3d> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(PhysicsError3d::InvalidTimestep);
        }
        self.inner.integration_parameters.dt = timestep;
        let collector =
            ChannelEventCollector::new(self.collision_send.clone(), self.force_send.clone());
        self.inner.step_with_events(&(), &collector);
        // Rapier keeps user forces until explicitly cleared. Vapour's command
        // API treats addForce/addForceAtPoint/addTorque as forces for the next
        // fixed step, matching how gameplay systems submit suspension, tyre,
        // and movement forces once per update.
        for (_, body) in self.inner.bodies.iter_mut() {
            body.reset_forces(false);
            body.reset_torques(false);
        }
        while self.force_recv.try_recv().is_ok() {}
        let mut events = Vec::new();
        while let Ok(event) = self.collision_recv.try_recv() {
            let Some(&a) = self.collider_ids.get(&event.collider1()) else {
                continue;
            };
            let Some(&b) = self.collider_ids.get(&event.collider2()) else {
                continue;
            };
            let (body_a, body_b) = if a <= b { (a, b) } else { (b, a) };
            let kind = match (event.started(), event.sensor()) {
                (true, true) => PhysicsEventKind::TriggerEnter,
                (false, true) => PhysicsEventKind::TriggerExit,
                (true, false) => PhysicsEventKind::CollisionEnter,
                (false, false) => PhysicsEventKind::CollisionExit,
            };
            events.push(PhysicsEvent3d {
                kind,
                body_a,
                body_b,
            });
        }
        events.sort_by_key(|event| (event.body_a, event.body_b, event_kind_order(event.kind)));
        Ok(events)
    }

    pub fn snapshots(&self) -> Vec<PhysicsBodySnapshot3d> {
        let mut snapshots: Vec<_> = self
            .bodies
            .iter()
            .filter_map(|(&id, &(handle, _))| {
                // Fixed bodies never move, so a world made mostly of static level
                // geometry does not pay to serialise them every step.
                self.inner.bodies.get(handle).filter(|body| !body.is_fixed()).map(|body| {
                    let rotation = body.rotation().to_array();
                    PhysicsBodySnapshot3d {
                        id,
                        position: body.translation().into(),
                        rotation,
                        linear_velocity: body.linvel().into(),
                        angular_velocity: body.angvel().into(),
                        sleeping: body.is_sleeping(),
                    }
                })
            })
            .collect();
        snapshots.sort_by_key(|snapshot| snapshot.id);
        snapshots
    }

    /// The mass, centre of mass, and principal inertia Rapier is simulating
    /// for a body, whether it came from density or from an explicit override.
    pub fn mass_properties_of(&self, id: u64) -> Result<MassProperties3d, PhysicsError3d> {
        let handle = self
            .bodies
            .get(&id)
            .map(|entry| entry.0)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        let body = self
            .inner
            .bodies
            .get(handle)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        let local = &body.mass_properties().local_mprops;
        Ok(MassProperties3d {
            mass: local.mass(),
            center_of_mass: local.local_com.into(),
            principal_inertia: Some(local.principal_inertia().into()),
        })
    }

    /// Every solver contact from the last step, in world space.
    ///
    /// Only pairs the solver actually resolved are included: sensors and
    /// filtered pairs have no solver contacts, which is also what a debug view
    /// wants to show.
    #[must_use]
    pub fn contacts(&self) -> Vec<ContactPoint3d> {
        let mut contacts = Vec::new();
        for pair in self.inner.narrow_phase.contact_pairs() {
            let (Some(&body_a), Some(&body_b)) = (
                self.collider_ids.get(&pair.collider1),
                self.collider_ids.get(&pair.collider2),
            ) else {
                continue;
            };
            for manifold in &pair.manifolds {
                for contact in &manifold.data.solver_contacts {
                    contacts.push(ContactPoint3d {
                        body_a,
                        body_b,
                        point: contact.point.into(),
                        normal: manifold.data.normal.into(),
                        distance: contact.dist,
                    });
                }
            }
        }
        contacts.sort_by(|a, b| {
            (a.body_a, a.body_b).cmp(&(b.body_a, b.body_b)).then(
                a.point
                    .partial_cmp(&b.point)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        contacts
    }

    #[must_use]
    pub fn stats(&self) -> PhysicsStats3d {
        let mut stats = PhysicsStats3d {
            bodies: self.bodies.len(),
            joints: self.joints.len(),
            materials: self.materials.len(),
            ..PhysicsStats3d::default()
        };
        for &(handle, _) in self.bodies.values() {
            let Some(body) = self.inner.bodies.get(handle) else {
                continue;
            };
            match body.body_type() {
                RigidBodyType::Dynamic => stats.dynamic += 1,
                RigidBodyType::Fixed => stats.fixed += 1,
                RigidBodyType::KinematicPositionBased | RigidBodyType::KinematicVelocityBased => {
                    stats.kinematic += 1;
                }
            }
            if body.is_sleeping() || body.is_fixed() {
                stats.sleeping += 1;
            } else {
                stats.awake += 1;
            }
        }
        for pair in self.inner.narrow_phase.contact_pairs() {
            stats.contact_pairs += 1;
            stats.manifolds += pair.manifolds.len();
            stats.contact_points += pair
                .manifolds
                .iter()
                .map(|manifold| manifold.data.solver_contacts.len())
                .sum::<usize>();
        }
        stats
    }

    /// `exclude` skips one body, which is what a ray cast from inside your
    /// own collider needs: a vehicle's wheel rays start inside the chassis.
    #[allow(clippy::too_many_arguments)]
    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        layer: u32,
        mask: u32,
        include_sensors: bool,
        exclude: Option<u64>,
    ) -> Result<Option<RaycastHit3d>, PhysicsError3d> {
        finite3(origin, "ray origin")?;
        finite3(direction, "ray direction")?;
        positive(max_distance, "ray maximum distance")?;
        let mut direction = Vector::from(direction);
        let length = direction.length();
        if length <= f32::EPSILON {
            return Err(PhysicsError3d::ZeroRayDirection);
        }
        direction /= length;
        let groups = InteractionGroups::new(
            Group::from_bits_truncate(layer),
            Group::from_bits_truncate(mask),
            InteractionTestMode::And,
        );
        let mut filter = QueryFilter::default().groups(groups);
        if !include_sensors {
            filter = filter.exclude_sensors();
        }
        if let Some(id) = exclude {
            let (body, _) = *self
                .bodies
                .get(&id)
                .ok_or(PhysicsError3d::UnknownBody(id))?;
            filter = filter.exclude_rigid_body(body);
        }
        let ray = Ray::new(origin.into(), direction);
        Ok(self
            .inner
            .cast_ray_and_get_normal(&ray, max_distance, true, filter)
            .and_then(|(handle, hit)| {
                self.collider_ids.get(&handle).map(|&body_id| {
                    let point = ray.point_at(hit.time_of_impact);
                    RaycastHit3d {
                        body_id,
                        distance: hit.time_of_impact,
                        point: point.into(),
                        normal: hit.normal.into(),
                    }
                })
            }))
    }

    pub fn overlap_shape(
        &self,
        position: [f32; 3],
        rotation: [f32; 4],
        shape: &ColliderShape3d,
        layer: u32,
        mask: u32,
        include_sensors: bool,
    ) -> Result<Vec<u64>, PhysicsError3d> {
        finite3(position, "overlap position")?;
        let pose = Pose::from_parts(position.into(), quaternion(rotation)?);
        let collider = collider_builder(shape)?.build();
        let mut filter = query_filter(layer, mask);
        if !include_sensors {
            filter = filter.exclude_sensors();
        }
        let mut ids: Vec<_> = self
            .inner
            .intersect_shape(pose, collider.shape(), filter)
            .filter_map(|(handle, _)| self.collider_ids.get(&handle).copied())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cast_shape(
        &self,
        position: [f32; 3],
        rotation: [f32; 4],
        direction: [f32; 3],
        max_distance: f32,
        shape: &ColliderShape3d,
        layer: u32,
        mask: u32,
        include_sensors: bool,
    ) -> Result<Option<ShapeCastHit3d>, PhysicsError3d> {
        finite3(position, "shape-cast position")?;
        finite3(direction, "shape-cast direction")?;
        positive(max_distance, "shape-cast maximum distance")?;
        let mut direction = Vector::from(direction);
        let length = direction.length();
        if length <= f32::EPSILON {
            return Err(PhysicsError3d::ZeroRayDirection);
        }
        direction /= length;
        let pose = Pose::from_parts(position.into(), quaternion(rotation)?);
        let collider = collider_builder(shape)?.build();
        let mut filter = query_filter(layer, mask);
        if !include_sensors {
            filter = filter.exclude_sensors();
        }
        let options = rapier3d::parry::query::ShapeCastOptions::with_max_time_of_impact(1.0);
        Ok(self
            .inner
            .cast_shape(
                &pose,
                direction * max_distance,
                collider.shape(),
                options,
                filter,
            )
            .and_then(|(handle, hit)| {
                self.collider_ids.get(&handle).map(|&body_id| {
                    let distance = hit.time_of_impact * max_distance;
                    let normal = self
                        .inner
                        .colliders
                        .get(handle)
                        .map_or(-hit.normal2, |collider| collider.rotation() * -hit.normal2);
                    ShapeCastHit3d {
                        body_id,
                        distance,
                        position: (Vector::from(position) + direction * distance).into(),
                        normal: normal.into(),
                    }
                })
            }))
    }

    pub fn move_character(
        &self,
        timestep: f32,
        position: [f32; 3],
        rotation: [f32; 4],
        desired_translation: [f32; 3],
        descriptor: &CharacterControllerDescriptor3d,
    ) -> Result<CharacterMovement3d, PhysicsError3d> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(PhysicsError3d::InvalidTimestep);
        }
        finite3(position, "character position")?;
        finite3(desired_translation, "character desired translation")?;
        positive(descriptor.offset, "character offset")?;
        positive(
            descriptor.autostep_min_width,
            "character autostep minimum width",
        )?;
        if let Some(value) = descriptor.autostep_max_height {
            positive(value, "character autostep maximum height")?;
        }
        if let Some(value) = descriptor.snap_to_ground {
            non_negative(value, "character ground snap distance")?;
        }
        slope(
            descriptor.max_slope_climb_degrees,
            "character maximum slope climb",
        )?;
        slope(
            descriptor.min_slope_slide_degrees,
            "character minimum slope slide",
        )?;
        let collider = collider_builder(&descriptor.shape)?.build();
        let pose = Pose::from_parts(position.into(), quaternion(rotation)?);
        let mut filter = query_filter(descriptor.layer, descriptor.mask);
        if !descriptor.include_sensors {
            filter = filter.exclude_sensors();
        }
        let pipeline = self.inner.query_pipeline_with_filter(filter);
        let controller = rapier3d::control::KinematicCharacterController {
            offset: rapier3d::control::CharacterLength::Absolute(descriptor.offset),
            slide: descriptor.slide,
            autostep: descriptor.autostep_max_height.map(|height| {
                rapier3d::control::CharacterAutostep {
                    max_height: rapier3d::control::CharacterLength::Absolute(height),
                    min_width: rapier3d::control::CharacterLength::Absolute(
                        descriptor.autostep_min_width,
                    ),
                    include_dynamic_bodies: descriptor.autostep_dynamic_bodies,
                }
            }),
            max_slope_climb_angle: descriptor.max_slope_climb_degrees.to_radians(),
            min_slope_slide_angle: descriptor.min_slope_slide_degrees.to_radians(),
            snap_to_ground: descriptor
                .snap_to_ground
                .map(rapier3d::control::CharacterLength::Absolute),
            ..Default::default()
        };
        let mut collisions = Vec::new();
        let movement = controller.move_shape(
            timestep,
            &pipeline,
            collider.shape(),
            &pose,
            desired_translation.into(),
            |collision| {
                if let Some(id) = self.collider_ids.get(&collision.handle) {
                    collisions.push(*id);
                }
            },
        );
        collisions.sort_unstable();
        collisions.dedup();
        Ok(CharacterMovement3d {
            translation: movement.translation.into(),
            grounded: movement.grounded,
            sliding_down_slope: movement.is_sliding_down_slope,
            collisions,
        })
    }

    fn body_mut(&mut self, id: u64) -> Result<&mut RigidBody, PhysicsError3d> {
        let handle = self
            .bodies
            .get(&id)
            .map(|entry| entry.0)
            .ok_or(PhysicsError3d::UnknownBody(id))?;
        self.inner
            .bodies
            .get_mut(handle)
            .ok_or(PhysicsError3d::UnknownBody(id))
    }
}

fn query_filter(layer: u32, mask: u32) -> QueryFilter<'static> {
    QueryFilter::default().groups(InteractionGroups::new(
        Group::from_bits_truncate(layer),
        Group::from_bits_truncate(mask),
        InteractionTestMode::And,
    ))
}

fn collider_builder(shape: &ColliderShape3d) -> Result<ColliderBuilder, PhysicsError3d> {
    Ok(match shape {
        ColliderShape3d::Box { half_extents } => {
            finite3(*half_extents, "box half extents")?;
            if half_extents.iter().any(|value| *value <= 0.0) {
                return Err(PhysicsError3d::Positive("box half extents"));
            }
            ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
        }
        ColliderShape3d::Sphere { radius } => {
            positive(*radius, "sphere radius")?;
            ColliderBuilder::ball(*radius)
        }
        ColliderShape3d::Capsule {
            half_height,
            radius,
        } => {
            positive(*half_height, "capsule half height")?;
            positive(*radius, "capsule radius")?;
            ColliderBuilder::capsule_y(*half_height, *radius)
        }
        ColliderShape3d::Cylinder {
            half_height,
            radius,
        } => {
            positive(*half_height, "cylinder half height")?;
            positive(*radius, "cylinder radius")?;
            ColliderBuilder::cylinder(*half_height, *radius)
        }
        ColliderShape3d::Plane { normal } => {
            finite3(*normal, "plane normal")?;
            let normal = Vector::from(*normal);
            if normal.length_squared() <= f32::EPSILON {
                return Err(PhysicsError3d::InvalidShape(
                    "plane normal cannot be zero".into(),
                ));
            }
            ColliderBuilder::halfspace(Unit::new_unchecked(normal.normalize()))
        }
        ColliderShape3d::ConvexMesh { points } => {
            validate_points(points, "convex mesh vertices", 4)?;
            let points: Vec<Vector> = points.iter().copied().map(Vector::from).collect();
            ColliderBuilder::convex_hull(&points).ok_or_else(|| {
                PhysicsError3d::InvalidShape(
                    "convex mesh points do not form a three-dimensional hull".into(),
                )
            })?
        }
        ColliderShape3d::TriangleMesh { vertices, indices } => {
            validate_points(vertices, "triangle mesh vertices", 3)?;
            if indices.is_empty() {
                return Err(PhysicsError3d::InvalidShape(
                    "triangle mesh needs at least one triangle".into(),
                ));
            }
            if indices
                .iter()
                .flatten()
                .any(|index| *index as usize >= vertices.len())
            {
                return Err(PhysicsError3d::InvalidShape(
                    "triangle mesh index is outside the vertex buffer".into(),
                ));
            }
            let vertices = vertices.iter().copied().map(Vector::from).collect();
            ColliderBuilder::trimesh(vertices, indices.clone()).map_err(|error| {
                PhysicsError3d::InvalidShape(format!("triangle mesh could not be built: {error}"))
            })?
        }
        ColliderShape3d::Heightfield {
            rows,
            columns,
            heights,
            scale,
        } => {
            if *rows < 2 || *columns < 2 {
                return Err(PhysicsError3d::InvalidShape(
                    "heightfield requires at least 2 rows and 2 columns".into(),
                ));
            }
            let expected = (*rows as usize)
                .checked_mul(*columns as usize)
                .ok_or_else(|| {
                    PhysicsError3d::InvalidShape("heightfield dimensions overflow".into())
                })?;
            if heights.len() != expected {
                return Err(PhysicsError3d::InvalidShape(format!(
                    "heightfield expected {expected} heights but received {}",
                    heights.len()
                )));
            }
            if !heights.iter().all(|value| value.is_finite()) {
                return Err(PhysicsError3d::NonFinite("heightfield heights"));
            }
            finite3(*scale, "heightfield scale")?;
            if scale.iter().any(|value| *value <= 0.0) {
                return Err(PhysicsError3d::Positive("heightfield scale"));
            }
            let column_major = (0..*columns as usize)
                .flat_map(|column| {
                    (0..*rows as usize).map(move |row| heights[row * *columns as usize + column])
                })
                .collect();
            let matrix = Array2::new(*rows as usize, *columns as usize, column_major);
            ColliderBuilder::heightfield(matrix, Vector::from(*scale))
        }
        ColliderShape3d::Compound { children } => {
            if children.is_empty() {
                return Err(PhysicsError3d::InvalidShape(
                    "compound collider needs at least one child".into(),
                ));
            }
            let mut shapes = Vec::with_capacity(children.len());
            for child in children {
                finite3(child.position, "compound child position")?;
                let rotation = quaternion(child.rotation)?;
                let pose = Pose::from_parts(Vector::from(child.position), rotation);
                shapes.push((pose, collider_builder(&child.shape)?.shape));
            }
            ColliderBuilder::compound(shapes)
        }
    })
}

/// Rapier mass properties for an explicit override.
///
/// When the caller gives no inertia, the shape's own inertia about its
/// centroid is scaled to the requested mass and reused about the new centre —
/// a "weighted crate" keeps a crate's inertia. It is an approximation, but the
/// exact answer needs to know where the extra mass sits, which is what
/// `principal_inertia` is for.
fn mass_properties(
    shape: &ColliderShape3d,
    value: MassProperties3d,
) -> Result<MassProperties, PhysicsError3d> {
    positive(value.mass, "mass")?;
    finite3(value.center_of_mass, "centre of mass")?;
    let center = Vector::from(value.center_of_mass);
    Ok(match value.principal_inertia {
        Some(inertia) => {
            finite3(inertia, "principal inertia")?;
            if inertia.iter().any(|value| *value < 0.0) {
                return Err(PhysicsError3d::NonNegative("principal inertia"));
            }
            MassProperties::new(center, value.mass, Vector::from(inertia))
        }
        None => {
            let shape_props = collider_builder(shape)?.build().mass_properties();
            let scale = value.mass / shape_props.mass().max(f32::EPSILON);
            MassProperties::with_principal_inertia_frame(
                center,
                value.mass,
                shape_props.principal_inertia() * scale,
                shape_props.principal_inertia_local_frame,
            )
        }
    })
}

fn validate_points(
    points: &[[f32; 3]],
    label: &'static str,
    minimum: usize,
) -> Result<(), PhysicsError3d> {
    if points.len() < minimum {
        return Err(PhysicsError3d::InvalidShape(format!(
            "{label} needs at least {minimum} points"
        )));
    }
    if points
        .iter()
        .all(|point| point.iter().all(|value| value.is_finite()))
    {
        Ok(())
    } else {
        Err(PhysicsError3d::NonFinite(label))
    }
}
fn validate_descriptor(value: &PhysicsBodyDescriptor3d) -> Result<(), PhysicsError3d> {
    if value.id == 0 {
        return Err(PhysicsError3d::ZeroId);
    }
    finite3(value.position, "physics body position")?;
    finite3(value.linear_velocity, "physics linear velocity")?;
    finite3(value.angular_velocity, "physics angular velocity")?;
    quaternion(value.rotation)?;
    non_negative(value.gravity_scale, "gravity scale")?;
    non_negative(value.linear_damping, "linear damping")?;
    non_negative(value.angular_damping, "angular damping")?;
    positive(value.density, "collider density")?;
    non_negative(value.friction, "collider friction")?;
    non_negative(value.restitution, "collider restitution")?;
    if value.layer == 0 {
        return Err(PhysicsError3d::EmptyLayer);
    }
    let _ = collider_builder(&value.shape)?;
    if let Some(mass) = value.mass_properties {
        let _ = mass_properties(&value.shape, mass)?;
    }
    Ok(())
}
fn quaternion(value: [f32; 4]) -> Result<Rotation, PhysicsError3d> {
    if !value.iter().all(|value| value.is_finite()) {
        return Err(PhysicsError3d::NonFinite("physics body rotation"));
    }
    let quaternion = Rotation::from_xyzw(value[0], value[1], value[2], value[3]);
    if quaternion.length_squared() <= f32::EPSILON {
        return Err(PhysicsError3d::ZeroQuaternion);
    }
    Ok(quaternion.normalize())
}
fn finite3(value: [f32; 3], label: &'static str) -> Result<(), PhysicsError3d> {
    if value.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PhysicsError3d::NonFinite(label))
    }
}
fn positive(value: f32, label: &'static str) -> Result<(), PhysicsError3d> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PhysicsError3d::Positive(label))
    }
}
fn non_negative(value: f32, label: &'static str) -> Result<(), PhysicsError3d> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PhysicsError3d::NonNegative(label))
    }
}
fn slope(value: f32, label: &'static str) -> Result<(), PhysicsError3d> {
    if value.is_finite() && (0.0..=90.0).contains(&value) {
        Ok(())
    } else {
        Err(PhysicsError3d::NonNegative(label))
    }
}
const fn event_kind_order(kind: PhysicsEventKind) -> u8 {
    match kind {
        PhysicsEventKind::CollisionEnter => 0,
        PhysicsEventKind::CollisionExit => 1,
        PhysicsEventKind::TriggerEnter => 2,
        PhysicsEventKind::TriggerExit => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body(
        id: u64,
        body_type: BodyType3d,
        y: f32,
        shape: ColliderShape3d,
    ) -> PhysicsBodyDescriptor3d {
        PhysicsBodyDescriptor3d {
            id,
            body_type,
            position: [0.0, y, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            continuous_collision_detection: true,
            shape,
            density: 1.0,
            friction: 0.7,
            restitution: 0.0,
            sensor: false,
            layer: 1,
            mask: u32::MAX,
            material: None,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
            mass_properties: None,
        }
    }

    /// A body at an arbitrary position, for joint tests.
    fn body_at(id: u64, body_type: BodyType3d, position: [f32; 3]) -> PhysicsBodyDescriptor3d {
        let mut descriptor = body(
            id,
            body_type,
            0.0,
            ColliderShape3d::Box {
                half_extents: [0.5; 3],
            },
        );
        descriptor.position = position;
        descriptor
    }

    fn position_of(world: &PhysicsWorld3d, id: u64) -> [f32; 3] {
        world
            .snapshots()
            .into_iter()
            .find(|snapshot| snapshot.id == id)
            .expect("body should exist")
            .position
    }

    #[test]
    fn fixed_joint_carries_a_hanging_body() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [0.0, 4.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Fixed {
                    anchor_a: [0.0, -1.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                },
                collide_connected: false,
            })
            .unwrap();
        assert_eq!(world.joint_count(), 1);

        for _ in 0..120 {
            world.step(1.0 / 60.0).unwrap();
        }
        // Without the joint the body would have fallen roughly 20 metres.
        let position = position_of(&world, 2);
        assert!(position[1] > 3.0, "welded body fell to {}", position[1]);
    }

    #[test]
    fn hinge_limits_confine_a_swinging_body() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [1.0, 5.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Hinge {
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [-1.0, 0.0, 0.0],
                    axis: [0.0, 1.0, 0.0],
                    limits: Some([-0.1, 0.1]),
                    motor: None,
                },
                collide_connected: false,
            })
            .unwrap();

        for _ in 0..120 {
            world.step(1.0 / 60.0).unwrap();
        }
        // The hinge axis is vertical, so gravity cannot swing the arm down and
        // the limits keep it from rotating away.
        let position = position_of(&world, 2);
        assert!(position[1] > 4.0, "hinged arm dropped to {}", position[1]);
        assert!(
            position[0] > 0.5,
            "hinged arm collapsed inward to {}",
            position[0]
        );
    }

    #[test]
    fn slider_constrains_motion_to_its_axis() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Slider {
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 0.0, 0.0],
                    axis: [1.0, 0.0, 0.0],
                    limits: Some([-2.0, 2.0]),
                    motor: None,
                },
                collide_connected: false,
            })
            .unwrap();

        world.apply_impulse(2, [50.0, -50.0, 50.0]).unwrap();
        for _ in 0..120 {
            world.step(1.0 / 60.0).unwrap();
        }
        let position = position_of(&world, 2);
        // Free along X within its limits, locked on the other two axes.
        assert!(
            position[0].abs() > 0.1,
            "slider did not move along its axis"
        );
        assert!(
            position[0].abs() <= 2.2,
            "slider escaped its limits at {}",
            position[0]
        );
        assert!(
            (position[1] - 5.0).abs() < 0.2,
            "slider drifted vertically to {}",
            position[1]
        );
        assert!(
            position[2].abs() < 0.2,
            "slider drifted sideways to {}",
            position[2]
        );
    }

    #[test]
    fn motor_drives_a_hinge() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [1.0, 5.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Hinge {
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [-1.0, 0.0, 0.0],
                    axis: [0.0, 1.0, 0.0],
                    limits: None,
                    motor: Some([4.0, 1000.0]),
                },
                collide_connected: false,
            })
            .unwrap();

        let start = position_of(&world, 2);
        for _ in 0..60 {
            world.step(1.0 / 60.0).unwrap();
        }
        let moved = position_of(&world, 2);
        // A driven hinge swings the arm around its axis.
        let travelled = (moved[0] - start[0]).hypot(moved[2] - start[2]);
        assert!(
            travelled > 0.3,
            "motor did not drive the hinge, travelled {travelled}"
        );

        world.set_joint_motor(1, 0.0, 1000.0).unwrap();
        assert!(world.set_joint_motor(99, 0.0, 1.0).is_err());
    }

    #[test]
    fn joints_are_validated() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [1.0, 5.0, 0.0]))
            .unwrap();
        let fixed = JointDescriptor3d::Fixed {
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
        };
        let spec = |id: u64, a: u64, b: u64, joint: JointDescriptor3d| JointSpec3d {
            id,
            body_a: a,
            body_b: b,
            joint,
            collide_connected: false,
        };

        assert!(matches!(
            world.add_joint(spec(0, 1, 2, fixed.clone())),
            Err(PhysicsError3d::ZeroId)
        ));
        assert!(matches!(
            world.add_joint(spec(1, 1, 1, fixed.clone())),
            Err(PhysicsError3d::SelfJoint(1))
        ));
        assert!(matches!(
            world.add_joint(spec(1, 1, 9, fixed.clone())),
            Err(PhysicsError3d::UnknownBody(9))
        ));

        world.add_joint(spec(1, 1, 2, fixed.clone())).unwrap();
        assert!(matches!(
            world.add_joint(spec(1, 1, 2, fixed)),
            Err(PhysicsError3d::DuplicateJoint(1))
        ));

        let zero_axis = JointDescriptor3d::Hinge {
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis: [0.0; 3],
            limits: None,
            motor: None,
        };
        assert!(matches!(
            world.add_joint(spec(2, 1, 2, zero_axis)),
            Err(PhysicsError3d::ZeroJointAxis)
        ));

        let bad_limits = JointDescriptor3d::Hinge {
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            limits: Some([1.0, -1.0]),
            motor: None,
        };
        assert!(matches!(
            world.add_joint(spec(3, 1, 2, bad_limits)),
            Err(PhysicsError3d::InvalidJointLimits)
        ));
    }

    #[test]
    fn removing_a_joint_frees_the_body() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [0.0, 4.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Fixed {
                    anchor_a: [0.0, -1.0, 0.0],
                    anchor_b: [0.0; 3],
                },
                collide_connected: false,
            })
            .unwrap();
        for _ in 0..30 {
            world.step(1.0 / 60.0).unwrap();
        }
        let held = position_of(&world, 2);

        world.remove_joint(1).unwrap();
        assert_eq!(world.joint_count(), 0);
        assert!(matches!(
            world.remove_joint(1),
            Err(PhysicsError3d::UnknownJoint(1))
        ));

        for _ in 0..60 {
            world.step(1.0 / 60.0).unwrap();
        }
        // Released, it falls — and it must wake to do so.
        assert!(position_of(&world, 2)[1] < held[1] - 1.0);
    }

    #[test]
    fn removing_a_body_drops_its_joints() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [0.0, 4.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Fixed {
                    anchor_a: [0.0; 3],
                    anchor_b: [0.0; 3],
                },
                collide_connected: false,
            })
            .unwrap();
        world.remove(2).unwrap();
        // A joint referencing a removed body must not survive as a dangling id.
        assert_eq!(world.joint_count(), 0);
        world.step(1.0 / 60.0).unwrap();
    }

    #[test]
    fn ball_joint_swings_but_stays_attached() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body_at(1, BodyType3d::Static, [0.0, 5.0, 0.0]))
            .unwrap();
        world
            .insert(body_at(2, BodyType3d::Dynamic, [0.0, 3.0, 0.0]))
            .unwrap();
        world
            .add_joint(JointSpec3d {
                id: 1,
                body_a: 1,
                body_b: 2,
                joint: JointDescriptor3d::Ball {
                    anchor_a: [0.0, 0.0, 0.0],
                    anchor_b: [0.0, 2.0, 0.0],
                },
                collide_connected: false,
            })
            .unwrap();
        world.apply_impulse(2, [30.0, 0.0, 0.0]).unwrap();
        for _ in 0..180 {
            world.step(1.0 / 60.0).unwrap();
        }
        let position = position_of(&world, 2);
        let distance =
            (position[0].powi(2) + (position[1] - 5.0).powi(2) + position[2].powi(2)).sqrt();
        // The anchor separation is held even while the body swings freely.
        assert!(
            (distance - 2.0).abs() < 0.5,
            "ball joint separation drifted to {distance}"
        );
    }

    #[test]
    fn dynamic_body_falls_and_collides_with_ground() {
        let mut world = PhysicsWorld3d::default();
        world
            .insert(body(
                1,
                BodyType3d::Static,
                -0.5,
                ColliderShape3d::Box {
                    half_extents: [5.0, 0.5, 5.0],
                },
            ))
            .unwrap();
        world
            .insert(body(
                2,
                BodyType3d::Dynamic,
                2.0,
                ColliderShape3d::Sphere { radius: 0.5 },
            ))
            .unwrap();
        let mut entered = false;
        for _ in 0..180 {
            entered |= world
                .step(1.0 / 60.0)
                .unwrap()
                .iter()
                .any(|event| event.kind == PhysicsEventKind::CollisionEnter);
        }
        let ball = world
            .snapshots()
            .into_iter()
            .find(|body| body.id == 2)
            .unwrap();
        assert!(entered);
        assert!(
            (ball.position[1] - 0.5).abs() < 0.05,
            "ball at {}",
            ball.position[1]
        );
    }
    #[test]
    fn sensors_emit_trigger_events_without_blocking() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        let mut trigger = body(
            1,
            BodyType3d::Static,
            0.0,
            ColliderShape3d::Sphere { radius: 1.0 },
        );
        trigger.sensor = true;
        world.insert(trigger).unwrap();
        let mut mover = body(
            2,
            BodyType3d::Dynamic,
            0.0,
            ColliderShape3d::Sphere { radius: 0.25 },
        );
        mover.position = [-2.0, 0.0, 0.0];
        mover.linear_velocity = [2.0, 0.0, 0.0];
        world.insert(mover).unwrap();
        let events: Vec<_> = (0..120)
            .flat_map(|_| world.step(1.0 / 60.0).unwrap())
            .collect();
        assert!(
            events
                .iter()
                .any(|event| event.kind == PhysicsEventKind::TriggerEnter)
        );
        assert!(
            events
                .iter()
                .any(|event| event.kind == PhysicsEventKind::TriggerExit)
        );
    }
    #[test]
    fn raycasts_return_stable_ids_points_and_normals() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                7,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Sphere { radius: 1.0 },
            ))
            .unwrap();
        world.step(1.0 / 60.0).unwrap();
        let hit = world
            .raycast(
                [0.0, 0.0, 4.0],
                [0.0, 0.0, -2.0],
                10.0,
                1,
                u32::MAX,
                false,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.body_id, 7);
        assert!((hit.distance - 3.0).abs() < 0.001);
        assert!((hit.normal[2] - 1.0).abs() < 0.001);
    }
    #[test]
    fn compound_children_keep_their_local_transforms() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                8,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Compound {
                    children: vec![
                        CompoundColliderChild3d {
                            position: [-2.0, 0.0, 0.0],
                            rotation: [0.0, 0.0, 0.0, 1.0],
                            shape: ColliderShape3d::Sphere { radius: 0.5 },
                        },
                        CompoundColliderChild3d {
                            position: [2.0, 0.0, 0.0],
                            rotation: [0.0, 0.0, 0.0, 1.0],
                            shape: ColliderShape3d::Box {
                                half_extents: [0.5; 3],
                            },
                        },
                    ],
                },
            ))
            .unwrap();
        world.step(1.0 / 60.0).unwrap();
        for x in [-2.0, 2.0] {
            let hit = world
                .raycast(
                    [x, 0.0, 3.0],
                    [0.0, 0.0, -1.0],
                    10.0,
                    1,
                    u32::MAX,
                    false,
                    None,
                )
                .unwrap();
            assert_eq!(hit.map(|value| value.body_id), Some(8));
        }
    }

    #[test]
    fn compound_colliders_reject_empty_and_invalid_children() {
        assert!(collider_builder(&ColliderShape3d::Compound { children: vec![] }).is_err());
        assert!(
            collider_builder(&ColliderShape3d::Compound {
                children: vec![CompoundColliderChild3d {
                    position: [f32::NAN, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    shape: ColliderShape3d::Sphere { radius: 1.0 },
                }],
            })
            .is_err()
        );
    }
    #[test]
    fn static_world_shapes_build_and_participate_in_queries() {
        let mut plane_world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        plane_world
            .insert(body(
                11,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Plane {
                    normal: [0.0, 1.0, 0.0],
                },
            ))
            .unwrap();
        plane_world.step(1.0 / 60.0).unwrap();
        assert_eq!(
            plane_world
                .raycast(
                    [0.0, 2.0, 0.0],
                    [0.0, -1.0, 0.0],
                    10.0,
                    1,
                    u32::MAX,
                    false,
                    None
                )
                .unwrap()
                .unwrap()
                .body_id,
            11
        );

        let mut terrain_world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        terrain_world
            .insert(body(
                12,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Heightfield {
                    rows: 2,
                    columns: 2,
                    heights: vec![0.0; 4],
                    scale: [10.0, 1.0, 10.0],
                },
            ))
            .unwrap();
        terrain_world
            .insert(body(
                13,
                BodyType3d::Static,
                5.0,
                ColliderShape3d::TriangleMesh {
                    vertices: vec![[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
                    indices: vec![[0, 1, 2]],
                },
            ))
            .unwrap();
        terrain_world
            .insert(body(
                14,
                BodyType3d::Static,
                10.0,
                ColliderShape3d::ConvexMesh {
                    points: vec![
                        [0.0, 1.0, 0.0],
                        [-1.0, -1.0, -1.0],
                        [1.0, -1.0, -1.0],
                        [0.0, -1.0, 1.0],
                    ],
                },
            ))
            .unwrap();
        assert_eq!(terrain_world.body_count(), 3);
    }
    #[test]
    fn malformed_world_shapes_return_actionable_errors() {
        let mut world = PhysicsWorld3d::default();
        let error = world
            .insert(body(
                1,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Heightfield {
                    rows: 3,
                    columns: 3,
                    heights: vec![0.0; 4],
                    scale: [1.0; 3],
                },
            ))
            .unwrap_err();
        assert!(error.to_string().contains("expected 9 heights"));
        let error = world
            .insert(body(
                2,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::TriangleMesh {
                    vertices: vec![[0.0; 3]; 3],
                    indices: vec![[0, 1, 9]],
                },
            ))
            .unwrap_err();
        assert!(error.to_string().contains("outside the vertex buffer"));
    }
    #[test]
    fn overlap_and_shape_cast_return_filtered_stable_ids() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                7,
                BodyType3d::Static,
                0.0,
                ColliderShape3d::Sphere { radius: 1.0 },
            ))
            .unwrap();
        world.step(1.0 / 60.0).unwrap();
        let overlaps = world
            .overlap_shape(
                [0.75, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                &ColliderShape3d::Sphere { radius: 0.5 },
                1,
                u32::MAX,
                false,
            )
            .unwrap();
        assert_eq!(overlaps, vec![7]);
        let hit = world
            .cast_shape(
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                10.0,
                &ColliderShape3d::Sphere { radius: 0.5 },
                1,
                u32::MAX,
                false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.body_id, 7);
        assert!(
            (hit.distance - 2.5).abs() < 0.01,
            "distance {}",
            hit.distance
        );
        assert!(hit.normal[2] > 0.99, "normal {:?}", hit.normal);
    }
    #[test]
    fn character_controller_stops_on_ground_and_reports_collision() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                1,
                BodyType3d::Static,
                -0.5,
                ColliderShape3d::Box {
                    half_extents: [5.0, 0.5, 5.0],
                },
            ))
            .unwrap();
        world.step(1.0 / 60.0).unwrap();
        let movement = world
            .move_character(
                1.0 / 60.0,
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -3.0, 0.0],
                &CharacterControllerDescriptor3d {
                    shape: ColliderShape3d::Capsule {
                        half_height: 0.5,
                        radius: 0.5,
                    },
                    offset: 0.01,
                    slide: true,
                    autostep_max_height: Some(0.3),
                    autostep_min_width: 0.2,
                    autostep_dynamic_bodies: false,
                    max_slope_climb_degrees: 45.0,
                    min_slope_slide_degrees: 50.0,
                    snap_to_ground: Some(0.2),
                    layer: 1,
                    mask: u32::MAX,
                    include_sensors: false,
                },
            )
            .unwrap();
        assert!(
            movement.translation[1] > -1.1 && movement.translation[1] < -0.9,
            "translation {:?}",
            movement.translation
        );
        assert!(movement.grounded);
        assert_eq!(movement.collisions, vec![1]);
    }
    #[test]
    fn validates_descriptors_and_stale_ids() {
        let mut world = PhysicsWorld3d::default();
        let mut invalid = body(
            0,
            BodyType3d::Dynamic,
            0.0,
            ColliderShape3d::Sphere { radius: 1.0 },
        );
        assert_eq!(world.insert(invalid.clone()), Err(PhysicsError3d::ZeroId));
        invalid.id = 1;
        invalid.layer = 0;
        assert_eq!(world.insert(invalid), Err(PhysicsError3d::EmptyLayer));
        assert_eq!(
            world.apply_impulse(99, [1.0, 0.0, 0.0]),
            Err(PhysicsError3d::UnknownBody(99))
        );
    }
    #[test]
    fn force_at_point_and_torque_drive_angular_velocity() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                4,
                BodyType3d::Dynamic,
                0.0,
                ColliderShape3d::Box {
                    half_extents: [0.5; 3],
                },
            ))
            .unwrap();
        world
            .add_force_at_point(4, [10.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            .unwrap();
        world.add_torque(4, [0.0, 0.0, 2.0]).unwrap();
        world.apply_torque_impulse(4, [0.0, 1.0, 0.0]).unwrap();
        world.step(1.0 / 60.0).unwrap();
        let angular = world.snapshots()[0].angular_velocity;
        assert!(
            angular.iter().any(|value| value.abs() > 0.01),
            "angular velocity {angular:?}"
        );
        world.set_angular_velocity(4, [1.0, 2.0, 3.0]).unwrap();
        assert_eq!(world.snapshots()[0].angular_velocity, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn submitted_forces_apply_to_one_step() {
        let mut world = PhysicsWorld3d::new([0.0, 0.0, 0.0]);
        world
            .insert(body(
                4,
                BodyType3d::Dynamic,
                0.0,
                ColliderShape3d::Sphere { radius: 0.5 },
            ))
            .unwrap();
        world.add_force(4, [10.0, 0.0, 0.0]).unwrap();
        world.step(1.0 / 60.0).unwrap();
        let after_force = world.snapshots()[0].linear_velocity[0];
        world.step(1.0 / 60.0).unwrap();
        let after_coast = world.snapshots()[0].linear_velocity[0];
        assert!(after_force > 0.0);
        assert!(
            after_coast <= after_force,
            "a force command persisted into the following step: {after_force} -> {after_coast}"
        );
    }

    fn ground() -> PhysicsBodyDescriptor3d {
        body(
            1,
            BodyType3d::Static,
            -0.5,
            ColliderShape3d::Box {
                half_extents: [10.0, 0.5, 10.0],
            },
        )
    }

    fn material(name: &str, friction: f32, rule: CombineRule) -> PhysicsMaterial {
        PhysicsMaterial {
            name: name.into(),
            friction,
            restitution: 0.0,
            friction_combine: rule,
            restitution_combine: rule,
        }
    }

    /// The effective friction Rapier solved a resting box against the ground with.
    fn solved_friction(ground_rule: CombineRule, box_rule: CombineRule) -> f32 {
        let mut world = PhysicsWorld3d::default();
        world
            .define_material(material("floor", 0.2, ground_rule))
            .unwrap();
        world
            .define_material(material("crate", 0.8, box_rule))
            .unwrap();
        let mut floor = ground();
        floor.material = Some("floor".into());
        world.insert(floor).unwrap();
        let mut crate_ = body(
            2,
            BodyType3d::Dynamic,
            0.5,
            ColliderShape3d::Box {
                half_extents: [0.5; 3],
            },
        );
        crate_.material = Some("crate".into());
        world.insert(crate_).unwrap();
        for _ in 0..5 {
            world.step(1.0 / 60.0).unwrap();
        }
        let pair = world
            .inner
            .narrow_phase
            .contact_pairs()
            .next()
            .expect("box should rest on the floor");
        pair.manifolds[0].data.solver_contacts[0].friction
    }

    #[test]
    fn combine_rules_produce_the_expected_coefficient() {
        // Rapier resolves conflicting rules by taking the "stronger" one in the
        // order Max > Multiply > Min > Average, so matched rules are the clear
        // cases and Average versus Min shows Min winning.
        let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(near(
            solved_friction(CombineRule::Average, CombineRule::Average),
            0.5
        ));
        assert!(near(
            solved_friction(CombineRule::Min, CombineRule::Min),
            0.2
        ));
        assert!(near(
            solved_friction(CombineRule::Max, CombineRule::Max),
            0.8
        ));
        assert!(near(
            solved_friction(CombineRule::Multiply, CombineRule::Multiply),
            0.16
        ));
        assert!(near(
            solved_friction(CombineRule::Average, CombineRule::Min),
            0.2
        ));
    }

    #[test]
    fn restitution_material_makes_a_ball_bounce() {
        let bounce = |restitution: f32| {
            let mut world = PhysicsWorld3d::default();
            world
                .define_material(PhysicsMaterial {
                    name: "rubber".into(),
                    friction: 0.5,
                    restitution,
                    friction_combine: CombineRule::Average,
                    restitution_combine: CombineRule::Max,
                })
                .unwrap();
            world.insert(ground()).unwrap();
            let mut ball = body(
                2,
                BodyType3d::Dynamic,
                3.0,
                ColliderShape3d::Sphere { radius: 0.5 },
            );
            ball.material = Some("rubber".into());
            world.insert(ball).unwrap();
            let mut peak: f32 = 0.0;
            let mut landed = false;
            for _ in 0..240 {
                world.step(1.0 / 60.0).unwrap();
                let y = position_of(&world, 2)[1];
                if y < 0.6 {
                    landed = true;
                }
                if landed {
                    peak = peak.max(y);
                }
            }
            peak
        };
        let dead = bounce(0.1);
        let lively = bounce(0.9);
        assert!(dead < 0.8, "restitution 0.1 rebounded to {dead}");
        assert!(lively > 1.5, "restitution 0.9 rebounded to {lively}");
    }

    #[test]
    fn set_material_changes_an_existing_collider() {
        let mut world = PhysicsWorld3d::default();
        world
            .define_material(material("ice", 0.02, CombineRule::Min))
            .unwrap();
        world.insert(ground()).unwrap();
        assert_eq!(
            world.set_material(1, "mud"),
            Err(PhysicsError3d::UnknownMaterial("mud".into()))
        );
        assert_eq!(
            world.set_material(9, "ice"),
            Err(PhysicsError3d::UnknownBody(9))
        );
        world.set_material(1, "ice").unwrap();
        let collider = world.inner.colliders.get(world.bodies[&1].1).unwrap();
        assert!((collider.friction() - 0.02).abs() < 1e-6);
        assert_eq!(
            collider.friction_combine_rule(),
            CoefficientCombineRule::Min
        );
        assert_eq!(world.material_count(), 1);
        assert_eq!(world.material("ice").map(|m| m.friction), Some(0.02));
    }

    #[test]
    fn explicit_mass_properties_set_the_inertia_frame() {
        let mut world = PhysicsWorld3d::new([0.0; 3]);
        let mut crate_ = body(
            1,
            BodyType3d::Dynamic,
            0.0,
            ColliderShape3d::Box {
                half_extents: [0.5; 3],
            },
        );
        crate_.mass_properties = Some(MassProperties3d {
            mass: 4.0,
            center_of_mass: [0.25, -0.4, 0.0],
            principal_inertia: Some([1.0, 2.0, 3.0]),
        });
        world.insert(crate_).unwrap();
        let props = world.mass_properties_of(1).unwrap();
        assert!((props.mass - 4.0).abs() < 1e-5, "mass {}", props.mass);
        assert_eq!(props.center_of_mass, [0.25, -0.4, 0.0]);
        let inertia = props.principal_inertia.unwrap();
        for (actual, expected) in inertia.iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-4, "inertia {inertia:?}");
        }

        // Omitting the inertia scales the shape's own tensor to the mass: a
        // unit cube of mass 4 has I = m/6 = 0.6667 about every axis.
        let mut derived = body(
            2,
            BodyType3d::Dynamic,
            0.0,
            ColliderShape3d::Box {
                half_extents: [0.5; 3],
            },
        );
        derived.mass_properties = Some(MassProperties3d {
            mass: 4.0,
            center_of_mass: [0.0, 0.0, 0.0],
            principal_inertia: None,
        });
        world.insert(derived).unwrap();
        let derived = world
            .mass_properties_of(2)
            .unwrap()
            .principal_inertia
            .unwrap();
        for value in derived {
            assert!(
                (value - 4.0 / 6.0).abs() < 1e-3,
                "derived inertia {derived:?}"
            );
        }

        let mut bad = body(
            3,
            BodyType3d::Dynamic,
            0.0,
            ColliderShape3d::Sphere { radius: 1.0 },
        );
        bad.mass_properties = Some(MassProperties3d {
            mass: 0.0,
            center_of_mass: [0.0; 3],
            principal_inertia: None,
        });
        assert_eq!(world.insert(bad), Err(PhysicsError3d::Positive("mass")));
    }

    #[test]
    fn offset_centre_of_mass_tips_a_box_toward_the_weight() {
        let mut world = PhysicsWorld3d::default();
        world.insert(ground()).unwrap();
        let mut crate_ = body(
            2,
            BodyType3d::Dynamic,
            0.5,
            ColliderShape3d::Box {
                half_extents: [0.5; 3],
            },
        );
        // Weight hanging past the +X edge of the footprint: gravity's torque
        // about that edge rolls the box over toward +X, i.e. negative about Z.
        crate_.mass_properties = Some(MassProperties3d {
            mass: 2.0,
            center_of_mass: [0.9, 0.3, 0.0],
            principal_inertia: None,
        });
        world.insert(crate_).unwrap();
        for _ in 0..60 {
            world.step(1.0 / 60.0).unwrap();
        }
        let snapshot = world.snapshots().into_iter().find(|s| s.id == 2).unwrap();
        assert!(
            snapshot.angular_velocity[2] < -0.1 || snapshot.rotation[2] < -0.05,
            "box did not tip toward +X: {snapshot:?}"
        );
        assert!(
            snapshot.position[0] > 0.05,
            "box did not roll toward +X: {snapshot:?}"
        );
    }

    #[test]
    fn stats_and_contacts_describe_the_resting_scene() {
        let mut world = PhysicsWorld3d::default();
        world.insert(ground()).unwrap();
        world
            .insert(body(
                2,
                BodyType3d::Dynamic,
                0.5,
                ColliderShape3d::Sphere { radius: 0.5 },
            ))
            .unwrap();
        world
            .insert(body(
                3,
                BodyType3d::KinematicPosition,
                5.0,
                ColliderShape3d::Sphere { radius: 0.5 },
            ))
            .unwrap();
        assert_eq!(world.stats().contact_points, 0);
        for _ in 0..10 {
            world.step(1.0 / 60.0).unwrap();
        }
        let stats = world.stats();
        assert_eq!(
            (stats.bodies, stats.dynamic, stats.kinematic, stats.fixed),
            (3, 1, 1, 1)
        );
        assert_eq!(stats.awake + stats.sleeping, 3);
        assert_eq!(stats.contact_pairs, 1);
        assert_eq!(stats.manifolds, 1);
        assert!(stats.contact_points >= 1);
        let contacts = world.contacts();
        assert_eq!(contacts.len(), stats.contact_points);
        let contact = contacts[0];
        assert_eq!((contact.body_a, contact.body_b), (1, 2));
        assert!(contact.point[1].abs() < 0.05, "contact {contact:?}");
        assert!(contact.normal[1].abs() > 0.99, "contact {contact:?}");
        assert!(contact.distance < 0.01, "contact {contact:?}");
    }
}

/// Translates a descriptor into a Rapier joint.
fn build_joint(
    descriptor: &JointDescriptor3d,
    collide_connected: bool,
) -> Result<GenericJoint, PhysicsError3d> {
    let unit_axis = |axis: [f32; 3]| -> Result<Vector, PhysicsError3d> {
        finite3(axis, "joint axis")?;
        let vector = Vector::new(axis[0], axis[1], axis[2]);
        if vector.length() < 1.0e-6 {
            return Err(PhysicsError3d::ZeroJointAxis);
        }
        Ok(vector.normalize())
    };
    let limits = |values: [f32; 2]| -> Result<[f32; 2], PhysicsError3d> {
        if !values[0].is_finite() || !values[1].is_finite() {
            return Err(PhysicsError3d::NonFinite("joint limits"));
        }
        if values[0] > values[1] {
            return Err(PhysicsError3d::InvalidJointLimits);
        }
        Ok(values)
    };

    let joint = match descriptor {
        JointDescriptor3d::Fixed { anchor_a, anchor_b } => {
            finite3(*anchor_a, "joint anchor")?;
            finite3(*anchor_b, "joint anchor")?;
            FixedJointBuilder::new()
                .local_anchor1(Vector::from(*anchor_a))
                .local_anchor2(Vector::from(*anchor_b))
                .build()
                .into()
        }
        JointDescriptor3d::Hinge {
            anchor_a,
            anchor_b,
            axis,
            limits: bounds,
            motor,
        } => {
            finite3(*anchor_a, "joint anchor")?;
            finite3(*anchor_b, "joint anchor")?;
            let mut builder = RevoluteJointBuilder::new(unit_axis(*axis)?)
                .local_anchor1(Vector::from(*anchor_a))
                .local_anchor2(Vector::from(*anchor_b));
            if let Some(values) = bounds {
                let checked = limits(*values)?;
                builder = builder.limits(checked);
            }
            if let Some(values) = motor {
                if !values[0].is_finite() {
                    return Err(PhysicsError3d::NonFinite("joint motor velocity"));
                }
                if !values[1].is_finite() || values[1] < 0.0 {
                    return Err(PhysicsError3d::NonNegative("joint motor force"));
                }
                builder = builder.motor_velocity(values[0], values[1]);
            }
            builder.build().into()
        }
        JointDescriptor3d::Slider {
            anchor_a,
            anchor_b,
            axis,
            limits: bounds,
            motor,
        } => {
            finite3(*anchor_a, "joint anchor")?;
            finite3(*anchor_b, "joint anchor")?;
            let mut builder = PrismaticJointBuilder::new(unit_axis(*axis)?)
                .local_anchor1(Vector::from(*anchor_a))
                .local_anchor2(Vector::from(*anchor_b));
            if let Some(values) = bounds {
                let checked = limits(*values)?;
                builder = builder.limits(checked);
            }
            if let Some(values) = motor {
                if !values[0].is_finite() {
                    return Err(PhysicsError3d::NonFinite("joint motor velocity"));
                }
                if !values[1].is_finite() || values[1] < 0.0 {
                    return Err(PhysicsError3d::NonNegative("joint motor force"));
                }
                builder = builder.motor_velocity(values[0], values[1]);
            }
            builder.build().into()
        }
        JointDescriptor3d::Ball { anchor_a, anchor_b } => {
            finite3(*anchor_a, "joint anchor")?;
            finite3(*anchor_b, "joint anchor")?;
            SphericalJointBuilder::new()
                .local_anchor1(Vector::from(*anchor_a))
                .local_anchor2(Vector::from(*anchor_b))
                .build()
                .into()
        }
        JointDescriptor3d::Spring {
            anchor_a,
            anchor_b,
            rest_length,
            stiffness,
            damping,
        } => {
            finite3(*anchor_a, "joint anchor")?;
            finite3(*anchor_b, "joint anchor")?;
            if !rest_length.is_finite() || *rest_length < 0.0 {
                return Err(PhysicsError3d::NonNegative("joint rest length"));
            }
            if !stiffness.is_finite() || *stiffness < 0.0 {
                return Err(PhysicsError3d::NonNegative("joint stiffness"));
            }
            if !damping.is_finite() || *damping < 0.0 {
                return Err(PhysicsError3d::NonNegative("joint damping"));
            }
            RopeJointBuilder::new(*rest_length)
                .local_anchor1(Vector::from(*anchor_a))
                .local_anchor2(Vector::from(*anchor_b))
                .build()
                .into()
        }
    };

    let mut generic: GenericJoint = joint;
    generic.set_contacts_enabled(collide_connected);
    Ok(generic)
}
