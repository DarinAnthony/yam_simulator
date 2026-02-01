# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.shapes import CuboidCfg, CylinderCfg, SphereCfg
from isaaclab.utils import configclass

from ....assets import YAM_CFG

INCH = 0.0254
TABLE_HEIGHT = 0.75
# Table: 23.9 in × 59.8 in → 0.60706 m × 1.51892 m (swapped X/Y)
TABLE_SIZE = (0.60706, 1.51892, 0.05)
# Robot base placement: 35.5 in (0.90170 m) from the left edge when facing -Y.
# Left edge (facing -Y) lies at Y = +TABLE_SIZE[1]/2 = +0.75946 m.
# Base Y = 0.75946 - 0.90170 = -0.14224 m
ROBOT_BASE_X = -0.26543
ROBOT_BASE_Y = 0.14224
BLOCK_SIZE = (INCH, INCH, INCH)
BLOCK_POSITIONS = (
    (-0.05, -0.22352, TABLE_HEIGHT + INCH / 2.0),
    (0.00, -0.22352, TABLE_HEIGHT + INCH / 2.0),
    (0.05, -0.22352, TABLE_HEIGHT + INCH / 2.0),
)
WALL_THICKNESS = 0.02
WALL_WIDTH = 1.0
WALL_HEIGHT = 1.0
WALL_SIZE = (WALL_THICKNESS, WALL_WIDTH, WALL_HEIGHT)
WALL_X = (TABLE_SIZE[0] / 2.0) + (WALL_THICKNESS / 2.0) + 0.05
WALL_POS = (WALL_X, 0.0, TABLE_HEIGHT + WALL_HEIGHT / 2.0)
START_AREA_RADIUS = 0.08
START_AREA_HEIGHT = 0.01
START_CENTER = (0.0, 0.4352, TABLE_HEIGHT + START_AREA_HEIGHT / 2.0)
BLOCK_CLEARANCE = 0.001
MIN_BLOCK_SEPARATION = INCH * 1.25
SITE_MARKER_RADIUS = 0.01
DROP_OFF_RADIUS = 0.16
DROP_OFF_HEIGHT = 0.02
DROP_Y = -0.06096  # Y = -0.75946 + 0.69850 (left edge at +0.75946, facing -Y)
DROP_OFF_POSITIONS = (
    (-0.20, DROP_Y, TABLE_HEIGHT + DROP_OFF_HEIGHT / 2.0),  # Red
    (0.00, DROP_Y, TABLE_HEIGHT + DROP_OFF_HEIGHT / 2.0),   # Blue
    (0.20, DROP_Y, TABLE_HEIGHT + DROP_OFF_HEIGHT / 2.0),   # Yellow
)
DROP_OFF_RADIUS_VIS = 0.05


@configclass
class YamManipEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    action_space = 6
    observation_space = 40
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    use_start_area_radius: bool = True
    block_clearance: float = BLOCK_CLEARANCE
    min_block_separation: float = MIN_BLOCK_SEPARATION

    # Actions: [dx, dy, dz, gripper]
    ee_delta_scale: float = 0.005
    ee_pos_limit: float = 0.005
    tilt_scale: float = 0.15
    yaw_rate_limit: float = 0.25
    yaw_update_eps: float = 1e-3
    gripper_open: float = -0.0425
    gripper_closed: float = -0.005
    home_joint_pos: list[float] = [
        -0.003242542153047978,
        0.5556191348134583,
        0.6071183337148085,
        -0.09899290455481768,
        0.0005722133211261138,
        0.02193484397650103,
    ]
    gripper_site_offset: tuple[float, float, float] = (0.0, 0.0, -0.08)

    # Reward/threshold parameters
    reach_k: float = 10.0
    goal_k: float = 10.0
    reach_w: float = 1.0
    grasp_w: float = 0.5
    lift_w: float = 2.0
    carry_w: float = 2.0
    place_bonus: float = 10.0
    step_penalty_w: float = 0.01
    smooth_w: float = 0.02
    open_until_contact_w: float = 0.2
    # Waypoint and smoothness shaping
    hover_height: float = 0.08
    pick_z_offset: float = 0.01
    place_z_offset: float = 0.01
    wp_gate_xy: float = 0.06
    wp_gate_k: float = 40.0
    wp_k: float = 10.0
    wp_pick_w: float = 0.3
    wp_place_w: float = 0.3
    wp_progress_w: float = 0.5
    tilt_smooth_w: float = 0.005
    ee_vel_w: float = 0.004
    ee_acc_w: float = 0.0004
    clearance_w: float = 0.2
    clearance_z: float = 0.04
    table_contact_w: float = 0.3
    table_clearance: float = 0.03
    # Descend and close shaping
    descend_w: float = 0.6
    descend_k: float = 20.0
    close_w: float = 0.8
    close_z_gate: float = 0.02

    near_thresh: float = 0.03
    grip_closed_thresh: float = 0.8
    grip_open_thresh: float = 0.2
    lift_height: float = 0.05
    goal_xy_thresh: float = 0.05
    place_z_thresh: float = 0.02
    fall_termination_margin: float = 0.02

    robot_entity: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"],
        body_names=["gripper"],
    )
    marker_body_names: list[str] = ["tip_left", "tip_right"]
    debug_print_joint_pos: bool = True
    debug_nan_checks: bool = True
    debug_nan_max_print: int = 10
    debug_abs_max: float = 1.0e3
    debug_print_obs: bool = False
    use_camera: bool = False
    camera_offset_pos: tuple[float, float, float] = (0.0, 0.02540, 0.11176)
    camera_pitch_deg: float = -60.0

    diff_ik_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )

    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        update_period=0.0,
        data_types=["rgb"],
        width=640,
        height=480,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = YAM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (ROBOT_BASE_X, ROBOT_BASE_Y, TABLE_HEIGHT)

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.22),
                roughness=0.75,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT - TABLE_SIZE[2] / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    wall_cfg: RigidObjectCfg | None = None

    dropoff_red_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DropoffRed",
        spawn=CylinderCfg(
            radius=DROP_OFF_RADIUS_VIS,
            height=DROP_OFF_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.1, 0.1),
                opacity=0.15,
                roughness=0.6,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DROP_OFF_POSITIONS[0],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    dropoff_blue_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DropoffBlue",
        spawn=CylinderCfg(
            radius=DROP_OFF_RADIUS_VIS,
            height=DROP_OFF_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.2, 1.0),
                opacity=0.15,
                roughness=0.6,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DROP_OFF_POSITIONS[1],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    dropoff_yellow_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DropoffYellow",
        spawn=CylinderCfg(
            radius=DROP_OFF_RADIUS_VIS,
            height=DROP_OFF_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.9, 0.1),
                opacity=0.15,
                roughness=0.6,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DROP_OFF_POSITIONS[2],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    start_area_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/StartArea",
        spawn=CylinderCfg(
            radius=START_AREA_RADIUS,
            height=START_AREA_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 1.0, 0.2),
                opacity=0.1,
                roughness=0.6,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=START_CENTER,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    site_marker_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GripperSiteMarker",
        spawn=SphereCfg(
            radius=SITE_MARKER_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 1.0),
                opacity=0.25,
                roughness=0.6,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=START_CENTER,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    red_block_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedBlock",
        spawn=CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BLOCK_POSITIONS[0],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    blue_block_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueBlock",
        spawn=CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BLOCK_POSITIONS[1],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    yellow_block_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/YellowBlock",
        spawn=CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BLOCK_POSITIONS[2],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
