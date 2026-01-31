# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass

from ....assets import YAM_CFG

INCH = 0.0254
TABLE_HEIGHT = 0.75
TABLE_SIZE = (1.0, 1.0, 0.05)
BLOCK_SIZE = (INCH, INCH, INCH)
BLOCK_POSITIONS = (
    (0.30, 0.00, TABLE_HEIGHT + INCH / 2.0),
    (0.35, 0.05, TABLE_HEIGHT + INCH / 2.0),
    (0.35, -0.05, TABLE_HEIGHT + INCH / 2.0),
)
WALL_THICKNESS = 0.02
WALL_WIDTH = 1.0
WALL_HEIGHT = 1.0
WALL_SIZE = (WALL_THICKNESS, WALL_WIDTH, WALL_HEIGHT)
WALL_X = (TABLE_SIZE[0] / 2.0) + (WALL_THICKNESS / 2.0) + 0.05
WALL_POS = (WALL_X, 0.0, TABLE_HEIGHT + WALL_HEIGHT / 2.0)


@configclass
class YamManipEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 1
    observation_space = 3
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)

    # robot(s)
    robot_cfg: ArticulationCfg = YAM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, TABLE_HEIGHT)

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT - TABLE_SIZE[2] / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Wall",
        spawn=CuboidCfg(
            size=WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=WALL_POS,
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
