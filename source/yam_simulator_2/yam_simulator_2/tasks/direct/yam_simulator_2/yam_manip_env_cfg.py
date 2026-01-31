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

    block_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        spawn=CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, TABLE_HEIGHT + INCH / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
