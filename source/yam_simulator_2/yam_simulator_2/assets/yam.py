# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the YAM robot with linear gripper."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(__file__).resolve().parent
YAM_USD_PATH = (
    ASSET_DIR
    / "yam_new"
    / "yam_st_urdf_with_linear_gripper"
    / "yam_st_urdf_with_linear_gripper.usd"
)

YAM_CFG = ArticulationCfg(
    prim_path="",  # override per scene
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(YAM_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # USD exposes only joint1..joint8, so bind actuators to those names.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-8]"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=400.0,
            damping=40.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
