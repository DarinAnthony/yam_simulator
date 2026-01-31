# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms

from .yam_manip_env_cfg import YamManipEnvCfg


class YamManipEnv(DirectRLEnv):
    cfg: YamManipEnvCfg

    def __init__(self, cfg: YamManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._ik = DifferentialIKController(self.cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.ee_pos_target_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat_target_b = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.wall = RigidObject(self.cfg.wall_cfg)
        self.dropoff_red = RigidObject(self.cfg.dropoff_red_cfg)
        self.dropoff_blue = RigidObject(self.cfg.dropoff_blue_cfg)
        self.dropoff_yellow = RigidObject(self.cfg.dropoff_yellow_cfg)
        self.start_area = RigidObject(self.cfg.start_area_cfg) if self.cfg.use_start_area_radius else None
        self.red_block = RigidObject(self.cfg.red_block_cfg)
        self.blue_block = RigidObject(self.cfg.blue_block_cfg)
        self.yellow_block = RigidObject(self.cfg.yellow_block_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["wall"] = self.wall
        self.scene.rigid_objects["dropoff_red"] = self.dropoff_red
        self.scene.rigid_objects["dropoff_blue"] = self.dropoff_blue
        self.scene.rigid_objects["dropoff_yellow"] = self.dropoff_yellow
        if self.start_area is not None:
            self.scene.rigid_objects["start_area"] = self.start_area
        self.scene.rigid_objects["red_block"] = self.red_block
        self.scene.rigid_objects["blue_block"] = self.blue_block
        self.scene.rigid_objects["yellow_block"] = self.yellow_block

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.cfg.robot_entity.resolve(self.scene)
        self.arm_joint_ids = torch.tensor(self.cfg.robot_entity.joint_ids[:6], device=self.device, dtype=torch.long)
        self.grip_joint_ids = torch.tensor(self.cfg.robot_entity.joint_ids[6:8], device=self.device, dtype=torch.long)
        self.ee_body_id = self.cfg.robot_entity.body_ids[0]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        delta_pos = torch.clamp(self.actions[:, 0:3], -1.0, 1.0)
        delta_pos = delta_pos * float(self.cfg.ee_delta_scale)
        delta_pos = torch.clamp(delta_pos, -float(self.cfg.ee_pos_limit), float(self.cfg.ee_pos_limit))
        grip_cmd = torch.clamp(self.actions[:, 3], -1.0, 1.0)

        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        self.ee_pos_target_b = self.ee_pos_target_b + delta_pos
        self.ee_quat_target_b = ee_quat_b

        jacobian_w = self.robot.root_physx_view.get_jacobians()
        jacobian = jacobian_w[:, self.ee_body_id - 1, :, self.arm_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]

        self._ik.set_command(self.ee_pos_target_b, ee_quat=ee_quat_b)
        arm_q_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(arm_q_des, joint_ids=self.arm_joint_ids)

        t = (grip_cmd + 1.0) * 0.5
        open_q = float(self.cfg.gripper_open)
        closed_q = float(self.cfg.gripper_closed)
        finger_q = open_q + t * (closed_q - open_q)
        grip_q_des = torch.stack([finger_q, finger_q], dim=1)
        self.robot.set_joint_position_target(grip_q_des, joint_ids=self.grip_joint_ids)
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        red_pos = self.red_block.data.root_pos_w - self.scene.env_origins
        blue_pos = self.blue_block.data.root_pos_w - self.scene.env_origins
        yellow_pos = self.yellow_block.data.root_pos_w - self.scene.env_origins
        observations = {"policy": torch.cat((red_pos, blue_pos, yellow_pos), dim=-1)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Empty reward for now (visualization-only task).
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        num_envs = env_ids.shape[0]
        start_center = torch.tensor(
            (self.cfg.start_area_cfg.init_state.pos[0], self.cfg.start_area_cfg.init_state.pos[1]),
            device=self.device,
            dtype=torch.float32,
        )
        radius = self.cfg.start_area_cfg.spawn.radius * 0.95
        min_sep = float(self.cfg.min_block_separation)
        xy_offsets = torch.zeros((3, num_envs, 2), device=self.device, dtype=torch.float32)

        if radius <= 0.0 or min_sep <= 0.0:
            xy_offsets[:, :, :] = 0.0
        else:
            for env_i in range(num_envs):
                placed = []
                for block_i in range(3):
                    for _ in range(20):
                        angle = 2.0 * torch.pi * torch.rand((), device=self.device)
                        r = torch.sqrt(torch.rand((), device=self.device)) * radius
                        candidate = torch.stack((r * torch.cos(angle), r * torch.sin(angle)))
                        if all(torch.linalg.norm(candidate - p) >= min_sep for p in placed):
                            placed.append(candidate)
                            break
                    else:
                        placed.append(candidate)
                    xy_offsets[block_i, env_i] = placed[-1]

        z = (
            self.cfg.table_cfg.init_state.pos[2]
            + self.cfg.table_cfg.spawn.size[2] / 2.0
            + self.cfg.red_block_cfg.spawn.size[2] / 2.0
            + self.cfg.block_clearance
        )
        for idx, block in enumerate((self.red_block, self.blue_block, self.yellow_block)):
            block_state = block.data.default_root_state[env_ids].clone()
            block_state[:, 0] = start_center[0] + xy_offsets[idx, :, 0] + self.scene.env_origins[env_ids, 0]
            block_state[:, 1] = start_center[1] + xy_offsets[idx, :, 1] + self.scene.env_origins[env_ids, 1]
            block_state[:, 2] = z + self.scene.env_origins[env_ids, 2]
            block_state[:, 7:] = 0.0
            block.write_root_pose_to_sim(block_state[:, 0:7], env_ids)
            block.write_root_velocity_to_sim(block_state[:, 7:], env_ids)
            block.reset()

        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        self.ee_pos_target_b[env_ids] = ee_pos_b[env_ids]
        self.ee_quat_target_b[env_ids] = ee_quat_b[env_ids]
