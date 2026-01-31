# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .yam_manip_env_cfg import YamManipEnvCfg


class YamManipEnv(DirectRLEnv):
    cfg: YamManipEnvCfg

    def __init__(self, cfg: YamManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.wall = RigidObject(self.cfg.wall_cfg)
        self.dropoff_red = RigidObject(self.cfg.dropoff_red_cfg)
        self.dropoff_blue = RigidObject(self.cfg.dropoff_blue_cfg)
        self.dropoff_yellow = RigidObject(self.cfg.dropoff_yellow_cfg)
        self.start_area = RigidObject(self.cfg.start_area_cfg)
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
        self.scene.rigid_objects["start_area"] = self.start_area
        self.scene.rigid_objects["red_block"] = self.red_block
        self.scene.rigid_objects["blue_block"] = self.blue_block
        self.scene.rigid_objects["yellow_block"] = self.yellow_block

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # No-op for now. This task is intended for visualization only.
        return

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
        angles = 2.0 * torch.pi * torch.rand((3, num_envs), device=self.device)
        radii = torch.sqrt(torch.rand((3, num_envs), device=self.device)) * radius
        xy_offsets = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=-1)

        z = (
            self.cfg.table_cfg.init_state.pos[2]
            + self.cfg.table_cfg.spawn.size[2] / 2.0
            + self.cfg.red_block_cfg.spawn.size[2] / 2.0
        )
        for idx, block in enumerate((self.red_block, self.blue_block, self.yellow_block)):
            block_state = block.data.default_root_state[env_ids].clone()
            block_state[:, 0] = start_center[0] + xy_offsets[idx, :, 0] + self.scene.env_origins[env_ids, 0]
            block_state[:, 1] = start_center[1] + xy_offsets[idx, :, 1] + self.scene.env_origins[env_ids, 1]
            block_state[:, 2] = z
            block_state[:, 7:] = 0.0
            block.write_root_pose_to_sim(block_state[:, 0:7], env_ids)
            block.write_root_velocity_to_sim(block_state[:, 7:], env_ids)
            block.reset()
