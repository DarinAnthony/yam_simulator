# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms
import isaacsim.core.utils.torch as torch_utils

from .yam_manip_env_cfg import YamManipEnvCfg


class YamManipEnv(DirectRLEnv):
    cfg: YamManipEnvCfg

    def __init__(self, cfg: YamManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._ik = DifferentialIKController(self.cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.ee_pos_target_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat_target_b = torch.zeros((self.num_envs, 4), device=self.device)
        self._entities_resolved = False
        self.phase = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.sorted_mask = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.bool)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.wall = RigidObject(self.cfg.wall_cfg)
        self.dropoff_red = RigidObject(self.cfg.dropoff_red_cfg)
        self.dropoff_blue = RigidObject(self.cfg.dropoff_blue_cfg)
        self.dropoff_yellow = RigidObject(self.cfg.dropoff_yellow_cfg)
        self.start_area = RigidObject(self.cfg.start_area_cfg) if self.cfg.use_start_area_radius else None
        self.site_marker = RigidObject(self.cfg.site_marker_cfg)
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
        self.scene.rigid_objects["site_marker"] = self.site_marker
        self.scene.rigid_objects["red_block"] = self.red_block
        self.scene.rigid_objects["blue_block"] = self.blue_block
        self.scene.rigid_objects["yellow_block"] = self.yellow_block

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Defer entity resolution until the simulation initializes (e.g., in reset).

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self._resolve_robot_entities()
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

        # Treat non-positive commands as "keep open" so default 0.0 stays open.
        t = torch.clamp(grip_cmd, 0.0, 1.0)
        open_q = float(self.cfg.gripper_open)
        closed_q = float(self.cfg.gripper_closed)
        finger_q = open_q + t * (closed_q - open_q)
        grip_q_des = torch.stack([finger_q, finger_q], dim=1)
        self.robot.set_joint_position_target(grip_q_des, joint_ids=self.grip_joint_ids)
        self.robot.write_data_to_sim()
        self._update_site_marker()

    def _get_observations(self) -> dict:
        self._resolve_robot_entities()

        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        grip_t = self._grip_close_t().unsqueeze(-1)

        red_pos = self.red_block.data.root_pos_w - self.scene.env_origins
        blue_pos = self.blue_block.data.root_pos_w - self.scene.env_origins
        yellow_pos = self.yellow_block.data.root_pos_w - self.scene.env_origins

        red_vel = self.red_block.data.root_lin_vel_w
        blue_vel = self.blue_block.data.root_lin_vel_w
        yellow_vel = self.yellow_block.data.root_lin_vel_w

        drop_r = self.dropoff_red.data.root_pos_w - self.scene.env_origins
        drop_b = self.dropoff_blue.data.root_pos_w - self.scene.env_origins
        drop_y = self.dropoff_yellow.data.root_pos_w - self.scene.env_origins

        phase_clamped = torch.clamp(self.phase, 0, 2)
        phase_oh = F.one_hot(phase_clamped, num_classes=3).float()
        sorted_f = self.sorted_mask.float()

        obs = torch.cat(
            [
                phase_oh,
                sorted_f,
                ee_pos_b,
                self.ee_pos_target_b,
                grip_t,
                red_pos,
                blue_pos,
                yellow_pos,
                red_vel,
                blue_vel,
                yellow_vel,
                drop_r,
                drop_b,
                drop_y,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._resolve_robot_entities()

        table_top_z = (
            self.cfg.table_cfg.init_state.pos[2] + self.cfg.table_cfg.spawn.size[2] / 2.0
        )
        block_half_z = self.cfg.red_block_cfg.spawn.size[2] / 2.0

        phase = self.phase
        active = phase < 3
        tgt_idx = torch.clamp(phase, 0, 2)

        block_pos = torch.stack(
            [
                self.blue_block.data.root_pos_w - self.scene.env_origins,
                self.red_block.data.root_pos_w - self.scene.env_origins,
                self.yellow_block.data.root_pos_w - self.scene.env_origins,
            ],
            dim=1,
        )
        goal_pos = torch.stack(
            [
                self.dropoff_blue.data.root_pos_w - self.scene.env_origins,
                self.dropoff_red.data.root_pos_w - self.scene.env_origins,
                self.dropoff_yellow.data.root_pos_w - self.scene.env_origins,
            ],
            dim=1,
        )

        env_ids = torch.arange(self.num_envs, device=self.device)
        tgt_block_pos = block_pos[env_ids, tgt_idx]
        tgt_goal_pos = goal_pos[env_ids, tgt_idx]

        grasp_pos_w, _ = self._ee_grasp_pos_w()
        grasp_pos = grasp_pos_w - self.scene.env_origins

        d_reach = torch.linalg.norm(grasp_pos - tgt_block_pos, dim=-1)
        d_goal_xy = torch.linalg.norm((tgt_block_pos - tgt_goal_pos)[:, 0:2], dim=-1)

        grip_t = self._grip_close_t()
        grip_closed = grip_t > float(self.cfg.grip_closed_thresh)
        grip_open = grip_t < float(self.cfg.grip_open_thresh)

        near = d_reach < float(self.cfg.near_thresh)
        lifted = (tgt_block_pos[:, 2] - table_top_z) > float(self.cfg.lift_height)
        at_goal_xy = d_goal_xy < float(self.cfg.goal_xy_thresh)
        near_table = torch.abs(tgt_block_pos[:, 2] - (table_top_z + block_half_z)) < float(
            self.cfg.place_z_thresh
        )

        grasped = (near & grip_closed) | lifted

        r_reach = (1.0 - torch.tanh(float(self.cfg.reach_k) * d_reach)) * (~grasped) * active
        r_grasp = (near.float() * grip_t) * (~lifted) * active
        r_lift = torch.clamp(
            (tgt_block_pos[:, 2] - table_top_z) / float(self.cfg.lift_height), 0.0, 1.0
        ) * grasped * active
        r_carry = (1.0 - torch.tanh(float(self.cfg.goal_k) * d_goal_xy)) * grasped * active

        placed = at_goal_xy & near_table & grip_open & active

        bonus = torch.zeros((self.num_envs,), device=self.device)
        newly_placed = placed.nonzero(as_tuple=False).squeeze(-1)
        if newly_placed.numel() > 0:
            idxs = tgt_idx[newly_placed]
            self.sorted_mask[newly_placed, idxs] = True
            self.phase[newly_placed] += 1
            bonus[newly_placed] = float(self.cfg.place_bonus)

        act_pen = float(self.cfg.step_penalty_w) * torch.sum(self.actions[:, 0:3] ** 2, dim=-1)

        reward = (
            float(self.cfg.reach_w) * r_reach
            + float(self.cfg.grasp_w) * r_grasp
            + float(self.cfg.lift_w) * r_lift
            + float(self.cfg.carry_w) * r_carry
            + bonus
            - act_pen
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._resolve_robot_entities()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = self.phase >= 3
        if torch.any(success):
            success_ids = success.nonzero(as_tuple=False).squeeze(-1)
            self._set_robot_home(success_ids)
        return success, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._resolve_robot_entities()
        self.phase[env_ids] = 0
        self.sorted_mask[env_ids] = False

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._set_robot_home(env_ids)

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

        self.actions[env_ids] = 0.0
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

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
        self._update_site_marker(env_ids=env_ids)

    def _resolve_robot_entities(self) -> None:
        if self._entities_resolved:
            return
        self.cfg.robot_entity.resolve(self.scene)
        joint_ids = self.cfg.robot_entity.joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = list(range(self.robot.num_joints))[joint_ids]
        joint_ids = list(joint_ids)
        if len(joint_ids) >= 8:
            arm_ids = joint_ids[:6]
            grip_ids = joint_ids[6:8]
        else:
            arm_ids = joint_ids[:-2]
            grip_ids = joint_ids[-2:]
        self.arm_joint_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self.grip_joint_ids = torch.tensor(grip_ids, device=self.device, dtype=torch.long)
        self.ee_body_id = self.cfg.robot_entity.body_ids[0]
        self.marker_body_ids, _ = self.robot.find_bodies(self.cfg.marker_body_names, preserve_order=True)
        self._entities_resolved = True

    def _set_robot_home(self, env_ids) -> None:
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if len(self.cfg.home_joint_pos) >= len(self.arm_joint_ids):
            joint_pos[:, self.arm_joint_ids] = torch.tensor(
                self.cfg.home_joint_pos[: len(self.arm_joint_ids)], device=self.device
            ).unsqueeze(0)
        joint_pos[:, self.grip_joint_ids] = float(self.cfg.gripper_open)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    def _ee_grasp_pos_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]
        offset = torch.tensor(self.cfg.gripper_site_offset, device=self.device, dtype=torch.float32)
        offset = offset.unsqueeze(0).expand(ee_quat_w.shape[0], 3)
        offset_w = torch_utils.quat_apply(ee_quat_w, offset)
        return ee_pos_w + offset_w, ee_quat_w

    def _grip_close_t(self) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self.grip_joint_ids]
        open_q = float(self.cfg.gripper_open)
        closed_q = float(self.cfg.gripper_closed)
        t = (q - open_q) / (closed_q - open_q)
        t = torch.clamp(t, 0.0, 1.0)
        return t.mean(dim=1)

    def _update_site_marker(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_quat_w = ee_pose_w[:, 3:7]
        if len(self.marker_body_ids) >= 2:
            tip_pos_w = self.robot.data.body_state_w[:, self.marker_body_ids[:2], 0:3].mean(dim=1)
        else:
            tip_pos_w = ee_pose_w[:, 0:3]
        offset = torch.tensor(self.cfg.gripper_site_offset, device=self.device, dtype=torch.float32)
        offset = offset.unsqueeze(0).expand(ee_quat_w.shape[0], 3)
        offset_w = torch_utils.quat_apply(ee_quat_w, offset)
        marker_pos = tip_pos_w + offset_w
        marker_quat = ee_quat_w
        marker_state = self.site_marker.data.default_root_state[env_ids].clone()
        marker_state[:, 0:3] = marker_pos[env_ids]
        marker_state[:, 3:7] = marker_quat[env_ids]
        marker_state[:, 7:] = 0.0
        self.site_marker.write_root_pose_to_sim(marker_state[:, 0:7], env_ids)
        self.site_marker.write_root_velocity_to_sim(marker_state[:, 7:], env_ids)
