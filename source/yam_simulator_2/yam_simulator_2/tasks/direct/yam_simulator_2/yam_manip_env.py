# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms
import isaacsim.core.utils.torch as torch_utils

from .yam_manip_env_cfg import YamManipEnvCfg


@torch.jit.script
def compute_policy_obs(
    phase: torch.Tensor,
    sorted_mask: torch.Tensor,
    ee_pos_b: torch.Tensor,
    ee_pos_target_b: torch.Tensor,
    grip_t: torch.Tensor,
    red_pos: torch.Tensor,
    blue_pos: torch.Tensor,
    yellow_pos: torch.Tensor,
    red_vel: torch.Tensor,
    blue_vel: torch.Tensor,
    yellow_vel: torch.Tensor,
    drop_r: torch.Tensor,
    drop_b: torch.Tensor,
    drop_y: torch.Tensor,
) -> torch.Tensor:
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    phase_clamped = torch.clamp(phase, 0, 2).to(torch.long)
    phase_oh = torch.zeros((phase.shape[0], 3), device=phase.device, dtype=torch.float32)
    phase_oh.scatter_(1, phase_clamped.unsqueeze(1), 1.0)
    sorted_f = sorted_mask.to(torch.float32)
    obs = torch.cat(
        [
            phase_oh,
            sorted_f,
            ee_pos_b,
            ee_pos_target_b,
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
    return obs


@torch.jit.script
def compute_reward_core(
    phase: torch.Tensor,
    actions_xyz: torch.Tensor,
    tgt_block_pos: torch.Tensor,
    tgt_goal_pos: torch.Tensor,
    grasp_pos: torch.Tensor,
    table_top_z: float,
    block_half_z: float,
    grip_t: torch.Tensor,
    grip_closed_thresh: float,
    grip_open_thresh: float,
    near_thresh: float,
    lift_height: float,
    goal_xy_thresh: float,
    place_z_thresh: float,
    reach_k: float,
    goal_k: float,
    reach_w: float,
    grasp_w: float,
    lift_w: float,
    carry_w: float,
    step_penalty_w: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    active = phase < 3
    d_reach = torch.linalg.norm(grasp_pos - tgt_block_pos, dim=-1)
    d_goal_xy = torch.linalg.norm((tgt_block_pos - tgt_goal_pos)[:, 0:2], dim=-1)

    grip_closed = grip_t > grip_closed_thresh
    grip_open = grip_t < grip_open_thresh

    near = d_reach < near_thresh
    lifted = (tgt_block_pos[:, 2] - table_top_z) > lift_height
    at_goal_xy = d_goal_xy < goal_xy_thresh
    near_table = torch.abs(tgt_block_pos[:, 2] - (table_top_z + block_half_z)) < place_z_thresh

    grasped = (near & grip_closed) | lifted

    r_reach = (1.0 - torch.tanh(reach_k * d_reach)) * (~grasped) * active
    r_grasp = (near.to(torch.float32) * grip_t) * (~lifted) * active
    r_lift = torch.clamp((tgt_block_pos[:, 2] - table_top_z) / lift_height, 0.0, 1.0) * grasped * active
    r_carry = (1.0 - torch.tanh(goal_k * d_goal_xy)) * grasped * active

    placed = at_goal_xy & near_table & grip_open & active

    act_pen = step_penalty_w * torch.sum(actions_xyz * actions_xyz, dim=-1)
    reward = reach_w * r_reach + grasp_w * r_grasp + lift_w * r_lift + carry_w * r_carry - act_pen
    return reward, placed


class YamManipEnv(DirectRLEnv):
    cfg: YamManipEnvCfg

    def __init__(self, cfg: YamManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._ik = DifferentialIKController(self.cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.ee_pos_target_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat_target_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_quat_nominal_b = torch.zeros((self.num_envs, 4), device=self.device)
        self._entities_resolved = False
        self.phase = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.sorted_mask = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_wp_dist = torch.zeros((self.num_envs,), device=self.device)
        self.prev_ee_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._debug_printed = False
        self._debug_bad_obs = False
        self._debug_bad_rew = False
        self._debug_bad_tensors = set()
        self._obs_step_counter = 0
        self._debug_enabled = getattr(self.cfg, "debug_print_obs", False)

    def _ensure_pose_buffers(self):
        """Create pose-related buffers if they are missing (e.g., after hot-reload)."""
        if not hasattr(self, "ee_pos_target_b"):
            self.ee_pos_target_b = torch.zeros((self.num_envs, 3), device=self.device)
        if not hasattr(self, "ee_quat_target_b"):
            self.ee_quat_target_b = torch.zeros((self.num_envs, 4), device=self.device)
        if not hasattr(self, "ee_quat_nominal_b"):
            self.ee_quat_nominal_b = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.dropoff_red = RigidObject(self.cfg.dropoff_red_cfg)
        self.dropoff_blue = RigidObject(self.cfg.dropoff_blue_cfg)
        self.dropoff_yellow = RigidObject(self.cfg.dropoff_yellow_cfg)
        self.start_area = RigidObject(self.cfg.start_area_cfg) if self.cfg.use_start_area_radius else None
        self.site_marker = RigidObject(self.cfg.site_marker_cfg)
        self.red_block = RigidObject(self.cfg.red_block_cfg)
        self.blue_block = RigidObject(self.cfg.blue_block_cfg)
        self.yellow_block = RigidObject(self.cfg.yellow_block_cfg)
        self.camera = Camera(self.cfg.camera_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["dropoff_red"] = self.dropoff_red
        self.scene.rigid_objects["dropoff_blue"] = self.dropoff_blue
        self.scene.rigid_objects["dropoff_yellow"] = self.dropoff_yellow
        if self.start_area is not None:
            self.scene.rigid_objects["start_area"] = self.start_area
        self.scene.rigid_objects["site_marker"] = self.site_marker
        self.scene.rigid_objects["red_block"] = self.red_block
        self.scene.rigid_objects["blue_block"] = self.blue_block
        self.scene.rigid_objects["yellow_block"] = self.yellow_block
        self.scene.sensors["camera"] = self.camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Defer entity resolution until the simulation initializes (e.g., in reset).

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Fail fast if caller passes a mismatched action vector. This keeps the
        # policy/action_dim consistent with YamManipEnvCfg.action_space (6):
        # [dx, dy, dz, d_roll, d_pitch, gripper].
        if actions.shape[1] != self.cfg.action_space:
            raise RuntimeError(
                f"Expected actions with {self.cfg.action_space} dims, got {tuple(actions.shape)}"
            )
        # Catch NaNs/Infs coming from the policy before they hit the sim.
        if not torch.isfinite(actions).all():
            bad_env = (~torch.isfinite(actions)).any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            e = int(bad_env[0].item())
            print(f"[BAD ACT] env={e} actions[e]={actions[e].detach().cpu().tolist()}")
            raise RuntimeError("Non-finite action input from policy")
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self._resolve_robot_entities()
        self._ensure_pose_buffers()
        delta_pos = torch.clamp(self.actions[:, 0:3], -1.0, 1.0)
        delta_pos = delta_pos * float(self.cfg.ee_delta_scale)
        delta_pos = torch.clamp(delta_pos, -float(self.cfg.ee_pos_limit), float(self.cfg.ee_pos_limit))
        tilt_cmd = torch.clamp(self.actions[:, 3:5], -1.0, 1.0)
        grip_cmd = torch.clamp(self.actions[:, 5], -1.0, 1.0)

        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # Make the target relative to the current EE pose to avoid drift to clamp corners.
        self.ee_pos_target_b = ee_pos_b + delta_pos
        ee_min = torch.tensor([-0.25, -0.25, -0.10], device=self.device)
        ee_max = torch.tensor([0.45, 0.25, 0.35], device=self.device)
        self.ee_pos_target_b = torch.clamp(self.ee_pos_target_b, ee_min, ee_max)

        # Orientation target: apply small roll/pitch deltas; yaw held by target
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3)
        roll = tilt_cmd[:, 0] * float(self.cfg.tilt_scale)
        pitch = tilt_cmd[:, 1] * float(self.cfg.tilt_scale)
        if torch.any(roll != 0.0) or torch.any(pitch != 0.0):
            q_roll = self._quat_from_angle_axis(roll, x_axis)
            q_pitch = self._quat_from_angle_axis(pitch, y_axis)
            q_tilt = self._quat_mul(q_pitch, q_roll)
            q_target = self._quat_mul(q_tilt, self.ee_quat_target_b)
            q_target = q_target / torch.linalg.norm(q_target, dim=-1, keepdim=True).clamp_min(1e-8)
            self.ee_quat_target_b = q_target

        jacobian_w = self.robot.root_physx_view.get_jacobians()
        jacobian = jacobian_w[:, self.ee_body_id - 1, :, self.arm_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        lims = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        self._debug_check_tensor("soft_joint_pos_limits", lims)
        self._debug_check_tensor("jacobian", jacobian)
        self._debug_check_tensor("joint_pos_arm", joint_pos)

        ik_cmd = torch.cat([self.ee_pos_target_b, self.ee_quat_target_b], dim=-1)
        self._ik.set_command(ik_cmd)
        arm_q_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        arm_q_des = torch.clamp(arm_q_des, lims[..., 0], lims[..., 1])
        self._debug_check_tensor("arm_q_des", arm_q_des)
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
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]

        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        grip_t = self._grip_close_t().unsqueeze(-1)

        def to_base(pos_w: torch.Tensor) -> torch.Tensor:
            return torch_utils.quat_rotate_inverse(root_quat_w, pos_w - root_pos_w)

        def vel_to_base(vel_w: torch.Tensor) -> torch.Tensor:
            return torch_utils.quat_rotate_inverse(root_quat_w, vel_w)

        red_pos = to_base(self.red_block.data.root_pos_w)
        blue_pos = to_base(self.blue_block.data.root_pos_w)
        yellow_pos = to_base(self.yellow_block.data.root_pos_w)

        red_vel = vel_to_base(self.red_block.data.root_lin_vel_w)
        blue_vel = vel_to_base(self.blue_block.data.root_lin_vel_w)
        yellow_vel = vel_to_base(self.yellow_block.data.root_lin_vel_w)

        drop_r = to_base(self.dropoff_red.data.root_pos_w)
        drop_b = to_base(self.dropoff_blue.data.root_pos_w)
        drop_y = to_base(self.dropoff_yellow.data.root_pos_w)

        self._debug_check_tensor("ee_pos_b", ee_pos_b)
        self._debug_check_tensor("ee_pos_target_b", self.ee_pos_target_b)
        self._debug_check_tensor("grip_t", grip_t)
        self._debug_check_tensor("red_pos", red_pos)
        self._debug_check_tensor("blue_pos", blue_pos)
        self._debug_check_tensor("yellow_pos", yellow_pos)
        self._debug_check_tensor("red_vel", red_vel)
        self._debug_check_tensor("blue_vel", blue_vel)
        self._debug_check_tensor("yellow_vel", yellow_vel)
        self._debug_check_tensor("drop_r", drop_r)
        self._debug_check_tensor("drop_b", drop_b)
        self._debug_check_tensor("drop_y", drop_y)

        # Keep ordering consistent: blue, red, yellow (for both positions and goals).
        obs = compute_policy_obs(
            self.phase,
            self.sorted_mask,
            ee_pos_b,
            self.ee_pos_target_b,
            grip_t,
            blue_pos,
            red_pos,
            yellow_pos,
            blue_vel,
            red_vel,
            yellow_vel,
            drop_b,
            drop_r,
            drop_y,
        )
        # Clamp to avoid huge magnitudes propagating into the policy (helps
        # prevent NaNs in logstd/std during early random exploration).
        max_abs = torch.max(torch.abs(obs))
        if max_abs > self.cfg.debug_abs_max:
            print(f"[DEBUG] obs magnitude clamp: max_abs={max_abs.item():.3e}")
            obs = torch.clamp(obs, -self.cfg.debug_abs_max, self.cfg.debug_abs_max)

        # Optional observation printout for debugging; controlled via cfg.debug_print_obs.
        if self._debug_enabled:
            self._obs_step_counter += 1
            sample = obs[0].detach().cpu()
            print(
                f"[OBS@{self._obs_step_counter}] env0 min={sample.min():.3f} "
                f"max={sample.max():.3f} first10={sample[:10].tolist()}"
            )
        # Hard check across all envs; stop immediately on any non-finite observation.
        if not torch.isfinite(obs).all():
            bad_env = (~torch.isfinite(obs)).any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            e = int(bad_env[0].item())
            print(f"[BAD OBS] env={e} obs[e]=", obs[e].detach().cpu().tolist())
            raise RuntimeError("Non-finite observation")

        self._debug_check_tensor("obs", obs, "_debug_bad_obs")
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._resolve_robot_entities()

        # -------------------------
        # Target selection by phase
        # -------------------------
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

        # EE grasp point (the one you already use)
        grasp_pos_w, _ = self._ee_grasp_pos_w()
        grasp_pos = grasp_pos_w - self.scene.env_origins

        grip_t = self._grip_close_t()  # 0=open, 1=closed (from joint positions)

        # Geometry constants
        table_top_z = self.cfg.table_cfg.init_state.pos[2] + self.cfg.table_cfg.spawn.size[2] / 2.0
        block_half_z = self.cfg.red_block_cfg.spawn.size[2] / 2.0  # all same size
        block_rest_center_z = table_top_z + block_half_z

        # -------------------------
        # Distances and soft gates
        # -------------------------
        xy = torch.linalg.norm((grasp_pos - tgt_block_pos)[:, 0:2], dim=-1)   # EE to block XY
        z = grasp_pos[:, 2]
        z_block = tgt_block_pos[:, 2]

        # Two key heights relative to the block center
        z_hover = z_block + float(self.cfg.hover_height)
        z_pick  = z_block + float(self.cfg.pick_z_offset)

        # Soft "aligned in XY" gate: ~1 when close, ~0 when far
        # (Use a slightly larger radius than near_thresh so it’s reachable early)
        align_r = 0.08
        k_align = 20.0
        g_align = torch.sigmoid(k_align * (align_r - xy))

        # Soft "at pick height" gate: ~1 when near z_pick
        k_z = 25.0
        z_gate_r = 0.03
        g_zpick = torch.sigmoid(k_z * (z_gate_r - torch.abs(z - z_pick)))

        # -------------------------
        # Stage 1-2: Pose shaping
        # -------------------------
        # Encourage XY alignment strongly, and encourage Z to be at hover when far,
        # then transition to pick height when aligned.
        k_xy = 15.0
        r_xy = torch.exp(-k_xy * xy)

        r_z_hover = torch.exp(-k_z * torch.abs(z - z_hover))
        r_z_pick  = torch.exp(-k_z * torch.abs(z - z_pick))

        # Blend: if not aligned -> prefer hover, if aligned -> prefer pick
        r_z = (1.0 - g_align) * r_z_hover + g_align * r_z_pick

        r_pose = r_xy * r_z

        # Progress term (only reward improvement; never penalize exploration)
        # Reuse your buffer prev_wp_dist as "previous xy"
        delta_xy = (self.prev_wp_dist - xy)
        r_prog = torch.clamp(delta_xy, min=0.0)
        self.prev_wp_dist = xy.detach()

        # -------------------------
        # Stage 3: Gripper timing shaping
        # -------------------------
        # Desired gripper state:
        # - open when not aligned
        # - closed when aligned AND near pick height
        grip_target = (g_align * g_zpick)  # in [0,1]
        r_grip = 1.0 - (grip_t - grip_target) ** 2  # in [0,1], max when matching target

        # -------------------------
        # Stage 4: Lift / Carry / Place
        # -------------------------
        # "Lifted" proxy based on block height above rest position on the table.
        block_clear = (tgt_block_pos[:, 2] - block_rest_center_z)
        lifted = block_clear > (0.5 * float(self.cfg.lift_height))  # easier than full lift_height early

        # Lift reward (bounded [0,1])
        r_lift = torch.clamp(block_clear / float(self.cfg.lift_height), 0.0, 1.0) * lifted.to(torch.float32)

        # Carry toward goal (bounded)
        d_goal_xy = torch.linalg.norm((tgt_block_pos - tgt_goal_pos)[:, 0:2], dim=-1)
        k_goal = 10.0
        r_carry = torch.exp(-k_goal * d_goal_xy) * lifted.to(torch.float32)

        # Place shaping: encourage lowering near goal
        place_z = block_rest_center_z + float(self.cfg.place_z_offset)
        r_place_z = torch.exp(-k_z * torch.abs(tgt_block_pos[:, 2] - place_z)) * lifted.to(torch.float32)

        near_goal = d_goal_xy < (2.0 * float(self.cfg.goal_xy_thresh))
        r_place = (0.5 * r_carry + 0.5 * r_place_z) * near_goal.to(torch.float32)

        # Placement success + phase advance (keep your logic)
        grip_open = grip_t < float(self.cfg.grip_open_thresh)
        at_goal_xy = d_goal_xy < float(self.cfg.goal_xy_thresh)
        near_table = torch.abs(tgt_block_pos[:, 2] - block_rest_center_z) < float(self.cfg.place_z_thresh)
        placed = at_goal_xy & near_table & grip_open & active

        bonus = torch.zeros((self.num_envs,), device=self.device)
        newly_placed = placed.nonzero(as_tuple=False).squeeze(-1)
        if newly_placed.numel() > 0:
            idxs = tgt_idx[newly_placed]
            self.sorted_mask[newly_placed, idxs] = True
            self.phase[newly_placed] += 1
            bonus[newly_placed] = float(self.cfg.place_bonus)

        # -------------------------
        # Regularization penalties
        # -------------------------
        # Small action penalty (position deltas only)
        act_pen = float(self.cfg.step_penalty_w) * torch.sum(self.actions[:, 0:3] ** 2, dim=-1)

        # Smoothness penalty
        r_smooth = -float(self.cfg.smooth_w) * torch.sum((self.actions[:, 0:3] - self.prev_actions[:, 0:3]) ** 2, dim=-1)

        # Optional: tiny EE velocity penalty (keep if you want)
        r_vel = torch.zeros((self.num_envs,), device=self.device)
        ee_state = self.robot.data.body_state_w[:, self.ee_body_id]
        if ee_state.shape[-1] >= 13:
            ee_lin_vel = ee_state[:, 7:10]
            r_vel = -float(self.cfg.ee_vel_w) * torch.sum(ee_lin_vel * ee_lin_vel, dim=-1)

        # -------------------------
        # Combine
        # -------------------------
        # Weights (start conservative; you can tune later)
        w_pose   = 2.0
        w_prog   = 1.0
        w_grip   = 0.5
        w_lift   = 2.0
        w_carry  = 2.0
        w_place  = 1.0

        reward = (
            active.to(torch.float32)
            * (
                w_pose  * r_pose
                + w_prog * r_prog
                + w_grip * r_grip
                + w_lift * r_lift
                + w_carry * r_carry
                + w_place * r_place
                + r_smooth
                + r_vel
                - act_pen
            )
            + bonus
        )

        if not torch.isfinite(reward).all():
            bad_env = (~torch.isfinite(reward)).nonzero(as_tuple=False).squeeze(-1)
            e = int(bad_env[0].item())
            print(f"[BAD REW] env={e} rew={float(reward[e].item())}")
            raise RuntimeError("Non-finite reward")

        return reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._resolve_robot_entities()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = self.phase >= 3
        # Terminate if any block falls below the table.
        block_pos = torch.stack(
            [
                self.red_block.data.root_pos_w - self.scene.env_origins,
                self.blue_block.data.root_pos_w - self.scene.env_origins,
                self.yellow_block.data.root_pos_w - self.scene.env_origins,
            ],
            dim=1,
        )
        min_z = block_pos[:, :, 2].min(dim=1).values
        table_top = self.cfg.table_cfg.init_state.pos[2] + self.cfg.table_cfg.spawn.size[2] / 2.0
        fallen = min_z < (table_top - float(self.cfg.fall_termination_margin))
        if torch.any(success):
            success_ids = success.nonzero(as_tuple=False).squeeze(-1)
            self._set_robot_home(success_ids)
        terminated = success | fallen
        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._resolve_robot_entities()
        self._ensure_pose_buffers()
        self.phase[env_ids] = 0
        self.sorted_mask[env_ids] = False
        self.prev_actions[env_ids] = 0.0
        self.prev_wp_dist[env_ids] = 0.0
        self.prev_ee_lin_vel[env_ids] = 0.0

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
        self.ee_quat_nominal_b[env_ids] = ee_quat_b[env_ids]
        self._ik.reset()
        self._update_site_marker(env_ids=env_ids)
        self._update_camera_pose(env_ids=env_ids)
        # Initialize EE velocity buffer from current state (if available).
        ee_state = self.robot.data.body_state_w[:, self.ee_body_id]
        if ee_state.shape[-1] >= 13:
            self.prev_ee_lin_vel[env_ids] = ee_state[env_ids, 7:10]
        # Initialize progress distance (XY) to avoid negative first-step progress.
        phase = self.phase
        tgt_idx = torch.clamp(phase, 0, 2)
        block_pos = torch.stack(
            [
                self.blue_block.data.root_pos_w - self.scene.env_origins,
                self.red_block.data.root_pos_w - self.scene.env_origins,
                self.yellow_block.data.root_pos_w - self.scene.env_origins,
            ],
            dim=1,
        )
        env_ids_all = torch.arange(self.num_envs, device=self.device)
        tgt_block_pos = block_pos[env_ids_all, tgt_idx]
        grasp_pos_w, _ = self._ee_grasp_pos_w()
        grasp_pos = grasp_pos_w - self.scene.env_origins
        xy_block = torch.linalg.norm((grasp_pos - tgt_block_pos)[:, 0:2], dim=-1)
        self.prev_wp_dist[env_ids] = xy_block[env_ids].detach()
        self._debug_print_robot_state()

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

    def _debug_print_robot_state(self) -> None:
        if self._debug_printed or not self.cfg.debug_print_joint_pos:
            return
        try:
            joint_names = list(self.robot.joint_names)
        except Exception:
            joint_names = [f"joint_{i}" for i in range(self.robot.num_joints)]
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().tolist()
        robot_name = getattr(self.robot, "prim_path", "robot")
        print(f"[DEBUG] {robot_name} joint positions:", dict(zip(joint_names, joint_pos)))
        self._debug_printed = True

    def _debug_check_tensor(self, name: str, tensor: torch.Tensor, flag_attr: str | None = None) -> None:
        if not self.cfg.debug_nan_checks:
            return
        if name in self._debug_bad_tensors:
            return
        if flag_attr is not None and getattr(self, flag_attr):
            return
        if not torch.isfinite(tensor).all():
            bad = torch.where(~torch.isfinite(tensor))
            idx = torch.stack(bad, dim=-1)
            max_print = int(self.cfg.debug_nan_max_print)
            sample_idx = idx[:max_print].detach().cpu().tolist()
            print(f"[DEBUG] non-finite {name} shape={tuple(tensor.shape)} idx={sample_idx}")
            if flag_attr is not None:
                setattr(self, flag_attr, True)
            self._debug_bad_tensors.add(name)

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

    def _wrap_to_pi(self, a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2 * torch.pi) - torch.pi

    def _quat_to_yaw(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(t0, t1)

    def _quat_from_angle_axis(self, angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        half = 0.5 * angle
        s = torch.sin(half).unsqueeze(-1)
        qw = torch.cos(half).unsqueeze(-1)
        qxyz = axis * s
        return torch.cat([qw, qxyz], dim=-1)

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

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

    def _update_camera_pose(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # midpoint of tips
        tip_pos_w = self.robot.data.body_state_w[:, self.marker_body_ids[:2], 0:3].mean(dim=1)
        tip_quat_w = self.robot.data.body_state_w[:, self.marker_body_ids[0], 3:7]

        offset = torch.tensor(self.cfg.camera_offset_pos, device=self.device, dtype=torch.float32)
        offset = offset.unsqueeze(0).expand(tip_quat_w.shape[0], 3)
        offset_w = torch_utils.quat_apply(tip_quat_w, offset)
        cam_pos = tip_pos_w + offset_w

        # pitch down about +X by camera_pitch_deg
        angle = math.radians(self.cfg.camera_pitch_deg)
        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(tip_quat_w.shape[0], 3)
        q_pitch = self._quat_from_angle_axis(torch.full((tip_quat_w.shape[0],), angle, device=self.device), axis)
        cam_quat = self._quat_mul(tip_quat_w, q_pitch)
        cam_quat = cam_quat / torch.linalg.norm(cam_quat, dim=-1, keepdim=True).clamp_min(1e-8)

        self.camera.set_world_poses(cam_pos[env_ids], cam_quat[env_ids], env_ids=env_ids)
