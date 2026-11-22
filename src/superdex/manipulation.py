"""LEAP cube manipulation in MuJoCo Warp, with two goal-conditioned rewards."""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.sim import Simulation
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from supermimic.representations import GenCoordRepresentation
from supermimic.tasks.task_spec import build_actor_obs_td
from tensordict import TensorDict

from superdex.grasp import reset_grasp
from superdex.leap import LEAP_JOINTS, grasp_spec
from superdex.mjlab_scene import leap_simulation_cfg

NUM_ACTIONS = len(LEAP_JOINTS)
TASKS = ("yaw", "reorient")
QUARTER_TASKS = ("positive", "negative")
HOLD_SECONDS = 1.0
QUARTER_CONTROL_SECONDS = 4.0
DECIMATION = 10
ACTION_SCALE = 0.35
CUBE_POSITION = slice(0, 3)
CUBE_ORIENTATION = slice(3, 7)
CUBE_ANGULAR_VELOCITY = slice(10 + NUM_ACTIONS, 13 + NUM_ACTIONS)
CUBE_YAW_RATE = CUBE_ANGULAR_VELOCITY.stop - 1


def orientation_error(quat: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Geodesic angular distance for scalar-first, unit quaternions."""
    delta = quat_mul(goal, quat_conjugate(quat))
    sine = torch.linalg.vector_norm(delta[..., 1:], dim=-1)
    return 2 * torch.atan2(sine, delta[..., 0].abs())


def signed_yaw_rate(state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    return state[..., CUBE_YAW_RATE] * command.sign() / command.abs().clamp_min(0.75)


def front_normal(quat: torch.Tensor) -> torch.Tensor:
    """World normal of side 1 (local +Y), for scalar-first unit quaternions."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), dim=-1)


def heading_step(state, next_state):
    a = front_normal(state[..., CUBE_ORIENTATION])
    b = front_normal(next_state[..., CUBE_ORIENTATION])
    cross = a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    dot = (a[..., :2]*b[..., :2]).sum(-1)
    valid = cross.square() + dot.square() > 1e-6
    return torch.atan2(torch.where(valid, cross, 0.), torch.where(valid, dot, 1.))


def advance_goal(goals, delta):
    following = goals.clone(False)
    following["yaw"] = goals["yaw"] - torch.where(
        goals["task_id"] == 0, delta / math.tau, 0.
    ).unsqueeze(-1)
    return following


def yawed_pose(origin, angle):
    half = angle / 2
    zero = torch.zeros_like(half)
    rotation = torch.stack((half.cos(), zero, zero, half.sin()), dim=-1)
    return quat_mul(rotation, origin.expand_as(rotation))


class HandCubeRepresentation(GenCoordRepresentation):
    """Treat the cube as the free body in SuperMimic's integrated representation."""

    LOSS_WEIGHT_CONFIG = {
        **GenCoordRepresentation.LOSS_WEIGHT_CONFIG,
        "cube_angular_velocity": "nominal_error_body_avel",
    }

    def __init__(self, joint_targets: torch.Tensor, dt: float, *, angular_velocity_loss: bool = True) -> None:
        super().__init__(NUM_ACTIONS, joint_targets)
        if not angular_velocity_loss:
            self.LOSS_WEIGHT_CONFIG = GenCoordRepresentation.LOSS_WEIGHT_CONFIG
        self.dt = dt
        self.substep_coeff = (DECIMATION + 1) / (2 * DECIMATION)
        self._base_dim = self.rep_state_dim
        # Absolute cube pose matters because the hand is fixed to the world.
        self.rep_state_dim += 7

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        return torch.cat((super().represent(full_state), full_state[..., :7]), dim=-1)

    def apply_prediction(self, aux_state, curr_rep_state, prediction, dt):
        aux, rep = super().apply_prediction(
            aux_state, curr_rep_state[..., : self._base_dim], prediction, dt
        )
        return aux, torch.cat((rep, aux), dim=-1)

    def reconstruct_full_state(self, aux_state, rep_state):
        return super().reconstruct_full_state(aux_state, rep_state[..., : self._base_dim])

    def loss_blocks(self, rep_state, aux_state):
        blocks = super().loss_blocks(rep_state[..., : self._base_dim], aux_state)
        if "cube_angular_velocity" in self.LOSS_WEIGHT_CONFIG:
            full_state = self.reconstruct_full_state(aux_state, rep_state)
            blocks["cube_angular_velocity"] = full_state[..., CUBE_ANGULAR_VELOCITY]
        return blocks


class CubeManipulation:
    """Batched LEAP worlds with privileged object pose and velocity observations."""

    def __init__(
        self, num_envs: int, device: str, *, angular_velocity_loss: bool = True,
        yaw_reward: str = "clipped", cube_mm: int = 36,
    ) -> None:
        self.quarter = yaw_reward == "quarter"
        if self.quarter and cube_mm != 42:
            raise ValueError("The quarter-turn task requires the calibrated 42 mm grasp")
        self.task_names = QUARTER_TASKS if self.quarter else TASKS
        self.cube_mm = cube_mm
        self.yaw_reward = yaw_reward
        self.num_envs = num_envs
        self.device = device
        model = grasp_spec(cube_mm).compile()
        fixture = mujoco.MjData(model)
        reset_grasp(model, fixture)
        self.sim = Simulation(
            num_envs=num_envs, cfg=leap_simulation_cfg(), model=model, device=device
        )
        self.dt = model.opt.timestep * DECIMATION
        self.hold_steps = math.ceil(HOLD_SECONDS / self.dt)
        if self.quarter:
            import warp as wp

            data = self.sim.wp_data
            self._contact_geom = wp.to_torch(data.contact.geom)
            self._contact_world = wp.to_torch(data.contact.worldid)
            self._contact_distance = wp.to_torch(data.contact.dist)
            self._contact_count = wp.to_torch(data.nacon).reshape(-1)[0]
            self._contact_slots = torch.arange(len(self._contact_world), device=device)
            self._cube_geom = model.geom("cube/surface").id
            self._tip_geoms = torch.tensor(
                [model.geom(name).id for name in ("if_tip", "mf_tip", "rf_tip", "th_tip")],
                device=device,
            )

        self._qpos = self.sim.data.qpos[:]
        self._qvel = self.sim.data.qvel[:]
        self._ctrl = self.sim.data.ctrl[:]
        self._reset_pos = torch.as_tensor(fixture.qpos, dtype=self._qpos.dtype, device=device)
        self._reset_ctrl = torch.as_tensor(fixture.ctrl, dtype=self._ctrl.dtype, device=device)
        self._joint_pos = [model.joint(name).qposadr.item() for name in LEAP_JOINTS]
        self._joint_vel = [model.joint(name).dofadr.item() for name in LEAP_JOINTS]
        self._actuators = [model.actuator(f"{name}_act").id for name in LEAP_JOINTS]
        cube_pos = model.joint("cube/free").qposadr.item()
        cube_vel = model.joint("cube/free").dofadr.item()
        self._cube_pose = slice(cube_pos, cube_pos + 7)
        self._cube_linear = slice(cube_vel, cube_vel + 3)
        self._cube_angular = slice(cube_vel + 3, cube_vel + 6)
        self.action_offset = self._reset_ctrl[self._actuators]
        ranges = torch.as_tensor(
            model.actuator_ctrlrange[self._actuators], dtype=self._ctrl.dtype, device=device
        )
        self._min_action, self._max_action = (
            (ranges - self.action_offset[:, None]) / ACTION_SCALE
        ).clamp(-1, 1).unbind(-1)
        self.origin = self._reset_pos[self._cube_pose]
        self.front_view = front_normal(self.origin[CUBE_ORIENTATION]).clone()
        self.front_view[2] = 0
        self.rep = HandCubeRepresentation(
            self.action_offset, self.dt, angular_velocity_loss=angular_velocity_loss
        )
        self.ids = torch.arange(num_envs, device=device)
        self.task_id = (self.ids >= num_envs // 2).long()

    def state(self) -> torch.Tensor:
        pose = self._qpos[:, self._cube_pose]
        # MuJoCo free joints store angular velocity in the moving body frame.
        omega = quat_apply(pose[:, 3:7], self._qvel[:, self._cube_angular])
        return torch.cat(
            (pose, self._qpos[:, self._joint_pos], self._qvel[:, self._cube_linear],
             omega, self._qvel[:, self._joint_vel]),
            dim=-1,
        )

    def reset(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        self.sim.reset(ids)
        ids = slice(None) if ids is None else ids
        self._qpos[ids] = self._reset_pos
        self._qvel[ids] = 0
        self._ctrl[ids] = self._reset_ctrl
        self.sim.forward()
        return self.state()

    def visible_face(self, state):
        view = self.front_view.expand(state.shape[0], -1)
        x, y, z = quat_apply_inverse(state[:, CUBE_ORIENTATION], view).unbind(-1)
        # Face numbers match the dot markers rendered from the fixed front camera.
        return torch.stack((y, x, -y, -x, z.abs()), dim=-1).argmax(dim=-1) + 1

    def fingertip_contacts(self):
        """Count distinct fingertips touching the cube in each Warp world."""
        first, second = self._contact_geom.unbind(-1)
        valid = (self._contact_slots < self._contact_count) & (self._contact_distance <= 0)
        touched = ((first == self._cube_geom)[:, None] & (second[:, None] == self._tip_geoms)
                   | (second == self._cube_geom)[:, None] & (first[:, None] == self._tip_geoms))
        counts = self.ids.new_zeros((self.num_envs, len(self._tip_geoms)))
        worlds = self._contact_world.long().clamp(0, self.num_envs - 1)
        counts.index_add_(0, worlds, (touched & valid[:, None]).long())
        return (counts > 0).sum(dim=1)

    def cube_still(self, state):
        linear = state[:, 7 + NUM_ACTIONS:CUBE_ANGULAR_VELOCITY.start]
        angular = state[:, CUBE_ANGULAR_VELOCITY]
        return (linear.norm(dim=-1) <= .01) & (angular.norm(dim=-1) <= .15)

    def start_stable(self, state):
        drift = (state[:, CUBE_POSITION] - self.origin[CUBE_POSITION]).norm(dim=-1)
        pose = state[:, CUBE_ORIENTATION]
        pose_error = orientation_error(pose, self.origin[CUBE_ORIENTATION].expand_as(pose))
        return ((self.visible_face(state) == 1) & (self.fingertip_contacts() == 4)
                & self.cube_still(state) & (drift <= .002) & (pose_error <= math.radians(3)))

    def observe(self, rep_state, prev_action, goals):
        if not self.quarter:
            return observations(rep_state, prev_action, goals)
        sensor = torch.cat((rep_state, prev_action), dim=-1)
        yaw = goals["yaw"]
        return build_actor_obs_td(
            dict.fromkeys(self.task_names, yaw), sensor.unsqueeze(1),
            torch.cat((sensor, yaw), dim=-1),
        )

    def bound_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action, self._min_action, self._max_action)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        target = self.action_offset + ACTION_SCALE * self.bound_action(action)
        self._ctrl[:, self._actuators] = target
        for _ in range(DECIMATION):
            self.sim.step()
        return self.state()

    def goals(
        self, ids: torch.Tensor, *, evaluation: bool = False
    ) -> TensorDict:
        task_id = self.task_id[ids]
        if self.quarter:
            yaw = torch.where(task_id == 0, .25, -.25)
            attitude = yawed_pose(self.origin[CUBE_ORIENTATION], yaw * math.tau)
            return TensorDict({"task_id": task_id, "yaw": yaw[:, None], "reorient": attitude},
                              batch_size=[ids.numel()], device=self.device)
        direction = torch.where(ids % 2 == 0, 1.0, -1.0)
        if evaluation:
            speed = torch.full_like(direction, 1.0)
            headings = ids * (math.pi * (3 - math.sqrt(5)))
            angles = torch.deg2rad(25.0 + 10.0 * (ids % 3))
        else:
            speed = 0.75 + 0.75 * torch.rand_like(direction)
            headings = torch.rand_like(direction) * (2 * math.pi)
            angles = torch.deg2rad(18.0 + 17.0 * torch.rand_like(direction))
        half = angles / 2
        tilt = torch.stack(
            (half.cos(), headings.cos() * half.sin(), headings.sin() * half.sin(),
             torch.zeros_like(half)),
            dim=-1,
        )
        neutral = self.origin[CUBE_ORIENTATION].expand_as(tilt)
        attitude = torch.where(task_id[:, None] == 1, quat_mul(tilt, neutral), neutral)
        # Reorientation slots use a zero-yaw command for the official latent anchor.
        yaw = torch.where(task_id == 0, direction if self.yaw_reward == "turn" else direction * speed, 0.0)
        return TensorDict(
            {"task_id": task_id, "yaw": yaw[:, None], "reorient": attitude},
            batch_size=[ids.numel()],
            device=self.device,
        )

    def drop_error(self, state: torch.Tensor) -> torch.Tensor:
        delta = state[..., CUBE_POSITION] - self.origin[CUBE_POSITION]
        return torch.maximum(
            torch.linalg.vector_norm(delta, dim=-1) / 0.08, -delta[..., 2] / 0.055
        )

    def score(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        goals: TensorDict,
        *,
        imagined: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        command = goals["yaw"].squeeze(-1)
        signed_rate = signed_yaw_rate(next_state, command)
        if self.quarter:
            target = goals["reorient"]
            old_error = orientation_error(state[..., CUBE_ORIENTATION], target)
            new_error = orientation_error(next_state[..., CUBE_ORIENTATION], target)
            progress = ((old_error - new_error) / self.dt).tanh()
            speed = next_state[..., CUBE_ANGULAR_VELOCITY].norm(dim=-1)
            settled = torch.exp(-new_error / math.radians(20) - speed)
            # Rotation and settling both matter. Penalize moving out of the grasp.
            drift = self.drop_error(next_state)
            spin = .6 * progress + .5 * settled - .4 * drift.square().clamp(max=1)
        elif self.yaw_reward == "turn":
            remaining = command * math.tau
            delta = heading_step(state, next_state)
            after = remaining - delta
            progress = (remaining.abs() - after.abs()) / self.dt
            speed_error = next_state[..., CUBE_YAW_RATE] - remaining.clamp(-1, 1)
            # Modulo one revolution, the desired path pose is independent of direction.
            path_error = orientation_error(next_state[..., CUBE_ORIENTATION],
                                           yawed_pose(self.origin[CUBE_ORIENTATION], -after))
            settled = torch.exp(-after.abs() / math.radians(10)) * torch.exp(-speed_error.abs())
            spin = (0.65 * progress.tanh() + 0.35 * settled
                    - 0.15 * torch.nn.functional.smooth_l1_loss(speed_error, torch.zeros_like(speed_error), reduction="none")
                    - 0.25 * path_error)
        elif self.yaw_reward == "huber":
            # Match the clipped reward at rest (-0.1) and at the target (1).
            error = torch.nn.functional.huber_loss(
                signed_rate, torch.ones_like(signed_rate), reduction="none"
            )
            spin = 1 - 2.2 * error
        else:
            spin = signed_rate.clamp(-1, 1) - 0.1 * (signed_rate - 1).abs().clamp(max=2)
        old_error = orientation_error(state[..., CUBE_ORIENTATION], goals["reorient"])
        new_error = orientation_error(next_state[..., CUBE_ORIENTATION], goals["reorient"])
        progress = ((old_error - new_error) / (2 * self.dt)).clamp(-1, 1)
        reorient = progress + 0.1 * torch.exp(-new_error / math.radians(15))
        skill = spin if self.quarter else torch.where(goals["task_id"] == 0, spin, reorient)

        error = self.drop_error(next_state)
        done = torch.sigmoid((error - 1) / 0.08) if imagined else (error > 1).float()
        smoothness = (action - prev_action).square().mean(dim=-1)
        return self.dt * (skill + 0.05 - 0.025 * smoothness) - done, done


def observations(
    rep_state: torch.Tensor, prev_action: torch.Tensor, goals: TensorDict
) -> TensorDict:
    sensor = torch.cat((rep_state, prev_action), dim=-1)
    critic = torch.cat((sensor, goals["yaw"], goals["reorient"]), dim=-1)
    return build_actor_obs_td(
        {name: goals[name] for name in TASKS}, sensor.unsqueeze(1), critic
    )


def verify_task_rewards(env: CubeManipulation) -> None:
    """Exercise both task slots and drop thresholds on known state transitions."""
    state = env.reset()
    goals = env.goals(env.ids, evaluation=True)
    actions = state.new_zeros((env.num_envs, NUM_ACTIONS))
    yaw_only = state.clone()
    yaw_only[:, CUBE_YAW_RATE] = goals["yaw"].squeeze(-1)
    if env.quarter:
        yaw_only[:, CUBE_ORIENTATION] = yawed_pose(
            env.origin[CUBE_ORIENTATION], goals["yaw"].squeeze(-1).sign() * .02
        )
    good, _ = env.score(state, yaw_only, actions, actions, goals, imagined=False)
    yaw_only[:, CUBE_YAW_RATE].neg_()
    if env.quarter:
        yaw_only[:, CUBE_ORIENTATION] = yawed_pose(
            env.origin[CUBE_ORIENTATION], -goals["yaw"].squeeze(-1).sign() * .02
        )
    bad, _ = env.score(state, yaw_only, actions, actions, goals, imagined=False)
    spin = (goals["task_id"] == 0) | env.quarter
    if not torch.all(good[spin] > bad[spin]):
        raise RuntimeError("Yaw reward must prefer the commanded direction")

    on_target = state.clone()
    on_target[:, CUBE_ORIENTATION] = goals["reorient"]
    arrived, _ = env.score(state, on_target, actions, actions, goals, imagined=False)
    idle, _ = env.score(state, state, actions, actions, goals, imagined=False)
    if not env.quarter and not torch.all(arrived[~spin] > idle[~spin]):
        raise RuntimeError("Reorientation reward must prefer progress to the 3D target")
    torch.testing.assert_close(
        orientation_error(-on_target[:, CUBE_ORIENTATION], goals["reorient"]),
        torch.zeros_like(arrived),
        atol=1e-6,
        rtol=0,
    )

    on_target[:, 2] -= 0.09
    _, actual = env.score(state, on_target, actions, actions, goals, imagined=False)
    _, imagined = env.score(state, on_target, actions, actions, goals, imagined=True)
    if not torch.all((actual == 1) & (imagined > 0.99)):
        raise RuntimeError("Real and imagined rewards must identify a dropped cube")
