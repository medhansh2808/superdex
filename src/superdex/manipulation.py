"""LEAP cube manipulation in MuJoCo Warp, with two goal-conditioned rewards."""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.sim import Simulation
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul
from supermimic.representations import GenCoordRepresentation
from supermimic.tasks.task_spec import build_actor_obs_td
from tensordict import TensorDict

from superdex.grasp import reset_grasp
from superdex.leap import LEAP_JOINTS, grasp_spec
from superdex.mjlab_scene import leap_simulation_cfg

NUM_ACTIONS = len(LEAP_JOINTS)
TASKS = ("yaw", "reorient")
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


class HandCubeRepresentation(GenCoordRepresentation):
    """Treat the cube as the free body in SuperMimic's integrated representation."""

    def __init__(self, joint_targets: torch.Tensor, dt: float) -> None:
        super().__init__(NUM_ACTIONS, joint_targets)
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
        return super().loss_blocks(rep_state[..., : self._base_dim], aux_state)


class CubeManipulation:
    """Batched LEAP worlds with privileged object pose and velocity observations."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.num_envs = num_envs
        self.device = device
        model = grasp_spec().compile()
        fixture = mujoco.MjData(model)
        reset_grasp(model, fixture)
        self.sim = Simulation(
            num_envs=num_envs, cfg=leap_simulation_cfg(), model=model, device=device
        )
        self.dt = model.opt.timestep * DECIMATION
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
        self.rep = HandCubeRepresentation(self.action_offset, self.dt)
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
        yaw = torch.where(task_id == 0, direction * speed, 0.0)
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
        signed_rate = (
            next_state[..., CUBE_YAW_RATE] * command.sign() / command.abs().clamp_min(0.75)
        )
        spin = signed_rate.clamp(-1, 1) - 0.1 * (signed_rate - 1).abs().clamp(max=2)
        old_error = orientation_error(state[..., CUBE_ORIENTATION], goals["reorient"])
        new_error = orientation_error(next_state[..., CUBE_ORIENTATION], goals["reorient"])
        progress = ((old_error - new_error) / (2 * self.dt)).clamp(-1, 1)
        reorient = progress + 0.1 * torch.exp(-new_error / math.radians(15))
        skill = torch.where(goals["task_id"] == 0, spin, reorient)

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
    good, _ = env.score(state, yaw_only, actions, actions, goals, imagined=False)
    yaw_only[:, CUBE_YAW_RATE].neg_()
    bad, _ = env.score(state, yaw_only, actions, actions, goals, imagined=False)
    spin = goals["task_id"] == 0
    if not torch.all(good[spin] > bad[spin]):
        raise RuntimeError("Yaw reward must prefer the commanded direction")

    on_target = state.clone()
    on_target[:, CUBE_ORIENTATION] = goals["reorient"]
    arrived, _ = env.score(state, on_target, actions, actions, goals, imagined=False)
    idle, _ = env.score(state, state, actions, actions, goals, imagined=False)
    if not torch.all(arrived[~spin] > idle[~spin]):
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
