"""Train and evaluate one SuperMimic actor on both LEAP manipulation tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.tasks.task_spec import TaskSpec
from tensordict import TensorDict

from superdex.manipulation import (
    ACTION_SCALE,
    CUBE_ORIENTATION,
    CUBE_POSITION,
    CUBE_YAW_RATE,
    NUM_ACTIONS,
    TASKS,
    CubeManipulation,
    observations,
    orientation_error,
    verify_task_rewards,
)

GOAL_KEYS = ("task_id", *TASKS)
MODEL_HORIZON = 8
REORIENTATION_TOLERANCE = math.radians(15)
EPISODE_SECONDS = 3


class JointReplay:
    """One replay interface with a partition per contiguous task block."""

    def __init__(self, env: CubeManipulation, capacity: int) -> None:
        self._slots = env.num_envs // 2
        schema = {
            "state": FieldSpec(env.rep.full_state_dim),
            "action": FieldSpec(NUM_ACTIONS),
            "prev_action": FieldSpec(NUM_ACTIONS),
            "task_id": FieldSpec(dtype=torch.long),
            "yaw": FieldSpec(1),
            "reorient": FieldSpec(4),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        }
        self._blocks = [
            ReplayBuffer(schema, capacity, self._slots, env.device) for _ in TASKS
        ]

    def insert(self, transitions: TensorDict) -> None:
        for block, replay in enumerate(self._blocks):
            start = block * self._slots
            replay.insert(transitions[start : start + self._slots])

    def sample(self, length: int, batch_size: int) -> TensorDict:
        return torch.cat(
            [replay.sample(length, batch_size // 2) for replay in self._blocks]
        )


class Collector:
    def __init__(self, env: CubeManipulation) -> None:
        self.env = env
        self.state = env.reset()
        self.goals = env.goals(env.ids)
        self.prev_action = self.state.new_zeros((env.num_envs, NUM_ACTIONS))
        self.age = torch.zeros_like(env.ids)
        self.episode_steps = round(EPISODE_SECONDS / env.dt)

    @torch.no_grad()
    def collect(
        self, steps: int, replay: JointReplay, actor: MultiEncoderAE | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actor is not None:
            actor.eval()
        env = self.env
        states = []
        actions = []
        for _ in range(steps):
            noise = 0.2 * torch.randn_like(self.prev_action)
            if actor is None:
                action = 0.75 * self.prev_action + noise
            else:
                obs = observations(env.rep.represent(self.state), self.prev_action, self.goals)
                action = actor(obs) + noise
            action = env.bound_action(action)
            states.append(self.state)
            actions.append(action.clone())
            next_state = env.step(action)
            self.age += 1
            terminated = env.drop_error(next_state) > 1
            truncated = self.age >= self.episode_steps
            replay.insert(
                TensorDict(
                    {
                        **{key: self.goals[key] for key in GOAL_KEYS},
                        "state": self.state,
                        "action": action,
                        "prev_action": self.prev_action,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                    batch_size=[env.num_envs],
                    device=env.device,
                )
            )

            self.state, self.prev_action = next_state, action
            ended = (terminated | truncated).nonzero(as_tuple=True)[0]
            if ended.numel():
                self.state = env.reset(ended)
                self.goals[ended] = env.goals(ended)
                self.age[ended] = 0
                self.prev_action[ended] = 0
        return torch.stack(states), torch.stack(actions)


class ImaginedManipulation:
    """LEAP adapter to the official actor-critic learner's rollout interface."""

    def __init__(self, env: CubeManipulation, dynamics: Dynamics) -> None:
        self.env = env
        self.dynamics = dynamics
        self.representation = env.rep
        self.max_step_reward = 1.15 * env.dt

    def reset(self, start: TensorDict) -> TensorDict:
        self.state = start["state"]
        self.aux = self.representation.extract_aux_state(self.state)
        self.rep_state = self.representation.represent(self.state)
        self.prev_action = start["prev_action"]
        self.goals = start.select(*GOAL_KEYS)
        self.context = self.state.new_empty((len(start), 0))
        return observations(self.rep_state, self.prev_action, self.goals)

    def step(self, action: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
        action = self.env.bound_action(action)
        self.aux, self.rep_state = self.dynamics(
            self.rep_state, action, self.aux, self.context
        )
        next_state = self.representation.reconstruct_full_state(self.aux, self.rep_state)
        reward, done = self.env.score(
            self.state, next_state, action, self.prev_action, self.goals, imagined=True
        )
        self.state, self.prev_action = next_state, action
        return observations(self.rep_state, action, self.goals), reward, done


def _make_actor(env: CubeManipulation) -> MultiEncoderAE:
    def use_live_goals(*_):
        raise RuntimeError("LEAP goals come from the manipulation environment")

    specs = [TaskSpec(name, dim, use_live_goals) for name, dim in zip(TASKS, (1, 4), strict=True)]
    actor = MultiEncoderAE(
        obs_dim=env.rep.rep_state_dim + NUM_ACTIONS,
        action_dim=NUM_ACTIONS,
        task_specs=specs,
        slot_mix=dict.fromkeys(TASKS, 0.5),
        latent_dim=32,
        hidden_dims=(2048, 1024, 512, 256),
        latent_bc_weight=0.1,
        history_length=1,
    ).to(env.device)
    # Small initial offsets keep the hand close to its calibrated grasp.
    with torch.no_grad():
        actor.decoder[-1].weight.mul_(0.05)
        actor.decoder[-1].bias.zero_()
    return actor


def _make_dynamics(env: CubeManipulation) -> Dynamics:
    return Dynamics(
        representation=env.rep,
        num_actions=NUM_ACTIONS,
        sim_ctx_dim=0,
        action_scale=torch.full_like(env.action_offset, ACTION_SCALE),
        action_offset=env.action_offset,
        hidden_dims=(2048, 2048, 2048),
    ).to(env.device)


def _with_context(window: TensorDict) -> TensorDict:
    window["sim_ctx"] = window["state"].new_empty((*window.batch_size, 0))
    return window


def _log(label: str, data: dict) -> None:
    print(f"{label} {json.dumps(data, sort_keys=True, allow_nan=False)}", flush=True)


@torch.no_grad()
def _world_model_errors(env: CubeManipulation, dynamics: Dynamics, window: TensorDict) -> dict:
    states, actions = window["state"], window["action"]
    aux = env.rep.extract_aux_state(states[:, 0])
    rep = env.rep.represent(states[:, 0])
    context = states.new_empty((len(window), 0))
    errors = {}
    for i in range(MODEL_HORIZON):
        aux, rep = dynamics(rep, actions[:, i], aux, context)
        horizon = i + 1
        if horizon in (1, 4, MODEL_HORIZON):
            predicted = env.rep.reconstruct_full_state(aux, rep)
            actual = states[:, horizon]
            displacement = predicted[:, CUBE_POSITION] - actual[:, CUBE_POSITION]
            angle = orientation_error(predicted[:, CUBE_ORIENTATION], actual[:, CUBE_ORIENTATION])
            for task_id, name in enumerate(TASKS):
                slots = window["task_id"][:, 0] == task_id
                errors[f"{name}/{horizon}"] = {
                    "position_mm": (1000 * displacement[slots].norm(dim=-1).mean()).item(),
                    "orientation_deg": torch.rad2deg(angle[slots].mean()).item(),
                    "joint_rmse_rad": (predicted[slots, 7:7 + NUM_ACTIONS] -
                                       actual[slots, 7:7 + NUM_ACTIONS]).square().mean().sqrt().item(),
                    "yaw_rate_mae_rad_s": (
                        predicted[slots, CUBE_YAW_RATE] - actual[slots, CUBE_YAW_RATE]
                    ).abs().mean().item(),
                }
    return errors


def _verify_bptt(imagination: ImaginedManipulation, starts: TensorDict) -> dict[str, float]:
    imagination.dynamics.requires_grad_(False)
    try:
        imagination.reset(starts)
        first = torch.zeros_like(starts["action"], requires_grad=True)
        imagination.step(first)
        imagination.step(torch.zeros_like(first))
        _, later_reward, _ = imagination.step(torch.zeros_like(first))
        gradient, = torch.autograd.grad(later_reward.sum(), first)
    finally:
        imagination.dynamics.requires_grad_(True)
    result = {}
    for task_id, name in enumerate(TASKS):
        norm = gradient[starts["task_id"] == task_id].norm().item()
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError(f"Later rewards supplied no finite BPTT gradient for {name}")
        result[name] = norm
    return result


@torch.no_grad()
def evaluate(env: CubeManipulation, actor: MultiEncoderAE, steps: int) -> dict:
    actor.eval()
    state = env.reset()
    goals = env.goals(env.ids, evaluation=True)
    prev_action = state.new_zeros((env.num_envs, NUM_ACTIONS))
    alive = torch.ones_like(env.ids, dtype=torch.bool)
    direction = goals["yaw"].squeeze(-1).sign()
    yaw = state.new_zeros(env.num_envs)
    survived_steps = torch.zeros_like(env.ids)
    hold = torch.zeros_like(env.ids)
    hold_steps = math.ceil(0.2 / env.dt)
    success_at = torch.zeros_like(env.ids)
    error = orientation_error(state[:, CUBE_ORIENTATION], goals["reorient"])
    for step in range(1, steps + 1):
        obs = observations(env.rep.represent(state), prev_action, goals)
        action = env.bound_action(actor(obs))
        state = env.step(torch.where(alive[:, None], action, 0.0))
        alive = alive & (env.drop_error(state) <= 1)
        survived_steps += alive
        yaw += torch.where(alive, direction * state[:, CUBE_YAW_RATE] * env.dt, 0.0)
        current_error = orientation_error(state[:, CUBE_ORIENTATION], goals["reorient"])
        error = torch.where(alive, current_error, error)
        holding = alive & (current_error <= REORIENTATION_TOLERANCE)
        hold = torch.where(holding, hold + 1, 0)
        success_at = torch.where((success_at == 0) & (hold >= hold_steps), step, success_at)
        prev_action = action

    spin = goals["task_id"] == 0
    tilt = ~spin
    successes = tilt & (success_at > 0) & alive
    return {
        "yaw": {
            "signed_rotation_deg": torch.rad2deg(yaw[spin].mean()).item(),
            "signed_rad_s_while_held": (yaw[spin].sum() /
                                        (env.dt * survived_steps[spin].sum().clamp_min(1))).item(),
            "drop_rate": (~alive[spin]).float().mean().item(),
        },
        "reorient": {
            "success_rate": successes.sum().item() / tilt.sum().item(),
            "median_final_held_error_deg": torch.rad2deg(error[tilt].median()).item(),
            "mean_time_to_success_s": (
                env.dt * success_at[successes].float().mean().item() if successes.any() else None
            ),
            "drop_rate": (~alive[tilt]).float().mean().item(),
        },
    }


def train(args: argparse.Namespace, env: CubeManipulation) -> None:
    smoke = args.smoke
    batch_size = 16 if smoke else 256
    collect_steps = 8 if smoke else 32
    warmup_steps = 24 if smoke else 64
    eval_steps = args.eval_steps or (20 if smoke else 150)
    iterations = args.iterations or (2 if smoke else 100)
    verify_task_rewards(env)
    print("Yaw, reorientation, and cube-drop reward checks passed.", flush=True)

    torch.manual_seed(args.seed + 1)
    held_out = JointReplay(env, warmup_steps)
    Collector(env).collect(warmup_steps, held_out)
    validation = held_out.sample(MODEL_HORIZON + 1, batch_size)
    torch.manual_seed(args.seed)

    actor = _make_actor(env)
    dynamics = _make_dynamics(env)
    critic = build_mlp(env.rep.rep_state_dim + NUM_ACTIONS + 5, (256, 128), 1).to(env.device)
    world_cfg = WorldModelLearnerCfg(
        horizon=MODEL_HORIZON, batch_size=batch_size, num_batches=1 if smoke else 10
    )
    actor_horizon = args.actor_horizon or (4 if smoke else 32)
    actor_cfg = ActorCriticPolicyLearnerCfg(batch_size=batch_size, horizon=actor_horizon)
    _log("rollout_horizons", {"actor": actor_horizon, "world_model": MODEL_HORIZON})
    world_learner = WorldModelLearner(dynamics, env.rep, world_cfg, device=env.device)
    imagined = ImaginedManipulation(env, dynamics)
    actor_learner = ActorCriticPolicyLearner(actor, critic, imagined, actor_cfg, device=env.device)
    _log("untrained_policy", evaluate(env, actor, eval_steps))

    replay = JointReplay(env, 256 if smoke else 1024)
    collector = Collector(env)
    states, actions = collector.collect(warmup_steps, replay)
    world_learner.update_normalization(states, actions)
    _log("world_model_heldout_before", _world_model_errors(env, dynamics, validation))

    for i in range(1, iterations + 1):
        windows = (
            _with_context(replay.sample(MODEL_HORIZON + 1, batch_size))
            for _ in range(world_cfg.num_batches)
        )
        model_stats = world_learner.update(windows)
        starts = replay.sample(1, batch_size)[:, 0]
        if i == 1:
            _log("later_reward_gradient", _verify_bptt(imagined, starts))

        with torch.no_grad():
            actor.train()
            actor.update_normalization(
                observations(env.rep.represent(starts["state"]), starts["prev_action"], starts)
            )
        policy_stats = actor_learner.update([starts])
        states, actions = collector.collect(collect_steps, replay, actor)
        world_learner.update_normalization(states, actions)
        stats = {k: v.item() for k, v in (model_stats | policy_stats).items()}
        _log(f"iteration {i}/{iterations}", stats)

    _log("world_model_heldout_after", _world_model_errors(env, dynamics, validation))
    _log("trained_policy", evaluate(env, actor, eval_steps))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as file:
            torch.save({"actor": actor.state_dict(), "dynamics": dynamics.state_dict(),
                        "critic": critic.state_dict(), "seed": args.seed, "actor_horizon": actor_horizon}, file)
        print(f"Saved {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--actor-horizon", type=int, help="Imagined control steps per actor update")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run a bounded end-to-end check")
    parser.add_argument("--output", type=Path, help="Write a new checkpoint at this path")
    parser.add_argument("--evaluate", type=Path, help="Evaluate one checkpoint on both tasks")
    args = parser.parse_args()
    if args.envs is None:
        args.envs = 4 if args.smoke else 64
    if args.envs < 4 or args.envs % 4:
        parser.error("--envs must be a positive multiple of four")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.actor_horizon is not None and args.actor_horizon < 1:
        parser.error("--actor-horizon must be positive")
    if args.eval_steps is not None and args.eval_steps < 1:
        parser.error("--eval-steps must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"checkpoint already exists: {args.output}")

    if args.smoke or args.device == "cpu":
        torch.compiler.set_stance("force_eager")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    env = CubeManipulation(args.envs, args.device)
    print(f"{args.device}: {args.envs // 2} worlds per skill, {env.dt:.3f} s control step", flush=True)
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location=args.device, weights_only=True)
        actor = _make_actor(env)
        actor.load_state_dict(checkpoint["actor"])
        _log("checkpoint_policy", evaluate(env, actor, args.eval_steps or 150))
    else:
        train(args, env)


if __name__ == "__main__":
    main()
