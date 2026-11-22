"""Compare checkpoint predictions with MuJoCo Warp on the actor's own actions."""

import argparse
import json
from pathlib import Path

import torch

from superdex.manipulation import (
    advance_goal, heading_step, CUBE_ORIENTATION, CUBE_POSITION, CUBE_YAW_RATE, NUM_ACTIONS,
    CubeManipulation, observations, orientation_error,
)
from superdex.train import ImaginedManipulation, _make_actor, _make_dynamics

HORIZONS = (1, 4, 8, 32)


@torch.no_grad()
def diagnose(checkpoint: Path, num_envs: int, device: str) -> None:
    weights = torch.load(checkpoint, weights_only=True, map_location=device)
    env = CubeManipulation(num_envs, device, cube_mm=weights.get("cube_mm", 36), yaw_reward=weights.get("yaw_reward", "clipped"))
    actor = _make_actor(env).eval()
    actor.load_state_dict(weights["actor"])
    dynamics = _make_dynamics(env).eval()
    dynamics.load_state_dict(weights["dynamics"])
    imagined = ImaginedManipulation(env, dynamics)
    state = env.reset()
    goals = env.goals(env.ids, evaluation=True)
    previous = state.new_zeros((env.num_envs, NUM_ACTIONS))
    alive = torch.ones_like(env.ids, dtype=torch.bool)
    direction = goals["yaw"].squeeze(-1).sign()
    groups = {"positive_yaw": direction > 0, "negative_yaw": direction < 0,
              "reorient": goals["task_id"] == 1}
    samples = {h: [] for h in HORIZONS}
    for _ in range(4):
        # Restart imagination from a measured state every 32 control steps.
        start = goals.clone()
        start["state"], start["prev_action"] = state, previous
        imagined.reset(start)
        eligible = alive.clone()
        predicted_alive = eligible.clone()
        real_yaw = torch.zeros_like(direction)
        predicted_yaw = torch.zeros_like(direction)
        for step in range(1, max(HORIZONS) + 1):
            obs = observations(env.rep.represent(state), previous, goals)
            action = env.bound_action(actor(obs))
            prior = state
            state = env.step(action)
            if env.yaw_reward == "turn":
                goals = advance_goal(goals, heading_step(prior, state))
            imagined.step(action)
            predicted = imagined.state
            previous = action
            alive &= env.drop_error(state) <= 1
            predicted_alive &= env.drop_error(predicted) <= 1
            real_yaw += direction * state[:, CUBE_YAW_RATE] * env.dt
            predicted_yaw += direction * predicted[:, CUBE_YAW_RATE] * env.dt
            if step in samples:
                q = state[:, CUBE_ORIENTATION]
                q_hat = predicted[:, CUBE_ORIENTATION]
                samples[step].append({
                    "eligible": eligible, "held": alive.clone(),
                    "predicted_held": predicted_alive.clone(),
                    "pose_error_deg": torch.rad2deg(orientation_error(q, q_hat)),
                    "position_error_mm": 1000 * (
                        state[:, CUBE_POSITION] - predicted[:, CUBE_POSITION]
                    ).norm(dim=-1),
                    "real_goal_error_deg": torch.rad2deg(orientation_error(q, goals["reorient"])),
                    "model_goal_error_deg": torch.rad2deg(orientation_error(q_hat, goals["reorient"])),
                    "real_signed_rad_s": real_yaw / (step * env.dt),
                    "model_signed_rad_s": predicted_yaw / (step * env.dt),
                })

    for horizon, rows in samples.items():
        values = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        for name, group in groups.items():
            valid = values["eligible"] & group
            held = valid & values["held"]
            report = {"skill": name, "horizon": horizon, "starts": valid.sum().item(),
                      "held_samples": held.sum().item()}
            if valid.any():
                for key in ("held", "predicted_held"):
                    report[key + "_fraction"] = values[key][valid].float().mean().item()
            if held.any():
                task_fields = (("real_goal_error_deg", "model_goal_error_deg")
                               if name == "reorient" else ("real_signed_rad_s", "model_signed_rad_s"))
                for key in ("pose_error_deg", "position_error_mm", *task_fields):
                    report[key] = round(values[key][held].mean().item(), 4)
            print(json.dumps(report, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", default=32, type=int)
    args = parser.parse_args()
    if args.envs < 4 or args.envs % 4:
        parser.error("--envs must be a positive multiple of four")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    diagnose(args.checkpoint, args.envs, args.device)


if __name__ == "__main__":
    main()
