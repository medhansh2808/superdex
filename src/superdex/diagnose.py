"""Compare checkpoint predictions with MuJoCo Warp on the actor's own actions."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from superdex.manipulation import (
    advance_goal, heading_step, CUBE_ORIENTATION, CUBE_POSITION, CUBE_YAW_RATE, NUM_ACTIONS,
    CubeManipulation, orientation_error,
)
from superdex.grasp import _TIP_NAMES
from superdex.train import ImaginedManipulation, _make_actor, _make_dynamics

HORIZONS = (1, 4, 8, 16, 32)


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
    if env.quarter:
        del groups["reorient"]
        for _ in range(env.hold_steps):
            state = env.step(previous)
            alive &= env.start_stable(state)
    model = env.sim.mj_model
    probe = mujoco.MjData(model)
    tip_ids = [model.geom(name).id for name in _TIP_NAMES]
    cube_id = model.geom("cube/surface").id

    def predicted_support(states):
        """Query geometry at the predicted pose in an independent native simulator."""
        counts = []
        for item in states.cpu().numpy():
            probe.qpos[env._cube_pose] = item[:7]
            probe.qpos[env._joint_pos] = item[7:7 + NUM_ACTIONS]
            mujoco.mj_forward(model, probe)
            contacts = probe.contact.geom[probe.contact.efc_address >= 0]
            pairs = contacts[np.any(contacts == cube_id, axis=1)]
            counts.append(np.isin(tip_ids, pairs).sum())
        return torch.tensor(counts, device=device) >= 2

    # Record the actor's real trajectory once. Restart imagination from measured
    # states every four steps so short horizons sample the approach to drops.
    trajectory = []
    for _ in range(128):
        obs = env.observe(env.rep.represent(state), previous, goals)
        action = env.bound_action(actor(obs))
        action = torch.where(alive[:, None], action, 0.)
        support = (env.fingertip_contacts() >= 2) if env.quarter else alive
        trajectory.append((state, previous, goals, action, alive.clone(), support))
        next_state = env.step(action)
        if env.yaw_reward == "turn":
            goals = advance_goal(goals, heading_step(state, next_state))
        alive &= env.drop_error(next_state) <= 1
        state, previous = next_state, action
    support = (env.fingertip_contacts() >= 2) if env.quarter else alive
    trajectory.append((state, previous, goals, None, alive.clone(), support))

    samples = {h: [] for h in HORIZONS}
    for offset in range(0, len(trajectory) - max(HORIZONS), 4):
        initial, previous, goals, _, eligible, supported_at_start = trajectory[offset]
        if not eligible.any() and offset:
            break
        start = goals.clone()
        start["state"], start["prev_action"] = initial, previous
        imagined.reset(start)
        predicted_alive = eligible.clone()
        real_yaw = torch.zeros_like(direction)
        predicted_yaw = torch.zeros_like(direction)
        measured_support = supported_at_start.clone()
        for step in range(1, max(HORIZONS) + 1):
            action = trajectory[offset + step - 1][3]
            state, _, goals, _, alive, support = trajectory[offset + step]
            imagined.step(action)
            predicted = imagined.state
            predicted_alive &= env.drop_error(predicted) <= 1
            measured_support &= support
            real_yaw += direction * state[:, CUBE_YAW_RATE] * env.dt
            predicted_yaw += direction * predicted[:, CUBE_YAW_RATE] * env.dt
            if step in samples:
                q = state[:, CUBE_ORIENTATION]
                q_hat = predicted[:, CUBE_ORIENTATION]
                samples[step].append({
                    "eligible": eligible, "held": alive.clone(),
                    "predicted_held": predicted_alive.clone(),
                    "started_supported": supported_at_start,
                    "measured_support": measured_support.clone(),
                    "measured_endpoint_support": support,
                    "predicted_endpoint_support": predicted_support(predicted) if env.quarter else predicted_alive.clone(),
                    "maximum_action": action.abs().amax(dim=-1),
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
            support_starts = valid & values["started_supported"]
            if env.quarter and support_starts.any():
                lost = ~values["measured_support"]
                endpoint_lost = ~values["measured_endpoint_support"]
                estimated_lost = ~values["predicted_endpoint_support"]
                report.update(
                    support_lost_fraction=lost[support_starts].float().mean().item(),
                    support_lost_while_model_predicts_held_fraction=(
                        (lost & values["predicted_held"])[support_starts].float().mean().item()
                    ),
                    endpoint_support_false_negative_fraction=(
                        (endpoint_lost & ~estimated_lost)[support_starts].float().mean().item()
                    ),
                    endpoint_support_false_positive_fraction=(
                        (~endpoint_lost & estimated_lost)[support_starts].float().mean().item()
                    ),
                    mean_maximum_action=values["maximum_action"][support_starts].mean().item(),
                )
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
        torch.compiler.set_stance("force_eager")
        torch.set_num_threads(min(4, torch.get_num_threads()))
    diagnose(args.checkpoint, args.envs, args.device)


if __name__ == "__main__":
    main()
