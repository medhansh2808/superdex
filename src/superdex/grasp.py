"""Verify a deterministic LEAP cube grasp in MuJoCo and optionally mjlab."""

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from superdex.leap import LEAP_JOINTS, grasp_spec

# Settled pose and its position targets, in LEAP_JOINTS order. The targets
# provide preload at the thumb, index and middle fingers.
_GRASP_JOINTS = (
    0.734278, 0.038642, 0.848978, 0.541187,
    0.867807, -0.039628, 0.508263, 0.599168,
    0.0, 0.010211, 0.000006, 0.0,
    1.241764, 0.995236, 0.662533, 0.601033,
)
_GRASP_TARGETS = (
    0.794940, 0.054711, 0.891253, 0.562656,
    0.922427, -0.046972, 0.542599, 0.614178,
    0.0, 0.0, 0.0, 0.0,
    1.150922, 0.887252, 0.625879, 0.578974,
)
_GRASP_CUBE_POSE = (
    -0.0141737, 0.1091255, 0.2275260,
    0.9705113, -0.1466234, -0.1716994, -0.0844320,
)
_TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
_HOLD_SECONDS = 2.0
_RELEASE_SECONDS = 0.8
_MAX_HOLD_DRIFT = 0.02
_MIN_RELEASE_DROP = 0.08
_TRACE_COLUMNS = (
    "time_s", "cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz",
    *LEAP_JOINTS, *(f"{name}_contact" for name in _TIP_NAMES),
)


def reset_grasp(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Restore the grasp's named joints, targets and free cube pose."""
    joint_addresses = [model.joint(name).qposadr.item() for name in LEAP_JOINTS]
    actuator_ids = [model.actuator(f"{name}_act").id for name in LEAP_JOINTS]
    cube_address = model.joint("cube/free").qposadr.item()

    mujoco.mj_resetData(model, data)
    data.qpos[joint_addresses] = _GRASP_JOINTS
    data.ctrl[actuator_ids] = _GRASP_TARGETS
    data.qpos[cube_address : cube_address + 7] = _GRASP_CUBE_POSE
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)


def _rollout(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, int]:
    reset_grasp(model, data)
    joint_addresses = [model.joint(name).qposadr.item() for name in LEAP_JOINTS]
    cube_address = model.joint("cube/free").qposadr.item()
    cube_geom = model.geom("cube/surface").id
    tip_ids = [model.geom(name).id for name in _TIP_NAMES]

    def sample() -> np.ndarray:
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Non-finite simulation state at {data.time:.3f} s")
        pairs = data.contact.geom[data.contact.efc_address >= 0]
        touching_cube = pairs[np.any(pairs == cube_geom, axis=1)]
        return np.concatenate((
            [data.time],
            data.qpos[cube_address : cube_address + 7],
            data.qpos[joint_addresses],
            np.isin(tip_ids, touching_cube),
        ))

    def advance(seconds: float) -> None:
        steps = round(seconds / model.opt.timestep)
        for start in range(0, steps, 10):
            mujoco.mj_step(model, data, nstep=min(10, steps - start))
            mujoco.mj_forward(model, data)
            rows.append(sample())

    rows = [sample()]
    advance(_HOLD_SECONDS)
    hold_samples = len(rows)
    data.ctrl[:] = 0
    advance(_RELEASE_SECONDS)
    return np.stack(rows), hold_samples


def check_grasp(trace_path: Path | None = None) -> None:
    model = grasp_spec().compile()
    data = mujoco.MjData(model)
    trace, hold_samples = _rollout(model, data)
    repeated, _ = _rollout(model, data)

    hold = trace[:hold_samples]
    last_hold = hold[-1]
    drift = float(np.linalg.norm(hold[:, 1:4] - hold[0, 1:4], axis=1).max())
    drop = float(last_hold[3] - trace[-1, 3])
    supported = (hold[:, -1] == 1) & np.any(hold[:, -4:-1] == 1, axis=1)

    if trace_path is not None:
        with trace_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(_TRACE_COLUMNS)
            writer.writerows(trace)

    if drift > _MAX_HOLD_DRIFT or not np.all(supported[1:]):
        raise RuntimeError(f"Cube grasp was unstable over the hold (drift {drift:.3f} m)")
    if drop < _MIN_RELEASE_DROP or np.any(trace[-1, -4:]):
        raise RuntimeError(f"Cube was not released after opening (drop {drop:.3f} m)")
    np.testing.assert_array_equal(repeated, trace)

    contacts = ", ".join(name for name, flag in zip(_TIP_NAMES, last_hold[-4:]) if flag)
    print(
        f"LEAP grasp held {_HOLD_SECONDS:.1f} s on {contacts} "
        f"with {drift * 1000:.2f} mm max drift. "
        f"Opening dropped the cube {drop:.3f} m. Replay matched exactly."
    )


def check_mjlab_grasp(device: str, num_envs: int) -> None:
    """Verify that batched worlds can hold and release the same grasp independently."""
    import torch
    from mjlab.sim import Simulation

    from superdex.mjlab_scene import leap_simulation_cfg

    if num_envs < 2:
        raise ValueError("The independent release check requires at least two worlds")

    model = grasp_spec().compile()
    data = mujoco.MjData(model)
    reset_grasp(model, data)
    sim = Simulation(num_envs=num_envs, cfg=leap_simulation_cfg(), model=model, device=device)
    sim.reset()

    qpos = sim.data.qpos[:]
    qvel = sim.data.qvel[:]
    ctrl = sim.data.ctrl[:]
    qpos[:] = torch.as_tensor(data.qpos, dtype=qpos.dtype, device=device)
    qvel[:] = 0
    ctrl[:] = torch.as_tensor(data.ctrl, dtype=ctrl.dtype, device=device)
    sim.forward()

    cube_address = model.joint("cube/free").qposadr.item()
    position = qpos[:, cube_address : cube_address + 3]
    origin = position.clone()

    def advance(seconds: float) -> torch.Tensor:
        steps = round(seconds / model.opt.timestep)
        samples = []
        for step in range(1, steps + 1):
            sim.step()
            if step % 10 == 0 or step == steps:
                samples.append(position.clone())
        positions = torch.stack(samples)
        if not torch.isfinite(positions).all() or not torch.isfinite(qvel).all():
            raise RuntimeError(f"Non-finite mjlab state on {device}")
        return positions

    held = advance(_HOLD_SECONDS)
    drift = torch.linalg.vector_norm(held - origin, dim=-1).max().item()
    if drift > _MAX_HOLD_DRIFT:
        raise RuntimeError(f"mjlab grasp was unstable on {device} (drift {drift:.3f} m)")

    ctrl[::2] = 0
    first_release = advance(_RELEASE_SECONDS)
    drop = (held[-1, ::2, 2] - first_release[-1, ::2, 2]).min().item()
    retained = torch.linalg.vector_norm(first_release[:, 1::2] - origin[1::2], dim=-1)
    if drop < _MIN_RELEASE_DROP or retained.max().item() > _MAX_HOLD_DRIFT:
        raise RuntimeError(f"mjlab independent release failed on {device} (drop {drop:.3f} m)")

    ctrl[1::2] = 0
    second_release = advance(_RELEASE_SECONDS)
    remaining_drop = (first_release[-1, 1::2, 2] - second_release[-1, 1::2, 2]).min().item()
    if remaining_drop < _MIN_RELEASE_DROP:
        raise RuntimeError(f"mjlab remaining worlds failed to release on {device}")

    print(
        f"mjlab {device}: {num_envs} independent grasp worlds, "
        f"{drift * 1000:.2f} mm max hold drift, "
        f"{min(drop, remaining_drop):.3f} m minimum release drop"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="Write joint, cube and contact samples to CSV")
    parser.add_argument("--mjlab-device", help="Also check batched grasps on cpu or CUDA")
    parser.add_argument("--mjlab-envs", type=int, default=2, help="Number of mjlab worlds")
    args = parser.parse_args()
    check_grasp(args.trace)
    if args.mjlab_device:
        check_mjlab_grasp(args.mjlab_device, args.mjlab_envs)


if __name__ == "__main__":
    main()
