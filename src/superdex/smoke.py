"""Bounded MuJoCo and optional mjlab smoke checks for the LEAP model."""

import argparse

import mujoco
import numpy as np

from superdex.leap import LEAP_JOINTS, calibration_spec


def _check_actuators(model: mujoco.MjModel, prefix: str = "") -> None:
    actuator_names = tuple(model.actuator(i).name for i in range(model.nu))
    expected_names = tuple(f"{prefix}{joint}_act" for joint in LEAP_JOINTS)
    if actuator_names != expected_names:
        raise RuntimeError(f"Unexpected actuator order: {actuator_names}")

    joint_ids = [model.joint(f"{prefix}{name}").id for name in LEAP_JOINTS]
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], joint_ids)
    np.testing.assert_array_equal(model.actuator_ctrllimited, 1)
    np.testing.assert_allclose(model.actuator_ctrlrange, model.jnt_range[joint_ids])


def check_cpu() -> None:
    model = calibration_spec().compile()
    data = mujoco.MjData(model)
    _check_actuators(model)
    for actuator_id, name in enumerate(LEAP_JOINTS):
        dof_id = model.joint(name).dofadr.item()
        for direction in (-1, 1):
            mujoco.mj_resetData(model, data)
            data.ctrl[actuator_id] = direction * 0.1
            mujoco.mj_step(model, data)
            if not direction * data.qvel[dof_id] > 0:
                raise RuntimeError(
                    f"{name} did not move in the commanded direction {direction}"
                )

    mujoco.mj_resetData(model, data)
    initial_qpos = data.qpos.copy()
    cube_geom = model.geom("cube/surface").id
    cube_qpos = model.joint("cube/free").qposadr.item()
    first_contact = None
    for step in range(1, 151):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Non-finite simulation state on step {step}")
        np.testing.assert_allclose(
            np.linalg.norm(data.qpos[cube_qpos + 3 : cube_qpos + 7]), 1
        )
        if first_contact is None and np.any(data.contact.geom == cube_geom):
            first_contact = step

    if first_contact is None:
        raise RuntimeError("The cube did not contact the hand")
    mujoco.mj_resetData(model, data)
    np.testing.assert_array_equal(data.qpos, initial_qpos)
    np.testing.assert_array_equal(data.qvel, 0)
    print(
        f"MuJoCo {mujoco.__version__}: 16 bidirectional actuators, cube contact at step "
        f"{first_contact}, 150 finite steps, deterministic reset"
    )


def check_mjlab(device: str) -> None:
    import torch
    from mjlab.scene import Scene
    from mjlab.sim import Simulation

    from superdex.mjlab_scene import calibration_scene_cfg, leap_simulation_cfg

    scene = Scene(calibration_scene_cfg(num_envs=2), device=device)
    model = scene.compile()
    _check_actuators(model, prefix="hand/")
    sim = Simulation(
        num_envs=scene.num_envs, cfg=leap_simulation_cfg(), model=model, device=device
    )
    scene.initialize(model, sim.model, sim.data)
    sim.reset()
    initial_qpos = sim.data.qpos[:].clone()

    positive = torch.linspace(0.05, 0.12, len(LEAP_JOINTS), device=device)
    targets = torch.stack((positive, -positive))
    scene["hand"].set_joint_position_target(targets)
    scene.write_data_to_sim()
    torch.testing.assert_close(sim.data.ctrl[:], targets)

    cube_qpos = model.joint("cube/free").qposadr.item()
    for _ in range(32):
        sim.step()
        qpos = sim.data.qpos[:]
        if not torch.isfinite(qpos).all() or not torch.isfinite(sim.data.qvel[:]).all():
            raise RuntimeError("Non-finite mjlab simulation state")
        quat_norms = torch.linalg.vector_norm(
            qpos[:, cube_qpos + 3 : cube_qpos + 7], dim=1
        )
        torch.testing.assert_close(quat_norms, torch.ones_like(quat_norms))

    if torch.allclose(sim.data.qpos[0, :cube_qpos], sim.data.qpos[1, :cube_qpos]):
        raise RuntimeError("The two hands did not respond independently")
    sim.reset()
    torch.testing.assert_close(sim.data.qpos[:], initial_qpos)
    print(
        f"mjlab {device}: 2 independent worlds, 16 mapped actuators, 32 steps, deterministic reset"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjlab-device", help="Also step mjlab on cpu or an explicit CUDA device"
    )
    args = parser.parse_args()
    check_cpu()
    if args.mjlab_device:
        check_mjlab(args.mjlab_device)


if __name__ == "__main__":
    main()
