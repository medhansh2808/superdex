"""Retarget v2.0 SO(3) samples into physically checked SuperDex reset poses."""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import mujoco
import numpy as np

from superdex.grasp import _TIP_NAMES, reset_grasp
from superdex.leap import LEAP_JOINTS, grasp_spec

CUBE_MM = 50


def source_samples(archive: Path):
    """Yield intermediate states from successful 50 mm goals in the archive."""
    from scipy.spatial.transform import Rotation

    # Source palm is mounted with Rx(180°). Map rotation (an axial vector)
    # through palm-local Y reflection to our Rx(90°) mounting.
    axial_map = -Rotation.from_euler("x", 90, degrees=True).as_matrix() @ np.diag([1, -1, 1])
    axial_map = axial_map @ Rotation.from_euler("x", -180, degrees=True).as_matrix()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if "/demos_cube5cm/" not in member.name or not member.name.endswith(".npz"):
                continue
            data = tar.extractfile(member)
            with np.load(io.BytesIO(data.read()), allow_pickle=False) as shard:
                alive = shard["alive"] & ~shard["by_student"]
                positions, quats = shard["joint_pos"], shard["cube_quat"]
                goal_ids, goals = shard["goal_id"], shard["goal_quat"]
                for end, world in np.argwhere(alive & shard["success"]):
                    begin = end
                    while begin > 0 and end - begin < 40 and alive[begin - 1, world]:
                        if goal_ids[begin - 1, world] != goal_ids[end, world]:
                            break
                        begin -= 1
                    frames = np.arange(begin, end, 3)
                    if not len(frames):
                        continue
                    relative = Rotation.from_quat(goals[end, world], scalar_first=True) * (
                        Rotation.from_quat(quats[frames, world], scalar_first=True).inv()
                    )
                    angular = np.rad2deg(relative.as_rotvec() @ axial_map.T)
                    for frame, (x, y, remaining) in zip(frames, angular, strict=True):
                        if not (14 <= abs(remaining) <= 65 and np.hypot(x, y) <= 20):
                            continue
                        progress = np.copysign(90 - abs(remaining), remaining)
                        delta = (positions[frame, world] - positions[begin, world]).copy()
                        delta[13] *= -1  # mirrored thumb axial coordinate, used as an IK seed
                        yield progress, delta, member.name, int(frame), int(world)


class GraspFitter:
    """Fit fingertip contact and preload, then test the resulting pose under gravity."""

    def __init__(self):
        self.model = grasp_spec(CUBE_MM).compile()
        self.data = mujoco.MjData(self.model)
        reset_grasp(self.model, self.data)
        self.pose = self.data.qpos.copy()
        self.control = self.data.ctrl.copy()
        self.joints = np.array([self.model.joint(name).qposadr.item() for name in LEAP_JOINTS])
        self.actuators = np.array([self.model.actuator(f"{name}_act").id for name in LEAP_JOINTS])
        self.tip_ids = np.array([self.model.geom(name).id for name in _TIP_NAMES])
        self.cube_id = self.model.geom("cube/surface").id
        start = self.model.joint("cube/free").qposadr.item()
        self.xyz, self.quat = slice(start, start + 3), slice(start + 3, start + 7)
        self.joint_limits = np.array([self.model.joint(name).range for name in LEAP_JOINTS])
        self.control_limits = self.model.actuator_ctrlrange[self.actuators].copy()
        self.control_limits[:, 0] = np.maximum(self.control_limits[:, 0], self.control[self.actuators] - .35)
        self.control_limits[:, 1] = np.minimum(self.control_limits[:, 1], self.control[self.actuators] + .35)

    def distance(self, tip):
        return mujoco.mj_geomDistance(self.model, self.data, self.cube_id, tip, .2, None)

    def supported(self):
        contacts = self.data.contact.geom[self.data.contact.efc_address >= 0]
        cube_contacts = contacts[np.any(contacts == self.cube_id, axis=1)]
        tips = np.isin(self.tip_ids, cube_contacts)
        return bool(tips[-1] and tips[:3].any())

    def fit(self, angle: float, delta: np.ndarray, rng):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self.pose
        data.qpos[self.xyz] += rng.normal(0, .004, 3)
        goal = (Rotation.from_euler("z", angle, degrees=True)
                * Rotation.from_quat(self.pose[self.quat], scalar_first=True))
        data.qpos[self.quat] = goal.as_quat(scalar_first=True)
        data.ctrl[:] = self.control
        for finger, tip in enumerate(self.tip_ids):
            section = slice(4*finger, 4*finger+4)
            joints, actuators = self.joints[section], self.actuators[section]
            anchor = self.pose[joints] + .5*np.clip(delta[section], -.25, .25) + rng.normal(0, .10, 4)
            low, high = self.joint_limits[section].T

            def residual(q):
                data.qpos[joints] = q
                mujoco.mj_fwdPosition(model, data)
                return np.r_[self.distance(tip) + .0006, .002*(q-anchor)]

            fit = least_squares(residual, np.clip(anchor, low, high), bounds=(low, high), max_nfev=24)
            data.qpos[joints] = fit.x
            gradient = fit.jac[0]
            preload = rng.uniform(.03, .16) * gradient / max(np.linalg.norm(gradient), 1e-8)
            data.ctrl[actuators] = np.clip(fit.x-preload, *self.control_limits[section].T)

        mujoco.mj_step(model, data, nstep=550)
        mujoco.mj_forward(model, data)
        pose, control = data.qpos.copy(), data.ctrl.copy()
        if not np.isfinite(pose).all() or np.linalg.norm(pose[self.xyz]-self.pose[self.xyz]) > .025:
            return None
        error = (Rotation.from_quat(pose[self.quat], scalar_first=True) * goal.inv()).magnitude()
        if error > np.deg2rad(10):
            return None

        # Validate the exact reset that will be used in training, with zero velocities.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = pose
        data.ctrl[:] = control
        for _ in range(50):
            mujoco.mj_step(model, data, nstep=10)
            mujoco.mj_forward(model, data)
            if (not self.supported() or np.linalg.norm(data.qpos[self.xyz]-pose[self.xyz]) > .002
                    or abs(data.qpos[self.quat] @ pose[self.quat]) < np.cos(np.deg2rad(1.5))):
                return None
        return pose, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-direction", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=160)
    args = parser.parse_args()
    if min(args.per_direction, args.attempts) < 1:
        parser.error("reset count and fitting attempts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    rng = np.random.default_rng(42)
    samples = list(source_samples(args.archive))
    fitter = GraspFitter()
    accepted = []
    for direction in (1, -1):
        candidates = [sample for sample in samples if np.sign(sample[0]) == direction]
        print(f"{direction:+}: {len(candidates)} candidate source frames", flush=True)
        count = 0
        for index in rng.permutation(len(candidates))[:args.attempts]:
            angle, delta, shard, frame, world = candidates[index]
            result = fitter.fit(angle, delta, rng)
            if result is None:
                continue
            accepted.append((*result, direction, angle, shard, frame, world))
            count += 1
            print(f"{direction:+}: accepted reset {count} at {angle:+.1f}°", flush=True)
            if count == args.per_direction:
                break
        if count == 0:
            raise SystemExit(f"No physically valid resets for {direction:+}; no bank written")

    pose, control, direction, angle, shard, frame, world = zip(*accepted, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, pose=pose, control=control, direction=direction, angle=angle,
                            shard=shard, frame=frame, world=world, cube_mm=CUBE_MM)
    print(f"Saved {len(accepted)} validated resets to {args.output}")


if __name__ == "__main__":
    main()
