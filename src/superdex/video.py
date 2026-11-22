"""Render selected worlds from the checkpoint evaluation at simulation speed."""

import math
import subprocess

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import mujoco
import torch
from PIL import Image, ImageDraw

from superdex.leap import grasp_spec
from superdex.manipulation import front_normal


class EvaluationClips:
    def __init__(self, env):
        self.slots = [0, 1, env.num_envs // 2]
        self.turn = env.yaw_reward == "turn"
        self.front = front_normal(env.origin[3:7]).cpu().numpy()
        self.cube_mm = env.cube_mm
        self.dt = env.dt
        self.frames = []
        self.origin = env.origin[:3].cpu().numpy()

    def __call__(self, env, step, alive, yaw, error):
        slots = self.slots
        qpos = env.sim.data.qpos[:][slots].cpu().numpy()
        # Yaw in evaluate() already includes the command sign.
        degrees = torch.rad2deg(torch.stack([yaw[slots[0]], yaw[slots[1]], error[slots[2]]]))
        self.frames.append((step, qpos, alive[slots].tolist(), degrees.tolist()))

    def write(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        spec = grasp_spec(self.cube_mm)
        normals = np.array([(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)])
        if self.turn:
            body = spec.body("cube/object")
            for face, normal in enumerate(normals, start=1):
                tangent = np.cross(normal, [0, 0, 1])
                for dot in range(face):
                    pos = normal * (self.cube_mm / 2000 + .0002) + tangent * ((dot - (face-1)/2)*.006)
                    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.0025, 0, 0),
                                  pos=pos, mass=0, contype=0, conaffinity=0, rgba=(1, .85, .25, 1))
        model = spec.compile()
        # Brighten the dark hand material on the render-only model.
        model.mat_rgba[model.material("black").id, :3] = (0.43, 0.46, 0.50)
        model.vis.headlight.ambient[:] = 0.55
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.origin
        camera.lookat[1] -= 0.025
        camera.distance = 0.32
        camera.azimuth = -105
        camera.elevation = -32
        if self.turn:
            camera.lookat[:] = self.origin
            camera.azimuth = math.degrees(math.atan2(-self.front[1], -self.front[0]))
            camera.elevation = 0
        width, height = 640, 480
        names = ("yaw-positive", "yaw-negative", "reorient")
        labels = ("Yaw +1 rad/s", "Yaw -1 rad/s", "Reorientation")
        if self.turn:
            labels = ("Turn +360 deg", "Turn -360 deg", "Reorientation")
        with mujoco.Renderer(model, height=height, width=width) as renderer, TemporaryDirectory(dir=directory) as tmp:
            for i, name in enumerate(names):
                path = Path(tmp) / f"{name}.mp4"
                command = [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(1 / (2 * self.dt)),
                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
                ]
                with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                    with encoder.stdin as stream:
                        for step, qpos, alive, degrees in self.frames:
                            data.qpos[:] = qpos[i]
                            mujoco.mj_forward(model, data)
                            renderer.update_scene(data, camera=camera)
                            frame = Image.fromarray(renderer.render())
                            draw = ImageDraw.Draw(frame)
                            metric = "goal error" if i == 2 else "command-aligned yaw"
                            if self.turn and i < 2:
                                rotation = data.body("cube/object").xmat.reshape(3, 3)
                                view = np.array([self.front[0], self.front[1], 0])
                                local_view = rotation.T @ view
                                facing = int(np.argmax(normals @ local_view)) + 1
                                face = str(facing) if abs(local_view[2]) <= max(normals @ local_view) else "cap"
                                metric = f"front {face}   physical yaw"
                            status = "held" if alive[i] else "dropped"
                            caption = f"{labels[i]}   {step * self.dt:.2f} s   {status}   {metric} {degrees[i]:+.1f} deg"
                            draw.rectangle((0, 0, width, 28), fill=(24, 28, 34))
                            draw.text((10, 4), caption, fill="white", font_size=16)
                            stream.write(frame.tobytes())
                    if encoder.wait():
                        raise RuntimeError(f"Video encoder failed for {name}")
            for name in names:
                path = directory / f"{name}.mp4"
                (Path(tmp) / path.name).replace(path)
                print(f"Video {path}", flush=True)
