# Superdex

Superdex is a simulation research project about learning multiple manipulation skills with a single LEAP Hand policy. Its first two tasks are continuous cube rotation in either yaw direction and reorienting the cube to a target 3D orientation. The research question is whether sharing a learned dynamics model and policy improves performance or sample efficiency compared with training each skill independently.

The planned learner follows [SuperMimic](https://github.com/HybridRobotics/supermimic). It fits one differentiable world model to simulator replay and trains an actor through imagined trajectories. Each task has its own goal encoder. The dynamics, actor decoder, and critic are shared. Evaluation will compare the same joint checkpoint on both tasks with separate baselines at matched simulator interaction budgets.

## Current status

The repository contains the simulator setup, a right-hand LEAP model, a fixed-palm cube calibration scene, and a deterministic cube grasp. The grasp check holds the cube on fingertip contacts, releases it under gravity, and verifies that the rollout repeats exactly after reset. The grasp reset distribution, manipulation tasks, and shared learner integration are upcoming milestones.

## Run the CPU smoke check

Use [uv](https://docs.astral.sh/uv/) and Git. The setup was tested on Linux x86-64 with uv 0.13.7. `uv sync` uses the Python version in `.python-version` and the package versions in `uv.lock`.

```bash
# CPU
uv sync --locked
uv run --locked python -m superdex.smoke
```

The check loads the bundled meshes, tests all 16 position actuators in both directions, and steps a free cube onto the palm. It checks the cube quaternion and verifies that resetting restores the initial state.

## Replay the cube grasp

```bash
uv run --locked python -m superdex.grasp --trace grasp.csv
```

The headless check uses CPU MuJoCo. The optional CSV records joint positions, cube pose in MuJoCo's scalar-first quaternion order, and fingertip contact flags during the two-second hold and subsequent release.

## Set up the upstream training environment

The Linux setup script uses pinned official SuperMimic and `mjlab_research` revisions in `.upstream/` and installs the packages in SuperMimic's lockfile. It creates a second virtual environment so the CPU smoke check stays lightweight. The training lock includes several gigabytes of CUDA packages.

```bash
# CPU setup and interface checks
bash scripts/setup-upstream.sh
PYTHONPATH=src .upstream/supermimic/.venv/bin/python -m superdex.smoke --mjlab-device cpu
.upstream/supermimic/.venv/bin/python -m supermimic.scripts.train SuperMimic-GenCoord-G1 --help
```

On a Linux host with a compatible NVIDIA driver, use the same setup and run the bounded MuJoCo Warp checks on the selected GPU.

```bash
# GPU simulation check
PYTHONPATH=src .upstream/supermimic/.venv/bin/python -m superdex.smoke --mjlab-device cuda:0
PYTHONPATH=src .upstream/supermimic/.venv/bin/python -m superdex.grasp --mjlab-device cuda:0 --mjlab-envs 32
```

The official G1 baseline requires a compatible retargeted motion clip. The first Superdex training experiment will use the LEAP cube grasp reset for yaw rotation.

## Credits

The unchanged right-hand MJCF and its ten referenced meshes are from [MuJoCo Menagerie at `370c439`](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/leap_hand). The bundled [MIT license](src/superdex/assets/leap_hand/LICENSE) credits Ananye Agarwal.

The baseline setup uses upstream [SuperMimic](https://github.com/HybridRobotics/supermimic/tree/698ea39bc901d8fe733b3c8621df3199d241f7e4) and [mjlab_research](https://github.com/HybridRobotics/mjlab_research/tree/6ea3e574b08605b79a290a6d464a812a26acd6fd), both under Apache-2.0.
