# Superdex

Superdex is a simulation research project about learning multiple manipulation skills with a single LEAP Hand policy. Its first two tasks are continuous cube rotation in either yaw direction and reorienting the cube to a target 3D orientation. The research question is whether sharing a learned dynamics model and policy improves performance or sample efficiency compared with training each skill independently.

The planned learner follows [SuperMimic](https://github.com/HybridRobotics/supermimic). It fits one differentiable world model to simulator replay and trains an actor through imagined trajectories. Each task has its own goal encoder. The dynamics, actor decoder, and critic are shared. Evaluation will compare the same joint checkpoint on both tasks with separate baselines at matched simulator interaction budgets.

## Current status

The repository contains the simulator setup, a right-hand LEAP model, a fixed-palm cube calibration scene, and a deterministic cube grasp. The grasp check holds the cube on fingertip contacts, releases it under gravity, and verifies that the rollout repeats exactly after reset. Continuous yaw and 3D reorientation are wired into one shared learner. The short run checks integration and reports per-skill results in MuJoCo Warp.

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

## Joint training smoke test

Run this from the Superdex root with the CUDA-capable SuperMimic virtualenv active. The pinned source checkouts under .upstream come from the setup step above. When using a working baseline virtualenv, the command below uses its installed dependencies and the pinned source trees.

```bash
export PYTHONPATH="$PWD/src:$PWD/.upstream/supermimic/src:$PWD/.upstream/mjlab_research/src"
python -u -m superdex.train --device cuda:0 --smoke
```

The short run uses four worlds and two learner updates. It checks both rewards and drop detection, fits the shared dynamics from contiguous eight-step simulator trajectories, and checks that later imagined rewards send gradients to the first action for both skills. It prints held-out world-model errors and untrained and trained policy metrics. To save a shared checkpoint, add `--output outputs/joint-smoke.pt`. Evaluate both skills from the same actor with `--evaluate outputs/joint-smoke.pt` without `--smoke`.

The learner imports the pinned official SuperMimic dynamics, world-model update, multi-encoder actor and BPTT actor-critic update. The LEAP adapter uses the calibrated grasp, cube pose and velocity from the simulator, ±0.35 rad position offsets, and fixed physics parameters. The final actor layer starts near zero so its initial actions stay close to the grasp. The smoke test uses four imagined steps and a small batch. The regular run uses the original eight-step model and 32-step actor horizons. The smoke test verifies the training loop. Skill performance is measured in MuJoCo Warp over longer runs.

## Credits

The unchanged right-hand MJCF and its ten referenced meshes are from [MuJoCo Menagerie at `370c439`](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/leap_hand). The bundled [MIT license](src/superdex/assets/leap_hand/LICENSE) credits Ananye Agarwal.

The baseline setup uses upstream [SuperMimic](https://github.com/HybridRobotics/supermimic/tree/698ea39bc901d8fe733b3c8621df3199d241f7e4) and [mjlab_research](https://github.com/HybridRobotics/mjlab_research/tree/6ea3e574b08605b79a290a6d464a812a26acd6fd), both under Apache-2.0.
