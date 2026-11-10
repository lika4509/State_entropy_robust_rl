# State Entropy Regularization for Robust RL

Official implementation of *State Entropy Regularization for Robust RL* (NeurIPS 2025, Oral).

## Setup

A single conda environment covers both experiment directories. Create it from the repository root:

```bash
conda env create -f environment.yml
conda activate state_entropy_robust_rl
```

This installs Python 3.9, PyTorch (CUDA 12.x), legacy `gym` (for Minigrid), modern `gymnasium`+`mujoco` (for MuJoCo), and all shared tooling. `torch-ac` is installed in editable mode from `Minigrid_robust_ent/torch-ac`.

## Structure

- [`Minigrid_robust_ent/`](Minigrid_robust_ent/) — grid-world experiments; see its README for commands and output locations.
- [`Mujoco_robust_ent/`](Mujoco_robust_ent/) — continuous-control (Pusher/Ant) experiments; see its README.

`environment.yml` at this root installs both.

## Weight & Biases

Both projects log to W&B by default. Run `wandb login` once (or set `WANDB_API_KEY`) before training. Minigrid uploads to the `RobustEnt_Minigrid_new` project; MuJoCo to `RobustEnt_Mujoco`. To disable tracking, use `WANDB_MODE=disabled` for Minigrid / omit `--track` for MuJoCo.

## On shared / remote servers

If `conda env create` or `pip` fails with `dataclass() got an unexpected keyword argument 'slots'`, a pre-existing `PYTHONPATH` is shadowing the env's interpreter with a Python 3.13 one. Clear it first:

```bash
unset PYTHONPATH
conda env create -f environment.yml
```

The scripts re-set their own `PYTHONPATH` where needed, so this is a one-time shell fix.
