# Minigrid robust state entropy experiments

This repo is based on the paper: *Maximum Value-Conditional State Entropy Exploration for Sample-Efficient Deep Reinforcement Learning*, available at https://sites.google.com/view/rl-vcse.

(We do **not** use value conditioning in our experiments.)

## Installation

The single `environment.yml` at the **repository root** installs everything for both Minigrid and MuJoCo. Run it from the repo root so the relative `torch-ac` path resolves:

```bash
cd ..                                  # repo root: State_entropy_robust_rl/
conda env create -f environment.yml
conda activate state_entropy_robust_rl
```

This creates the `state_entropy_robust_rl` env (Python 3.9), installs `torch-ac` in editable mode from `./Minigrid_robust_ent/torch-ac`, and brings in `gym` (legacy pre-0.26 API), `minigrid`, `gym-minigrid`, `tensorboardX`, `scikit-image`, `pygame`, `array2gif`, and `wandb`, plus all MuJoCo deps. The earlier manual steps (separate `requirements.txt`, cloning Minigrid, `pip install -e torch-ac`) are now baked into the env file and no longer needed.

## Reproducing our main results

`train.py` uploads to wandb, so log in first (or set `WANDB_API_KEY`):

```bash
wandb login

export RL_STORAGE="$HOME/data/rl_logs"    # absolute path; where models + logs land

cd Minigrid_robust_ent
bash run_a2c.sh
```

If wandb cannot authenticate, training still runs — wandb is downgraded to `mode="disabled"`, the real error is printed, and nothing crashes. Metrics remain in `log.csv` and TensorBoard; they just are not uploaded.

Credentials are resolved in this order, so a **stale `WANDB_API_KEY` silently wins over a fresh `wandb login`**:

1. `WANDB_API_KEY` environment variable
2. `~/.netrc` (written by `wandb login`)

`wandb init` printing "W&B API key is configured" only means *a* key was **found** — not that the server accepted it. If a run reports `CommError: user is not logged in` despite `wandb login`, check both sources:

```bash
env | grep WANDB_API_KEY                  # stale export in .bashrc / job script?
python -c "import wandb; print(wandb.Api().viewer)"   # must not raise AuthenticationError
```

Fix with `unset WANDB_API_KEY` and/or `wandb login --relogin`.

This trains and evaluates three variants across **20 seeds each** (60 runs total), up to 10 in parallel:

- **No regularization** — `beta=0.0`, `entropy_coef=0.0`
- **Policy-entropy regularization** — `beta=0.0`, `entropy_coef=0.01`
- **State-entropy regularization** — `beta=0.07`, `entropy_coef=0.01`

All metrics and plots upload to the `RobustEnt_Minigrid_new` wandb project. To skip uploading (local logging only), prepend `WANDB_MODE=disabled bash run_a2c.sh`; output still lands in `$RL_STORAGE`. For a fast sanity check instead, call `train.py` directly:

```bash
PYTHONPATH=./torch-ac python3 rl-starter-files/rl-starter-files/scripts/train.py \
  --algo a2c --seed 1 --frames 512 --procs 2
```

## Where outputs land

`RL_STORAGE` is read by [scripts/utils/storage.py](rl-starter-files/rl-starter-files/scripts/utils/storage.py#L17). When set, all artifacts go under it:

```
$RL_STORAGE/
└── <run_name>/                 # e.g. MiniGrid-EmptyEnv-v0_seed1_beta_0.07_alpha_0.01
    ├── log.txt                 # console log
    ├── log.csv                 # per-update metrics
    ├── status.pt               # training checkpoint
    ├── agent.pt                # saved agent for evaluate/visualize
    └── events.out.tfevents.*   # TensorBoard stream
```

If unset it defaults to `rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW/`. Use an absolute path so `run_a2c.sh` and any later eval resolve consistently regardless of the shell's working directory.

## Remote / shared servers (Renku)

If `conda env create` or `pip` fails with `dataclass() got an unexpected keyword argument 'slots'`, a pre-existing global `PYTHONPATH` is shadowing the env's interpreter with a Python 3.13 one. Clear it first:

```bash
unset PYTHONPATH
conda env create -f environment.yml
```

Make it permanent for this account with `echo 'unset PYTHONPATH' >> ~/.bashrc`. The scripts re-set their own `PYTHONPATH` where needed, so this is a one-time shell fix.
