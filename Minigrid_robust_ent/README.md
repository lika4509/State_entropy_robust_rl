# Minigrid robust state entropy experiments

This repo is based on the paper: *Maximum Value-Conditional State Entropy Exploration for Sample-Efficient Deep Reinforcement Learning*, available at https://sites.google.com/view/rl-vcse.

(We do **not** use value conditioning in our experiments.)

## Installation

The single `environment.yml` at the **repository root** installs everything for both Minigrid and MuJoCo. Run it from the repo root so the relative `torch-ac` path resolves:

```bash
conda env create -f environment.yml
conda activate state_entropy_robust_rl
```

This creates the `state_entropy_robust_rl` env (Python 3.9), installs `torch-ac` in editable mode from `./Minigrid_robust_ent/torch-ac`, and brings in `gym` (legacy pre-0.26 API), `minigrid`, `gym-minigrid`, `tensorboardX`, `scikit-image`, `pygame`, `array2gif`, and `wandb`, plus all MuJoCo deps. The earlier manual steps (separate `requirements.txt`, cloning Minigrid, `pip install -e torch-ac`) are now baked into the env file and no longer needed.

## Reproducing our main results

`train.py` uploads to wandb, so log in first (or set `WANDB_API_KEY`):

```bash
wandb login

cd Minigrid_robust_ent
bash run_a2c.sh
```

This trains and evaluates three variants across **20 seeds each** (60 runs total), up to 10 in parallel:

- **No regularization** — `beta=0.0`, `entropy_coef=0.0`
- **Policy-entropy regularization** — `beta=0.0`, `entropy_coef=0.01`
- **State-entropy regularization** — `beta=0.07`, `entropy_coef=0.01`

By default, each invocation creates one unique `RobustEnt_Minigrid_<timestamp>_<random>` W&B project and a matching `utils/STORAGE_<timestamp>_<random>/` directory shared by all 60 runs. The resolved destinations are printed before training.

### Choosing project and storage

Set the `WANDB_PROJECT` and `RL_STORAGE` environment variables to override the generated defaults for a launch:

```bash
WANDB_PROJECT=my_project RL_STORAGE=/absolute/path/to/checkpoints bash run_a2c.sh
```

Run the launcher from `Minigrid_robust_ent/`. Either variable can be supplied independently, and relative paths resolve from that working directory. These variables also work when calling `train.py` directly. Set `WANDB_ENTITY` to override the default W&B account/team. If you previously exported destination variables, use `unset WANDB_PROJECT RL_STORAGE` to return to unique defaults.

### Policy entropy decay

Use `--ent-coef-decay` to enable linear policy-entropy decay and `--no-ent-coef-decay` to disable it. The default and launcher setting disable decay. These flags take no value. Earlier versions parsed `--ent_coef_decay False` with `type=bool`, which accidentally enabled decay. The paper does not explicitly specify this policy-entropy schedule.

### Local logging and a short training check

To skip uploading, run `WANDB_MODE=disabled bash run_a2c.sh`. For a short training check, call `train.py` directly:

```bash
PYTHONPATH=./torch-ac python3 rl-starter-files/rl-starter-files/scripts/train.py \
  --algo a2c --seed 1 --frames 512 --procs 2
```

The short training check still runs the automatic evaluation afterward.

## Where outputs land

Training uses `RL_STORAGE` when set, otherwise a unique directory under `scripts/utils`. All artifacts go under that root:

```
$RL_STORAGE/
└── <run_name>/                 # e.g. MiniGrid-EmptyEnv-v0_seed1_beta_0.07alpha_0.01
    ├── log.txt                 # console log
    ├── log.csv                 # per-update metrics
    ├── status.pt               # training checkpoint
    ├── experiment.json         # resolved configuration, project and W&B run ID
    └── events.out.tfevents.*   # TensorBoard stream
```

Direct `train.py` invocations also generate unique default projects and storage roots. Model subfolder names keep the original format, such as `MiniGrid-EmptyEnv-v0_seed1_beta_0.07alpha_0.01`. An explicit `--model` overrides that subfolder name. Evaluation/visualization still use the legacy `STORAGE_NEW/` fallback: set `RL_STORAGE` to the new experiment's root when loading its policies. Use an absolute path for evaluation so resolution is independent of the working directory.

## Resuming an experiment

Existing model directories require explicit resumption:

```bash
WANDB_PROJECT=my_project RL_STORAGE=/absolute/path/to/checkpoints bash run_a2c.sh --resume
```

Run this from `Minigrid_robust_ent/`, using the original training settings, including the decay flag. Resume expects a checkpoint for each selected model and uses the recorded W&B run ID. Model names are derived from the environment, seed, beta and entropy coefficient, as in the original implementation. For an individual policy, set the same environment variables and call `train.py` with its original training settings and `--resume`. Supply `--model` only for a custom subfolder name. Older checkpoints without metadata use the original model name as their W&B ID.

## Troubleshooting

### Remote / shared servers (Renku)

If `conda env create` or `pip` fails with `dataclass() got an unexpected keyword argument 'slots'`, a pre-existing global `PYTHONPATH` is shadowing the env's interpreter with a Python 3.13 one. Clear it first:

```bash
unset PYTHONPATH
conda env create -f environment.yml
```

Make it permanent for this account with `echo 'unset PYTHONPATH' >> ~/.bashrc`. The scripts re-set their own `PYTHONPATH` where needed, so this is a one-time shell fix.

### W&B authentication

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
