# Minigrid robust state entropy experiments

This repo is based on the paper: *Maximum Value-Conditional State Entropy Exploration for Sample-Efficient Deep Reinforcement Learning*, available at https://sites.google.com/view/rl-vcse.

(We do **not** use value conditioning in our experiments.)

Current robustness variants, their limitations, and proposed stronger perturbations are documented in [PERTURBATIONS.md](PERTURBATIONS.md).

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

## Evaluating existing checkpoints on perturbed environments

`eval_perturbed.sh` evaluates a checkpoint root on the four perturbation modes without retraining. It scans `<checkpoint_root>/*/status.pt`, reads the seed, beta and alpha from each model directory name, and starts one W&B run per checkpoint. From the repository root:

```bash
cd Minigrid_robust_ent
bash eval_perturbed.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW" \
  5
```

The second argument is the number of episodes **per worker**; with the default 8 workers, `5` means 40 episodes per perturbation mode and checkpoint. Use `5` for a quick check and `100` for the full evaluation:

```bash
bash eval_perturbed.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW" \
  100
```

To evaluate the newer Nov 23 checkpoint set, pass its root and keep its outputs separate:

```bash
EVAL_RL_STORAGE="$PWD/eval_runs/2026" \
EVAL_WANDB_PROJECT="RobustEnt_Minigrid_PerturbEval_2026" \
bash eval_perturbed.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_20261123_030107_e3e130c7" \
  100
```

Each checkpoint is evaluated on `wall`, `elbow`, `random_blocks` and `reward_shift`. The script uses `train.py --resume --frames 0`, so it loads the checkpoint but does not train. Progress and per-mode summaries are written to `<EVAL_RL_STORAGE>/<run_name>/log.txt`. W&B receives `Evaluation/<mode>_success_rate` plotted against `<mode>_eval_step`: the fraction of episodes that reach the goal within each step budget, matching the original evaluation curves. A full 60-checkpoint run with `100` episodes per worker typically takes several hours.

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `EVAL_RL_STORAGE` | `./eval_runs` | Eval logs and checkpoint symlinks |
| `EVAL_WANDB_PROJECT` | `RobustEnt_Minigrid_PerturbEval` | W&B project; created on first run |
| `EVAL_WANDB_MODE` | `online` | Set to `disabled` for a local-only test |
| `EVAL_PROCS` | `8` | Parallel environments per checkpoint |
| `EVAL_CONDA_ENV` | `state_entropy_robust_rl` | Conda environment to activate |
| `EVAL_SKIP_EXISTING` | `0` | Set to `1` to skip checkpoints with a completed `Eval` log line |

Use separate `EVAL_RL_STORAGE` and `EVAL_WANDB_PROJECT` values for different checkpoint roots, otherwise run names collide. To resume an interrupted sweep, set `EVAL_SKIP_EXISTING=1` with the original storage and project.

## Evaluating the stronger perturbations

`eval_perturbed_strong.sh` runs only the new stronger modes; the original four modes and `eval_perturbed.sh` are unchanged. It evaluates each checkpoint on five tasks:

- `wall_random`: random straight wall, either partial or full with one gap.
- `random_blocks_10`, `random_blocks_15`, `random_blocks_20`: dense random blocks, rejection-sampled until the goal is reachable.
- `reward_corners`: one goal in the top-right `(8,1)` or bottom-left `(1,8)`, never the training corner.

From `Minigrid_robust_ent/`:

```bash
bash eval_perturbed_strong.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW" \
  5
```

As with the original launcher, the second argument is episodes **per worker**. With the default 8 workers, `5` means 40 episodes per task; use `100` for the full evaluation:

```bash
bash eval_perturbed_strong.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW" \
  100
```

For the newer Nov 23 checkpoint set:

```bash
bash eval_perturbed_strong.sh \
  "$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_20261123_030107_e3e130c7" \
  100
```

Defaults are `EVAL_RL_STORAGE=./eval_runs/strong` and `EVAL_WANDB_PROJECT=RobustEnt_Minigrid_PerturbEval_Strong`. The same overrides as the original launcher apply: `EVAL_WANDB_MODE`, `EVAL_PROCS`, `EVAL_CONDA_ENV` and `EVAL_SKIP_EXISTING`. The skip check requires all five new eval lines in `log.txt`. Because this runs five tasks per checkpoint, expect roughly 1.25–2× the runtime of the original four-mode launcher.

All new wall and block layouts are checked with BFS before use, so every evaluated layout has a path from the start to the goal. Layouts are sampled from the per-environment seed and reset counter, so rerunning with the same checkpoint root, `EVAL_PROCS` and `EVAL_EPISODES` produces the same layouts.

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
