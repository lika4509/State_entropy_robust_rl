#!/usr/bin/env bash
set -euo pipefail

# Evaluate every checkpoint on the stronger perturbation suite.
#
# Modes:
#   wall_random          random straight wall, partial or full with one gap
#   random_blocks_10     dense random blocks, 10 cells
#   random_blocks_15     dense random blocks, 15 cells
#   random_blocks_20     dense random blocks, 20 cells
#   reward_corners       goal in a non-training corner
#
# The original modes and eval_perturbed.sh are unchanged.
#
# Usage:
#   bash eval_perturbed_strong.sh [CHECKPOINT_ROOT] [EPISODES_PER_WORKER]
#
# Defaults:
#   CHECKPOINT_ROOT     = rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW
#   EPISODES_PER_WORKER = 5
#
# Environment overrides:
#   EVAL_RL_STORAGE      where logs and checkpoint symlinks are written
#                        (default: <repo>/eval_runs/strong)
#   EVAL_WANDB_PROJECT   W&B project (default: RobustEnt_Minigrid_PerturbEval_Strong)
#   EVAL_WANDB_MODE      online | offline | disabled (default: online)
#   EVAL_CONDA_ENV       conda environment (default: state_entropy_robust_rl)
#   EVAL_PROCS           number of parallel environments per run (default: 8)
#   EVAL_SKIP_EXISTING   set to 1 to skip checkpoints whose log already has all
#                        five eval lines (default: 0)
#
# This launches train.py with --resume --frames 0, so no training happens: it
# loads each checkpoint, runs the stronger perturbations, and writes to W&B.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT_ROOT="${1:-$SCRIPT_DIR/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW}"
EPISODES="${2:-5}"

EVAL_MODES="wall_random,random_blocks_10,random_blocks_15,random_blocks_20,reward_corners"
EVAL_LABELS=(wall_random random_blocks_10 random_blocks_15 random_blocks_20 reward_corners)

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
    exit 1
fi
CHECKPOINT_ROOT="$(cd "$CHECKPOINT_ROOT" && pwd)"

EVAL_STORAGE="${EVAL_RL_STORAGE:-$SCRIPT_DIR/eval_runs/strong}"
mkdir -p "$EVAL_STORAGE"
EVAL_STORAGE="$(cd "$EVAL_STORAGE" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${EVAL_CONDA_ENV:-state_entropy_robust_rl}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${EVAL_CONDA_ENV:-state_entropy_robust_rl}"
    else
        echo "conda not found; activate '${EVAL_CONDA_ENV:-state_entropy_robust_rl}' manually." >&2
    fi
fi

export RL_STORAGE="$EVAL_STORAGE"
export WANDB_PROJECT="${EVAL_WANDB_PROJECT:-RobustEnt_Minigrid_PerturbEval_Strong}"
export WANDB_MODE="${EVAL_WANDB_MODE:-online}"
export PYTHONPATH="$SCRIPT_DIR/torch-ac"

cd "$SCRIPT_DIR"

shopt -s nullglob
CHECKPOINTS=("$CHECKPOINT_ROOT"/*/status.pt)
if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "No checkpoints found under: $CHECKPOINT_ROOT/*/status.pt" >&2
    exit 1
fi

TOTAL=${#CHECKPOINTS[@]}
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Found:           $TOTAL checkpoints"
echo "Eval logs:       $EVAL_STORAGE"
echo "W&B:             $WANDB_PROJECT (mode=$WANDB_MODE)"
echo "Modes:           $EVAL_MODES"
echo "Episodes:        $EPISODES per worker with ${EVAL_PROCS:-8} workers"
echo

FAILED=()
INDEX=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    INDEX=$((INDEX + 1))
    CHECKPOINT="$(realpath "$CHECKPOINT")"
    CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
    CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"

    RUN_NAME="eval_strong_${CHECKPOINT_NAME}_ep${EPISODES}"
    RUN_NAME="$(printf '%s' "$RUN_NAME" | tr -c 'A-Za-z0-9_-' '_')"
    MODEL_DIR="$EVAL_STORAGE/$RUN_NAME"

    if [[ "${EVAL_SKIP_EXISTING:-0}" == "1" && -f "$MODEL_DIR/log.txt" ]]; then
        ALL_DONE=1
        for label in "${EVAL_LABELS[@]}"; do
            if ! grep -qs "^Eval ${label} " "$MODEL_DIR/log.txt"; then
                ALL_DONE=0
                break
            fi
        done
        if [[ "$ALL_DONE" == "1" ]]; then
            echo "[$INDEX/$TOTAL] skip (already evaluated): $CHECKPOINT_NAME"
            continue
        fi
    fi

    # Parse beta, alpha and seed from names like
    # MiniGrid-EmptyEnv-v0_seed1_beta_0.07alpha_0.01
    BETA="0.0"
    ALPHA="0.0"
    SEED="1"
    if [[ "$CHECKPOINT_NAME" =~ _beta_([^a]+)alpha_(.+)$ ]]; then
        BETA="${BASH_REMATCH[1]}"
        ALPHA="${BASH_REMATCH[2]}"
    fi
    if [[ "$CHECKPOINT_NAME" =~ _seed([0-9]+)_ ]]; then
        SEED="${BASH_REMATCH[1]}"
    fi

    mkdir -p "$MODEL_DIR"
    ln -sfn "$CHECKPOINT" "$MODEL_DIR/status.pt"

    if [[ -f "$MODEL_DIR/experiment.json" ]]; then
        echo "  note: reusing existing eval directory; use a new EVAL_RL_STORAGE to reset." >&2
    fi

    echo "[$INDEX/$TOTAL] $CHECKPOINT_NAME | beta=$BETA alpha=$ALPHA seed=$SEED"

    if python3 rl-starter-files/rl-starter-files/scripts/train.py \
        --algo a2c \
        --env MiniGrid-EmptyEnv-v0 \
        --model "$RUN_NAME" \
        --resume \
        --frames 0 \
        --episodes "$EPISODES" \
        --procs "${EVAL_PROCS:-8}" \
        --seed "$SEED" \
        --beta "$BETA" \
        --entropy-coef "$ALPHA" \
        --eval-modes "$EVAL_MODES"; then
        echo "  done: $RUN_NAME"
    else
        echo "  FAILED: $RUN_NAME" >&2
        FAILED+=("$CHECKPOINT_NAME")
    fi
    echo
done

if (( ${#FAILED[@]} > 0 )); then
    echo "Failed checkpoints (${#FAILED[@]}):" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi

echo "All ${TOTAL} checkpoints evaluated on the stronger perturbations."
