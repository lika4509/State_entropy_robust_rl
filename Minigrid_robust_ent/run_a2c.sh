#!/bin/bash

# Share one experiment across all seeds; environment variables override the defaults.
experiment_id="$(python3 -c 'import datetime, uuid; print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])')"
export WANDB_PROJECT="${WANDB_PROJECT:-RobustEnt_Minigrid_${experiment_id}}"
export RL_STORAGE="${RL_STORAGE:-$PWD/rl-starter-files/rl-starter-files/scripts/utils/STORAGE_${experiment_id}}"
configs=(
  "beta=0.07 entropy_coef=0.01"
  "beta=0.0 entropy_coef=0.01"
  "beta=0.0 entropy_coef=0.0"

)

max_parallel=10
count=0

for config in "${configs[@]}"; do
  eval $config
  for seed in {1..20}; do
    echo "Running: seed=$seed, beta=$beta, entropy=$entropy_coef"
    PYTHONPATH=./torch-ac python3 rl-starter-files/rl-starter-files/scripts/train.py \
      --algo a2c \
      --seed $seed \
      --entropy-coef $entropy_coef \
      --no-ent-coef-decay \
      --beta $beta \
      "$@" &
    ((count++))
    if (( count % max_parallel == 0 )); then
      wait
    fi
  done
done

