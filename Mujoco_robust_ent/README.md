# Continuous-Control State-Entropy Regularization for Robust RL

This directory builds on the PPO implementation from **CleanRL**: https://github.com/vwxyzjn/cleanrl.

---

## Requirements

The single `environment.yml` at the **repository root** covers these experiments too. Run it from the repo root:

```bash
cd ..                                  # repo root: State_entropy_robust_rl/
conda env create -f environment.yml
conda activate state_entropy_robust_rl
```

This brings in `gymnasium==0.28.1`, `mujoco==2.3.3`, `cffi`, `imageio`, `tyro`, `tensorboard`, and `torch==2.5.1`. `mujoco` needs `cffi` and `imageio` at runtime — they are already in the env file. If you see `No module named '_cffi_backend'` or `No module named 'imageio'`, rerun the env creation/update.

## Training

`lstm_continuous_action_ppo.py` runs PPO with a state-entropy intrinsic reward. Key flags:

- **`--ent_coef`** – policy-entropy coefficient.
- **`--starting_beta`** – initial state-entropy coefficient.
- **`--beta`** – final state-entropy coefficient (where `starting_beta` decays to).
- **`--beta_discount`** – discount factor for the state-entropy intrinsic reward.
- **`--ent_coef_decay`** – linearly anneal `ent_coef` to 0 over training (boolean).
- **`--num_envs`** – parallel environments; sets rollouts per PPO update and batch size for state-entropy estimation.
- **`--num_steps`** – steps per rollout; batch size = `num_envs × num_steps`.
- **`--save_dir`** – where the model and TensorBoard events are written.
- **`--track`** – log to Weight & Biases. Enabled in `train_and_evaluate.sh`, so it requires `wandb login` or `WANDB_API_KEY`.
- **`--wandb_project_name`** / **`--wandb_entity`** – wandb destination. The script sets the project to `RobustEnt_Mujoco`; entity defaults to your own account.

## Training + evaluation (the "wall" experiment)

```bash
wandb login

cd Mujoco_robust_ent
bash train_and_evaluate.sh
```

Models, TensorBoard events, and the eval CSV all land under the `save_dir` set at the top of the script (currently `$HOME/data/mj_results/pusher_puck/`). To skip wandb, remove `--track True` from the script and pass nothing — `--track` defaults to `False`.

## Where outputs land

```
$save_dir/
├── CustomPusher-v1__<seed>__alpha_.../    # model + TensorBoard events
└── pusher_wall_alpha.csv                   # eval results (pinned to --load_dir)
```

`Eval_pusher_wall.py` writes its CSV **into the same `--load_dir` you read the model from**, not the current working directory. The other eval scripts (`Eval_pusher_reward_shift.py`, `Eval_Ant_hurdels.py`) follow the same pattern.

## Remote / shared servers (Renku)

Same caveat as the Minigrid README: a tainted global `PYTHONPATH` shadows the env's interpreter and produces the `dataclass() got an unexpected keyword argument 'slots'` install error. Clear it first:

```bash
unset PYTHONPATH
conda env create -f environment.yml
```

For TensorBoard on these scripts, `tensorboard --logdir $HOME/data/mj_results/pusher_puck --port 6006 &`.
