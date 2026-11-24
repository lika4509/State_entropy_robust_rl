# Minigrid perturbation benchmark: current state and ideas

This note documents the current `EmptyEnv` perturbation variants in
`rl-starter-files/rl-starter-files/scripts/utils/custom_env.py`, their
weaknesses, and possible stronger perturbations for robustness evaluation.

The important context is that policies are trained only on the default empty
10×10 room (`blocked=False`). All other variants are used for evaluation in
`train.py` (`eval_tasks = {1: "wall", 2: "elbow", 3: "random_blocks", 4:
"reward_shift"}`). Even a mild map change is therefore out-of-distribution for
the policy.

## Current setup

### Shared configuration

- 10×10 grid with a surrounding wall.
- Agent starts at `(1, 1)` facing east (`agent_dir=0`) in every episode.
- Goal is at `(8, 8)` except in mode 4.
- Action space is `Discrete(3)`: turn left, turn right, move forward. Pickup,
  drop, toggle and done are disabled.
- Observation is a 7×7×3 egocentric RGB image.
- `see_through_walls=True`, so walls do not occlude cells behind them.
- `max_steps=200`. Reaching the goal gives
  `1 - 0.9 * step_count / max_steps`.
- Mission is always `"reach the goal"`.
- Training uses only the empty room. Modes 1–4 are evaluation-only.

### Current variants

| `blocked` | Label | What it does | Random per reset? |
|---|---|---|---|
| `False` / `0` | training | Empty room, goal at `(8,8)` | No |
| `1` | wall | Vertical wall at `x=4`, `y=1..5` | No* |
| `2` | elbow | L-wall: `(5..7, y)` plus `(5, y+1)`, `(5, y+2)`, with `y ∈ {1..4}` | Yes, `y` |
| `3` | random_blocks | 7 random interior walls, excluding `(2,2)` and `(8,8)` | Yes |
| `4` | reward_shift | Empty room, goal uniformly sampled from `x,y ∈ 2..7` | Yes |

\* `blocked=1` draws a random `x_blocked`, but immediately overwrites it with
`x_blocked=4`, so there is only one wall layout. The call only consumes RNG
state.

## Weaknesses of the current suite

The shortest action paths below were computed with BFS over `(x, y, direction)`
using the actual three-action space, including turns, over 1000 random seeds.
"Goal reward" is the reward received at the shortest path, not a learned
policy's return.

| Variant | Shortest actions | Mean goal reward | Observation |
|---|---:|---:|---|
| Empty / training | 15 | 0.9325 | Baseline |
| `blocked=1` wall | 16 | 0.9280 | Only one extra action |
| `blocked=2` elbow | 15–16, mean 15.25 | 0.9314 | 75% of layouts are still 15 actions |
| `blocked=3` random blocks | 15–20, mean 15.82 | 0.9288 | Strongest, but 3.7% unsolvable |
| `blocked=4` reward shift | 3–13, mean 7.98 | 0.9641 | Actually easier than training |

### Learned policy returns

Intrinsic difficulty is only half of the story. The 60 completed checkpoints in
`rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW` (20 seeds for each
of the three training variants) were evaluated on all five modes. Each seed was
run for 100 episodes per mode with stochastic actions (`argmax=False`, matching
`train.py` and `Evaluate.py`) and paired layout seeds across policies.
Unsolvable mode-3 layouts were skipped by BFS (2 of the 100 sampled layouts in
this evaluation set).

All evaluations are zero-shot: the policy weights are frozen, and no optimizer
step or gradient update occurs. Only environment state and recurrent memory (if
the model uses it) change during an episode. The reported degradation is
therefore generalization failure of the trained policy, not failed adaptation
during evaluation.

Values are mean ± std of the 20 seed-level mean returns; parentheses show the
mean success rate. These are stochastic-policy returns, not greedy returns.

| Variant | State entropy (`β=0.07`, `α=0.01`) | Policy entropy (`β=0`, `α=0.01`) | Unregularized (`β=0`, `α=0`) |
|---|---:|---:|---:|
| Empty / training | 0.921 ± 0.002 (100%) | 0.931 ± 0.002 (100%) | 0.932 ± 0.001 (100%) |
| `blocked=1` wall | 0.788 ± 0.208 (90.0%) | 0.232 ± 0.410 (25.2%) | 0.054 ± 0.206 (6.5%) |
| `blocked=2` elbow | 0.623 ± 0.160 (73.3%) | 0.680 ± 0.179 (75.0%) | 0.545 ± 0.204 (60.0%) |
| `blocked=3` random blocks | 0.296 ± 0.090 (39.4%) | 0.367 ± 0.046 (40.3%) | 0.365 ± 0.066 (39.7%) |
| `blocked=4` reward shift | 0.855 ± 0.141 (96.3%) | 0.031 ± 0.042 (5.2%) | 0.011 ± 0.041 (1.5%) |

Main issues, updated in light of the learned returns:

- **The wall is weak in path length but not weak for learned policies.** It
  blocks the direct east-then-south trajectory, but a monotone down-then-east
  path remains. The optimal cost changes from 15 to 16 actions, and the optimal
  reward changes by only about 0.5% relative. Most of the wall is already visible
  in the initial 7×7 view. Nevertheless, it is strongly method-discriminating:
  state entropy reaches `0.788` return / `90%` success, policy entropy
  `0.232` / `25%`, and unregularized `0.054` / `6.5%`. The per-seed results are
  highly uneven: many policies either solve the wall near-optimally or time
  out, with relatively few in between. The wall is therefore a good test for
  whether the policy memorized the empty room, but not a hard navigation problem
  in its own right.
- **The wall is a single deterministic layout.** Mode 1 is the same map in every
  evaluation episode, so it measures one point perturbation, not a distribution
  over wall placements. It also makes the bimodal seed-level results easy to
  misinterpret: 100 episodes are repeated evaluations of one layout.
- **The elbow is often irrelevant in path length, but not for policies.** In 75%
  of layouts it does not increase the shortest path at all; otherwise it costs
  one extra action. Learned returns are still only `0.55–0.68` with `60–75%`
  success, and the three methods are not clearly separated.
- **Random blocks are varied but not a coherent obstacle.** Seven independent
  wall cells can be scattered anywhere. They can occasionally make the goal
  unreachable, e.g. walls at both `(7,8)` and `(8,7)`, and there is no
  connectivity check. Intrinsically this is the hardest mode, but all three
  methods collapse to roughly `0.30–0.37` return / `40%` success, so the mode is
  close to a performance floor and does not discriminate between methods.
- **"Reward shift" is misnamed.** It does not change the reward function; it only
  moves the goal. Since the goal is sampled from `x,y ∈ 2..7`, every goal is
  closer than the training goal at `(8,8)`, and the shortest-path reward is
  higher on average. Empirically it is still a strong out-of-distribution test
  of goal location: state entropy gets `0.855` / `96%`, while policy entropy and
  unregularized collapse to `0.031` / `5.2%` and `0.011` / `1.5%`. The near-zero
  returns are a floor effect, not evidence that the shift is intrinsically hard.
- **Everything else is fixed.** Start position, start orientation, map size,
  view size, colors, dynamics and observation process are unchanged.
- **There is no stochastic dynamics.** Transitions and the goal reward are
  deterministic. The `PenaltyTile` class exists, but no penalty tile or lava is
  ever placed, so its 75% penalty branch is dead code.
- **Seeding is fragile.** `_gen_grid` uses global `np.random`, `reset(seed=...)`
  ignores its seed, and `seed()` shadows itself with an integer. Layouts are
  reproducible only if the global RNG, environment creation order, process
  count and interaction order are identical. This prevents clean paired
  comparisons across policies.
- **Mode coordinates assume a 10×10 grid.** Mode 1 uses `x=4`, mode 2 uses
  `x=5..7`, and mode 3 removes `(8,8)` explicitly. Several of these break or
  become unsolvable at other sizes.

The learned returns change the interpretation of the suite. The wall is not a
hard navigation problem, but it is an effective test of path memorization; random
blocks and reward shift show strong floor effects. The current variants are
therefore useful as diagnostic smoke tests, but they are not a graded general
robustness benchmark. Report success rate, per-seed spread and the number of
solvable layouts alongside mean return.

## Perturbation ideas

### 1. Smaller environments

Train on 10×10 and evaluate on 8×8, 7×7 or 6×6 maps, keeping the agent and goal
at relative positions such as `(1,1)` and `(width-2,height-2)`.

This tests:

- sensitivity to absolute map scale and absolute coordinates;
- whether wall configurations remain solvable when the room is smaller;
- shorter-horizon behaviour and whether the policy still reaches the goal
  efficiently.

Implementation notes:

- Parameterize every hard-coded coordinate relative to `width` and `height`.
- Mode 1 becomes a full barrier at size 7 unless the wall is shortened or given
  a gap. Give every layout a BFS reachability check.
- Mode 2 must not write walls at `x=width-1` or outside the grid.
- Mode 3 must exclude `(width-2,height-2)` instead of `(8,8)`, and the number of
  random walls must be capped for small grids.
- Adjust `max_steps`; otherwise the 200-step budget becomes very generous.
- If the view size stays 7×7, the observation is still 7×7 even on a smaller
  map. Reduce the view size separately if perception should become harder.

Larger maps (12×12, 15×15) are the opposite perturbation and test long-horizon
exploration. They are also worth including if evaluation time permits.

### 2. Turn off `see_through_walls`

Add a `see_through_walls` argument to `EmptyEnv` and pass it to
`MiniGridEnv.__init__`. The current code hard-codes `True`.

With `see_through_walls=False`, walls occlude the cells behind them, so the
7×7 image no longer reveals everything in the view. This tests:

- robustness to partial observability;
- whether the policy relies on information it would not normally have;
- memory and exploration requirements.

Caveats:

- This changes the observation process, not just the layout. A feedforward
  policy may fail for reasons unrelated to spatial robustness.
- The effect is much stronger when combined with a smaller view (5×5 or 3×3).
- Report it separately from layout perturbations so the cause of a performance
  drop is clear.
- Keep the observation space shape consistent with `agent_view_size`.

### 3. Stronger layout perturbations

- **Barrier with one gap.** Use a wall that spans most of the room with a single
  opening, placed so the shortest path must go through a specific region.
- **Non-monotone detours.** Design layouts where the agent must move away from
  the goal, or backtrack, instead of following a monotone right/down path.
- **Wall segments instead of isolated blocks.** Sample horizontal and vertical
  segments, or use a random maze generator such as recursive division.
- **Density sweep.** Evaluate with 5%, 10%, 15% and 25% of interior cells
  blocked, and report performance as a function of density.
- **Connectivity and path-length checks.** Run BFS after generating a layout.
  Reject layouts where the goal is unreachable, and optionally stratify by
  shortest-path length.
- **Held-out layout families.** Train on empty/wall/elbow and test on mazes,
  corridors, multi-room maps or ObstructedMaze-style maps.
- **Distractor objects.** Add keys, balls or open doors as visual clutter without
  changing the action space. Note that closed/locked doors need the toggle
  action, which the current policy does not have.
- **Color/texture shift.** Recolor walls, goal or agent, or apply a global color
  jitter. This tests whether the policy overfits to the training appearance.

### 4. Start and goal perturbations

- **Random start cell.** Sample a reachable empty cell for the agent, with a
  connectivity check. This prevents the policy from memorizing the top-left
  start.
- **Random initial direction.** Sample `agent_dir` from `{0,1,2,3}` or from a
  small set of held-out orientations.
- **Uniform reachable goal.** Sample the goal from all reachable cells rather
  than only the `2..7` interior. Include goals farther away than the training
  goal.
- **Distance-stratified goals.** Report results separately for easy, baseline
  and harder goal distances. Ensure the benchmark does not consist only of
  easier goals.
- **Farthest-cell goal.** Place the goal at the farthest reachable cell to test
  long-horizon planning.
- **Corner/room swaps.** Evaluate the same map with the goal in different
  corners or rooms.

### 5. Perception and observation perturbations

- **Smaller view size.** Change `agent_view_size` from 7 to 5 or 3. The view
  size must be odd.
- **Occlusion.** Combine a small view with `see_through_walls=False`.
- **Visual noise.** Add Gaussian or uniform pixel noise to the image.
- **Blur or downsampling.** Apply a blur or reduce the image resolution before
  the policy.
- **Color jitter/grayscale.** Change object colors or remove color information.
- **Random masking/dropout.** Hide random patches of the observation.
- **Sensor aliasing.** Make different wall/goal configurations look identical
  from the agent's current view.

Perceptual perturbations should be evaluated separately from layout
perturbations, because they change what the policy can observe rather than what
the environment requires.

### 6. Dynamics and action perturbations

- **Action slip.** With probability ε, execute a random action or a
  perpendicular action instead of the requested one.
- **Sticky actions.** Repeat the previous action with some probability.
- **Forward failure.** With probability ε, a forward move does not change the
  agent's position.
- **Dynamic obstacles.** Move one or more walls/objects between steps. This is
  a larger implementation change and should be considered a separate benchmark.
- **Time pressure.** Evaluate with `max_steps` of 50, 100 or 400. Note that the
  current reward formula depends on `max_steps`, so either keep the reward
  normalization fixed or report success and path length separately.
- **Action-space variants.** Adding toggle/pickup/drop enables doors and keys,
  but changes the model output dimension and requires retraining. Do not mix it
  with the current three-action benchmark without saying so.

### 7. Reward and risk perturbations

- **Actual reward shift.** Change the goal reward, add a per-step cost, or add a
  collision/penalty cost. Keep success rate and episode length as separate
  metrics because returns are not directly comparable across reward functions.
- **Lava fields.** Place deterministic lava that terminates the episode. This
  tests risk-sensitive navigation.
- **Stochastic penalty tiles.** Place `PenaltyTile` or a stochastic hazard and
  use the existing 75% `-0.2` branch. The class already exists but is currently
  never instantiated.
- **Risk–return trade-off.** Create maps with a short hazardous route and a
  longer safe route, then measure whether the policy chooses appropriately.
- **Reward noise.** Add small stochastic reward noise if robustness to reward
  uncertainty is part of the claim.

### 8. Evaluation protocol

These are not environment perturbations, but they are necessary for the
perturbations to be meaningful:

- Use one independent RNG per environment, preferably `self.np_random`, and
  pass `seed` through `reset(seed=...)`.
- Generate a fixed set of layout seeds and evaluate every policy on the same
  layouts (paired comparison).
- Reject unsolvable layouts or report the unsolvable rate explicitly. Do not
  silently count an invalid map as a robustness failure.
- Report success rate, return, episode length, timeout rate and shortest-path
  regret (actual steps minus BFS shortest path).
- Stratify results by perturbation type and difficulty instead of reporting a
  single average.
- For deterministic modes such as the current wall, note that repeated episodes
  are repeated evaluations of the same map, not independent samples.
- Include a no-perturbation baseline and report confidence intervals over
  layouts, not only over episodes.

### 9. Standard MiniGrid benchmarks

Where possible, compare against established MiniGrid distribution-shift
benchmarks instead of hand-built variants:

- `DistShift` for distribution shift;
- `LavaCrossing` and `SimpleCrossing` for obstacle generalization;
- `MultiRoom` for multi-room navigation;
- `ObstructedMaze` for harder exploration;
- `DoorKey` for key/door subgoals (requires toggle/pickup actions).

These provide recognized reference points and reduce the risk that conclusions
are specific to a few hand-coded maps.

## Suggested priority

| Priority | Perturbations |
|---|---|
| First | Fix per-environment seeding; add solvability checks; barrier with one gap; distance-stratified goal sampling; `see_through_walls=False`; view size 5; random start and direction; lava/hazard tiles |
| Second | Random mazes with connectivity checks; wall-density sweep; smaller and larger maps; action slip; actual reward shift |
| Third | Multi-room layouts, doors/keys (with an expanded action space), dynamic obstacles, visual noise, large-scale domain randomization |

The cheapest high-value additions are probably disabling see-through walls,
reducing the view size, adding a barrier with a single gap, sampling harder
goals, randomizing the start orientation, and adding a real hazard.

## Implementation touchpoints

- `EmptyEnv.__init__`: currently hard-codes `see_through_walls=True`, the
  7×7×3 observation shape, the three-action space and `max_steps=200`.
- `EmptyEnv._gen_grid`: currently hard-codes 10×10 coordinates and the modes
  described above.
- `EmptyEnv.reset` and `EmptyEnv.seed`: need proper seed propagation before
  paired evaluation is possible.
- `EmptyEnv.step`: already adapts the Gymnasium 5-tuple to the old Gym 4-tuple;
  any new stochastic dynamics should use the environment RNG, not global
  NumPy randomness.
- `EmptyEnv.PenaltyTile`: exists but is never placed; it is an easy starting
  point for hazard experiments.
- `train.py`: add new evaluation modes near the existing `eval_tasks` loop.
- `utils/env.py`: the `make_env` factory and any new environment arguments
  should be kept in sync.
