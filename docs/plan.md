# Project Plan: Discovery vs. Propagation in Sparse-Reward RL

## 0. Core Goal

This project tests whether sparse-reward failure in MiniGrid DoorKey is driven mainly by:

1. **Reward discovery bottleneck**: the agent rarely reaches reward-bearing or prerequisite states.
2. **Reward propagation bottleneck**: after a rare successful trajectory is observed, the reward signal is not efficiently assigned to earlier decisions.
3. **Interaction**: exploration and credit assignment help each other non-additively.

The original proposal uses a 2x2 factorial design:

| Backup target | No intrinsic discovery module | RIDE intrinsic discovery module |
|---|---|---|
| 1-step TD | DQN | DQN + RIDE |
| n-step TD | n-step DQN | n-step DQN + RIDE |

This plan keeps that core design, but adds environment difficulty sweeps, mechanism-specific metrics, a fixed-replay propagation test, and a pre-reward discovery analysis that reduce confounding between discovery and propagation.

The key methodological principle is:

> Do not infer mechanism from final success rate alone. Infer mechanism only when performance effects agree with mechanism-specific diagnostics.

Compute assumption:

- The recommended full design assumes access to A100/H100-class GPUs.
- MiniGrid symbolic DQN is usually not FLOP-bound on these GPUs. Use the extra compute primarily to increase seed count, evaluation precision, difficulty coverage, and diagnostic logging rather than to enlarge the agent.
- Avoid changing the main agent into a much larger or higher-update-ratio learner merely to saturate the GPU, because that changes the discovery/propagation mechanisms being tested.

---

## 1. Experimental Environment Specification

### 1.1 Benchmark Family

Primary benchmark:

- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-DoorKey-8x8-v0`
- Custom or registered intermediate DoorKey variant, preferably `DoorKey-12x12`
- `MiniGrid-DoorKey-16x16-v0`

Optional robustness benchmark if compute allows:

- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-MultiRoom-N4-S5-v0` or a comparable MultiRoom variant

Rationale:

- DoorKey has ordered sparse-reward structure: find key, pick up key, unlock/open door, reach goal.
- The environment is partially observable with a local egocentric field of view.
- The official MiniGrid documentation describes DoorKey as difficult for classical RL because of sparse rewards and useful for curiosity or curriculum learning.
- The difficulty sweep is necessary because DoorKey-6x6 alone may be too easy, producing ceiling effects that obscure whether discovery or propagation is the real bottleneck.
- With A100/H100 compute, include an intermediate 10x10 or 12x12 DoorKey variant if 8x8 is too easy and 16x16 is too hard. The most informative regime is one where some seeds solve and some do not.

### 1.2 Observation Space

Use the default MiniGrid partial observation:

- Local egocentric image: `7 x 7 x 3`
- Agent direction
- Mission string

Recommended preprocessing:

- Use symbolic MiniGrid observations rather than RGB pixels.
- Convert tile object/color/state channels to one-hot or embedding features.
- Include agent direction as a one-hot vector.
- Ignore mission text for DoorKey, because the mission is constant within the task family. If included, encode it as a fixed task id rather than a learned language model.

Do not use privileged full-state observations in the experiments.

### 1.3 Action Space

Use the native MiniGrid discrete action space:

| Action id | Action |
|---|---|
| 0 | turn left |
| 1 | turn right |
| 2 | move forward |
| 3 | pick up |
| 4 | drop |
| 5 | toggle |
| 6 | done |

Do not use action masking in the main experiment.

Reason:

- Action masking would inject prior task knowledge and reduce exploration difficulty.
- Keeping invalid actions preserves the original sparse-reward challenge.

### 1.4 Reward Function

Evaluation reward:

- Use only the original extrinsic MiniGrid reward.
- Success reward is the environment-provided sparse reward, typically positive only when the goal is reached.
- Failed episodes receive zero extrinsic reward.

Training reward:

- For non-RIDE agents: extrinsic reward only.
- For RIDE agents: extrinsic reward plus intrinsic RIDE bonus during training only.
- Evaluation must disable intrinsic reward completely.

Important reporting rule:

- Always report extrinsic evaluation performance separately from intrinsic training reward.
- Never compare algorithms using training return when RIDE is active, because the reward scale is algorithm-dependent.

### 1.5 Episode Termination and Time Limits

Use the environment default maximum episode length unless explicitly overridden.

Record for every episode:

- Episode length
- Whether goal was reached
- Whether timeout occurred
- Whether key was picked up
- Whether door was toggled open
- Whether the agent reached the room beyond the door

### 1.6 Training and Evaluation Seeds

Use paired random seeds across all algorithms.

Recommended seed count:

- Minimum: 10 seeds per condition, matching the original proposal.
- Preferred: 20 seeds per condition for the main DoorKey sweep.
- A100/H100 full package: 30 paired seeds per main condition.

Seed pairing:

- For seed `s`, all four main agents should use the same initialization seed, environment seed stream, replay sampling seed, and evaluation layout set where possible.
- This reduces variance when estimating factorial effects.

Evaluation layout protocol:

- Create a fixed held-out evaluation set of environment seeds for each environment size.
- Suggested: 100 evaluation episodes per checkpoint for DoorKey-6x6 and DoorKey-8x8; 200 for DoorKey-16x16 if variance is high.
- A100/H100 full package: 200 evaluation episodes per checkpoint for DoorKey-6x6 and DoorKey-8x8; 300-500 for DoorKey-16x16 or other high-variance tasks.
- Evaluation episodes use greedy policy or low-exploration policy.
- No learning occurs during evaluation.

### 1.7 Training Budget

Use a difficulty-dependent budget.

Initial recommended budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k | every 5k |
| DoorKey-8x8 | 1M | every 10k |
| DoorKey-12x12 or intermediate custom DoorKey | 3M | every 10k-25k |
| DoorKey-16x16 | 5M | every 25k |
| KeyCorridorS3R3 | 5M | every 25k |

A100/H100 full-package budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k-500k | every 5k |
| DoorKey-8x8 | 2M | every 10k |
| DoorKey-12x12 or intermediate custom DoorKey | 5M | every 10k-25k |
| DoorKey-16x16 | 10M | every 10k-25k |
| KeyCorridorS3R3 | 10M | every 25k |

Before final runs, perform a pilot with 5-8 paired seeds to check whether:

- All agents fail completely.
- All agents solve too early.
- The evaluation interval is too coarse to estimate first success and post-discovery speed.

If all agents solve DoorKey-6x6 quickly, keep it as a sanity check but do not make it the main evidence.

If DoorKey-16x16 remains a floor-effect regime even with the A100/H100 budget, do not force conclusions from it. Use the intermediate DoorKey size as the primary mechanism-identification environment.

### 1.8 Base Agent Specification

Use one shared value-based agent implementation for all four main cells.

Main agent:

- DQN with experience replay and target network.
- Same network architecture across all conditions.
- Same optimizer, learning rate, replay buffer size, target update period, epsilon schedule, batch size, discount, and training budget.

Recommended architecture for symbolic partial observations:

- Input: one-hot encoded `7 x 7` symbolic observation plus direction.
- Encoder: small convolutional network or compact MLP over flattened symbolic features.
- Head: fully connected layers producing Q-values for 7 actions.
- No recurrent memory in the main experiment.

Recommended stability variant:

- Use Double DQN if vanilla DQN is unstable, but then call the baseline Double DQN consistently.
- Do not combine dueling networks, prioritized replay, noisy nets, distributional RL, or other Rainbow components in the main factorial design, because they introduce additional mechanisms.

A100/H100 implementation guidance:

- Keep the main network small and fixed across conditions.
- Keep the main update-to-data ratio at 1. Higher ratios can artificially strengthen propagation and confound the n-step comparison.
- Keep the main batch size at 128 unless stability requires otherwise.
- Use GPU parallelism by running many independent seed/condition jobs concurrently. Do not change the single-agent data-collection process just to increase GPU utilization.

### 1.9 n-step Backup Specification

Compare:

- `n = 1`
- `n = 3`

Main comparison should use one pre-registered `n`, preferably `n = 3`, because Rainbow-style DQN commonly uses 3-step returns and multi-step targets are known to propagate newly observed rewards faster.

Implementation details:

- Store transitions in replay with enough sequence information to compute n-step targets.
- Use the same replay buffer capacity and sampling strategy for 1-step and n-step agents.
- For terminal transitions, truncate the n-step return correctly.
- Ensure that the effective discount is `gamma^n` on the bootstrap term.

### 1.10 RIDE Specification

RIDE agents receive an intrinsic reward bonus based on impact in learned representation space.

Required components:

- State embedding network.
- Forward dynamics model.
- Inverse dynamics model.
- Intrinsic reward proportional to representation change caused by action.

Training reward:

```text
r_total = r_extrinsic + beta * r_RIDE
```

RIDE reporting requirements:

- Report `beta`.
- Report intrinsic reward normalization method.
- Report whether intrinsic reward is clipped.
- Report whether RIDE shares an encoder with the Q-network.
- Report auxiliary loss weights for forward and inverse dynamics.

Important fairness rule:

- If RIDE adds parameters and auxiliary losses, report parameter count and wall-clock training time.
- Main comparisons are based on environment steps, but compute overhead should still be disclosed.

### 1.11 Software Environment and Dependencies

Use one reproducible Python virtual environment for all experiments.

Recommended Python runtime:

- Python: `3.11`
- Minimum patch recommendation: `3.11.15` or newer security patch in the Python 3.11 series.
- Environment manager: `venv`, `uv`, `conda`, or `mamba` are all acceptable, but the resolved lockfile must be saved with the experiment code.

GPU and deep-learning stack:

- Main target: PyTorch `2.12.0` with CUDA `13.0` wheels.
- Fallback target: PyTorch `2.12.0` with CUDA `12.6` wheels if the cluster driver does not support CUDA 13.x.
- Do not change the PyTorch/CUDA build between compared algorithms inside the same experimental batch.
- Record `torch.__version__`, CUDA runtime version, CUDA driver version, GPU model, and cuDNN version in every run config.

Core RL and environment dependencies:

| Package | Recommended version | Purpose |
|---|---:|---|
| `torch` | `2.12.0` | DQN, RIDE auxiliary models, GPU training |
| `torchvision` | matching PyTorch wheel | Compatibility with PyTorch install set |
| `torchaudio` | matching PyTorch wheel | Compatibility with PyTorch install set |
| `gymnasium` | `1.3.0` | RL environment API |
| `minigrid` | `3.1.0` | DoorKey and related sparse-reward environments |
| `numpy` | `>=2.2,<3` | Array operations and replay preprocessing |
| `scipy` | `>=1.15` | Statistical utilities |
| `pandas` | `>=2.2` | Tabular experiment summaries |
| `polars` | `>=1.30` | Fast large-scale log analysis |
| `pyarrow` | `>=20.0` | Parquet storage for metrics and trajectories |

Experiment management and logging:

| Package | Recommended version | Purpose |
|---|---:|---|
| `wandb` | `>=0.19` | Run tracking, artifacts, config snapshots, sweeps |
| `tensorboard` | `>=2.19` | Local scalar, histogram, and Q-value diagnostics |
| `hydra-core` | `>=1.3` | Structured experiment configuration |
| `omegaconf` | `>=2.3` | Config composition and serialization |
| `pyyaml` | `>=6.0` | Lightweight config and metadata files |
| `tqdm` | `>=4.67` | Progress display |
| `rich` | `>=13.9` | Readable console logs |

Analysis, statistics, and visualization:

| Package | Recommended version | Purpose |
|---|---:|---|
| `matplotlib` | `>=3.10` | Publication-quality static plots |
| `seaborn` | `>=0.13` | Statistical plotting wrappers |
| `plotly` | `>=6.0` | Interactive diagnostics and report appendices |
| `kaleido` | `>=0.2` | Static export for Plotly figures |
| `rliable` | latest available stable release | Bootstrap confidence intervals, probability of improvement, IQM-style reporting |
| `lifelines` | `>=0.30` | Kaplan-Meier survival curves for first-success analysis |
| `statsmodels` | `>=0.14` | Additional statistical checks |

Development and reproducibility utilities:

| Package | Recommended version | Purpose |
|---|---:|---|
| `pytest` | `>=8.3` | Unit tests for replay, n-step targets, wrappers, and metrics |
| `ruff` | `>=0.11` | Linting and formatting |
| `mypy` | optional | Static checks for experiment infrastructure |
| `psutil` | `>=6.1` | System and process diagnostics |

Recommended install pattern:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install gymnasium==1.3.0 minigrid==3.1.0 numpy scipy pandas polars pyarrow wandb tensorboard hydra-core omegaconf pyyaml tqdm rich matplotlib seaborn plotly kaleido rliable lifelines statsmodels pytest ruff psutil
```

If CUDA 13.0 wheels are not compatible with the cluster driver, use the CUDA 12.6 PyTorch wheel index instead and keep every compared run on that same stack.

Reproducibility requirements:

- Save `pip freeze` or an equivalent lockfile with every experiment batch.
- Save the Git commit hash, uncommitted diff status, config file, and random seeds.
- Save the exact MiniGrid/Gymnasium environment ids and any custom DoorKey registration code.
- Store large metrics and trajectory diagnostics as Parquet files, with W&B artifacts or another immutable artifact store for long-running batches.

---

## 2. Team Implementation Pipeline and Experiments

The project should be implemented as a sequential handoff:

1. Team member 1 builds the shared experiment foundation: DQN, n-step DQN, replay logic, environment wrappers, common helper functions, metric logging, evaluation code, and hyperparameter config files.
2. Team member 2 receives that foundation, adds the RIDE intrinsic reward module, integrates it through the shared training interface, and runs the complete four-condition experiment suite.

This split keeps the project coherent: Team member 1 defines the common experimental contract, and Team member 2 extends that contract without changing baseline behavior.

### 2.0 Shared Handoff Contract

Team member 1 must deliver:

- A single training entry point that can run either `algorithm=dqn` or `algorithm=nstep_dqn`.
- Config files for environment id, seed, training budget, evaluation interval, replay buffer, optimizer, epsilon schedule, target network update, batch size, discount, and n-step horizon.
- Common helpers for seeding, environment construction, observation preprocessing, replay storage, n-step return construction, checkpointing, evaluation, metric logging, and result export.
- A stable transition schema containing at least `obs`, `action`, `reward_ext`, `next_obs`, `done`, `truncated`, `env_seed`, `episode_id`, `timestep`, and subgoal flags.
- Unit tests or smoke tests for replay sampling, terminal n-step truncation, seed determinism, and evaluation without learning.
- Baseline pilot results for DQN and n-step DQN on at least DoorKey-6x6 and DoorKey-8x8.

Team member 2 must preserve that contract and add:

- A RIDE module with state embedding, forward dynamics model, inverse dynamics model, intrinsic reward normalization, and auxiliary losses.
- A training reward interface:

```text
r_train = reward_ext + beta * reward_ride
```

- Config extensions for `use_ride`, `beta`, intrinsic reward normalization, intrinsic clipping, RIDE loss weights, and whether the RIDE encoder is shared with the Q-network.
- Logging for intrinsic reward magnitude, RIDE auxiliary losses, and parameter count.
- Final training runs for all four conditions using the same seeds, environment layouts, budgets, and evaluation code.

If Team member 2 needs to change a shared helper because of a bug, all affected baseline and RIDE conditions must be rerun or clearly marked as non-comparable.

---

## Experiment 1: Main 2x2 Factorial Study

### RQ1

In sparse-reward DoorKey, are learning outcomes primarily improved by discovery-oriented intrinsic reward, propagation-oriented n-step backups, or their interaction?

### Conditions

Run the following four conditions:

1. DQN, 1-step backup, no RIDE.
2. DQN + RIDE, 1-step backup.
3. n-step DQN, no RIDE.
4. n-step DQN + RIDE.

Run all four conditions on:

- DoorKey-6x6
- DoorKey-8x8
- DoorKey-16x16

### Team Workflow

Team member 1 phase:

1. Implement the common DQN trainer, target network update, replay buffer, epsilon schedule, evaluation loop, and logging schema.
2. Implement n-step returns behind the same trainer by changing only the backup target configuration.
3. Register configs for `dqn_1step` and `dqn_nstep`.
4. Run smoke tests and pilot baseline runs to verify that DQN and n-step DQN produce comparable logs, checkpoints, and evaluation files.
5. Freeze the baseline interface before handoff.

Team member 2 phase:

1. Add RIDE as an optional intrinsic reward module called from the same training loop.
2. Register configs for `dqn_ride_1step` and `dqn_ride_nstep`.
3. Verify that intrinsic rewards are used during training only and disabled during evaluation.
4. Run the complete four-condition grid with paired seeds.
5. Compute factorial estimates using Team member 1's baseline outputs and Team member 2's RIDE outputs.

### Method

For each environment and seed:

1. Initialize environment, network weights, replay buffer, RIDE module if applicable, and RNG streams using the paired seed protocol.
2. Train each agent for the pre-specified number of environment steps.
3. Evaluate periodically on the fixed held-out evaluation layout set.
4. Log both training diagnostics and evaluation diagnostics using the shared schema.
5. Disable RIDE intrinsic reward during evaluation.

Primary factorial estimates:

Let:

- `Y_00` = DQN
- `Y_10` = DQN + RIDE
- `Y_01` = n-step DQN
- `Y_11` = n-step DQN + RIDE

Then estimate:

```text
RIDE main effect        = mean(Y_10, Y_11) - mean(Y_00, Y_01)
n-step main effect      = mean(Y_01, Y_11) - mean(Y_00, Y_10)
interaction effect      = (Y_11 - Y_01) - (Y_10 - Y_00)
```

Compute these for:

- Final success rate
- Success-rate AUC
- First-success timestep
- First key pickup timestep
- First door-open timestep
- Post-discovery learning speed

### Expected Interpretation

Evidence for discovery bottleneck:

- RIDE strongly improves first-success time.
- RIDE increases key pickup and door opening frequency before first reward.
- n-step has little effect before first success.

Evidence for propagation bottleneck:

- n-step does not necessarily improve first discovery.
- After first successful episode appears, n-step reaches high success thresholds faster.
- n-step improves post-discovery AUC more than pre-discovery subgoal coverage.

Evidence for interaction:

- RIDE creates more successful trajectories.
- n-step exploits those trajectories faster.
- `Y_11` is larger than expected from adding RIDE and n-step main effects.

Ambiguous case:

- If RIDE improves both discovery and post-discovery speed, do not conclude pure discovery.
- If n-step changes exploration behavior before first success, do not conclude pure propagation.
- Use Experiments 2 and 3 to disambiguate.

---

## Experiment 2: Difficulty Sweep and Ceiling/Floor Check

### RQ2

Does the apparent bottleneck depend on environment difficulty?

### Conditions

Use the same four conditions as Experiment 1:

- DQN
- DQN + RIDE
- n-step DQN
- n-step DQN + RIDE

Run across DoorKey sizes:

- 6x6
- 8x8
- 12x12 or another intermediate custom DoorKey size if available
- 16x16

Optional extension:

- KeyCorridorS3R3
- MultiRoom variant

### Team Workflow

Team member 1 phase:

1. Make environment id, custom DoorKey size, training budget, and evaluation interval fully config-driven.
2. Implement fixed evaluation layout generation for every environment size.
3. Run DQN and n-step DQN baselines across the selected difficulty sweep.
4. Export one result table per `environment x algorithm x seed` with the same metric columns used in Experiment 1.

Team member 2 phase:

1. Reuse the exact same environment configs and evaluation layout files.
2. Run DQN + RIDE and n-step DQN + RIDE across the same difficulty sweep.
3. Combine baseline and RIDE results into one analysis table.
4. Report how RIDE main effect, n-step main effect, and interaction change with environment difficulty.

### Method

Train and evaluate using the same protocol as Experiment 1.

Analyze effect sizes separately by environment difficulty.

For each environment, report:

- Final success rate
- AUC
- First-success survival curve
- Subgoal completion curves
- Factorial main effects and interaction

### Interpretation

If DoorKey-6x6 is solved by all agents:

- Treat it as a sanity check.
- Do not use it as decisive evidence for bottleneck structure.

If DoorKey-16x16 is failed by all agents:

- Treat it as a floor-effect regime.
- Increase training budget or include an intermediate task.
- Under the A100/H100 full package, prefer adding an intermediate DoorKey size over overfitting conclusions to an all-failure 16x16 regime.

The most informative regime is where:

- Some agents solve.
- Some agents fail.
- Learning curves are not saturated.

---

## Experiment 3: Fixed Successful Replay Propagation Test

### RQ3

When reward discovery is controlled, does n-step backup propagate rare successful reward signals faster than 1-step backup?

This experiment directly targets the largest confound in the main design: ordinary training mixes discovery and propagation.

### Conditions

Compare:

1. 1-step DQN trained with a fixed replay dataset containing rare successful trajectories.
2. n-step DQN trained with the same fixed replay dataset.

This experiment does not require RIDE conditions. It isolates propagation by holding discovery data fixed.

### Team Workflow

Team member 1 phase:

1. Implement a dataset builder that collects random, epsilon-random, and optional scripted successful DoorKey trajectories.
2. Save fixed replay datasets with metadata for environment id, dataset seed, success ratio, episode count, and transition count.
3. Implement offline replay training for 1-step DQN and n-step DQN using the same Q-network and target update code as Experiment 1.
4. Add diagnostics for Q-values and Bellman error along successful trajectories.
5. Produce pilot fixed-replay results on DoorKey-8x8.

Team member 2 phase:

1. Validate that the fixed replay datasets load through the same transition schema used by online training.
2. Run the final 1-step vs n-step fixed-replay comparison with paired initialization seeds.
3. Periodically evaluate learned policies in the actual environment.
4. Integrate propagation diagnostics into the final result analysis.

### Method

Dataset construction:

1. Collect a replay buffer with a controlled mix of trajectories:
   - Random or epsilon-random trajectories.
   - A small number of successful trajectories.
   - Optional scripted expert trajectories for DoorKey.
2. Keep the exact same dataset for all compared agents.
3. Freeze data collection. No additional online interaction during this test.

Suggested dataset compositions:

- 0.01% successful episodes, if enough compute is available.
- 0.1% successful episodes.
- 1% successful episodes.
- 5% successful episodes.

Training:

1. Initialize agents from the same seed.
2. Train only from the fixed replay dataset.
3. Compare 1-step and n-step targets under identical data.
4. Periodically evaluate the learned policy in the actual environment.

Diagnostics:

- Track Q-values for states/actions along successful trajectories.
- Track how quickly positive value estimates appear at early trajectory states.
- Track Bellman error along the successful trajectory prefix.
- With A100/H100 compute, run 20 paired seeds for each replay composition and include the 0.01% success condition to stress-test reward propagation under extremely rare discovery.

### Interpretation

Strong evidence for propagation benefit:

- n-step learns positive values for early successful-trajectory states faster.
- n-step reaches nonzero evaluation success faster from the same replay data.
- The effect increases when successful trajectories are rare.

Weak evidence:

- n-step only helps online but not fixed replay.
- This suggests the online effect may be partly exploration or replay-distribution related, not pure reward propagation.

---

## Experiment 4: Discovery-Only Pre-Reward Analysis

### RQ4

Does RIDE improve exploration before any extrinsic reward is observed?

### Conditions

Compare:

1. DQN
2. DQN + RIDE
3. n-step DQN
4. n-step DQN + RIDE

The key analysis window is before the first successful episode for each seed.

### Team Workflow

Team member 1 phase:

1. Add common subgoal logging to the environment wrapper or episode logger.
2. Log key pickup, door open, room transition, goal reached, timeout, episode length, object interactions, and coverage fields for every algorithm.
3. Ensure these diagnostics are never exposed to the agent as observations.
4. Produce baseline pre-reward traces for DQN and n-step DQN.

Team member 2 phase:

1. Add RIDE-specific logging for intrinsic reward magnitude, normalized intrinsic reward, and auxiliary losses.
2. Run RIDE conditions using the same pre-reward logging fields.
3. Compare pre-reward event distributions between non-RIDE and RIDE agents.
4. Check whether RIDE increases ordered key-door-goal progress rather than only increasing generic interaction counts.

### Method

For every training episode before first success:

1. Log whether the agent picked up the key.
2. Log whether the agent toggled/opened the door.
3. Log whether the agent entered the second room.
4. Log unique grid cells visited if available from environment metadata.
5. Log object interactions:
   - pickup attempts
   - successful key pickup
   - toggle attempts
   - successful door opening
6. Log intrinsic reward magnitude for RIDE agents.

Do not use these privileged logs as agent inputs.

Analyze:

- Time to first key pickup.
- Time to first door opening.
- Time to first room transition.
- Time to first goal.
- Coverage growth over environment steps.
- Ordered subgoal completion probability.

### Interpretation

Evidence that RIDE improves discovery:

- RIDE shifts key/door/goal first-event distributions earlier.
- RIDE increases object interaction rates before any extrinsic reward.
- RIDE improves state coverage or meaningful subgoal coverage, not merely random wandering.

Potential failure mode:

- RIDE may over-reward controllable but irrelevant interactions.
- If RIDE increases pickup/toggle attempts but not ordered key-door-goal progress, then it may improve novelty without solving task-relevant discovery.

---

## 3. Evaluation Metrics

## 3.1 Primary Performance Metrics

### Final Success Rate

Definition:

```text
success_rate = number of successful evaluation episodes / total evaluation episodes
```

Use the final checkpoint and optionally the best checkpoint.

Report both:

- Final checkpoint success rate.
- Best checkpoint success rate.

Reason:

- Final performance captures convergence.
- Best performance detects instability or collapse.

### Success-Rate AUC

Definition:

Area under the evaluation success-rate curve over environment steps.

Use normalized AUC:

```text
AUC = integral success_rate(t) dt / total_training_steps
```

Reason:

- Captures sample efficiency.
- Avoids over-focusing on final checkpoint.

### Extrinsic Return

Definition:

Mean original environment reward during evaluation.

Report separately from success rate because MiniGrid success reward often includes a step penalty. Two agents can have the same success rate but different efficiency.

### Episode Length on Success

Definition:

Mean number of steps among successful evaluation episodes.

Reason:

- Shorter successful trajectories indicate more efficient policies.
- Useful when success rate saturates.

---

## 3.2 Discovery Metrics

### First Success Timestep

Definition:

The first training timestep at which the agent completes a successful episode.

Use survival analysis style reporting:

- Some seeds may never succeed.
- Treat never-success seeds as right-censored at the training budget.

Report:

- Median time to first success.
- Fraction of seeds succeeding by each timestep.
- Kaplan-Meier-style survival curve if possible.

### First Key Pickup Timestep

Definition:

The first training timestep at which the agent successfully picks up the key.

Reason:

- Key pickup is the first major prerequisite for DoorKey.
- It can occur before any extrinsic reward.

### First Door Open Timestep

Definition:

The first training timestep at which the agent successfully opens/unlocks the door.

Reason:

- Door opening is a later ordered subgoal.
- It is more task-relevant than generic state coverage.

### First Room Transition Timestep

Definition:

The first timestep at which the agent reaches the area beyond the door.

Reason:

- Distinguishes merely opening the door from actually exploiting the opened path.

### Ordered Subgoal Completion Rate

For each evaluation or training episode, record whether the following ordered chain occurred:

```text
key pickup -> door open -> goal reached
```

Report:

- `P(key)`
- `P(door)`
- `P(goal)`
- `P(door | key)`
- `P(goal | door)`
- `P(goal | key and door)`

Reason:

- This decomposes where the agent fails in the task sequence.

### State Coverage

Definition:

Number or fraction of unique underlying grid positions visited during training.

Use only for logging, not as agent input.

Report:

- Unique cells visited per episode.
- Cumulative unique cells visited.
- Unique object-adjacent states visited.

Caution:

- High coverage is not necessarily task-relevant discovery.
- Always pair coverage with ordered subgoal metrics.

### Object Interaction Frequency

Count per episode:

- pickup actions
- successful key pickup
- toggle actions
- successful door opening
- attempts to toggle irrelevant objects or walls

Reason:

- RIDE should encourage controllable object interactions.
- But excessive irrelevant interaction may indicate misdirected curiosity.

---

## 3.3 Propagation Metrics

### Post-Discovery Learning Speed

Definition:

For each seed that obtains a first success, measure the number of environment steps from first success to reaching a fixed evaluation success-rate threshold.

Suggested thresholds:

- 25% success
- 50% success
- 75% success

Important:

- Treat seeds that never reach threshold as censored.
- Do not drop failed seeds silently.

Report:

- Time from first success to 25%, 50%, 75% success.
- Fraction of seeds reaching each threshold.
- Censored-time summary.

### Post-Discovery AUC

Definition:

Success-rate AUC computed only after the first successful training episode for each seed.

Reason:

- Separates reward exploitation after discovery from the pre-discovery search phase.

Caution:

- Must handle seeds with no first success.
- Report how many seeds are excluded or censored.

### Value Propagation Along Successful Trajectory

Use saved successful trajectories.

For states/actions along a successful trajectory, log:

```text
Q(s_t, a_t)
```

at multiple training checkpoints.

Report:

- How quickly Q-values become positive near the goal.
- How quickly positive values move backward toward early states.
- Slope of value increase by distance-to-goal.

Reason:

- Directly measures whether reward information is propagating backward through the trajectory.

### Bellman Error Along Successful Trajectory

For saved successful trajectories, compute TD error or n-step target error.

Report:

- Mean TD error by position in trajectory.
- TD error decay over training.
- Difference between 1-step and n-step targets.

Reason:

- Helps distinguish value propagation from behavioral exploration.

---

## 3.4 Interaction Metrics

### Factorial Interaction on AUC

Definition:

```text
interaction_AUC = (AUC_nstep_RIDE - AUC_nstep_noRIDE)
                - (AUC_1step_RIDE - AUC_1step_noRIDE)
```

Positive interaction means:

- RIDE helps more when n-step backup is present than when it is absent.

### Factorial Interaction on Discovery

Same formula, but using negative time-to-event metrics or event probabilities.

Examples:

- First success timestep.
- First door-open timestep.
- Probability of reaching goal by timestep `T`.

### Factorial Interaction on Propagation

Same formula, but using:

- Time from first success to 50% success.
- Post-discovery AUC.
- Value propagation slope.

Interpretation:

- If interaction appears mainly in discovery metrics, the combination may help create successful trajectories.
- If interaction appears mainly in propagation metrics, n-step may be exploiting RIDE-generated trajectories better.

---

## 3.5 Statistical Reporting

Use paired analysis wherever possible.

Required reporting:

- Mean and median across seeds.
- 95% bootstrap confidence intervals.
- Stratified bootstrap across environment difficulty when aggregating multiple environments.
- Probability of improvement for key pairwise comparisons.
- Individual seed learning curves in appendix or supplementary plots.

Recommended aggregate metrics:

- Median
- Interquartile mean if aggregating across several tasks
- Probability of improvement

Avoid:

- Claiming superiority from point estimates alone.
- Selecting the best seed.
- Reporting only smoothed curves without raw variability.

Pairwise comparisons to report:

1. DQN vs DQN + RIDE
2. DQN vs n-step DQN
3. DQN + RIDE vs n-step DQN + RIDE
4. n-step DQN vs n-step DQN + RIDE
5. DQN vs n-step DQN + RIDE

Factorial effects to report:

- RIDE main effect
- n-step main effect
- RIDE x n-step interaction

Report these for:

- Final success rate
- Success-rate AUC
- First success
- First key pickup
- First door open
- Post-discovery learning speed

---

## 4. Decision Rules

### Discovery-Dominant Bottleneck

Conclude discovery is dominant only if:

- RIDE improves first key pickup, first door opening, and first success.
- RIDE improves pre-reward exploration or ordered subgoal completion.
- n-step does not substantially improve pre-discovery metrics.
- Post-discovery differences are smaller than discovery differences.

### Propagation-Dominant Bottleneck

Conclude propagation is dominant only if:

- n-step improves post-discovery learning speed.
- n-step improves value propagation along successful trajectories.
- Fixed successful replay experiment shows faster learning with n-step.
- Discovery metrics are similar between 1-step and n-step before first success.

### Complementary Interaction

Conclude complementarity only if:

- The RIDE x n-step interaction is positive with uncertainty intervals mostly above zero.
- The interaction appears in AUC or success rate.
- Mechanism metrics explain why:
  - RIDE increases successful trajectories.
  - n-step exploits them faster.

### No Clear Bottleneck

Conclude no clear bottleneck if:

- Effects are inconsistent across environment sizes.
- Confidence intervals are wide.
- First-success and post-discovery metrics disagree.
- Fixed-replay propagation results and pre-reward discovery diagnostics disagree with the online factorial results.

---

## 5. Minimum Viable Experimental Package

If time is limited, run this reduced but defensible package:

1. Main 2x2 factorial on DoorKey-6x6, 8x8, and 16x16.
2. 10 paired seeds per condition.
3. Fixed held-out evaluation layouts.
4. Metrics:
   - final success rate
   - success-rate AUC
   - first success timestep
   - first key pickup timestep
   - first door open timestep
   - post-discovery time to 50% success
5. Fixed successful replay propagation test on DoorKey-8x8.
6. Bootstrap confidence intervals and probability of improvement.

This minimum package is much stronger than the original proposal because it directly checks whether the planned interventions actually affect the intended mechanisms.

---

## 6. A100/H100 Full Experimental Package

If A100/H100 compute is available, use it to improve statistical reliability and mechanism identification:

1. Main 2x2 factorial on DoorKey-6x6, 8x8, an intermediate DoorKey size such as 12x12, and 16x16.
2. 30 paired seeds per main condition.
3. 200 evaluation episodes per checkpoint for DoorKey-6x6 and DoorKey-8x8; 300-500 for DoorKey-16x16.
4. Increased training budgets:
   - DoorKey-6x6: 250k-500k steps.
   - DoorKey-8x8: 2M steps.
   - Intermediate DoorKey: 5M steps.
   - DoorKey-16x16: 10M steps.
5. Fixed successful replay propagation test on DoorKey-8x8 with 0.01%, 0.1%, 1%, and 5% successful-episode mixtures.
6. 20 paired seeds for the fixed replay propagation test if feasible.
7. Pre-reward discovery analysis for all main conditions.
8. Keep the main model, batch size, and update-to-data ratio fixed across compared conditions.
9. Schedule compute as many independent `environment x algorithm x seed` jobs rather than one oversized learner.

Rationale:

- A100/H100-class hardware does not remove sparse-reward variance.
- The main benefit of additional compute is narrower confidence intervals, better survival analysis, more precise post-discovery timing, and stronger fixed-replay propagation tests.
- Increasing model size, update ratio, or parallel actor count can change the causal mechanism being studied, so do not change them inside the main experimental package.

---

## 7. References to Use in Final Report

- Mnih et al. (2015), "Human-level control through deep reinforcement learning." Nature.
- Sutton and Barto (2018/2020), "Reinforcement Learning: An Introduction." n-step returns and temporal-difference learning.
- Hessel et al. (2018), "Rainbow: Combining Improvements in Deep Reinforcement Learning." Multi-step returns as a DQN extension.
- Raileanu and Rocktaschel (2020), "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS.
- Chevalier-Boisvert et al. (2023), "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks." NeurIPS Datasets and Benchmarks.
