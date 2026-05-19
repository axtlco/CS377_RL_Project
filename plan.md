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

This plan keeps that core design, but adds environment difficulty sweeps, mechanism-specific metrics, and control experiments that reduce confounding between discovery and propagation.

The key methodological principle is:

> Do not infer mechanism from final success rate alone. Infer mechanism only when performance effects agree with mechanism-specific diagnostics.

---

## 1. Experimental Environment Specification

### 1.1 Benchmark Family

Primary benchmark:

- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-DoorKey-8x8-v0`
- `MiniGrid-DoorKey-16x16-v0`

Optional robustness benchmark if compute allows:

- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-MultiRoom-N4-S5-v0` or a comparable MultiRoom variant

Rationale:

- DoorKey has ordered sparse-reward structure: find key, pick up key, unlock/open door, reach goal.
- The environment is partially observable with a local egocentric field of view.
- The official MiniGrid documentation describes DoorKey as difficult for classical RL because of sparse rewards and useful for curiosity or curriculum learning.
- The difficulty sweep is necessary because DoorKey-6x6 alone may be too easy, producing ceiling effects that obscure whether discovery or propagation is the real bottleneck.

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

Do not use privileged full-state observations in the main experiment.

Add one diagnostic oracle condition using full observations only as a control, not as the main result. This checks whether apparent discovery/propagation failures are actually caused by partial observability.

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

Optional diagnostic:

- A separate action-masked condition can be run to estimate how much failure comes from wasting actions on irrelevant commands, but it should not be mixed with the main comparisons.

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

Seed pairing:

- For seed `s`, all four main agents should use the same initialization seed, environment seed stream, replay sampling seed, and evaluation layout set where possible.
- This reduces variance when estimating factorial effects.

Evaluation layout protocol:

- Create a fixed held-out evaluation set of environment seeds for each environment size.
- Suggested: 100 evaluation episodes per checkpoint for DoorKey-6x6 and DoorKey-8x8; 200 for DoorKey-16x16 if variance is high.
- Evaluation episodes use greedy policy or low-exploration policy.
- No learning occurs during evaluation.

### 1.7 Training Budget

Use a difficulty-dependent budget.

Initial recommended budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k | every 5k |
| DoorKey-8x8 | 1M | every 10k |
| DoorKey-16x16 | 5M | every 25k |
| KeyCorridorS3R3 | 5M | every 25k |

Before final runs, perform a small pilot with 3 seeds to check whether:

- All agents fail completely.
- All agents solve too early.
- The evaluation interval is too coarse to estimate first success and post-discovery speed.

If all agents solve DoorKey-6x6 quickly, keep it as a sanity check but do not make it the main evidence.

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

### 1.9 n-step Backup Specification

Compare:

- `n = 1`
- `n = 3`

Optional sensitivity:

- `n = 5`
- `n = 10`

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

---

## 2. Experiments

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

### Method

For each environment and seed:

1. Initialize environment, network weights, replay buffer, and RNG streams using paired seed protocol.
2. Train each agent for the pre-specified number of environment steps.
3. Evaluate periodically on the fixed held-out evaluation layout set.
4. Log both training diagnostics and evaluation diagnostics.
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
- 16x16

Optional extension:

- KeyCorridorS3R3
- MultiRoom variant

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

Optional:

3. 1-step DQN with prioritized replay.
4. n-step DQN with prioritized replay.

The optional prioritized replay conditions should be clearly marked as diagnostics, not part of the main factorial design.

### Method

Dataset construction:

1. Collect a replay buffer with a controlled mix of trajectories:
   - Random or epsilon-random trajectories.
   - A small number of successful trajectories.
   - Optional scripted expert trajectories for DoorKey.
2. Keep the exact same dataset for all compared agents.
3. Freeze data collection. No additional online interaction during this test.

Suggested dataset compositions:

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

### Method

For every training episode before first success:

1. Log whether the agent picked up the key.
2. Log whether the agent toggled/opened the door.
3. Log whether the agent entered the second room.
4. Log unique grid cells visited if full state is available for logging.
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

## Experiment 5: Partial Observability Control

### RQ5

Are apparent discovery or propagation bottlenecks actually caused by partial observability and missing memory?

### Conditions

Run a reduced set:

1. Main DQN with partial observation.
2. Main n-step DQN with partial observation.
3. Full-observation DQN.
4. Full-observation n-step DQN.

Optional:

5. Recurrent DQN with partial observation.
6. Recurrent n-step DQN with partial observation.

### Method

Full-observation diagnostic:

- Provide a full symbolic grid observation to the network.
- Keep action space and reward unchanged.
- This is an oracle diagnostic, not the main benchmark.

Recurrent diagnostic:

- Add a GRU or LSTM after the observation encoder.
- Train with truncated backpropagation through time.
- Keep all other hyperparameters as close as possible.

### Interpretation

If full-observation agents learn much faster:

- Some failures attributed to discovery/propagation may actually be observability failures.

If recurrent agents close the gap:

- Memory is a relevant mechanism and should be discussed as a limitation of feedforward DQN.

If full observation does not change the pattern:

- The original discovery/propagation interpretation becomes more credible.

---

## Experiment 6: Hyperparameter Sensitivity and Fairness Checks

### RQ6

Are conclusions robust to key hyperparameters, or are they artifacts of one RIDE coefficient or one n-step horizon?

### Conditions

RIDE coefficient sweep:

- `beta = 0.01`
- `beta = 0.05`
- `beta = 0.1`
- `beta = 0.5`

n-step horizon sweep:

- `n = 1`
- `n = 3`
- `n = 5`
- `n = 10`

Run sensitivity primarily on DoorKey-8x8, because it is likely to be less saturated than 6x6 and cheaper than 16x16.

### Method

Use fewer seeds for sensitivity:

- 5 seeds per hyperparameter setting for screening.
- Re-run the selected setting with the full seed count for main results.

Report:

- Best setting chosen before final test.
- Whether final conclusions change under nearby settings.
- Whether RIDE or n-step is unusually brittle.

### Interpretation

Robust conclusion:

- Same qualitative pattern appears across several plausible beta and n values.

Fragile conclusion:

- Interaction appears only for one beta or one n value.
- Then report the result as hyperparameter-dependent rather than general.

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
- Full-observation or recurrent controls change the pattern substantially.

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

## 6. References to Use in Final Report

- Mnih et al. (2015), "Human-level control through deep reinforcement learning." Nature.
- Sutton and Barto (2018/2020), "Reinforcement Learning: An Introduction." n-step returns and temporal-difference learning.
- Hessel et al. (2018), "Rainbow: Combining Improvements in Deep Reinforcement Learning." Multi-step returns as a DQN extension.
- Raileanu and Rocktaschel (2020), "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS.
- Chevalier-Boisvert et al. (2023), "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks." NeurIPS Datasets and Benchmarks.
