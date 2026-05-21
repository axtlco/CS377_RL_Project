# Phase 1 Plan: Team Member 1 Shared RL Experiment Foundation

## 1. Role and Boundary

이 문서는 `plan.md`에서 팀원 1이 맡아야 하는 Phase 1 작업물을 별도로 정리한 구현 설계 문서다. 팀원 1의 목표는 RIDE를 제외한 모든 공통 실험 기반을 안정적으로 만드는 것이다. 즉, DQN과 n-step DQN이 같은 trainer, 같은 replay schema, 같은 evaluation loop, 같은 logging/export contract 위에서 실행되어야 한다.

팀원 1이 구현할 범위는 다음과 같다.

- DQN 및 n-step DQN 학습 루프
- replay buffer 및 n-step return logic
- MiniGrid DoorKey environment factory
- symbolic observation preprocessing
- checkpointing, evaluation, metric logging, result export
- experiment package config
- fixed replay propagation test scaffold
- smoke/unit tests

팀원 1은 RIDE 자체를 구현하지 않으며, RIDE의 내부 API 모양도 미리 고정하지 않는다. RIDE의 state embedding, forward dynamics model, inverse dynamics model, intrinsic reward normalization, auxiliary losses, reward composition 방식은 팀원 2의 범위다.

Phase 1에서 고정해야 하는 것은 RIDE 구현 자리가 아니라 baseline과 future RIDE 조건이 공유할 실험 contract다.

- 같은 environment factory
- 같은 symbolic observation preprocessing
- 같은 evaluation protocol
- 같은 seed/layout protocol
- 같은 episode/evaluation logging schema
- 같은 result export format

팀원 2가 RIDE를 추가할 때 shared helper를 수정해야 한다면, 영향을 받는 baseline과 RIDE 조건은 다시 실행하거나 non-comparable로 명확히 표시한다.

팀원 1의 산출물은 팀원 2가 RIDE를 추가하기 전에도 독립적으로 실행 가능해야 하며, `algorithm=dqn_1step`과 `algorithm=dqn_nstep` baseline 결과를 생성할 수 있어야 한다.

## 2. Deliverables

Phase 1의 필수 산출물은 다음과 같다.

- 단일 training entrypoint: `python -m rl_project.train`
- 실행 가능한 algorithm config: `algorithm=dqn_1step`, `algorithm=dqn_nstep`
- Hydra/OmegaConf 기반 config hierarchy
- MiniGrid DoorKey environment factory
- DoorKey-12x12 custom/intermediate variant 등록 자리
- symbolic partial observation preprocessing
- DQN/Double DQN Q-network 및 agent
- replay buffer, transition schema, n-step target construction
- target network update, epsilon schedule, optimizer setup
- greedy evaluation loop
- fixed held-out evaluation layout generation
- metric logging, Parquet/CSV export, TensorBoard logging
- checkpoint save/load utility
- fixed replay dataset builder scaffold
- offline 1-step/n-step propagation trainer scaffold
- replay, n-step, seeding, evaluation, config override에 대한 smoke/unit tests

Phase 1 종료 시점에는 최소한 DoorKey-6x6과 DoorKey-8x8에서 DQN 및 n-step DQN pilot run을 실행할 수 있어야 한다.

## 3. Config Package Design

실험 난이도는 사용자가 config에서 `package=minimum` 또는 `package=full`만 바꿔 조절할 수 있게 설계한다.

권장 실행 예시는 다음과 같다.

```bash
python -m rl_project.train package=minimum algorithm=dqn_1step env=doorkey_8x8 seed=0
python -m rl_project.train package=minimum algorithm=dqn_nstep env=doorkey_8x8 seed=0
python -m rl_project.train package=full algorithm=dqn_nstep env=doorkey_12x12 seed=0
python -m rl_project.offline_train package=minimum replay=fixed_replay env=doorkey_8x8 seed=0
```

`package=minimum`은 `plan.md` Section 5의 Minimum Viable Experimental Package를 따른다.

```yaml
name: minimum
envs: [doorkey_6x6, doorkey_8x8, doorkey_16x16]
paired_seeds: 10
eval_episodes:
  default: 100
training_steps:
  doorkey_6x6: 250000
  doorkey_8x8: 1000000
  doorkey_16x16: 5000000
eval_interval:
  doorkey_6x6: 5000
  doorkey_8x8: 10000
  doorkey_16x16: 25000
fixed_replay_env: doorkey_8x8
fixed_replay_success_mixtures: [0.001, 0.01, 0.05]
fixed_replay_seeds: 10
```

`package=full`은 `plan.md` Section 6의 A100/H100 Full Experimental Package를 따른다.

```yaml
name: full
envs: [doorkey_6x6, doorkey_8x8, doorkey_12x12, doorkey_16x16]
paired_seeds: 30
eval_episodes:
  doorkey_6x6: 200
  doorkey_8x8: 200
  doorkey_12x12: 300
  doorkey_16x16: 500
training_steps:
  doorkey_6x6: 500000
  doorkey_8x8: 2000000
  doorkey_12x12: 5000000
  doorkey_16x16: 10000000
eval_interval:
  doorkey_6x6: 5000
  doorkey_8x8: 10000
  doorkey_12x12: 25000
  doorkey_16x16: 25000
fixed_replay_env: doorkey_8x8
fixed_replay_success_mixtures: [0.0001, 0.001, 0.01, 0.05]
fixed_replay_seeds: 20
```

algorithm config는 backup target만 바꿔 동일 trainer를 사용하게 한다.

```yaml
# algorithm/dqn_1step.yaml
name: dqn_1step
n_step: 1
```

```yaml
# algorithm/dqn_nstep.yaml
name: dqn_nstep
n_step: 3
```

공통 agent 설정은 모든 baseline 조건에서 동일해야 한다.

- `gamma`
- `batch_size: 128`
- `replay_capacity`
- `learning_rate`
- `target_update_interval`
- `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`
- `update_to_data_ratio: 1`
- `double_dqn: true`

`double_dqn: true`를 기본 안정성 옵션으로 사용할 수 있지만, 이 경우 보고서와 log에는 baseline 이름을 Double DQN 계열로 명확히 기록한다.

## 4. Project Structure

권장 프로젝트 구조는 다음과 같다.

```text
configs/
  config.yaml
  package/
    minimum.yaml
    full.yaml
  algorithm/
    dqn_1step.yaml
    dqn_nstep.yaml
  env/
    doorkey_6x6.yaml
    doorkey_8x8.yaml
    doorkey_12x12.yaml
    doorkey_16x16.yaml
  replay/
    online.yaml
    fixed_replay.yaml

src/rl_project/
  train.py
  evaluate.py
  offline_train.py
  config_schema.py
  envs.py
  preprocessing.py
  replay.py
  nstep.py
  networks.py
  dqn_agent.py
  trainer.py
  logging_utils.py
  metrics.py
  datasets.py
  seeding.py
  checkpointing.py

tests/
  test_replay.py
  test_nstep.py
  test_seeding.py
  test_evaluation.py
  test_config_packages.py
```

## 5. Shared Training Contract

모든 online/offline baseline 학습 코드는 같은 transition schema를 사용한다. Phase 1에서 이 schema를 고정해야 팀원 2가 RIDE를 추가해도 baseline과 RIDE 결과를 같은 분석 코드로 합칠 수 있다.

필수 transition fields는 다음과 같다.

```text
obs
action
reward_ext
next_obs
done
truncated
env_seed
episode_id
timestep
picked_key
opened_door
entered_second_room
reached_goal
timeout
pickup_attempt
toggle_attempt
cell_position
```

이 schema의 privileged diagnostic fields는 logging과 analysis 전용이다. agent observation, network input, action selection에는 절대 사용하지 않는다.

episode log schema는 다음 fields를 포함한다.

```text
run_id
algorithm
package
env_id
seed
global_step
episode_id
episode_return_ext
episode_return_train
episode_length
success
timeout
picked_key
opened_door
entered_second_room
pickup_attempts
toggle_attempts
unique_cells
first_key_step
first_door_step
first_room_step
first_success_step
```

evaluation log schema는 다음 fields를 포함한다.

```text
run_id
algorithm
package
env_id
seed
checkpoint_step
eval_seed
success
return_ext
episode_length
picked_key
opened_door
entered_second_room
```

evaluation은 반드시 extrinsic reward 기준으로만 수행한다.

- evaluation reward는 원본 MiniGrid extrinsic reward만 사용한다.
- evaluation 중 replay insert를 하지 않는다.
- evaluation 중 optimizer step을 하지 않는다.
- evaluation 중 epsilon exploration을 사용하지 않는다.
- evaluation 중 model parameter와 replay buffer size가 변하면 안 된다.

저장 형식은 다음을 기본값으로 한다.

- scalar logs: TensorBoard 및 CSV
- episode/evaluation tables: Parquet
- config snapshot: YAML
- checkpoints: PyTorch `.pt`
- fixed replay metadata: YAML 또는 JSON
- large fixed replay transition data: Parquet 또는 compressed array format

## 6. Future RIDE Integration Boundary

Phase 1은 RIDE holder, placeholder config, intrinsic reward provider interface를 만들지 않는다. 대신 팀원 2가 보존해야 할 boundary를 문서화한다.

- RIDE 조건도 Phase 1의 environment factory, preprocessing, seed/layout protocol, evaluation loop, episode/evaluation export schema를 재사용한다.
- RIDE-specific scalar logs, auxiliary losses, intrinsic reward diagnostics는 팀원 2가 별도 schema extension으로 추가한다.
- RIDE가 shared trainer, replay, evaluation helper의 behavior를 바꾸면 baseline comparability가 깨질 수 있으므로, 영향을 받는 baseline은 rerun하거나 non-comparable로 표시한다.
- Evaluation은 RIDE 구현 여부와 무관하게 항상 extrinsic reward 기준이며 학습 side effect가 없어야 한다.

## 7. Fixed Replay Propagation Test

Phase 1은 Experiment 3의 fixed replay propagation test를 실행 가능한 scaffold로 제공한다.

필수 구성은 다음과 같다.

- fixed replay dataset builder
- random trajectory collector
- epsilon-random trajectory collector
- optional scripted successful DoorKey trajectory collector 자리
- dataset metadata 저장
- offline 1-step DQN trainer
- offline n-step DQN trainer
- successful trajectory prefix에 대한 `Q(s_t, a_t)` logging
- successful trajectory prefix에 대한 Bellman error logging
- fixed replay dataset을 online replay와 같은 transition schema로 load하는 기능

dataset metadata는 다음 fields를 포함한다.

```text
dataset_id
env_id
package
dataset_seed
success_ratio
episode_count
transition_count
successful_episode_count
collector_policy
created_at
schema_version
```

`package=minimum`에서는 DoorKey-8x8 fixed replay test를 기본 대상으로 삼는다. `package=full`에서는 성공 episode mixture를 더 세밀하게 둔다.

Phase 1의 목표는 RIDE 없는 1-step vs n-step propagation pilot 결과를 만드는 것이다. 팀원 2는 같은 dataset loader를 검증하고 final paired-seed comparison과 전체 분석에 통합한다.

## 8. Required Tests

Phase 1에서 반드시 포함할 테스트는 다음과 같다.

- replay sampling이 expected shape, dtype, device를 반환하는지 확인한다.
- replay sampling이 `done`, `truncated`, subgoal flags를 보존하는지 확인한다.
- terminal transition에서 n-step return이 정확히 truncation되는지 확인한다.
- `n_step=1` target이 일반 DQN 1-step target과 동일한지 확인한다.
- bootstrap term이 `gamma ** actual_n`을 사용하는지 확인한다.
- 같은 seed/config에서 environment seed stream과 evaluation layout이 재현되는지 확인한다.
- evaluation 실행 중 model weights, replay buffer size, optimizer step count가 변하지 않는지 확인한다.
- baseline 상태에서 `episode_return_train == episode_return_ext`인지 확인한다.
- `package=minimum`과 `package=full`이 env list, seed count, budget, eval episodes를 올바르게 override하는지 확인한다.
- DoorKey-6x6 짧은 smoke run에서 `dqn_1step` checkpoint/log/eval file이 생성되는지 확인한다.
- DoorKey-6x6 짧은 smoke run에서 `dqn_nstep` checkpoint/log/eval file이 생성되는지 확인한다.
- fixed replay smoke run에서 동일 dataset으로 1-step/n-step offline trainer가 모두 완료되는지 확인한다.

## 9. Assumptions

이 설계는 다음 전제를 둔다.

- 현재 repository에는 실험 구현 코드가 없으므로 새 Python package를 생성한다.
- Python runtime은 `3.11` 계열을 사용한다.
- deep-learning framework는 PyTorch를 사용한다.
- environment는 Gymnasium 및 MiniGrid를 사용한다.
- config system은 Hydra/OmegaConf를 사용한다.
- 기본 관측은 MiniGrid symbolic partial observation이다.
- mission string은 DoorKey task family에서 constant이므로 무시하거나 fixed task id로 처리한다.
- action masking은 사용하지 않는다.
- main network architecture, batch size, update-to-data ratio는 compared conditions 사이에서 고정한다.
- 팀원 1은 RIDE model, intrinsic reward, RIDE API, RIDE placeholder config를 구현하지 않는다.
- 기본 실행 난이도는 `package=minimum`이다.
- A100/H100 compute가 가능하면 사용자는 `package=full`로 바꿔 더 많은 seeds, 더 긴 budgets, 더 많은 evaluation episodes를 실행한다.
- 팀원 2는 팀원 1의 shared contract를 보존하면서 RIDE 조건을 추가한다.
