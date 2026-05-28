# CS377 RL Project README (26.05.21 - DDQN/n-step DDQN + RIDE foundation)

이 패키지는 DQN / n-step DQN baseline과 RIDE intrinsic reward 조건, 공통 replay schema, n-step return 구성, MiniGrid 환경 생성, symbolic observation preprocessing, greedy evaluation, metric logging, checkpoint 저장/복원, fixed successful replay dataset 수집, offline propagation training, successful trajectory diagnostic, 기초 분석 table 생성을 제공한다.
 
| Backup target | RIDE 없음 | RIDE 있음 |
|---|---|---|
| 1-step TD | DQN | DQN + RIDE |
| n-step TD | n-step DQN | n-step DQN + RIDE |

현재 이 레포지토리는 상기된 2*2 설계의 네 조건, 즉 `dqn_1step`, `dqn_nstep`, `dqn_ride_1step`, `dqn_ride_nstep`을 같은 trainer와 같은 logging/evaluation contract 위에서 실행하는 foundation이다.

RIDE 조건도 seed protocol, environment factory, preprocessing, replay transition schema, evaluation loop, episode/evaluation log schema를 baseline과 공유한다.

이 코드베이스에서 DQN baseline은 기본적으로 Double DQN target을 사용하는 DQN 계열을 뜻한다. Main comparison에서는 `agent.double_dqn=true`를 모든 condition에 고정하고, vanilla DQN ablation을 같은 비교 grid에 섞지 않는다.

## 실험 번호 빠른 참조

| 번호 | 문서상 이름 | README에서 자주 쓰는 표현 |
|---|---|---|
| Exp. 1 | Main 2x2 Factorial Study | main factorial, 2x2 factorial design, DQN/RIDE/n-step 4-condition comparison |
| Exp. 2 | Difficulty Sweep and Ceiling/Floor Check | difficulty sweep, DoorKey size sweep, DoorKey-6x6/8x8/12x12/16x16 sweep |
| Exp. 3 | Fixed Successful Replay Propagation Test | fixed replay propagation test, fixed replay, offline propagation test |
| Exp. 4 | Discovery-Only Pre-Reward Analysis | pre-reward discovery analysis, pre-reward subgoal/coverage diagnostics |

## 구현 범위

`src` 디렉터리에는 실제 패키지 코드인 `src/rl_project/`와 editable install 이후 생기는 packaging metadata인 `src/rl_project.egg-info/`가 있다. 사용자가 읽고 수정해야 하는 코드는 `src/rl_project/` 아래의 Python 모듈이다. `__pycache__`, `.DS_Store`, `rl_project.egg-info`는 실행/설치 과정에서 생기는 보조 산출물이며 실험 로직이 아니다.

- MiniGrid DoorKey 6x6, 8x8, 12x12, 16x16 difficulty sweep(Exp. 2) 설정 지원
- `algorithm=dqn_1step`: 1-step Double DQN baseline
- `algorithm=dqn_nstep`: 3-step Double DQN baseline
- `algorithm=dqn_ride_1step`: 1-step Double DQN + RIDE intrinsic reward
- `algorithm=dqn_ride_nstep`: 3-step Double DQN + RIDE intrinsic reward
- Hydra/OmegaConf config 기반 실행
- symbolic partial observation flattening 및 channel scaling
- replay buffer와 `Transition` schema
- terminal/truncated episode에서 truncate되는 n-step return
- timeout transition에서는 DQN bootstrap target 유지
- epsilon-greedy action selection, target network update, Huber loss, gradient clipping
- fixed held-out evaluation seed 생성 및 greedy evaluation
- evaluation 중 model weight, optimizer step, replay size가 변하지 않는지 runtime check
- episode/evaluation table을 CSV/Parquet으로 저장
- scalar log를 CSV 및 TensorBoard로 저장
- PyTorch checkpoint 저장/복원 및 run metadata 저장
- random/epsilon-random/scripted-success/mixed fixed replay dataset 수집
- RIDE state embedding, forward dynamics, inverse dynamics, episodic count scaled intrinsic reward, auxiliary loss logging
- offline DQN/n-step DQN propagation training 및 successful trajectory prefix diagnostic(Exp. 3)
- run output aggregation, success-rate AUC, bootstrap summary, factorial effect table 생성
- smoke/unit test: replay, n-step, seeding, evaluation side effect, config package, online/offline smoke run

범위 밖 구성요소는 서버 job orchestration, 최종 논문용 plotting/report generation, fixed replay propagation diagnostic의 최종 통계/시각화 해석, pre-reward discovery analysis(Exp. 4)의 최종 보고서용 분석 루프다.

## 설치 및 실행 전제

패키지는 `pyproject.toml` 기준 Python 3.11 이상을 요구한다. 루트 디렉터리에서 editable 설치 후 실행하는 흐름을 권장한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

GPU가 있으면 `device=auto`가 CUDA, macOS MPS, CPU 순서로 장치를 선택한다. 재현성과 비교 가능성을 위해 같은 experimental batch 안에서는 PyTorch/CUDA build, config, seed protocol을 고정해야 한다.

## 주요 실행 명령

온라인 1-step DQN smoke run:

```bash
python -m rl_project.train algorithm=dqn_1step package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
```

온라인 n-step DQN smoke run:

```bash
python -m rl_project.train algorithm=dqn_nstep package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
```

온라인 RIDE smoke run:

```bash
python -m rl_project.train algorithm=dqn_ride_1step package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
python -m rl_project.train algorithm=dqn_ride_nstep package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
```

DoorKey-8x8 minimum package pilot(Exp. 1 baseline 및 Exp. 2 difficulty sweep 일부):

```bash
python -m rl_project.train algorithm=dqn_1step package=minimum env=doorkey_8x8 seed=0
python -m rl_project.train algorithm=dqn_nstep package=minimum env=doorkey_8x8 seed=0
```

DoorKey-12x12 full package run(Exp. 2 difficulty sweep):

```bash
python -m rl_project.train algorithm=dqn_nstep package=full env=doorkey_12x12 seed=0
```

fixed replay offline smoke run(Exp. 3 fixed replay propagation test):

```bash
python -m rl_project.offline_train algorithm=dqn_nstep package=minimum replay=fixed_replay env=doorkey_6x6 seed=0 smoke.enabled=true
```

pytest:

```bash
pytest
```

실험 결과 aggregation:

```bash
python -m rl_project.analyze outputs --out analysis
```

`tests/` 디렉터리는 논문용 실험 결과를 생성하는 곳이 아니라, `pytest`로 실행되는 코드 검증용 test suite다. 실제 실험 조건은 `configs/`로 지정하고, 학습/평가 결과는 `outputs/`에 저장된다.

여기서 smoke run은 “논문용 결과를 얻는 본 실험”이 아니라, 전체 학습 파이프라인이 깨지지 않았는지 빠르게 확인하는 짧은 실행이다. `smoke.enabled=true`를 켜면 training steps, evaluation interval, evaluation episodes, batch size, replay capacity 등이 작게 override되어 checkpoint/log/evaluation 파일 생성 여부를 몇십-몇백 step 안에서 확인할 수 있다.

## Config 구조

Hydra config의 entrypoint는 루트의 `configs/config.yaml`이다.

```text
configs/
  config.yaml
  algorithm/
    dqn_1step.yaml
    dqn_nstep.yaml
  env/
    doorkey_6x6.yaml
    doorkey_8x8.yaml
    doorkey_12x12.yaml
    doorkey_16x16.yaml
  package/
    minimum.yaml
    full.yaml
  replay/
    online.yaml
    fixed_replay.yaml
```

기본값은 `package=minimum`, `algorithm=dqn_1step`, `env=doorkey_6x6`, `replay=online`, `seed=0`, `device=auto`다.

주요 config 항목:

| 항목 | 의미 |
|---|---|
| `algorithm.name` | 실행 알고리즘 이름. 지원 값은 `dqn_1step`, `dqn_nstep` |
| `algorithm.n_step` | n-step horizon. 1-step은 `1`, n-step baseline은 `3` |
| `package.name` | 실험 규모. `minimum` 또는 `full` |
| `package.training_steps` | 환경별 training step budget |
| `package.eval_interval` | 환경별 evaluation 주기 |
| `package.eval_episodes` | checkpoint당 평가 episode 수 |
| `agent.gamma` | discount factor |
| `agent.batch_size` | replay sampling batch size |
| `agent.replay_capacity` | replay buffer capacity |
| `agent.learning_rate` | Adam learning rate |
| `agent.target_update_interval` | target network hard update 주기 |
| `agent.epsilon_start/end/decay_steps` | epsilon-greedy exploration schedule |
| `agent.learning_starts` | update 시작 전 최소 환경 step |
| `agent.update_to_data_ratio` | environment step당 update 반복 수. 비교 실험에서는 1 유지 권장 |
| `agent.double_dqn` | Double DQN target 사용 여부 |
| `replay.collector_policy` | fixed replay collector. `random`, `epsilon_random`, `scripted_success`, `mixed` |
| `replay.episode_count` | fixed replay dataset episode 수 |
| `replay.target_success_ratio` | mixed replay에서 목표 successful episode 비율 |
| `replay.target_success_episodes` | mixed replay에서 목표 successful episode 수. 설정되면 ratio보다 우선 |
| `replay.offline_updates` | offline propagation trainer update 횟수 |
| `replay.diagnostic_interval` | successful-prefix diagnostic 기록 주기 |
| `smoke.enabled` | 빠른 테스트용 budget override 사용 여부 |
| `env_max_steps` | MiniGrid default max steps를 override할 때 사용 |

`minimum` package는 DoorKey 6x6, 8x8, 16x16 및 condition당 10 paired seeds를 기준으로 한다. `full` package는 DoorKey 12x12를 포함하고 condition당 30 paired seeds, 더 긴 budget, 더 많은 evaluation episodes를 사용한다. 이 DoorKey size 구성은 difficulty sweep(Exp. 2)의 핵심 축이다.

## 출력물

Hydra 실행 디렉터리는 기본적으로 다음 패턴을 따른다.

```text
outputs/YYYY-MM-DD/HH-MM-SS_${algorithm.name}_${env}_seed${seed}/
```

각 run directory에는 다음 파일이 생성된다.

```text
resolved_config.yaml
run_metadata.json
scalars.csv
tensorboard/
tables/
  episodes.parquet
  episodes.csv
  evaluation.parquet
  evaluation.csv
checkpoints/
  step_<global_step>.pt
```

offline fixed replay run(Exp. 3)에서는 추가로 다음이 생긴다.

```text
fixed_replay/
  transitions.parquet
  metadata.json
tables/
  offline_success_prefix_diagnostics.parquet
  offline_success_prefix_diagnostics.csv
checkpoints/
  offline_final.pt
```

## 데이터 흐름

온라인 학습은 다음 순서로 진행된다.

1. `train.py`가 Hydra config를 읽고 `trainer.run_training`을 호출한다.
2. `Trainer`가 seed를 고정하고, `envs.make_env`로 MiniGrid DoorKey 환경을 만든다.
3. `ImgObsWrapper`가 mission/direction dict 대신 symbolic image observation을 반환한다.
4. `preprocessing.preprocess_obs`가 `7 x 7 x 3` symbolic image를 scaled flat vector로 바꾼다.
5. `DQNAgent.act`가 epsilon-greedy policy로 action을 선택한다.
6. 환경 step 이후 `envs.extract_diagnostics`가 key/door/room/goal/timeout/action-attempt diagnostic을 추출한다.
7. 원 transition이 `NStepTransitionBuffer`에 들어가고, ready transition은 n-step return으로 변환되어 `ReplayBuffer`에 저장된다.
8. `learning_starts`, `batch_size`, `train_interval` 조건을 만족하면 replay에서 batch를 sample해 `DQNAgent.update`를 수행한다.
9. episode 종료 시 `EpisodeTracker`가 episode-level metric row를 만든다.
10. `eval_interval`마다 `evaluate_agent`가 fixed eval seeds에서 greedy evaluation을 수행한다.
11. `RunLogger`가 scalars, tables, checkpoints, resolved config를 저장한다.

## Transition schema

`replay.Transition`은 online replay, fixed replay dataset(Exp. 3), offline trainer가 공유하는 핵심 schema다.

| field | 의미 |
|---|---|
| `obs` | 전처리된 현재 observation vector |
| `action` | MiniGrid action id |
| `reward_ext` | extrinsic reward 또는 n-step discounted extrinsic return |
| `next_obs` | bootstrap에 사용할 다음 observation vector |
| `done` | environment terminal 여부 |
| `truncated` | time-limit truncate 여부 |
| `env_seed` | 해당 episode reset seed |
| `episode_id` | training run 내부 episode 번호 |
| `timestep` | episode 내부 timestep |
| `picked_key` | key를 들고 있는 상태였는지 |
| `opened_door` | door가 열린 상태였는지 |
| `entered_second_room` | door 너머 영역에 진입했는지 |
| `reached_goal` | positive reward와 함께 terminal goal에 도달했는지 |
| `timeout` | truncate로 episode가 끝났는지 |
| `pickup_attempt` | action id 3을 실행했는지 |
| `toggle_attempt` | action id 5를 실행했는지 |
| `cell_position` | privileged diagnostic용 underlying grid position |
| `actual_n` | terminal/truncated 때문에 실제로 사용된 n-step 길이 |

`cell_position`과 subgoal flags는 logging/analysis 전용 privileged diagnostic이다. agent observation이나 action selection에는 사용하지 않아야 한다.

## 평가 원칙

평가는 항상 extrinsic reward 기준 greedy evaluation이다. `evaluate_agent`는 다음 side effect가 없도록 체크한다.

- model weight 변경 금지
- optimizer step count 변경 금지
- replay buffer size 변경 금지
- evaluation 중 replay insert 및 optimizer update 없음

RIDE가 추가되더라도 evaluation에서는 intrinsic reward를 완전히 꺼야 한다.

## 파일별 역할과 사용법

### `__init__.py`

패키지 설명 문자열과 `__version__ = "0.1.0"`만 둔 가벼운 package marker다.

사용법:

```python
import rl_project
print(rl_project.__version__)
```

### `checkpointing.py`

PyTorch checkpoint 저장/복원 helper다. `save_checkpoint`는 agent state, `global_step`, `episode_id`, config snapshot, run metadata를 `.pt` 파일에 저장한다. `load_checkpoint`는 agent의 device에 맞춰 checkpoint를 읽고 `agent.load_state_dict`를 호출한다.

사용법:

```python
from rl_project.checkpointing import save_checkpoint, load_checkpoint

save_checkpoint("outputs/run/checkpoints/step_1000.pt", agent, 1000, 12, cfg)
state = load_checkpoint("outputs/run/checkpoints/step_1000.pt", agent)
```

주의: checkpoint 내부 agent state format은 `DQNAgent.state_dict()`와 맞아야 한다.

### `config_utils.py`

Hydra config에서 선택된 environment key를 읽고 실제 environment config mapping을 resolve한다.

- `env_key(cfg)`: `cfg.env`를 문자열로 반환한다.
- `resolve_env_config(cfg)`: `cfg.environments[cfg.env]`를 OmegaConf object로 반환한다. `env_max_steps`가 지정되어 있으면 해당 값을 `max_steps`로 override한다.

사용법:

```python
from rl_project.config_utils import env_key, resolve_env_config

selected = env_key(cfg)
env_cfg = resolve_env_config(cfg)
```

잘못된 `env` 이름을 넘기면 가능한 environment 목록을 포함한 `KeyError`를 던진다.

### `datasets.py`

fixed replay propagation test(Exp. 3)를 위한 dataset builder/loader다. RIDE 없이 random/epsilon-random/scripted-success/mixed fixed replay dataset을 만들 수 있다.

- `collect_fixed_replay(cfg, output_dir)`: `random`, `epsilon_random`, `scripted_success`, `mixed` policy로 fixed replay dataset을 수집해 `transitions.parquet`과 `metadata.json`을 저장하고 parquet path를 반환한다.
- `collect_random_replay(cfg, output_dir)`: random replay 수집용 convenience wrapper다.
- `load_replay_rows(path)`: parquet transition table을 list of dict로 읽는다.
- `scripted_doorkey_actions(env)`: DoorKey grid에서 key, door, goal 위치를 읽어 성공 trajectory action sequence를 만든다.

사용법:

```python
from rl_project.datasets import collect_fixed_replay, load_replay_rows

data_path = collect_fixed_replay(cfg, "outputs/run/fixed_replay")
rows = load_replay_rows(data_path)
```

metadata에는 `dataset_id`, `env_id`, `package`, `dataset_seed`, `target_success_ratio`, `target_success_episodes`, `success_ratio`, `episode_count`, `transition_count`, `successful_episode_count`, `collector_policy`, `collector_episode_counts`, `collection_attempts`, `created_at`, `schema_version`이 저장된다.

`mixed` collector는 scripted successful episode를 목표 개수만큼 먼저 수집한 뒤 random 또는 epsilon-random episode로 나머지를 채운다. filler episode가 우연히 성공해 목표 성공 수를 초과하면 해당 episode는 dataset에 넣지 않고 다시 수집한다.

### `dqn_agent.py`

DQN/Double DQN agent 구현이다. 하나의 `QNetwork`를 online network로, deep copy를 target network로 사용한다. Adam optimizer, Smooth L1 loss, epsilon schedule, Double DQN target, gradient clipping, target hard update를 포함한다.

주요 API:

- `DQNAgent(obs_dim, action_dim, cfg, device)`: agent 생성
- `epsilon(step)`: linear epsilon decay 값 반환
- `act(obs, step, greedy=False)`: observation에서 action 선택
- `update(batch, gamma, n_step_default)`: replay batch로 한 번 학습하고 `DQNUpdate(loss, q_mean, target_mean)` 반환
- `state_dict()`, `load_state_dict(state)`: checkpoint용 state 변환

사용법:

```python
from rl_project.dqn_agent import DQNAgent

agent = DQNAgent(obs_dim, action_dim, cfg.agent, device)
action = agent.act(obs, step=global_step)
update = agent.update(batch, gamma=cfg.agent.gamma, n_step_default=cfg.algorithm.n_step)
```

`update`는 batch의 `actual_n`을 사용해 bootstrap discount를 `gamma ** actual_n`으로 계산한다. batch에 `actual_n`이 없으면 `n_step_default`를 사용한다. `done=True` transition에서는 bootstrap을 끊고, `truncated=True, done=False` timeout transition에서는 bootstrap을 유지한다.

### `envs.py`

MiniGrid DoorKey 환경 생성과 diagnostic 추출을 담당한다.

- `CUSTOM_DOORKEY_IDS`: `MiniGrid-DoorKey-12x12-v0`, `MiniGrid-DoorKey-16x16-v0` 등록 정보
- `register_custom_doorkey_envs()`: custom DoorKey id를 Gymnasium registry에 등록한다.
- `make_env(env_cfg, seed=None)`: 환경을 만들고 `ImgObsWrapper`를 씌운 뒤 seed/action_space seed를 설정한다.
- `extract_diagnostics(env, action, reward, done, truncated)`: key/door/room/goal/timeout/action-attempt/cell-position diagnostic을 dict로 반환한다.

사용법:

```python
from rl_project.envs import make_env, extract_diagnostics

env = make_env(env_cfg, seed=0)
obs, _ = env.reset(seed=0)
next_obs, reward, done, truncated, _ = env.step(action)
diag = extract_diagnostics(env, action, reward, done, truncated)
```

주의: `extract_diagnostics`는 `env.unwrapped`의 internal grid와 agent position을 읽는다. 이 정보는 logging/analysis 전용이며 agent input으로 쓰면 안 된다.

### `evaluate.py`

greedy evaluation loop와 summary helper다.

- `evaluate_agent(agent, cfg, eval_seeds, checkpoint_step, run_id)`: fixed eval seeds에서 greedy policy로 평가하고 row list를 반환한다.
- `summarize_eval(rows)`: `eval/success_rate`, `eval/return_ext_mean` scalar를 계산한다.

사용법:

```python
from rl_project.evaluate import evaluate_agent, summarize_eval

rows = evaluate_agent(agent, cfg, eval_seeds, checkpoint_step=global_step, run_id=run_id)
summary = summarize_eval(rows)
```

이 함수는 평가 전후 online network state와 optimizer step count를 비교해 evaluation side effect를 감지한다.

### `logging_utils.py`

run output directory 안에 scalar, TensorBoard, episode table, evaluation table, resolved config, run metadata, checkpoint directory를 관리하는 logger다.

- `RunLogger(run_dir, cfg)`: directory 생성, `resolved_config.yaml` 및 `run_metadata.json` 저장, CSV/TensorBoard writer 준비
- `scalar(name, value, step)`: scalar를 `scalars.csv`와 TensorBoard에 기록
- `episode(row)`: episode row buffer에 추가
- `eval(rows)`: evaluation row buffer에 추가
- `flush_tables()`: episode/evaluation rows를 CSV/Parquet으로 저장
- `close()`: table flush, TensorBoard writer close, scalar file close

사용법:

```python
from rl_project.logging_utils import RunLogger

logger = RunLogger(run_dir, cfg)
logger.scalar("train/loss", loss, step)
logger.episode(episode_row)
logger.eval(eval_rows)
logger.close()
```

### `metrics.py`

episode-level training diagnostics를 누적하는 `EpisodeTracker`를 정의한다. 한 episode 안에서 return, length, success, timeout, key/door/room progress, pickup/toggle attempt count, unique cell count, first-event step을 누적한다.

사용법:

```python
from rl_project.metrics import EpisodeTracker

tracker = EpisodeTracker(run_id, algorithm, package, env_id, seed, episode_id, global_step)
tracker.update(reward, diag)
row = tracker.row()
```

baseline에서는 `episode_return_train`이 `episode_return_ext`와 같다. RIDE가 추가되면 training reward는 intrinsic reward를 포함할 수 있으므로 두 값을 분리해서 유지해야 한다.

### `networks.py`

Q-value approximation에 쓰는 작은 MLP다.

- 입력: flattened symbolic observation vector
- hidden: `Linear -> ReLU -> Linear -> ReLU`
- 출력: action별 Q-value

사용법:

```python
from rl_project.networks import QNetwork

q_net = QNetwork(obs_dim=147, action_dim=7, hidden_dim=128)
q_values = q_net(obs_batch)
```

MiniGrid partial symbolic observation은 일반적으로 `7 x 7 x 3 = 147` 차원으로 flatten된다.

### `nstep.py`

n-step return construction을 담당한다.

- `NStepTransitionBuffer(n_step, gamma)`: episode-local transition queue
- `append(transition)`: 새 transition을 넣고 replay에 넣을 준비가 된 n-step `Transition` list를 반환한다.

사용법:

```python
from rl_project.nstep import NStepTransitionBuffer

nstep = NStepTransitionBuffer(n_step=3, gamma=0.99)
for ready in nstep.append(transition):
    replay.add(ready)
```

`append`는 queue 길이가 `n_step`에 도달하면 첫 transition 기준 n-step return을 만든다. `done` 또는 `truncated`가 들어오면 queue에 남은 transition들을 terminal/truncated 지점까지 flush한다. 반환 transition의 `reward_ext`는 discounted n-step return이고, `actual_n`은 실제 누적 길이다.

### `offline_train.py`

fixed replay propagation test(Exp. 3)용 offline training entrypoint다. Hydra로 실행할 수 있으며, replay dataset이 없으면 `datasets.collect_fixed_replay`로 fixed replay를 먼저 만든다.

주요 흐름:

1. config와 smoke override 적용
2. output logger 생성
3. `cfg.replay.dataset_path`가 있으면 해당 parquet 사용, 없으면 fixed replay 수집
4. parquet row를 `Transition`으로 복원
5. offline buffer에 n-step 변환 transition 저장
6. `DQNAgent`를 고정 dataset에서 제한된 횟수만 update
7. `offline/loss`, `offline/q_mean` 기록
8. successful trajectory prefix의 Q-value/target/Bellman error diagnostic table 저장
9. `offline_final.pt` 저장

사용법:

```bash
python -m rl_project.offline_train replay=fixed_replay algorithm=dqn_1step env=doorkey_8x8 seed=0
python -m rl_project.offline_train replay=fixed_replay algorithm=dqn_nstep env=doorkey_8x8 seed=0 replay.collector_policy=mixed replay.target_success_episodes=1 replay.episode_count=20
python -m rl_project.offline_train replay=fixed_replay algorithm=dqn_nstep env=doorkey_8x8 seed=0 replay.dataset_path=/path/to/transitions.parquet
```

diagnostic table에는 `update_step`, `algorithm`, `episode_id`, `prefix_timestep`, `action`, `n_step`, `actual_n`, `q_value`, `target`, `bellman_error`, `reward_ext`, `done`, `truncated`가 기록된다.

### `preprocessing.py`

MiniGrid symbolic image observation을 network input vector로 바꾼다.

- `preprocess_obs(obs)`: `HxWx3` 배열을 `float32`로 변환하고 object/color/state channel을 각각 `10.0`, `5.0`, `3.0`으로 나눈 뒤 flatten한다.
- `preprocess_obs_tensor(obs, device)`: 위 전처리를 적용한 후 batch dimension을 추가한 Torch tensor를 만든다.

사용법:

```python
from rl_project.preprocessing import preprocess_obs, preprocess_obs_tensor

obs_vec = preprocess_obs(obs)
obs_tensor = preprocess_obs_tensor(obs, device)
```

잘못된 shape이 들어오면 `ValueError`를 던진다. 전처리는 one-hot encoding이 아니라 channel scaling 방식을 사용한다.

### `replay.py`

공통 transition schema와 circular replay buffer 구현이다.

- `DIAGNOSTIC_FIELDS`: diagnostic field 이름 모음
- `Transition`: replay/fixed dataset/offline training이 공유하는 dataclass
- `Transition.to_row()`: numpy 배열과 tuple을 parquet/JSON 친화적인 list로 변환
- `ReplayBuffer(capacity, obs_shape, device)`: fixed-capacity circular buffer
- `ReplayBuffer.add(transition)`: transition 저장
- `ReplayBuffer.sample(batch_size)`: Torch tensor batch 반환
- `ReplayBuffer.rows()`: buffer 내용을 row list로 반환

사용법:

```python
from rl_project.replay import ReplayBuffer, Transition

transition = Transition(obs, action, reward_ext, next_obs, done, truncated, env_seed, episode_id, timestep)
replay = ReplayBuffer(capacity=100000, obs_shape=obs.shape, device=device)
replay.add(transition)
batch = replay.sample(128)
```

`sample` 결과에는 `obs`, `actions`, `reward_ext`, `next_obs`, `done`, `truncated`, `actual_n`, subgoal diagnostic tensors가 포함된다.

### `run_context.py`

Hydra 실행 환경과 일반 Python 호출 환경 모두에서 output directory를 결정하는 helper다.

- `resolve_output_dir()`: Hydra가 초기화되어 있으면 `HydraConfig.get().runtime.output_dir`, 아니면 현재 작업 디렉터리 아래 `outputs/manual` 반환

사용법:

```python
from rl_project.run_context import resolve_output_dir

run_dir = resolve_output_dir()
```

### `run_metadata.py`

실험 재현성 확인에 필요한 실행 환경 metadata를 모은다.

- `collect_run_metadata(cfg, device=None)`: Python/platform/Numpy/Torch/device 정보, git commit, dirty status, seed protocol을 dict로 반환한다.

`RunLogger`는 이 metadata를 `run_metadata.json`에 저장하고, checkpoint 저장 시에도 같은 metadata가 포함된다. replay buffer와 RNG state를 포함하는 완전 resume checkpoint는 제공하지 않는다.

### `seeding.py`

재현 가능한 paired seed protocol을 위한 helper다.

- `set_global_seeds(seed)`: Python `random`, NumPy, PyTorch, CUDA seed 설정
- `SeedStream(seed, stride=100000)`: training episode seed와 eval seed를 disjoint range로 생성
- `fixed_eval_seeds(seed, count)`: 고정 held-out evaluation seed list 반환

사용법:

```python
from rl_project.seeding import set_global_seeds, SeedStream, fixed_eval_seeds

set_global_seeds(0)
stream = SeedStream(0)
env_seed = stream.env_seed(episode_id=3)
eval_seeds = fixed_eval_seeds(seed=0, count=100)
```

seed mapping은 training episode seed를 `seed * 100000 + episode_id`, evaluation seed를 `(seed + 1) * 100000 + 50000 + index`로 만들어 둘을 분리한다.

### `train.py`

온라인 학습 CLI entrypoint다. Hydra config를 읽어 `trainer.run_training(cfg)`를 호출하고 완료된 run directory를 출력한다.

사용법:

```bash
python -m rl_project.train algorithm=dqn_1step package=minimum env=doorkey_6x6 seed=0
python -m rl_project.train algorithm=dqn_nstep package=minimum env=doorkey_8x8 seed=1
```

직접 import해서 쓰려면 보통 `trainer.run_training`을 호출하는 편이 낫다.

### `trainer.py`

온라인 학습 loop의 중심 모듈이다.

주요 함수/클래스:

- `resolve_device(device_name)`: `auto`, `cuda`, `mps`, `cpu` 등 device 문자열을 Torch device로 변환
- `env_value(mapping, key, default=None)`: 환경별 config mapping에서 key 또는 `default` 값을 읽음
- `apply_smoke_overrides(cfg)`: `smoke.enabled=true`일 때 training/eval/batch/replay budget을 줄임
- `Trainer(cfg)`: 온라인 학습에 필요한 env, agent, replay, n-step buffer, logger, seed stream 초기화
- `Trainer.train()`: 전체 online training loop 실행
- `run_training(cfg)`: `Trainer(cfg).train()` convenience wrapper

사용법:

```python
from rl_project.trainer import run_training, Trainer

run_dir = run_training(cfg)
# 또는
trainer = Trainer(cfg)
run_dir = trainer.train()
```

`Trainer.train()`의 핵심 contract:

- 동일 trainer에서 `cfg.algorithm.n_step`만 바꿔 1-step DQN과 n-step DQN을 비교한다.
- baseline에서는 training reward와 extrinsic reward가 같다.
- evaluation은 replay size와 model/optimizer state를 바꾸지 않는다.
- checkpoint는 `logging.save_interval` 및 마지막 step에 저장된다.
- episode table과 evaluation table은 run 종료 시 flush된다.

## 테스트가 보장하는 것

test suite는 다음을 확인한다.

- replay sample shape/dtype 및 diagnostic flag 보존
- terminal transition에서 n-step return truncate
- `n_step=1`이 일반 1-step DQN target input과 일치
- bootstrap discount가 `gamma ** actual_n`을 사용
- timeout으로 인한 `truncated=True` transition은 DQN target에서 bootstrap을 유지
- scripted DoorKey collector가 successful fixed replay episode를 생성
- mixed fixed replay collector가 목표 successful episode count와 metadata를 보존
- fixed eval seeds가 재현 가능하고 training env seeds와 겹치지 않음
- evaluation이 model weights와 optimizer step count를 바꾸지 않음
- `minimum`/`full` package override 값이 문서 의도와 일치
- 짧은 online smoke run에서 checkpoint/scalar/evaluation output 생성
- fixed replay offline smoke run(Exp. 3) 완료 및 `offline_final.pt`, success-prefix diagnostic table 생성

## 확장 시 지켜야 할 계약

RIDE나 추가 알고리즘을 붙일 때는 다음을 유지해야 한다.

- agent가 보는 observation은 `preprocessing.py` output에 한정한다.
- diagnostic fields는 logging/analysis 전용으로 유지한다.
- evaluation은 extrinsic reward만 사용하고 학습 side effect가 없어야 한다.
- baseline과 RIDE condition은 같은 seed stream, evaluation seeds, environment config, training budget을 공유한다.
- shared helper의 behavior가 바뀌면 영향을 받은 baseline 결과는 다시 실행하거나 non-comparable로 표시한다.
- `Transition` schema를 수정할 때는 offline replay와 downstream analysis까지 함께 migration해야 한다.
- main factorial comparison(Exp. 1)에서는 network architecture, batch size, replay capacity, target update interval, update-to-data ratio를 condition 간 고정한다.
