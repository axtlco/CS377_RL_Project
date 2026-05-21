# 프로젝트 계획서: Sparse-Reward RL에서 Discovery와 Propagation 분리 분석

## 0. 핵심 목표

이 프로젝트는 MiniGrid DoorKey에서 sparse-reward 강화학습이 실패하는 주된 이유가 무엇인지 검증한다.

1. **Reward discovery bottleneck**: 에이전트가 보상이 있는 상태나 보상을 얻기 위한 선행 상태에 거의 도달하지 못하는 문제.
2. **Reward propagation bottleneck**: 드물게 성공 trajectory를 관측하더라도, 그 보상 신호가 앞쪽 의사결정으로 충분히 효율적으로 전파되지 않는 문제.
3. **Interaction**: exploration과 credit assignment가 서로 비가산적으로 돕는 경우.

원래 proposal의 핵심은 다음 2x2 factorial design이다.

| Backup target | Intrinsic discovery module 없음 | RIDE intrinsic discovery module 있음 |
|---|---|---|
| 1-step TD | DQN | DQN + RIDE |
| n-step TD | n-step DQN | n-step DQN + RIDE |

이 계획서는 위 핵심 설계를 유지하되, 환경 난이도 sweep, mechanism-specific metric, fixed-replay propagation test, pre-reward discovery analysis를 추가해 discovery와 propagation의 confounding을 줄인다.

가장 중요한 방법론적 원칙은 다음과 같다.

> 최종 success rate만 보고 메커니즘을 추론하지 않는다. 성능 변화와 mechanism-specific diagnostic이 일관되게 맞아떨어질 때만 mechanism을 해석한다.

Compute assumption:

- 권장 full design은 A100/H100급 GPU 접근을 가정한다.
- MiniGrid symbolic DQN은 보통 이런 GPU에서 FLOP-bound가 아니다. 추가 compute는 agent를 키우기보다 seed 수, evaluation precision, difficulty coverage, diagnostic logging을 늘리는 데 사용한다.
- GPU를 더 채우기 위해 main agent를 훨씬 큰 모델이나 더 높은 update-to-data ratio의 learner로 바꾸지 않는다. 그렇게 하면 discovery/propagation mechanism 자체가 달라질 수 있다.

---

## 1. 실험 환경 Specification

### 1.1 Benchmark Family

Primary benchmark:

- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-DoorKey-8x8-v0`
- Custom 또는 registered intermediate DoorKey variant, 가급적 `DoorKey-12x12`
- `MiniGrid-DoorKey-16x16-v0`

Compute가 허용될 경우 optional robustness benchmark:

- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-MultiRoom-N4-S5-v0` 또는 유사한 MultiRoom variant

선정 이유:

- DoorKey는 순서가 있는 sparse-reward 구조를 가진다. 에이전트는 key를 찾고, key를 집고, door를 unlock/open한 뒤, goal에 도달해야 한다.
- 환경은 partial observable이며 local egocentric field of view를 사용한다.
- 공식 MiniGrid 문서는 DoorKey가 sparse reward 때문에 classical RL에 어렵고 curiosity 또는 curriculum learning에 유용하다고 설명한다.
- DoorKey-6x6만 쓰면 너무 쉬워 ceiling effect가 생길 수 있고, discovery와 propagation 중 무엇이 진짜 bottleneck인지 가리기 어렵다.
- A100/H100 compute가 있다면 8x8이 너무 쉽고 16x16이 너무 어려울 때 중간 난이도인 10x10 또는 12x12 DoorKey variant를 포함한다. 가장 정보량이 큰 regime은 일부 seed는 solve하고 일부 seed는 fail하는 구간이다.

### 1.2 Observation Space

MiniGrid의 기본 partial observation을 사용한다.

- Local egocentric image: `7 x 7 x 3`
- Agent direction
- Mission string

권장 preprocessing:

- RGB pixel 대신 symbolic MiniGrid observation을 사용한다.
- Tile object/color/state channel을 one-hot 또는 embedding feature로 변환한다.
- Agent direction은 one-hot vector로 포함한다.
- DoorKey에서는 mission이 task family 안에서 고정되어 있으므로 mission text는 무시한다. 포함할 경우 learned language model보다 fixed task id로 encode한다.

실험에서는 privileged full-state observation을 사용하지 않는다.

### 1.3 Action Space

MiniGrid의 native discrete action space를 그대로 사용한다.

| Action id | Action |
|---|---|
| 0 | turn left |
| 1 | turn right |
| 2 | move forward |
| 3 | pick up |
| 4 | drop |
| 5 | toggle |
| 6 | done |

Main experiment에서는 action masking을 사용하지 않는다.

이유:

- Action masking은 task prior knowledge를 주입하며 exploration difficulty를 낮춘다.
- Invalid action을 유지해야 원래 sparse-reward challenge가 보존된다.

### 1.4 Reward Function

Evaluation reward:

- 원래 MiniGrid extrinsic reward만 사용한다.
- Success reward는 보통 goal에 도달했을 때만 주어지는 positive sparse reward이다.
- 실패 episode는 extrinsic reward 0을 받는다.

Training reward:

- Non-RIDE agent: extrinsic reward만 사용.
- RIDE agent: training 중에만 extrinsic reward + intrinsic RIDE bonus 사용.
- Evaluation에서는 intrinsic reward를 완전히 비활성화한다.

중요 reporting rule:

- Extrinsic evaluation performance와 intrinsic training reward를 항상 분리해서 보고한다.
- RIDE가 켜진 상태에서 training return으로 알고리즘을 비교하지 않는다. Reward scale이 알고리즘별로 달라지기 때문이다.

### 1.5 Episode Termination and Time Limits

명시적으로 override하지 않는 한 environment default maximum episode length를 사용한다.

각 episode마다 다음을 기록한다.

- Episode length
- Goal 도달 여부
- Timeout 여부
- Key pickup 여부
- Door toggle/open 여부
- Door 너머 room에 도달했는지 여부

### 1.6 Training and Evaluation Seeds

모든 알고리즘에서 paired random seed를 사용한다.

권장 seed 수:

- 최소: condition당 10 seeds. 원래 proposal과 동일하다.
- 권장: main DoorKey sweep에서는 condition당 20 seeds.
- A100/H100 full package: main condition당 30 paired seeds.

Seed pairing:

- Seed `s`에 대해 네 가지 main agent가 가능한 한 같은 initialization seed, environment seed stream, replay sampling seed, evaluation layout set을 사용하게 한다.
- 이렇게 하면 factorial effect 추정 variance를 줄일 수 있다.

Evaluation layout protocol:

- 각 environment size마다 fixed held-out evaluation seed set을 만든다.
- 권장: DoorKey-6x6과 DoorKey-8x8은 checkpoint당 100 evaluation episodes, DoorKey-16x16은 variance가 높으면 200 episodes.
- A100/H100 full package: DoorKey-6x6과 DoorKey-8x8은 checkpoint당 200 evaluation episodes, DoorKey-16x16 또는 high-variance task는 300-500 episodes.
- Evaluation episode는 greedy policy 또는 low-exploration policy를 사용한다.
- Evaluation 중에는 학습하지 않는다.

### 1.7 Training Budget

난이도에 따라 training budget을 다르게 사용한다.

초기 권장 budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k | every 5k |
| DoorKey-8x8 | 1M | every 10k |
| DoorKey-12x12 또는 intermediate custom DoorKey | 3M | every 10k-25k |
| DoorKey-16x16 | 5M | every 25k |
| KeyCorridorS3R3 | 5M | every 25k |

A100/H100 full-package budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k-500k | every 5k |
| DoorKey-8x8 | 2M | every 10k |
| DoorKey-12x12 또는 intermediate custom DoorKey | 5M | every 10k-25k |
| DoorKey-16x16 | 10M | every 10k-25k |
| KeyCorridorS3R3 | 10M | every 25k |

Final run 전에 5-8 paired seeds로 pilot을 수행해 다음을 확인한다.

- 모든 agent가 완전히 실패하는지.
- 모든 agent가 너무 빨리 solve하는지.
- Evaluation interval이 first success와 post-discovery speed를 추정하기에 너무 성긴지.

모든 agent가 DoorKey-6x6을 빠르게 solve한다면 sanity check로만 유지하고 main evidence로 삼지 않는다.

A100/H100 budget에서도 DoorKey-16x16이 floor-effect regime으로 남는다면 거기서 결론을 억지로 내리지 않는다. Mechanism identification의 primary environment로 intermediate DoorKey size를 사용한다.

### 1.8 Base Agent Specification

네 가지 main cell이 하나의 shared value-based agent implementation을 사용한다.

Main agent:

- Experience replay와 target network를 사용하는 DQN.
- 모든 condition에서 동일한 network architecture.
- 동일한 optimizer, learning rate, replay buffer size, target update period, epsilon schedule, batch size, discount, training budget.

Symbolic partial observation을 위한 권장 architecture:

- Input: one-hot encoded `7 x 7` symbolic observation + direction.
- Encoder: small convolutional network 또는 flattened symbolic feature에 대한 compact MLP.
- Head: 7개 action의 Q-value를 출력하는 fully connected layers.
- Main experiment에서는 recurrent memory를 사용하지 않는다.

권장 stability variant:

- Vanilla DQN이 불안정하면 Double DQN을 사용해도 된다. 단, 이 경우 baseline을 일관되게 Double DQN이라고 부른다.
- Main factorial design에는 dueling network, prioritized replay, noisy nets, distributional RL 또는 다른 Rainbow component를 섞지 않는다. 추가 mechanism을 도입하기 때문이다.

A100/H100 implementation guidance:

- Main network는 작게 유지하고 condition 간 고정한다.
- Main update-to-data ratio는 1로 유지한다. 더 높은 ratio는 propagation을 인위적으로 강화해 n-step 비교를 confound할 수 있다.
- Main batch size는 안정성 문제가 없는 한 128로 유지한다.
- GPU parallelism은 많은 independent `seed/condition` job을 동시에 돌리는 방식으로 사용한다. GPU utilization을 높이기 위해 single-agent data collection process를 바꾸지 않는다.

### 1.9 n-step Backup Specification

비교:

- `n = 1`
- `n = 3`

Main comparison은 사전에 정한 하나의 `n`을 사용하며, 권장은 `n = 3`이다. Rainbow-style DQN에서 3-step return이 흔히 쓰이고, multi-step target은 새롭게 관측된 reward를 더 빠르게 전파하는 것으로 알려져 있기 때문이다.

Implementation details:

- n-step target을 계산할 수 있도록 replay에 충분한 sequence 정보를 저장한다.
- 1-step agent와 n-step agent는 같은 replay buffer capacity와 sampling strategy를 사용한다.
- Terminal transition에서는 n-step return을 올바르게 truncate한다.
- Bootstrap term에는 `gamma^n`이 적용되어야 한다.

### 1.10 RIDE Specification

RIDE agent는 learned representation space에서 action이 만든 impact에 기반한 intrinsic reward bonus를 받는다.

필수 components:

- State embedding network.
- Forward dynamics model.
- Inverse dynamics model.
- Action으로 인한 representation 변화량에 비례하는 intrinsic reward.

Training reward:

```text
r_total = r_extrinsic + beta * r_RIDE
```

RIDE reporting requirements:

- `beta`를 보고한다.
- Intrinsic reward normalization method를 보고한다.
- Intrinsic reward clipping 여부를 보고한다.
- RIDE가 Q-network와 encoder를 공유하는지 보고한다.
- Forward/inverse dynamics auxiliary loss weight를 보고한다.

중요 fairness rule:

- RIDE가 parameter와 auxiliary loss를 추가한다면 parameter count와 wall-clock training time을 보고한다.
- Main comparison은 environment steps 기준이지만 compute overhead도 공개해야 한다.

### 1.11 Software Environment and Dependencies

모든 실험에 하나의 reproducible Python virtual environment를 사용한다.

권장 Python runtime:

- Python: `3.11`
- 최소 patch 권장: Python 3.11 series의 `3.11.15` 또는 더 최신 security patch.
- Environment manager: `venv`, `uv`, `conda`, `mamba` 모두 가능하지만 resolved lockfile을 experiment code와 함께 저장해야 한다.

GPU and deep-learning stack:

- Main target: CUDA `13.0` wheel을 사용하는 PyTorch `2.12.0`.
- Fallback target: cluster driver가 CUDA 13.x를 지원하지 않으면 CUDA `12.6` wheel을 사용하는 PyTorch `2.12.0`.
- 같은 experimental batch 안에서 비교되는 알고리즘끼리는 PyTorch/CUDA build를 바꾸지 않는다.
- 매 run config에 `torch.__version__`, CUDA runtime version, CUDA driver version, GPU model, cuDNN version을 기록한다.

Core RL and environment dependencies:

| Package | Recommended version | Purpose |
|---|---:|---|
| `torch` | `2.12.0` | DQN, RIDE auxiliary models, GPU training |
| `torchvision` | matching PyTorch wheel | PyTorch install set compatibility |
| `torchaudio` | matching PyTorch wheel | PyTorch install set compatibility |
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
| `tensorboard` | `>=2.19` | Local scalar, histogram, Q-value diagnostics |
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
| `pytest` | `>=8.3` | Unit tests for replay, n-step targets, wrappers, metrics |
| `ruff` | `>=0.11` | Linting and formatting |
| `mypy` | optional | Static checks for experiment infrastructure |
| `psutil` | `>=6.1` | System and process diagnostics |

권장 install pattern:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install gymnasium==1.3.0 minigrid==3.1.0 numpy scipy pandas polars pyarrow wandb tensorboard hydra-core omegaconf pyyaml tqdm rich matplotlib seaborn plotly kaleido rliable lifelines statsmodels pytest ruff psutil
```

CUDA 13.0 wheel이 cluster driver와 맞지 않으면 CUDA 12.6 PyTorch wheel index를 사용하고, 비교되는 모든 run은 같은 stack에서 실행한다.

Reproducibility requirements:

- 각 experiment batch마다 `pip freeze` 또는 equivalent lockfile을 저장한다.
- Git commit hash, uncommitted diff status, config file, random seeds를 저장한다.
- 정확한 MiniGrid/Gymnasium environment id와 custom DoorKey registration code를 저장한다.
- 큰 metric과 trajectory diagnostic은 Parquet file로 저장하고, W&B artifact 또는 다른 immutable artifact store를 사용한다.

---

## 2. 팀 구현 파이프라인과 실험

프로젝트는 순차 handoff 방식으로 구현한다.

1. 팀원 1은 shared experiment foundation을 만든다: DQN, n-step DQN, replay logic, environment wrapper, common helper functions, metric logging, evaluation code, hyperparameter config files.
2. 팀원 2는 팀원 1의 foundation을 받아 RIDE intrinsic reward module을 추가하고, shared training interface에 통합한 뒤, 전체 four-condition experiment suite를 실행한다.

이 분업은 프로젝트를 일관되게 유지한다. 팀원 1은 공통 experimental contract를 정의하고, 팀원 2는 baseline behavior를 바꾸지 않으면서 그 contract를 확장한다.

### 2.0 Shared Handoff Contract

팀원 1의 deliverables:

- `algorithm=dqn` 또는 `algorithm=nstep_dqn`으로 실행 가능한 단일 training entry point.
- Environment id, seed, training budget, evaluation interval, replay buffer, optimizer, epsilon schedule, target network update, batch size, discount, n-step horizon config files.
- Seeding, environment construction, observation preprocessing, replay storage, n-step return construction, checkpointing, evaluation, metric logging, result export용 common helpers.
- 최소한 `obs`, `action`, `reward_ext`, `next_obs`, `done`, `truncated`, `env_seed`, `episode_id`, `timestep`, subgoal flags를 포함하는 stable transition schema.
- Replay sampling, terminal n-step truncation, seed determinism, evaluation without learning에 대한 unit test 또는 smoke test.
- DoorKey-6x6과 DoorKey-8x8에서 DQN과 n-step DQN baseline pilot results.

팀원 2는 이 contract를 보존하면서 다음을 추가한다.

- State embedding, forward dynamics model, inverse dynamics model, intrinsic reward normalization, auxiliary losses를 포함한 RIDE module.
- Training reward interface:

```text
r_train = reward_ext + beta * reward_ride
```

- `use_ride`, `beta`, intrinsic reward normalization, intrinsic clipping, RIDE loss weights, RIDE encoder와 Q-network encoder 공유 여부에 대한 config extension.
- Intrinsic reward magnitude, RIDE auxiliary losses, parameter count logging.
- 같은 seeds, environment layouts, budgets, evaluation code를 사용한 네 가지 condition의 final training runs.

팀원 2가 bug 때문에 shared helper를 바꿔야 한다면, 영향을 받는 baseline과 RIDE condition은 모두 rerun하거나 non-comparable로 명확히 표시한다.

---

## Experiment 1: Main 2x2 Factorial Study

### RQ1

Sparse-reward DoorKey에서 learning outcome은 discovery-oriented intrinsic reward, propagation-oriented n-step backup, 또는 둘의 interaction 중 무엇에 의해 주로 개선되는가?

### Conditions

다음 네 condition을 실행한다.

1. DQN, 1-step backup, no RIDE.
2. DQN + RIDE, 1-step backup.
3. n-step DQN, no RIDE.
4. n-step DQN + RIDE.

네 condition을 모두 다음 환경에서 실행한다.

- DoorKey-6x6
- DoorKey-8x8
- DoorKey-16x16

### Team Workflow

팀원 1 phase:

1. Common DQN trainer, target network update, replay buffer, epsilon schedule, evaluation loop, logging schema를 구현한다.
2. 같은 trainer에서 backup target config만 바꿔 n-step return이 작동하도록 구현한다.
3. `dqn_1step`, `dqn_nstep` config를 등록한다.
4. Smoke test와 baseline pilot run을 통해 DQN과 n-step DQN이 comparable log, checkpoint, evaluation file을 생성하는지 확인한다.
5. Handoff 전에 baseline interface를 freeze한다.

팀원 2 phase:

1. 같은 training loop에서 호출되는 optional intrinsic reward module로 RIDE를 추가한다.
2. `dqn_ride_1step`, `dqn_ride_nstep` config를 등록한다.
3. Intrinsic reward가 training 중에만 사용되고 evaluation에서는 비활성화되는지 확인한다.
4. Paired seeds로 complete four-condition grid를 실행한다.
5. 팀원 1의 baseline output과 팀원 2의 RIDE output을 사용해 factorial estimates를 계산한다.

### Method

각 environment와 seed에 대해:

1. Paired seed protocol을 사용해 environment, network weights, replay buffer, 필요한 경우 RIDE module, RNG streams를 초기화한다.
2. 사전에 정한 environment steps만큼 각 agent를 train한다.
3. Fixed held-out evaluation layout set에서 주기적으로 evaluate한다.
4. Shared schema를 사용해 training diagnostics와 evaluation diagnostics를 모두 log한다.
5. Evaluation에서는 RIDE intrinsic reward를 비활성화한다.

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

다음 metric에 대해 계산한다.

- Final success rate
- Success-rate AUC
- First-success timestep
- First key pickup timestep
- First door-open timestep
- Post-discovery learning speed

### Expected Interpretation

Discovery bottleneck의 evidence:

- RIDE가 first-success time을 크게 개선한다.
- RIDE가 first reward 이전 key pickup과 door opening frequency를 증가시킨다.
- n-step은 first success 이전에는 영향이 작다.

Propagation bottleneck의 evidence:

- n-step이 반드시 first discovery를 개선하지는 않는다.
- 첫 successful episode가 나타난 뒤 n-step이 높은 success threshold에 더 빨리 도달한다.
- n-step이 pre-discovery subgoal coverage보다 post-discovery AUC를 더 크게 개선한다.

Interaction의 evidence:

- RIDE가 successful trajectory를 더 많이 만든다.
- n-step이 그 trajectory를 더 빠르게 exploit한다.
- `Y_11`이 RIDE main effect와 n-step main effect를 단순히 더한 것보다 크다.

Ambiguous case:

- RIDE가 discovery와 post-discovery speed를 모두 개선하면 pure discovery라고 결론내리지 않는다.
- n-step이 first success 이전 exploration behavior를 바꾸면 pure propagation이라고 결론내리지 않는다.
- Experiment 2와 3으로 disambiguate한다.

---

## Experiment 2: Difficulty Sweep and Ceiling/Floor Check

### RQ2

Apparent bottleneck은 environment difficulty에 따라 달라지는가?

### Conditions

Experiment 1과 같은 네 condition을 사용한다.

- DQN
- DQN + RIDE
- n-step DQN
- n-step DQN + RIDE

다음 DoorKey size에 대해 실행한다.

- 6x6
- 8x8
- 12x12 또는 사용 가능한 다른 intermediate custom DoorKey size
- 16x16

Optional extension:

- KeyCorridorS3R3
- MultiRoom variant

### Team Workflow

팀원 1 phase:

1. Environment id, custom DoorKey size, training budget, evaluation interval을 모두 config-driven으로 만든다.
2. 각 environment size에 대한 fixed evaluation layout generation을 구현한다.
3. 선택한 difficulty sweep 전체에서 DQN과 n-step DQN baseline을 실행한다.
4. Experiment 1과 같은 metric column을 사용해 `environment x algorithm x seed`별 result table을 export한다.

팀원 2 phase:

1. 정확히 같은 environment configs와 evaluation layout files를 재사용한다.
2. 같은 difficulty sweep에서 DQN + RIDE와 n-step DQN + RIDE를 실행한다.
3. Baseline과 RIDE result를 하나의 analysis table로 결합한다.
4. Environment difficulty에 따라 RIDE main effect, n-step main effect, interaction이 어떻게 변하는지 보고한다.

### Method

Experiment 1과 같은 protocol로 train/evaluate한다.

Effect size를 environment difficulty별로 나누어 분석한다.

각 environment에 대해 다음을 보고한다.

- Final success rate
- AUC
- First-success survival curve
- Subgoal completion curves
- Factorial main effects and interaction

### Interpretation

DoorKey-6x6이 모든 agent에게 solve된다면:

- Sanity check로 취급한다.
- Bottleneck structure에 대한 결정적 evidence로 사용하지 않는다.

DoorKey-16x16이 모든 agent에게 fail된다면:

- Floor-effect regime으로 취급한다.
- Training budget을 늘리거나 intermediate task를 포함한다.
- A100/H100 full package에서는 all-failure 16x16 regime에 결론을 과적합하기보다 intermediate DoorKey size를 추가하는 것을 선호한다.

가장 정보량이 큰 regime은 다음 조건을 만족한다.

- 일부 agent는 solve한다.
- 일부 agent는 fail한다.
- Learning curve가 saturated되지 않는다.

---

## Experiment 3: Fixed Successful Replay Propagation Test

### RQ3

Reward discovery를 control했을 때, n-step backup은 rare successful reward signal을 1-step backup보다 더 빠르게 propagate하는가?

이 실험은 main design의 가장 큰 confound를 직접 겨냥한다. 일반 online training은 discovery와 propagation을 섞어버린다.

### Conditions

다음을 비교한다.

1. Rare successful trajectory가 포함된 fixed replay dataset으로 학습한 1-step DQN.
2. 같은 fixed replay dataset으로 학습한 n-step DQN.

이 실험은 RIDE condition이 필요 없다. Discovery data를 고정해 propagation만 isolate한다.

### Team Workflow

팀원 1 phase:

1. Random, epsilon-random, optional scripted successful DoorKey trajectory를 수집하는 dataset builder를 구현한다.
2. Environment id, dataset seed, success ratio, episode count, transition count metadata와 함께 fixed replay dataset을 저장한다.
3. Experiment 1과 같은 Q-network와 target update code를 사용해 1-step DQN과 n-step DQN의 offline replay training을 구현한다.
4. Successful trajectory에 대한 Q-value와 Bellman error diagnostic을 추가한다.
5. DoorKey-8x8에서 pilot fixed-replay result를 만든다.

팀원 2 phase:

1. Fixed replay dataset이 online training과 같은 transition schema로 load되는지 검증한다.
2. Paired initialization seed로 final 1-step vs n-step fixed-replay comparison을 실행한다.
3. Learned policy를 실제 environment에서 주기적으로 evaluate한다.
4. Propagation diagnostic을 final result analysis에 통합한다.

### Method

Dataset construction:

1. Controlled mix의 trajectory로 replay buffer를 수집한다.
   - Random 또는 epsilon-random trajectories.
   - 소수의 successful trajectories.
   - Optional scripted expert trajectories for DoorKey.
2. 모든 비교 agent가 정확히 같은 dataset을 사용하게 한다.
3. Data collection을 freeze한다. 이 test에서는 추가 online interaction을 하지 않는다.

권장 dataset compositions:

- Compute가 충분하면 0.01% successful episodes.
- 0.1% successful episodes.
- 1% successful episodes.
- 5% successful episodes.

Training:

1. 같은 seed로 agent를 초기화한다.
2. Fixed replay dataset에서만 train한다.
3. 동일한 data에서 1-step target과 n-step target을 비교한다.
4. 실제 environment에서 learned policy를 주기적으로 evaluate한다.

Diagnostics:

- Successful trajectory의 states/actions에 대한 Q-values를 track한다.
- Early trajectory state에 positive value estimate가 얼마나 빨리 나타나는지 track한다.
- Successful trajectory prefix에 대한 Bellman error를 track한다.
- A100/H100 compute가 있다면 replay composition마다 20 paired seeds를 실행하고, extremely rare discovery에서 reward propagation을 stress-test하기 위해 0.01% success condition을 포함한다.

### Interpretation

Propagation benefit에 대한 strong evidence:

- n-step이 successful-trajectory early state에 대해 positive value를 더 빨리 학습한다.
- 같은 replay data에서 n-step이 nonzero evaluation success에 더 빨리 도달한다.
- Successful trajectory가 rare할수록 effect가 커진다.

Weak evidence:

- n-step이 fixed replay에서는 도움이 되지 않고 online에서만 도움이 된다.
- 이 경우 online effect가 pure reward propagation이 아니라 exploration 또는 replay-distribution과 관련되었을 수 있다.

---

## Experiment 4: Discovery-Only Pre-Reward Analysis

### RQ4

RIDE는 extrinsic reward가 관측되기 전 exploration을 개선하는가?

### Conditions

다음을 비교한다.

1. DQN
2. DQN + RIDE
3. n-step DQN
4. n-step DQN + RIDE

핵심 analysis window는 각 seed에서 첫 successful episode 이전이다.

### Team Workflow

팀원 1 phase:

1. Environment wrapper 또는 episode logger에 common subgoal logging을 추가한다.
2. 모든 algorithm에 대해 key pickup, door open, room transition, goal reached, timeout, episode length, object interactions, coverage fields를 log한다.
3. 이 diagnostic이 agent observation으로 절대 노출되지 않도록 보장한다.
4. DQN과 n-step DQN의 baseline pre-reward traces를 만든다.

팀원 2 phase:

1. Intrinsic reward magnitude, normalized intrinsic reward, auxiliary losses에 대한 RIDE-specific logging을 추가한다.
2. 같은 pre-reward logging fields를 사용해 RIDE condition을 실행한다.
3. Non-RIDE agent와 RIDE agent 사이의 pre-reward event distribution을 비교한다.
4. RIDE가 generic interaction count만 늘리는 것이 아니라 ordered key-door-goal progress를 늘리는지 확인한다.

### Method

First success 이전의 모든 training episode에 대해:

1. Agent가 key를 집었는지 log한다.
2. Agent가 door를 toggle/open했는지 log한다.
3. Agent가 second room에 들어갔는지 log한다.
4. Environment metadata에서 가능하다면 unique grid cells visited를 log한다.
5. Object interactions를 log한다.
   - pickup attempts
   - successful key pickup
   - toggle attempts
   - successful door opening
6. RIDE agent에 대해서는 intrinsic reward magnitude를 log한다.

이 privileged log를 agent input으로 사용하지 않는다.

Analyze:

- Time to first key pickup.
- Time to first door opening.
- Time to first room transition.
- Time to first goal.
- Coverage growth over environment steps.
- Ordered subgoal completion probability.

### Interpretation

RIDE가 discovery를 개선한다는 evidence:

- RIDE가 key/door/goal first-event distribution을 더 이른 시점으로 이동시킨다.
- RIDE가 extrinsic reward 이전 object interaction rate를 증가시킨다.
- RIDE가 단순 random wandering이 아니라 state coverage 또는 meaningful subgoal coverage를 개선한다.

Potential failure mode:

- RIDE가 controllable하지만 irrelevant한 interaction에 과도한 reward를 줄 수 있다.
- RIDE가 pickup/toggle attempts만 늘리고 ordered key-door-goal progress를 늘리지 못한다면, task-relevant discovery가 아니라 novelty만 개선한 것일 수 있다.

---

## 3. Evaluation Metrics

## 3.1 Primary Performance Metrics

### Final Success Rate

Definition:

```text
success_rate = number of successful evaluation episodes / total evaluation episodes
```

Final checkpoint와 optional best checkpoint를 모두 사용한다.

Report both:

- Final checkpoint success rate.
- Best checkpoint success rate.

이유:

- Final performance는 convergence를 반영한다.
- Best performance는 instability 또는 collapse를 감지한다.

### Success-Rate AUC

Definition:

Evaluation success-rate curve의 environment steps에 대한 area under curve.

Normalized AUC를 사용한다.

```text
AUC = integral success_rate(t) dt / total_training_steps
```

이유:

- Sample efficiency를 포착한다.
- Final checkpoint에만 과도하게 집중하는 것을 피한다.

### Extrinsic Return

Definition:

Evaluation 중 original environment reward의 평균.

MiniGrid success reward에는 step penalty가 포함되는 경우가 많으므로 success rate와 별도로 보고한다. 두 agent가 같은 success rate를 가져도 efficiency는 다를 수 있다.

### Episode Length on Success

Definition:

Successful evaluation episode들에 대한 평균 step 수.

이유:

- 더 짧은 successful trajectory는 더 효율적인 policy를 의미한다.
- Success rate가 saturated될 때 유용하다.

---

## 3.2 Discovery Metrics

### First Success Timestep

Definition:

Agent가 successful episode를 처음 완수한 training timestep.

Survival analysis style reporting을 사용한다.

- 일부 seed는 끝까지 성공하지 못할 수 있다.
- Never-success seed는 training budget에서 right-censored로 처리한다.

Report:

- Median time to first success.
- 각 timestep까지 성공한 seed fraction.
- 가능하다면 Kaplan-Meier-style survival curve.

### First Key Pickup Timestep

Definition:

Agent가 key pickup에 처음 성공한 training timestep.

이유:

- Key pickup은 DoorKey의 첫 major prerequisite이다.
- Extrinsic reward 이전에도 발생할 수 있다.

### First Door Open Timestep

Definition:

Agent가 door unlock/open에 처음 성공한 training timestep.

이유:

- Door opening은 더 늦은 ordered subgoal이다.
- Generic state coverage보다 task-relevant하다.

### First Room Transition Timestep

Definition:

Agent가 door 너머 영역에 처음 도달한 timestep.

이유:

- 단순히 door를 여는 것과 열린 path를 실제로 exploit하는 것을 구분한다.

### Ordered Subgoal Completion Rate

각 evaluation 또는 training episode에 대해 다음 ordered chain이 발생했는지 기록한다.

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

이유:

- Agent가 task sequence의 어느 지점에서 실패하는지 분해한다.

### State Coverage

Definition:

Training 중 방문한 underlying grid position의 unique count 또는 fraction.

Logging에만 사용하고 agent input으로 사용하지 않는다.

Report:

- Episode당 unique cells visited.
- Cumulative unique cells visited.
- Unique object-adjacent states visited.

Caution:

- High coverage가 반드시 task-relevant discovery를 의미하지는 않는다.
- Coverage는 항상 ordered subgoal metrics와 함께 해석한다.

### Object Interaction Frequency

Episode당 다음을 count한다.

- pickup actions
- successful key pickup
- toggle actions
- successful door opening
- irrelevant object 또는 wall에 대한 toggle attempts

이유:

- RIDE는 controllable object interaction을 장려할 수 있다.
- 하지만 irrelevant interaction이 과도하면 curiosity가 잘못된 방향으로 작동한다는 신호일 수 있다.

---

## 3.3 Propagation Metrics

### Post-Discovery Learning Speed

Definition:

First success를 얻은 각 seed에 대해, first success 이후 fixed evaluation success-rate threshold에 도달하기까지의 environment steps.

권장 thresholds:

- 25% success
- 50% success
- 75% success

Important:

- Threshold에 도달하지 못한 seed는 censored로 처리한다.
- Failed seed를 조용히 제거하지 않는다.

Report:

- First success 이후 25%, 50%, 75% success까지 걸린 시간.
- 각 threshold에 도달한 seed fraction.
- Censored-time summary.

### Post-Discovery AUC

Definition:

각 seed의 첫 successful training episode 이후 구간에서만 계산한 success-rate AUC.

이유:

- Discovery 이전 search phase와 discovery 이후 reward exploitation을 분리한다.

Caution:

- First success가 없는 seed를 처리해야 한다.
- 제외 또는 censor된 seed 수를 보고한다.

### Value Propagation Along Successful Trajectory

Saved successful trajectories를 사용한다.

Successful trajectory의 states/actions에 대해 여러 training checkpoint에서 다음을 log한다.

```text
Q(s_t, a_t)
```

Report:

- Goal 근처에서 Q-value가 얼마나 빨리 positive가 되는지.
- Positive value가 early state 쪽으로 얼마나 빨리 backward 이동하는지.
- Distance-to-goal에 따른 value increase slope.

이유:

- Reward information이 trajectory를 따라 backward propagation되는지를 직접 측정한다.

### Bellman Error Along Successful Trajectory

Saved successful trajectories에 대해 TD error 또는 n-step target error를 계산한다.

Report:

- Trajectory position별 mean TD error.
- Training에 따른 TD error decay.
- 1-step target과 n-step target의 차이.

이유:

- Behavioral exploration과 value propagation을 구분하는 데 도움을 준다.

---

## 3.4 Interaction Metrics

### Factorial Interaction on AUC

Definition:

```text
interaction_AUC = (AUC_nstep_RIDE - AUC_nstep_noRIDE)
                - (AUC_1step_RIDE - AUC_1step_noRIDE)
```

Positive interaction의 의미:

- RIDE가 n-step backup이 있을 때, 없을 때보다 더 크게 도움이 된다.

### Factorial Interaction on Discovery

같은 formula를 negative time-to-event metric 또는 event probability에 적용한다.

Examples:

- First success timestep.
- First door-open timestep.
- Timestep `T`까지 goal에 도달할 probability.

### Factorial Interaction on Propagation

같은 formula를 다음에 적용한다.

- First success 이후 50% success까지의 시간.
- Post-discovery AUC.
- Value propagation slope.

Interpretation:

- Interaction이 주로 discovery metric에서 나타나면, combination이 successful trajectory를 만드는 데 도움을 주는 것일 수 있다.
- Interaction이 주로 propagation metric에서 나타나면, n-step이 RIDE-generated trajectory를 더 잘 exploit하는 것일 수 있다.

---

## 3.5 Statistical Reporting

가능한 한 paired analysis를 사용한다.

Required reporting:

- Seed across mean과 median.
- 95% bootstrap confidence intervals.
- 여러 environment difficulty를 aggregate할 때 stratified bootstrap.
- Key pairwise comparison에 대한 probability of improvement.
- Appendix 또는 supplementary plot에 individual seed learning curves.

Recommended aggregate metrics:

- Median
- 여러 task를 aggregate한다면 interquartile mean
- Probability of improvement

Avoid:

- Point estimate만 보고 superiority를 주장하는 것.
- Best seed를 선택해 보고하는 것.
- Raw variability 없이 smoothed curve만 보고하는 것.

보고할 pairwise comparisons:

1. DQN vs DQN + RIDE
2. DQN vs n-step DQN
3. DQN + RIDE vs n-step DQN + RIDE
4. n-step DQN vs n-step DQN + RIDE
5. DQN vs n-step DQN + RIDE

보고할 factorial effects:

- RIDE main effect
- n-step main effect
- RIDE x n-step interaction

다음 metric에 대해 보고한다.

- Final success rate
- Success-rate AUC
- First success
- First key pickup
- First door open
- Post-discovery learning speed

---

## 4. Decision Rules

### Discovery-Dominant Bottleneck

다음 조건이 모두 맞을 때만 discovery가 dominant하다고 결론낸다.

- RIDE가 first key pickup, first door opening, first success를 개선한다.
- RIDE가 pre-reward exploration 또는 ordered subgoal completion을 개선한다.
- n-step이 pre-discovery metric을 크게 개선하지 않는다.
- Post-discovery difference보다 discovery difference가 더 크다.

### Propagation-Dominant Bottleneck

다음 조건이 모두 맞을 때만 propagation이 dominant하다고 결론낸다.

- n-step이 post-discovery learning speed를 개선한다.
- n-step이 successful trajectory를 따라 value propagation을 개선한다.
- Fixed successful replay experiment에서 n-step이 더 빠른 learning을 보인다.
- First success 이전 discovery metric은 1-step과 n-step이 비슷하다.

### Complementary Interaction

다음 조건이 모두 맞을 때만 complementarity라고 결론낸다.

- RIDE x n-step interaction이 positive이고 uncertainty interval이 대부분 0보다 크다.
- Interaction이 AUC 또는 success rate에서 나타난다.
- Mechanism metric이 이유를 설명한다.
  - RIDE가 successful trajectory를 늘린다.
  - n-step이 그것을 더 빠르게 exploit한다.

### No Clear Bottleneck

다음 경우 no clear bottleneck으로 결론낸다.

- Effect가 environment size에 따라 inconsistent하다.
- Confidence interval이 넓다.
- First-success metric과 post-discovery metric이 서로 맞지 않는다.
- Fixed-replay propagation result와 pre-reward discovery diagnostic이 online factorial result와 맞지 않는다.

---

## 5. Minimum Viable Experimental Package

시간이 제한되어 있다면 다음 reduced but defensible package를 실행한다.

1. DoorKey-6x6, 8x8, 16x16에서 main 2x2 factorial.
2. Condition당 10 paired seeds.
3. Fixed held-out evaluation layouts.
4. Metrics:
   - final success rate
   - success-rate AUC
   - first success timestep
   - first key pickup timestep
   - first door open timestep
   - post-discovery time to 50% success
5. DoorKey-8x8에서 fixed successful replay propagation test.
6. Bootstrap confidence intervals와 probability of improvement.

이 minimum package는 planned intervention이 의도한 mechanism에 실제로 영향을 주는지 직접 확인하므로 원래 proposal보다 훨씬 강하다.

---

## 6. A100/H100 Full Experimental Package

A100/H100 compute가 있다면 statistical reliability와 mechanism identification을 강화하는 데 사용한다.

1. DoorKey-6x6, 8x8, 12x12 같은 intermediate DoorKey size, 16x16에서 main 2x2 factorial.
2. Main condition당 30 paired seeds.
3. DoorKey-6x6과 DoorKey-8x8은 checkpoint당 200 evaluation episodes, DoorKey-16x16은 300-500 episodes.
4. Increased training budgets:
   - DoorKey-6x6: 250k-500k steps.
   - DoorKey-8x8: 2M steps.
   - Intermediate DoorKey: 5M steps.
   - DoorKey-16x16: 10M steps.
5. DoorKey-8x8에서 0.01%, 0.1%, 1%, 5% successful-episode mixture를 사용하는 fixed successful replay propagation test.
6. 가능하다면 fixed replay propagation test에 20 paired seeds.
7. 모든 main condition에 대해 pre-reward discovery analysis.
8. 비교 condition 간 main model, batch size, update-to-data ratio를 고정한다.
9. 하나의 oversized learner보다 많은 independent `environment x algorithm x seed` job으로 compute를 schedule한다.

Rationale:

- A100/H100급 hardware가 sparse-reward variance를 없애주지는 않는다.
- Additional compute의 주요 이점은 더 좁은 confidence interval, 더 나은 survival analysis, 더 정확한 post-discovery timing, 더 강한 fixed-replay propagation test이다.
- Model size, update ratio, parallel actor count를 키우면 연구 중인 causal mechanism이 바뀔 수 있으므로 main experimental package 안에서는 바꾸지 않는다.

---

## 7. References to Use in Final Report

- Mnih et al. (2015), "Human-level control through deep reinforcement learning." Nature.
- Sutton and Barto (2018/2020), "Reinforcement Learning: An Introduction." n-step returns and temporal-difference learning.
- Hessel et al. (2018), "Rainbow: Combining Improvements in Deep Reinforcement Learning." Multi-step returns as a DQN extension.
- Raileanu and Rocktaschel (2020), "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS.
- Chevalier-Boisvert et al. (2023), "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks." NeurIPS Datasets and Benchmarks.
