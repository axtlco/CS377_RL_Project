# 프로젝트 계획서: Sparse-Reward RL에서 Discovery와 Propagation 분리 분석

## 0. 핵심 목표

이 프로젝트는 MiniGrid DoorKey에서 sparse-reward 강화학습이 실패하는 이유가 주로 무엇인지 검증한다.

1. **Reward discovery bottleneck**: 에이전트가 보상이 있는 상태나 보상에 필요한 선행 상태에 거의 도달하지 못하는 문제.
2. **Reward propagation bottleneck**: 드물게 성공 trajectory를 관측하더라도, 그 보상 신호가 초반 의사결정으로 충분히 빠르게 전파되지 않는 문제.
3. **Interaction**: 탐험(exploration)과 credit assignment가 서로 비가산적으로 돕는 경우. 즉, RIDE와 n-step을 같이 썼을 때 단순 합보다 더 큰 효과가 나는 경우.

원래 proposal의 핵심 실험은 다음 2x2 factorial design이다.

| Backup target | Intrinsic discovery module 없음 | RIDE intrinsic discovery module 있음 |
|---|---|---|
| 1-step TD | DQN | DQN + RIDE |
| n-step TD | n-step DQN | n-step DQN + RIDE |

이 계획서는 위 핵심 설계를 유지하되, 다음 요소를 추가한다.

- 환경 난이도 sweep
- discovery와 propagation을 직접 측정하는 mechanism-specific metric
- discovery와 propagation의 confounding을 줄이는 control experiment
- partial observability, hyperparameter sensitivity, ceiling/floor effect 점검

가장 중요한 방법론적 원칙은 다음과 같다.

> 최종 success rate만 보고 메커니즘을 추론하지 않는다. 성능 변화와 mechanism-specific diagnostic이 일관되게 맞아떨어질 때만 discovery 또는 propagation 병목이라고 해석한다.

---

## 1. 실험 환경 Specification

### 1.1 Benchmark Family

주요 benchmark:

- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-DoorKey-8x8-v0`
- `MiniGrid-DoorKey-16x16-v0`

컴퓨팅 자원이 허용될 경우 추가 robustness benchmark:

- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-MultiRoom-N4-S5-v0` 또는 유사한 MultiRoom variant

선정 이유:

- DoorKey는 순서가 있는 sparse-reward task이다. 에이전트는 key를 찾아서 집고, locked door를 열고, goal에 도달해야 한다.
- MiniGrid는 partial observation 환경이다. 에이전트는 전체 grid가 아니라 자기 주변의 egocentric local view만 본다.
- DoorKey는 sparse reward 때문에 classical RL이 어려워하는 환경이며, curiosity나 curriculum learning을 시험하기 적합하다.
- DoorKey-6x6만 사용하면 너무 쉬워서 ceiling effect가 생길 수 있다. 이 경우 discovery 병목과 propagation 병목을 구분하기 어렵다. 따라서 6x6, 8x8, 16x16 난이도 sweep이 필요하다.

### 1.2 Observation Space

메인 실험에서는 MiniGrid의 기본 partial observation을 사용한다.

- Local egocentric image: `7 x 7 x 3`
- Agent direction
- Mission string

권장 preprocessing:

- RGB pixel 대신 symbolic MiniGrid observation을 사용한다.
- tile object, color, state channel은 one-hot 또는 embedding feature로 변환한다.
- agent direction은 one-hot vector로 포함한다.
- DoorKey에서는 mission text가 task family 안에서 거의 고정되어 있으므로, main experiment에서는 mission text를 무시해도 된다. 포함한다면 language model을 쓰기보다 fixed task id처럼 encode한다.

메인 실험에서는 privileged full-state observation을 사용하지 않는다.

다만 diagnostic oracle condition으로 full observation을 사용하는 control을 하나 추가한다. 이 control은 main result가 아니라, discovery/propagation 실패처럼 보이는 현상이 사실 partial observability 때문에 생긴 것인지 확인하기 위한 목적이다.

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

메인 실험에서는 action masking을 사용하지 않는다.

이유:

- action masking은 task prior knowledge를 주입하는 것이며, exploration 난이도를 낮춘다.
- invalid action까지 포함해야 원래 sparse-reward exploration challenge가 유지된다.

선택적 diagnostic:

- action-masked condition을 별도로 실행할 수 있다. 이는 실패가 irrelevant command 낭비 때문인지 추정하기 위한 보조 실험이다.
- 단, action-masked result를 main comparison에 섞지 않는다.

### 1.4 Reward Function

Evaluation reward:

- evaluation에서는 원래 MiniGrid extrinsic reward만 사용한다.
- success reward는 goal에 도달했을 때만 주어지는 sparse reward이다.
- 실패 episode는 extrinsic reward 0을 받는다.

Training reward:

- RIDE를 쓰지 않는 agent: extrinsic reward만 사용.
- RIDE agent: training 중에만 extrinsic reward + RIDE intrinsic bonus 사용.
- evaluation에서는 RIDE intrinsic reward를 완전히 비활성화한다.

중요한 reporting rule:

- extrinsic evaluation performance와 intrinsic training reward를 항상 분리해서 보고한다.
- RIDE가 켜져 있을 때 training return으로 알고리즘을 비교하지 않는다. RIDE는 reward scale 자체를 바꾸기 때문이다.

### 1.5 Episode Termination and Time Limits

명시적으로 바꾸지 않는 한 environment default maximum episode length를 사용한다.

각 episode마다 다음을 기록한다.

- episode length
- goal 도달 여부
- timeout 여부
- key pickup 성공 여부
- door toggle/open 성공 여부
- door 너머 room에 도달했는지 여부

### 1.6 Training and Evaluation Seeds

모든 알고리즘에서 paired random seed를 사용한다.

권장 seed 수:

- 최소: condition당 10 seeds. 원래 proposal과 동일하다.
- 권장: main DoorKey sweep에서는 condition당 20 seeds.

Seed pairing:

- seed `s`에 대해 네 가지 main agent가 가능한 한 같은 initialization seed, environment seed stream, replay sampling seed, evaluation layout set을 사용하게 한다.
- 이렇게 하면 factorial effect를 추정할 때 variance가 줄어든다.

Evaluation layout protocol:

- 각 environment size마다 fixed held-out evaluation seed set을 만든다.
- 권장 evaluation episode 수:
  - DoorKey-6x6: checkpoint당 100 episodes
  - DoorKey-8x8: checkpoint당 100 episodes
  - DoorKey-16x16: variance가 높으면 checkpoint당 200 episodes
- evaluation에서는 greedy policy 또는 매우 낮은 exploration policy를 사용한다.
- evaluation 중에는 학습하지 않는다.

### 1.7 Training Budget

환경 난이도에 따라 training budget을 다르게 설정한다.

초기 권장 budget:

| Environment | Training steps per seed | Evaluation interval |
|---|---:|---:|
| DoorKey-6x6 | 250k | every 5k |
| DoorKey-8x8 | 1M | every 10k |
| DoorKey-16x16 | 5M | every 25k |
| KeyCorridorS3R3 | 5M | every 25k |

최종 run 전에 3 seeds 정도로 pilot run을 수행한다.

Pilot에서 확인할 것:

- 모든 agent가 완전히 실패하는지.
- 모든 agent가 너무 빨리 solve해서 ceiling effect가 생기는지.
- evaluation interval이 first success와 post-discovery speed를 측정하기에 너무 성긴지.

DoorKey-6x6을 모든 agent가 빠르게 solve한다면, 6x6은 sanity check로만 사용하고 핵심 근거로 삼지 않는다.

### 1.8 Base Agent Specification

네 가지 main cell이 하나의 shared value-based agent implementation을 사용해야 한다.

Main agent:

- Experience replay와 target network를 사용하는 DQN.
- 모든 condition에서 동일한 network architecture.
- 동일한 optimizer, learning rate, replay buffer size, target update period, epsilon schedule, batch size, discount, training budget.

Symbolic partial observation을 위한 권장 architecture:

- Input: one-hot encoded `7 x 7` symbolic observation + direction.
- Encoder: small convolutional network 또는 flattened symbolic feature에 대한 compact MLP.
- Head: 7개 action에 대한 Q-value를 출력하는 fully connected layer.
- Main experiment에서는 recurrent memory를 사용하지 않는다.

Stability variant:

- vanilla DQN이 불안정하면 Double DQN을 사용해도 된다.
- 단, 그렇게 한다면 baseline 이름을 일관되게 Double DQN으로 부른다.
- Main factorial design에는 dueling network, prioritized replay, noisy net, distributional RL 등 다른 Rainbow component를 섞지 않는다. 이런 요소들은 추가 메커니즘을 도입하기 때문이다.

### 1.9 n-step Backup Specification

비교할 backup:

- `n = 1`
- `n = 3`

선택적 sensitivity:

- `n = 5`
- `n = 10`

Main comparison에서는 사전에 하나의 `n`을 정한다. 권장은 `n = 3`이다. Rainbow-style DQN에서 3-step return이 자주 사용되며, multi-step target은 새롭게 관측된 reward를 더 빠르게 뒤쪽 state로 전파하는 데 도움을 주기 때문이다.

Implementation detail:

- n-step target을 계산할 수 있도록 replay에 충분한 sequence 정보를 저장한다.
- 1-step agent와 n-step agent가 같은 replay buffer capacity와 sampling strategy를 사용하게 한다.
- terminal transition에서는 n-step return을 올바르게 truncate한다.
- bootstrap term에는 `gamma^n` discount가 적용되어야 한다.

### 1.10 RIDE Specification

RIDE agent는 learned representation space에서 action이 만든 impact를 기반으로 intrinsic reward bonus를 받는다.

필수 구성요소:

- state embedding network
- forward dynamics model
- inverse dynamics model
- action으로 인한 representation change에 비례하는 intrinsic reward

Training reward:

```text
r_total = r_extrinsic + beta * r_RIDE
```

RIDE reporting requirement:

- `beta` 값을 보고한다.
- intrinsic reward normalization 방식을 보고한다.
- intrinsic reward clipping 여부를 보고한다.
- RIDE가 Q-network와 encoder를 공유하는지 보고한다.
- forward dynamics loss와 inverse dynamics loss의 auxiliary weight를 보고한다.

Fairness rule:

- RIDE는 parameter와 auxiliary loss를 추가한다. 따라서 parameter count와 wall-clock training time을 보고한다.
- Main comparison은 environment step 기준으로 한다. 하지만 compute overhead는 반드시 공개한다.

---

## 2. 수행해야 할 실험

## Experiment 1: Main 2x2 Factorial Study

### RQ1

Sparse-reward DoorKey에서 학습 성능은 discovery-oriented intrinsic reward, propagation-oriented n-step backup, 또는 둘의 interaction 중 무엇에 의해 주로 개선되는가?

### Conditions

다음 네 가지 condition을 실행한다.

1. DQN, 1-step backup, RIDE 없음.
2. DQN + RIDE, 1-step backup.
3. n-step DQN, RIDE 없음.
4. n-step DQN + RIDE.

모든 condition을 다음 환경에서 실행한다.

- DoorKey-6x6
- DoorKey-8x8
- DoorKey-16x16

### Method

각 environment와 seed에 대해:

1. Paired seed protocol에 따라 environment, network weights, replay buffer, RNG stream을 초기화한다.
2. 각 agent를 사전에 정한 environment step 수만큼 train한다.
3. 고정된 held-out evaluation layout set에서 주기적으로 평가한다.
4. training diagnostic과 evaluation diagnostic을 모두 log한다.
5. RIDE agent도 evaluation에서는 intrinsic reward를 끈다.

Primary factorial estimate:

다음과 같이 정의한다.

- `Y_00` = DQN
- `Y_10` = DQN + RIDE
- `Y_01` = n-step DQN
- `Y_11` = n-step DQN + RIDE

그러면 effect는 다음처럼 추정한다.

```text
RIDE main effect        = mean(Y_10, Y_11) - mean(Y_00, Y_01)
n-step main effect      = mean(Y_01, Y_11) - mean(Y_00, Y_10)
interaction effect      = (Y_11 - Y_01) - (Y_10 - Y_00)
```

이 추정을 다음 metric에 대해 계산한다.

- final success rate
- success-rate AUC
- first-success timestep
- first key pickup timestep
- first door-open timestep
- post-discovery learning speed

### Expected Interpretation

Discovery bottleneck의 근거:

- RIDE가 first-success time을 크게 앞당긴다.
- RIDE가 first reward 이전의 key pickup, door opening frequency를 증가시킨다.
- n-step은 first success 이전 metric에는 큰 영향을 주지 않는다.

Propagation bottleneck의 근거:

- n-step이 first discovery 자체를 반드시 개선하지는 않는다.
- 하지만 첫 successful episode가 나온 뒤 높은 success threshold에 더 빠르게 도달한다.
- n-step이 pre-discovery subgoal coverage보다 post-discovery AUC를 더 개선한다.

Interaction의 근거:

- RIDE가 successful trajectory를 더 많이 만들어낸다.
- n-step이 그 trajectory를 더 빠르게 exploit한다.
- `Y_11`이 RIDE effect와 n-step effect를 단순히 더한 것보다 더 크다.

애매한 경우:

- RIDE가 discovery와 post-discovery speed를 모두 개선하면 pure discovery라고 단정하지 않는다.
- n-step이 first success 이전 exploration behavior까지 바꾸면 pure propagation이라고 단정하지 않는다.
- 이런 경우 Experiment 2와 Experiment 3으로 해석을 보강한다.

---

## Experiment 2: Difficulty Sweep and Ceiling/Floor Check

### RQ2

관측되는 병목이 환경 난이도에 따라 달라지는가?

### Conditions

Experiment 1과 같은 네 condition을 사용한다.

- DQN
- DQN + RIDE
- n-step DQN
- n-step DQN + RIDE

DoorKey size별로 실행한다.

- 6x6
- 8x8
- 16x16

선택적 확장:

- KeyCorridorS3R3
- MultiRoom variant

### Method

Experiment 1과 같은 protocol로 training과 evaluation을 수행한다.

Effect size를 environment difficulty별로 따로 분석한다.

각 environment마다 다음을 보고한다.

- final success rate
- AUC
- first-success survival curve
- subgoal completion curve
- factorial main effect와 interaction

### Interpretation

DoorKey-6x6을 모든 agent가 solve한다면:

- 6x6은 sanity check로 취급한다.
- bottleneck structure에 대한 결정적 증거로 사용하지 않는다.

DoorKey-16x16을 모든 agent가 실패한다면:

- floor-effect regime으로 취급한다.
- training budget을 늘리거나 중간 난이도 task를 추가한다.

가장 정보량이 많은 regime은 다음 조건을 만족하는 환경이다.

- 일부 agent는 solve한다.
- 일부 agent는 실패한다.
- learning curve가 완전히 saturated되지 않는다.

---

## Experiment 3: Fixed Successful Replay Propagation Test

### RQ3

Reward discovery를 통제했을 때, n-step backup은 rare successful reward signal을 1-step backup보다 더 빠르게 전파하는가?

이 실험은 main design의 가장 큰 confound를 직접 겨냥한다. 일반 online training에서는 discovery와 propagation이 섞여 있기 때문이다.

### Conditions

다음 두 condition을 비교한다.

1. rare successful trajectory가 포함된 fixed replay dataset으로 학습한 1-step DQN.
2. 같은 fixed replay dataset으로 학습한 n-step DQN.

선택적 추가 condition:

3. prioritized replay를 사용하는 1-step DQN.
4. prioritized replay를 사용하는 n-step DQN.

Prioritized replay condition은 main factorial design이 아니라 diagnostic으로 명확히 분리한다.

### Method

Dataset construction:

1. 통제된 trajectory mixture를 가진 replay buffer를 수집한다.
   - random 또는 epsilon-random trajectory
   - 소수의 successful trajectory
   - 선택적으로 scripted expert trajectory for DoorKey
2. 모든 비교 agent가 정확히 같은 dataset을 사용한다.
3. data collection을 freeze한다. 이 실험에서는 추가 online interaction을 하지 않는다.

권장 dataset composition:

- successful episode 0.1%
- successful episode 1%
- successful episode 5%

Training:

1. 같은 seed에서 agent를 초기화한다.
2. fixed replay dataset에서만 학습한다.
3. 동일 data 조건에서 1-step target과 n-step target을 비교한다.
4. 주기적으로 실제 environment에서 learned policy를 평가한다.

Diagnostics:

- successful trajectory 위의 state/action에 대한 Q-value를 추적한다.
- early trajectory state에 positive value estimate가 얼마나 빨리 생기는지 추적한다.
- successful trajectory prefix에 대한 Bellman error를 추적한다.

### Interpretation

Propagation benefit의 강한 근거:

- n-step이 early successful-trajectory state의 positive value를 더 빨리 학습한다.
- 같은 replay data에서 n-step이 evaluation success를 더 빨리 만든다.
- successful trajectory가 rare할수록 n-step 효과가 커진다.

약한 근거:

- n-step이 fixed replay에서는 도움이 되지 않고 online에서만 도움이 된다.
- 이 경우 online 효과는 pure reward propagation이 아니라 exploration 또는 replay-distribution 변화 때문일 수 있다.

---

## Experiment 4: Discovery-Only Pre-Reward Analysis

### RQ4

RIDE는 extrinsic reward가 관측되기 전에 exploration을 개선하는가?

### Conditions

다음 네 condition을 비교한다.

1. DQN
2. DQN + RIDE
3. n-step DQN
4. n-step DQN + RIDE

핵심 분석 구간은 각 seed에서 first successful episode가 나오기 전이다.

### Method

First success 이전의 모든 training episode에 대해 다음을 log한다.

1. agent가 key를 pick up했는지.
2. agent가 door를 toggle/open했는지.
3. agent가 second room에 들어갔는지.
4. logging 목적으로 full state가 가능하다면 unique grid cell 방문 수.
5. object interaction:
   - pickup attempts
   - successful key pickup
   - toggle attempts
   - successful door opening
6. RIDE agent의 intrinsic reward magnitude.

이 privileged log 정보는 agent input으로 사용하지 않는다.

분석할 것:

- time to first key pickup
- time to first door opening
- time to first room transition
- time to first goal
- coverage growth over environment steps
- ordered subgoal completion probability

### Interpretation

RIDE가 discovery를 개선한다는 근거:

- RIDE가 key, door, goal first-event distribution을 앞당긴다.
- RIDE가 extrinsic reward 이전의 object interaction rate를 증가시킨다.
- RIDE가 단순 random wandering이 아니라 task-relevant subgoal coverage를 개선한다.

가능한 failure mode:

- RIDE가 controllable하지만 task와 무관한 interaction을 과도하게 보상할 수 있다.
- RIDE가 pickup/toggle attempt는 늘리지만 ordered key-door-goal progress를 늘리지 못한다면, novelty는 늘렸지만 task-relevant discovery는 개선하지 못한 것이다.

---

## Experiment 5: Partial Observability Control

### RQ5

Discovery 또는 propagation bottleneck처럼 보이는 현상이 사실 partial observability와 memory 부족 때문에 생긴 것인가?

### Conditions

축약된 condition set을 실행한다.

1. Partial observation을 사용하는 main DQN.
2. Partial observation을 사용하는 main n-step DQN.
3. Full-observation DQN.
4. Full-observation n-step DQN.

선택적 추가 condition:

5. Partial observation을 사용하는 recurrent DQN.
6. Partial observation을 사용하는 recurrent n-step DQN.

### Method

Full-observation diagnostic:

- network에 full symbolic grid observation을 제공한다.
- action space와 reward는 바꾸지 않는다.
- 이는 oracle diagnostic이며 main benchmark가 아니다.

Recurrent diagnostic:

- observation encoder 뒤에 GRU 또는 LSTM을 추가한다.
- truncated backpropagation through time으로 학습한다.
- 다른 hyperparameter는 가능한 한 동일하게 유지한다.

### Interpretation

Full-observation agent가 훨씬 빨리 학습한다면:

- discovery/propagation으로 해석한 실패의 일부가 사실 observability failure일 수 있다.

Recurrent agent가 격차를 줄인다면:

- memory가 중요한 mechanism이며, feedforward DQN의 한계로 논의해야 한다.

Full observation이 pattern을 바꾸지 않는다면:

- 원래 discovery/propagation 해석의 신뢰도가 올라간다.

---

## Experiment 6: Hyperparameter Sensitivity and Fairness Checks

### RQ6

결론이 핵심 hyperparameter에 robust한가, 아니면 특정 RIDE coefficient나 특정 n-step horizon의 artifact인가?

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

Sensitivity 실험은 주로 DoorKey-8x8에서 수행한다. 6x6보다 saturation이 덜하고, 16x16보다 비용이 낮기 때문이다.

### Method

Sensitivity에는 seed 수를 줄인다.

- screening 단계: hyperparameter setting당 5 seeds.
- 최종 선택된 setting: full seed count로 재실행.

보고할 것:

- 최종 test 전에 어떤 setting을 선택했는지.
- 가까운 setting에서도 qualitative conclusion이 유지되는지.
- RIDE 또는 n-step이 특정 값에 지나치게 민감한지.

### Interpretation

Robust conclusion:

- 여러 plausible beta와 n 값에서 같은 qualitative pattern이 나타난다.

Fragile conclusion:

- interaction이 특정 beta 또는 특정 n에서만 나타난다.
- 이 경우 결과를 일반적 결론이 아니라 hyperparameter-dependent result로 보고한다.

---

## 3. Evaluation Metrics

## 3.1 Primary Performance Metrics

### Final Success Rate

정의:

```text
success_rate = successful evaluation episodes 수 / total evaluation episodes 수
```

Final checkpoint와 best checkpoint를 모두 보고한다.

보고 항목:

- final checkpoint success rate
- best checkpoint success rate

이유:

- final performance는 convergence를 보여준다.
- best performance는 instability나 collapse가 있었는지 보여준다.

### Success-Rate AUC

정의:

Environment step에 대한 evaluation success-rate curve의 area under curve.

Normalized AUC:

```text
AUC = integral success_rate(t) dt / total_training_steps
```

이유:

- sample efficiency를 반영한다.
- final checkpoint에만 과도하게 의존하지 않는다.

### Extrinsic Return

정의:

Evaluation 중 원래 environment reward의 평균.

Success rate와 별도로 보고한다. MiniGrid success reward는 step penalty를 포함하는 경우가 많으므로, 두 agent가 같은 success rate를 가져도 효율성은 다를 수 있다.

### Episode Length on Success

정의:

Successful evaluation episode들 중 평균 step 수.

이유:

- 더 짧은 successful trajectory는 더 효율적인 policy를 의미한다.
- success rate가 saturated될 때 특히 유용하다.

---

## 3.2 Discovery Metrics

### First Success Timestep

정의:

agent가 처음으로 successful episode를 완료한 training timestep.

Survival analysis 스타일로 보고한다.

- 어떤 seed는 끝까지 성공하지 못할 수 있다.
- never-success seed는 training budget 시점에서 right-censored로 처리한다.

보고 항목:

- median time to first success
- timestep별 success 경험 seed 비율
- 가능하면 Kaplan-Meier-style survival curve

### First Key Pickup Timestep

정의:

agent가 처음으로 key pickup에 성공한 training timestep.

이유:

- key pickup은 DoorKey의 첫 주요 prerequisite이다.
- extrinsic reward가 나오기 전에도 발생할 수 있다.

### First Door Open Timestep

정의:

agent가 처음으로 door unlock/open에 성공한 training timestep.

이유:

- door opening은 더 뒤쪽의 ordered subgoal이다.
- generic state coverage보다 task-relevant하다.

### First Room Transition Timestep

정의:

agent가 door 너머 영역에 처음 도달한 timestep.

이유:

- door를 여는 것과 열린 문을 실제로 이용하는 것을 구분한다.

### Ordered Subgoal Completion Rate

각 evaluation 또는 training episode에서 다음 ordered chain이 발생했는지 기록한다.

```text
key pickup -> door open -> goal reached
```

보고 항목:

- `P(key)`
- `P(door)`
- `P(goal)`
- `P(door | key)`
- `P(goal | door)`
- `P(goal | key and door)`

이유:

- agent가 task sequence의 어느 단계에서 실패하는지 분해해서 볼 수 있다.

### State Coverage

정의:

Training 동안 방문한 unique underlying grid position의 수 또는 비율.

이 metric은 logging에만 사용하고 agent input으로는 사용하지 않는다.

보고 항목:

- episode당 unique cells visited
- cumulative unique cells visited
- unique object-adjacent states visited

주의:

- coverage가 높다고 반드시 task-relevant discovery가 좋은 것은 아니다.
- 항상 ordered subgoal metric과 함께 해석한다.

### Object Interaction Frequency

Episode마다 다음을 count한다.

- pickup actions
- successful key pickup
- toggle actions
- successful door opening
- irrelevant object 또는 wall에 대한 toggle attempt

이유:

- RIDE는 controllable object interaction을 장려할 가능성이 있다.
- 하지만 irrelevant interaction이 과도하면 curiosity가 잘못된 방향으로 작동한 것일 수 있다.

---

## 3.3 Propagation Metrics

### Post-Discovery Learning Speed

정의:

각 seed에서 first success 이후, evaluation success-rate가 특정 threshold에 도달하기까지 걸린 environment step 수.

권장 threshold:

- 25% success
- 50% success
- 75% success

중요:

- threshold에 도달하지 못한 seed는 censored로 처리한다.
- 실패 seed를 조용히 제외하지 않는다.

보고 항목:

- first success 이후 25%, 50%, 75% success 도달 시간
- 각 threshold에 도달한 seed 비율
- censored-time summary

### Post-Discovery AUC

정의:

각 seed의 첫 successful training episode 이후 구간에서 계산한 success-rate AUC.

이유:

- discovery 전 탐색 phase와 discovery 후 reward exploitation phase를 분리한다.

주의:

- first success가 없는 seed를 어떻게 처리했는지 명확히 보고한다.
- 제외한 seed 수 또는 censored 처리 방식을 보고한다.

### Value Propagation Along Successful Trajectory

저장된 successful trajectory를 사용한다.

Successful trajectory 위의 state/action에 대해 여러 training checkpoint에서 다음 값을 log한다.

```text
Q(s_t, a_t)
```

보고 항목:

- goal 근처 state에서 Q-value가 얼마나 빨리 양수가 되는지.
- positive value가 early state 쪽으로 얼마나 빨리 뒤로 이동하는지.
- distance-to-goal에 따른 value increase slope.

이유:

- reward 정보가 trajectory를 따라 backward propagation되는지를 직접 측정한다.

### Bellman Error Along Successful Trajectory

저장된 successful trajectory에 대해 TD error 또는 n-step target error를 계산한다.

보고 항목:

- trajectory position별 mean TD error
- training이 진행되며 TD error가 감소하는 양상
- 1-step target과 n-step target의 차이

이유:

- value propagation과 behavioral exploration을 구분하는 데 도움이 된다.

---

## 3.4 Interaction Metrics

### Factorial Interaction on AUC

정의:

```text
interaction_AUC = (AUC_nstep_RIDE - AUC_nstep_noRIDE)
                - (AUC_1step_RIDE - AUC_1step_noRIDE)
```

Positive interaction의 의미:

- RIDE가 n-step backup이 있을 때 더 큰 도움을 준다.

### Factorial Interaction on Discovery

같은 공식을 discovery metric에 적용한다.

예시:

- first success timestep
- first door-open timestep
- timestep `T`까지 goal에 도달할 확률

Time-to-event metric은 값이 작을수록 좋은 metric이므로, interaction을 해석할 때 방향을 명확히 맞춘다. 예를 들어 `-first_success_timestep` 또는 `P(success by T)`처럼 클수록 좋은 값으로 바꿔 계산할 수 있다.

### Factorial Interaction on Propagation

같은 공식을 propagation metric에 적용한다.

예시:

- first success 이후 50% success까지 걸린 시간
- post-discovery AUC
- value propagation slope

해석:

- interaction이 주로 discovery metric에서 나타나면, 조합이 successful trajectory를 만들어내는 데 도움을 준 것일 수 있다.
- interaction이 주로 propagation metric에서 나타나면, n-step이 RIDE가 만든 trajectory를 더 잘 exploit한 것일 수 있다.

---

## 3.5 Statistical Reporting

가능하면 paired analysis를 사용한다.

필수 reporting:

- seed across mean과 median
- 95% bootstrap confidence interval
- 여러 environment difficulty를 aggregate할 경우 stratified bootstrap
- 주요 pairwise comparison에 대한 probability of improvement
- appendix 또는 supplementary plot에 individual seed learning curve

권장 aggregate metric:

- median
- 여러 task를 aggregate할 경우 interquartile mean
- probability of improvement

피해야 할 것:

- point estimate만 보고 우월성을 주장하는 것
- best seed만 고르는 것
- raw variability 없이 smoothed curve만 보고하는 것

보고할 pairwise comparison:

1. DQN vs DQN + RIDE
2. DQN vs n-step DQN
3. DQN + RIDE vs n-step DQN + RIDE
4. n-step DQN vs n-step DQN + RIDE
5. DQN vs n-step DQN + RIDE

보고할 factorial effect:

- RIDE main effect
- n-step main effect
- RIDE x n-step interaction

이 effect들을 다음 metric에 대해 보고한다.

- final success rate
- success-rate AUC
- first success
- first key pickup
- first door open
- post-discovery learning speed

---

## 4. Decision Rules

### Discovery-Dominant Bottleneck

다음 조건이 모두 대체로 만족될 때 discovery가 dominant bottleneck이라고 결론낸다.

- RIDE가 first key pickup, first door opening, first success를 개선한다.
- RIDE가 pre-reward exploration 또는 ordered subgoal completion을 개선한다.
- n-step이 pre-discovery metric에는 큰 영향을 주지 않는다.
- post-discovery 차이보다 discovery 차이가 더 크다.

### Propagation-Dominant Bottleneck

다음 조건이 모두 대체로 만족될 때 propagation이 dominant bottleneck이라고 결론낸다.

- n-step이 post-discovery learning speed를 개선한다.
- n-step이 successful trajectory 위의 value propagation을 개선한다.
- fixed successful replay experiment에서 n-step이 더 빠른 학습을 보인다.
- first success 이전 discovery metric은 1-step과 n-step 사이에 비슷하다.

### Complementary Interaction

다음 조건이 모두 대체로 만족될 때 complementarity가 있다고 결론낸다.

- RIDE x n-step interaction이 positive이고 uncertainty interval이 대부분 0보다 크다.
- interaction이 AUC 또는 success rate에서 나타난다.
- mechanism metric이 왜 interaction이 생겼는지 설명한다.
  - RIDE가 successful trajectory를 늘린다.
  - n-step이 그것을 더 빠르게 exploit한다.

### No Clear Bottleneck

다음 경우에는 명확한 bottleneck 결론을 내리지 않는다.

- effect가 environment size별로 일관되지 않다.
- confidence interval이 넓다.
- first-success metric과 post-discovery metric이 서로 다른 이야기를 한다.
- full-observation 또는 recurrent control이 pattern을 크게 바꾼다.

---

## 5. Minimum Viable Experimental Package

시간이 제한되어 있다면 다음 축약 package를 실행한다.

1. DoorKey-6x6, 8x8, 16x16에서 main 2x2 factorial.
2. condition당 10 paired seeds.
3. fixed held-out evaluation layouts.
4. 다음 metric:
   - final success rate
   - success-rate AUC
   - first success timestep
   - first key pickup timestep
   - first door open timestep
   - post-discovery time to 50% success
5. DoorKey-8x8에서 fixed successful replay propagation test.
6. Bootstrap confidence interval과 probability of improvement.

이 minimum package는 원래 proposal보다 훨씬 강하다. 계획된 intervention이 실제로 의도한 mechanism을 건드리는지 직접 확인하기 때문이다.

---

## 6. 최종 보고서에 사용할 참고문헌

- Mnih et al. (2015), "Human-level control through deep reinforcement learning." Nature.
- Sutton and Barto (2018/2020), "Reinforcement Learning: An Introduction." n-step returns and temporal-difference learning.
- Hessel et al. (2018), "Rainbow: Combining Improvements in Deep Reinforcement Learning." Multi-step returns as a DQN extension.
- Raileanu and Rocktaschel (2020), "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments." ICLR.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS.
- Chevalier-Boisvert et al. (2023), "Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks." NeurIPS Datasets and Benchmarks.
