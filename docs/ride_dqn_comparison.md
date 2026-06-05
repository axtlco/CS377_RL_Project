# RIDE-DQN Integration Comparison

작성일: 2026-06-04

## 1. 목적

DoorKey-8x8 환경에서 RIDE와 DQN을 결합하는 방식이 성능에 어떤 영향을 주는지 비교했다.

초기 구현에서는 RIDE intrinsic reward를 transition 수집 시점에 계산한 뒤 `reward_train = reward_ext + reward_ride`로 replay buffer에 저장했다. 이 방식은 replay에서 오래된 intrinsic reward를 계속 학습하게 만들 수 있다. 이후 구현을 수정하여 DQN update 시점에 현재 RIDE embedding으로 intrinsic reward를 다시 계산하도록 바꾸었다.

이번 비교의 핵심 질문은 다음과 같다.

1. 기존 stored RIDE reward 방식이 실제로 불안정했는가?
2. update-time recompute 방식이 성능을 개선하는가?
3. recompute 방식에 reward normalization, clipping, 낮은 beta를 함께 적용하면 성능이 개선되는가?

## 2. 실험 조건

공통 조건:

| 항목 | 값 |
|---|---|
| Environment | MiniGrid DoorKey-8x8 |
| Training steps | 1,000,000 |
| Evaluation interval | 10,000 |
| Evaluation episodes | 100 |
| Seeds | 0, 1, 2, 3, 4 |
| Epsilon schedule | `epsilon_decay_steps=1000000`, `epsilon_end=0.1` |
| Algorithm focus | `dqn_nstep`, `dqn_ride_nstep` |

비교한 조건:

| 이름 | 설명 | RIDE beta | RIDE reward normalization | RIDE clipping |
|---|---|---:|---|---|
| `dqn_nstep_baseline` | RIDE 없는 n-step DQN baseline | N/A | N/A | N/A |
| `ride_stored_old` | 기존 RIDE 구현. 수집 시점의 `reward_train`을 replay에 저장 | 0.1 | none | none |
| `ride_recompute` | DQN update 시점에 현재 RIDE embedding으로 intrinsic reward 재계산 | 0.1 | none | none |
| `ride_recompute_normclip_beta001` | recompute 방식에 reward scale 제어 추가 | 0.01 | `ema_std` | `reward_clip_max=5.0` |

## 3. Aggregate 결과

| Condition | Final Success Mean | Final Median | Best Success Mean | Best Median | AUC Mean | AUC Median | Train Success Episodes Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dqn_nstep_baseline` | 0.146 | 0.00 | 0.224 | 0.27 | 0.03721 | 0.05005 | 1626.4 |
| `ride_stored_old` | 0.126 | 0.01 | 0.322 | 0.11 | 0.06743 | 0.00640 | 2860.4 |
| `ride_recompute` | 0.038 | 0.00 | 0.178 | 0.10 | 0.03511 | 0.02010 | 1367.4 |
| `ride_recompute_normclip_beta001` | 0.192 | 0.00 | 0.506 | 0.70 | 0.12806 | 0.12650 | 4922.6 |

## 4. Seed별 결과

### 4.1 Stored Old vs Recompute vs Recompute + Normalization/Clipping

| Version | Seed | Final Success | Best Success | AUC | First Success Step | Post Discovery to 50% | Train Success Episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stored_old` | 0 | 0.47 | 0.75 | 0.15855 | 22997 | 737003 | 6271 |
| `recompute` | 0 | 0.05 | 0.10 | 0.02045 | 27350 | N/A | 1798 |
| `recompute_normclip_beta001` | 0 | 0.48 | 0.70 | 0.12650 | 14041 | 835959 | 5718 |
| `stored_old` | 1 | 0.00 | 0.11 | 0.00640 | 34496 | N/A | 1107 |
| `recompute` | 1 | 0.00 | 0.10 | 0.01080 | 65196 | N/A | 886 |
| `recompute_normclip_beta001` | 1 | 0.00 | 0.81 | 0.17940 | 36371 | 713629 | 6803 |
| `stored_old` | 2 | 0.01 | 0.09 | 0.00425 | 15787 | N/A | 687 |
| `recompute` | 2 | 0.00 | 0.00 | 0.00000 | 89495 | N/A | 53 |
| `recompute_normclip_beta001` | 2 | 0.00 | 0.02 | 0.00070 | 15787 | N/A | 303 |
| `stored_old` | 3 | 0.14 | 0.60 | 0.16160 | 53858 | 706142 | 5387 |
| `recompute` | 3 | 0.14 | 0.53 | 0.12420 | 37117 | 752883 | 2908 |
| `recompute_normclip_beta001` | 3 | 0.48 | 0.91 | 0.31220 | 19030 | 520970 | 10458 |
| `stored_old` | 4 | 0.01 | 0.06 | 0.00635 | 12119 | N/A | 850 |
| `recompute` | 4 | 0.00 | 0.16 | 0.02010 | 68222 | N/A | 1192 |
| `recompute_normclip_beta001` | 4 | 0.00 | 0.09 | 0.02150 | 20453 | N/A | 1331 |

### 4.2 Best Checkpoint for `ride_recompute_normclip_beta001`

| Seed | Best Step | Best Success | Final Step | Final Success |
|---:|---:|---:|---:|---:|
| 0 | 870000 | 0.70 | 1000000 | 0.48 |
| 1 | 910000 | 0.81 | 1000000 | 0.00 |
| 2 | 610000 | 0.02 | 1000000 | 0.00 |
| 3 | 840000 | 0.91 | 1000000 | 0.48 |
| 4 | 890000 | 0.09 | 1000000 | 0.00 |

## 5. 해석

### 5.1 n-step은 여전히 필수적이다

DoorKey-8x8에서는 1-step DQN 계열이 거의 성능을 내지 못했다. 반면 n-step baseline은 일부 seed에서 의미 있는 success signal을 보였다. 이는 sparse reward 환경에서 reward propagation이 주요 병목이라는 기존 해석과 일치한다.

### 5.2 기존 stored RIDE reward 방식은 peak를 만들지만 불안정하다

`ride_stored_old`는 `dqn_nstep_baseline`보다 best success와 AUC 평균이 높았다. 하지만 median AUC는 낮고 seed별 편차가 컸다.

특히 seed0과 seed3에서는 강한 성능을 보였지만, 다른 seed에서는 거의 실패했다. 따라서 기존 방식은 RIDE가 도움이 될 가능성을 보여주지만, 안정적인 개선이라고 보기는 어렵다.

### 5.3 recompute만 적용하면 성능이 떨어졌다

`ride_recompute`는 stale reward 문제를 직접 해결한 구조다. 그러나 beta와 reward scale을 그대로 둔 상태에서는 성능이 오히려 떨어졌다.

이는 stale reward가 유일한 병목이 아니었음을 의미한다. update-time recompute는 intrinsic reward를 현재 RIDE embedding에 맞게 다시 계산하지만, 동시에 DQN target을 더 non-stationary하게 만들 수 있다. 따라서 reward scale 제어 없이 recompute만 적용하는 것은 안정적이지 않았다.

### 5.4 recompute + reward scale control은 가장 좋은 결과를 보였다

`ride_recompute_normclip_beta001`는 지금까지 비교한 RIDE 계열 중 가장 좋은 결과를 보였다.

주요 개선:

| Metric | `ride_stored_old` | `ride_recompute_normclip_beta001` |
|---|---:|---:|
| Final success mean | 0.126 | 0.192 |
| Best success mean | 0.322 | 0.506 |
| Best success median | 0.11 | 0.70 |
| AUC mean | 0.06743 | 0.12806 |
| AUC median | 0.00640 | 0.12650 |
| Train success episodes mean | 2860.4 | 4922.6 |

이 결과는 RIDE-DQN 결합 문제의 핵심이 stale reward 하나만은 아니며, intrinsic reward scale control이 함께 필요하다는 것을 보여준다.

## 6. 현재 결론

현재까지의 가장 중요한 결론은 다음과 같다.

1. DoorKey-8x8에서 n-step은 필수적이다.
2. RIDE는 n-step과 결합될 때 peak performance와 AUC를 개선할 가능성이 있다.
3. 단순 update-time recompute는 성능을 개선하지 못했다.
4. `beta=0.01`, EMA normalization, clipping을 함께 적용한 recompute RIDE가 가장 좋은 결과를 보였다.
5. 다만 final success는 여전히 불안정하다. seed1은 best 0.81까지 올라갔지만 final은 0.00으로 떨어졌다.

따라서 현재 가장 타당한 해석은 다음과 같다.

> RIDE+n-step can improve exploration and peak performance in DoorKey-8x8, but only when intrinsic reward scale is controlled. The original stored-reward implementation was unstable, and recomputing intrinsic rewards alone was insufficient. The best-performing variant combined update-time recomputation with beta reduction, normalization, and clipping.

## 7. 다음 실험 제안

### 7.1 Best checkpoint visualization

`ride_recompute_normclip_beta001`에서 seed0, seed1, seed3의 best checkpoint를 시각화하는 것이 우선이다.

추천 대상:

| Seed | Best Step | Best Success |
|---:|---:|---:|
| 0 | 870000 | 0.70 |
| 1 | 910000 | 0.81 |
| 3 | 840000 | 0.91 |

확인할 점:

- key pickup이 안정적으로 나오는가?
- door opening이 성공적으로 이어지는가?
- goal까지 도달하는 trajectory가 실제로 학습된 것인가?
- best 이후 final collapse가 policy degradation인지, evaluation variance인지 확인할 수 있는가?

### 7.2 Longer training

현재 best success는 높지만 final success가 불안정하다. 따라서 같은 조건으로 DoorKey-8x8 2M steps를 돌려 final stability가 개선되는지 확인할 필요가 있다.

추천 조건:

| 항목 | 값 |
|---|---|
| Algorithm | `dqn_ride_nstep` |
| RIDE beta | `0.01` |
| RIDE normalization | `ema_std` |
| RIDE clipping | `reward_clip_max=5.0` |
| Training steps | 2,000,000 |
| Seeds | 0, 1, 2, 3, 4 |

### 7.3 Additional seeds

seed별 variance가 크기 때문에, 최종 보고서에서는 seed 5-9 추가가 유용하다. 특히 seed2, seed4 실패가 환경 layout 난이도 때문인지, 알고리즘 불안정성 때문인지 확인해야 한다.

## 8. 보고서용 요약 문장

DoorKey-8x8 experiments show that n-step returns are necessary for sparse reward propagation. RIDE alone does not solve the task, but RIDE combined with n-step returns can improve peak success and success-rate AUC. A naive update-time recomputation of intrinsic rewards under the original reward scale performed worse than the stored-reward implementation. However, recomputation combined with a smaller intrinsic reward coefficient (`beta=0.01`), EMA reward normalization, and reward clipping substantially improved performance, achieving a best-success mean of 0.506 and AUC mean of 0.128 across five seeds. This suggests that RIDE's effectiveness in DQN depends critically on controlling intrinsic reward scale, not only on avoiding stale replayed intrinsic rewards.
