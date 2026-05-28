# Server Check Notes(읽으면 좋음)
## 1. Python 환경 준비

서버에 `python3.11` 명령이 없으면 먼저 사용 가능한 파이썬 버전 체크

```bash
python --version
python3 --version
which python
which python3
ls /usr/bin/python*
```

`python3`가 3.11 이상이면 다음처럼 가상환경 생성 

```bash
cd ~/CS377_RL_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

CUDA 서버라면 서버 CUDA 버전에 맞는 PyTorch wheel을 먼저 설치 

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

설치 확인은 이렇게: 

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pytest -q
```

## 2. Feasible Test 

```bash
python -m rl_project.train algorithm=dqn_1step package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
python -m rl_project.train algorithm=dqn_nstep package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
python -m rl_project.train algorithm=dqn_ride_1step package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
python -m rl_project.train algorithm=dqn_ride_nstep package=minimum env=doorkey_6x6 seed=0 smoke.enabled=true
```

성공하면 각 run directory에 다음 파일이 생길거임

```text
scalars.csv
tables/evaluation.parquet
tables/episodes.parquet
checkpoints/step_*.pt
tensorboard/
```

## 3. 짧은 Pilot Run

변수 budget 짧게 잡고 테스트 

```bash
python -m rl_project.train \
  algorithm=dqn_ride_nstep \
  package=minimum \
  env=doorkey_8x8 \
  seed=0 \
  package.training_steps.doorkey_8x8=10000 \
  package.eval_interval.doorkey_8x8=1000 \
  package.eval_episodes.doorkey_8x8=10 \
  logging.progress=true
```

**서버에서는 `tmux` 써서**

간단한 `tmux` 사용법:

```bash
# 새 tmux session 생성
tmux new -s cs377

# tmux 안에서 실험 실행
source .venv/bin/activate
python -m rl_project.train algorithm=dqn_ride_nstep package=minimum env=doorkey_8x8 seed=0 logging.progress=true
```

실행 중인 session에서 빠져나오려면 `Ctrl-b`를 누른 뒤 `d`를 누른다. 프로세스는 서버에서 계속 실행

다시 접속하려면:

```bash
tmux attach -t cs377
```

실행 중인 session 목록 확인:

```bash
tmux ls
```

session을 종료하려면 tmux 안에서 실행 중인 프로세스를 끝낸 뒤:

```bash
exit
```

## 4. 결과 Aggregation

전체 `outputs` 아래 모든 run을 aggregate하려면:

```bash
python -m rl_project.analyze outputs --out analysis
```

주의: 이 명령은 가장 최근 run 하나만 보는 것이 아니라 `outputs` 아래에서 `tables/evaluation.*`가 있는 모든 run directory를 재귀적으로 찾아 합치니까 Smoke run, pilot run, final run이 섞일 수 있음 

특정 run 하나만 분석하려면 run directory를 직접 넘기면 됨

```bash
python -m rl_project.analyze outputs/2026-05-28/08-03-xx_dqn_ride_nstep_doorkey_6x6_seed0 --out analysis/latest
```

특정 날짜만 분석하려면:

```bash
python -m rl_project.analyze outputs/2026-05-28 --out analysis/2026-05-28
```

여러 run만 골라 분석하려면:

```bash
python -m rl_project.analyze \
  outputs/2026-05-28/run_a \
  outputs/2026-05-28/run_b \
  --out analysis/selected
```

생성되는 주요 파일:

```text
analysis/checkpoint_eval.csv
analysis/run_summary.csv
analysis/aggregate_summary.csv
analysis/factorial_effects.csv
```

각 파일의 의미:

- `checkpoint_eval.csv`: checkpoint별 success curve
- `run_summary.csv`: run별 final success, best success, AUC, first-success 요약
- `aggregate_summary.csv`: package/env/algorithm별 mean, median, bootstrap CI
- `factorial_effects.csv`: RIDE main effect, n-step main effect, RIDE x n-step interaction

## 5. TensorBoard 시각화

학습 중/후 scalar는 TensorBoard로 바로 볼 수 있음. `logging.tensorboard=true`가 기본값

```bash
tensorboard --logdir outputs --host 0.0.0.0 --port 6006
```

SSH port forwarding이 필요하면 로컬에서:

```bash
ssh -L 6006:localhost:6006 elicer@서버주소
```

그 다음 브라우저에서 다음 주소로 접속

```text
http://localhost:6006
```

볼 만한 값들: 

```text
eval/success_rate
eval/return_ext_mean
train/loss
train/reward_ext_mean
train/reward_train_mean
train/reward_ride_mean
ride/auxiliary_loss
ride/forward_loss
ride/inverse_loss
episode/return_ext
episode/return_ride
```

## 6. 시각화

- TensorBoard를 통한 scalar curve 시각화
- `rl_project.analyze`가 만든 CSV/Parquet table을 이용한 외부 plotting
- checkpoint에서 greedy policy rollout을 렌더링한 GIF 저장

agent가 실제 MiniGrid에서 움직이는 모습을 GIF로 저장하려면 checkpoint path를 넘긴다.

```bash
python -m rl_project.visualize \
  outputs/2026-05-28/08-03-xx_dqn_ride_nstep_doorkey_6x6_seed0/checkpoints/step_200.pt \
  --out visualizations/ride_nstep_seed0.gif \
  --seed 123 \
  --fps 6
```

`--seed`를 바꾸면 다른 DoorKey layout에서 policy가 어떻게 움직이는지 확인할 수 있다. 너무 긴 episode를 자르고 싶으면 `--max-steps`를 쓴다.

```bash
python -m rl_project.visualize \
  outputs/.../checkpoints/step_10000.pt \
  --out visualizations/rollout.gif \
  --seed 123 \
  --max-steps 200
```

GIF 생성에는 `imageio`가 필요하다. 새로 추가된 의존성이므로 기존 venv에서는 한 번 다시 설치한다.
기본 GIF에는 step/action/position overlay가 들어가고, 같은 이름의 `.csv` trace도 같이 생성된다.

```bash
python -m pip install -e ".[dev]"
```

trace만 빠르게 확인하려면:

```bash
head visualizations/ride_nstep_seed0.csv
```

만약 action이 `done`, `pickup`, `toggle`처럼 제자리 action만 반복되면 GIF가 움직이지 않는 것이 정상이다. 이 경우는 시각화 문제가 아니라 해당 checkpoint의 greedy policy가 아직 이동하는 policy를 학습하지 못한 것이다.

아직 `python -m rl_project.plot analysis --out figures` 같은 보고서용 static plotting CLI은 추가해야함. 

`checkpoint_eval.csv`, `aggregate_summary.csv`, `factorial_effects.csv`를 읽어 success curve, final success bar, AUC summary, factorial interaction plot을 생성하는 모듈을 추가해야함.
