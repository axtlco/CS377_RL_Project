#!/usr/bin/env bash
set -euo pipefail

# Experiment 3 pilot from docs/plan.md:
# isolate propagation by holding discovery data fixed.

ENV_NAME=${ENV_NAME:-doorkey_8x8}
SEEDS=${SEEDS:-"0 1 2 3 4"}
EPISODE_COUNT=${EPISODE_COUNT:-100}
TARGET_SUCCESS_EPISODES=${TARGET_SUCCESS_EPISODES:-1}
OFFLINE_UPDATES=${OFFLINE_UPDATES:-2000}
DEVICE=${DEVICE:-auto}
LOG_DIR=${LOG_DIR:-logs/fixed_replay_pilot}
RUN_ROOT=${RUN_ROOT:-outputs/fixed_replay_pilot/$(date +%Y-%m-%d_%H-%M-%S)}
ANALYSIS_OUT=${ANALYSIS_OUT:-analysis/fixed_replay_pilot}

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_ROOT}"

RUN_DIRS=()

for seed in ${SEEDS}; do
  dataset_dir="fixed_replay_datasets/${ENV_NAME}_seed${seed}_success${TARGET_SUCCESS_EPISODES}"
  mkdir -p "${dataset_dir}"

  echo "===== collect fixed replay ${ENV_NAME} seed=${seed} ====="
  python -m rl_project.offline_train \
    algorithm=dqn_nstep \
    package=minimum \
    replay=fixed_replay \
    "env=${ENV_NAME}" \
    "seed=${seed}" \
    "device=${DEVICE}" \
    replay.collector_policy=mixed \
    "replay.episode_count=${EPISODE_COUNT}" \
    "replay.target_success_episodes=${TARGET_SUCCESS_EPISODES}" \
    "replay.offline_updates=1" \
    logging.progress=false \
    hydra.run.dir="${dataset_dir}" \
    2>&1 | tee "${LOG_DIR}/${ENV_NAME}_collect_seed${seed}.log"

  dataset_path="${dataset_dir}/fixed_replay/transitions.parquet"
  for algorithm in dqn_1step dqn_nstep; do
    run_dir="${RUN_ROOT}/${ENV_NAME}/${algorithm}/seed${seed}"
    echo "===== offline ${algorithm} ${ENV_NAME} seed=${seed} ====="
    python -m rl_project.offline_train \
      "algorithm=${algorithm}" \
      package=minimum \
      replay=fixed_replay \
      "env=${ENV_NAME}" \
      "seed=${seed}" \
      "device=${DEVICE}" \
      "replay.dataset_path=${dataset_path}" \
      "replay.offline_updates=${OFFLINE_UPDATES}" \
      replay.diagnostic_interval=100 \
      logging.progress=false \
      "hydra.run.dir=${run_dir}" \
      2>&1 | tee "${LOG_DIR}/${ENV_NAME}_${algorithm}_seed${seed}.log"
    RUN_DIRS+=("${run_dir}")
  done
done

python -m rl_project.analyze "${RUN_DIRS[@]}" --out "${ANALYSIS_OUT}"
echo "Fixed replay runs written under ${RUN_ROOT}"
echo "Fixed replay analysis written to ${ANALYSIS_OUT}"
