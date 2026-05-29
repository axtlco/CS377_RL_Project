#!/usr/bin/env bash
set -euo pipefail

# Pilot grid from docs/plan.md:
# - Check whether agents solve too early, fail completely, or need a harder/intermediate task.
# - Keep architecture, update ratio, batch size, seeds, and evaluation protocol fixed.

ENVS=${ENVS:-"doorkey_6x6 doorkey_8x8"}
ALGORITHMS=${ALGORITHMS:-"dqn_1step dqn_nstep dqn_ride_1step dqn_ride_nstep"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DEVICE=${DEVICE:-auto}
LOG_DIR=${LOG_DIR:-logs/pilot_grid}
RUN_ROOT=${RUN_ROOT:-outputs/pilot_grid/$(date +%Y-%m-%d_%H-%M-%S)}
ANALYSIS_OUT=${ANALYSIS_OUT:-analysis/pilot_grid}

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_ROOT}"

RUN_DIRS=()

run_one() {
  local env_name=$1
  local algorithm=$2
  local seed=$3
  local steps
  local interval
  local eval_episodes

  case "${env_name}" in
    doorkey_6x6)
      steps=${DOORKEY_6X6_STEPS:-250000}
      interval=${DOORKEY_6X6_EVAL_INTERVAL:-5000}
      eval_episodes=${DOORKEY_6X6_EVAL_EPISODES:-100}
      ;;
    doorkey_8x8)
      steps=${DOORKEY_8X8_STEPS:-300000}
      interval=${DOORKEY_8X8_EVAL_INTERVAL:-10000}
      eval_episodes=${DOORKEY_8X8_EVAL_EPISODES:-100}
      ;;
    doorkey_12x12)
      steps=${DOORKEY_12X12_STEPS:-500000}
      interval=${DOORKEY_12X12_EVAL_INTERVAL:-25000}
      eval_episodes=${DOORKEY_12X12_EVAL_EPISODES:-100}
      ;;
    doorkey_16x16)
      steps=${DOORKEY_16X16_STEPS:-500000}
      interval=${DOORKEY_16X16_EVAL_INTERVAL:-25000}
      eval_episodes=${DOORKEY_16X16_EVAL_EPISODES:-100}
      ;;
    *)
      echo "Unknown env: ${env_name}" >&2
      exit 2
      ;;
  esac

  local log_path="${LOG_DIR}/${env_name}_${algorithm}_seed${seed}.log"
  local run_dir="${RUN_ROOT}/${env_name}/${algorithm}/seed${seed}"
  echo "===== ${env_name} ${algorithm} seed=${seed} steps=${steps} ====="
  python -m rl_project.train \
    "algorithm=${algorithm}" \
    package=minimum \
    "env=${env_name}" \
    "seed=${seed}" \
    "device=${DEVICE}" \
    "package.training_steps.${env_name}=${steps}" \
    "package.eval_interval.${env_name}=${interval}" \
    "package.eval_episodes.${env_name}=${eval_episodes}" \
    logging.progress=true \
    "hydra.run.dir=${run_dir}" \
    2>&1 | tee "${log_path}"
  RUN_DIRS+=("${run_dir}")
}

for env_name in ${ENVS}; do
  for seed in ${SEEDS}; do
    for algorithm in ${ALGORITHMS}; do
      run_one "${env_name}" "${algorithm}" "${seed}"
    done
  done
done

python -m rl_project.analyze "${RUN_DIRS[@]}" --out "${ANALYSIS_OUT}"
echo "Pilot runs written under ${RUN_ROOT}"
echo "Pilot analysis written to ${ANALYSIS_OUT}"
