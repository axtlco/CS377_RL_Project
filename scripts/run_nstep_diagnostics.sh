#!/usr/bin/env bash
set -euo pipefail

# Diagnostic runner for strengthening the n-step DQN baseline before expanding
# seed count. This is not a final factorial experiment; it tests whether the
# default exploration schedule is too short for DoorKey-8x8.

ENV_NAME=${ENV_NAME:-doorkey_8x8}
SEEDS=${SEEDS:-"0"}
STEPS=${STEPS:-1000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
DEVICE=${DEVICE:-cuda}
LOG_DIR=${LOG_DIR:-logs/nstep_diagnostics}
RUN_ROOT=${RUN_ROOT:-outputs/nstep_diagnostics/$(date +%Y-%m-%d_%H-%M-%S)}
ANALYSIS_OUT=${ANALYSIS_OUT:-analysis/nstep_diagnostics}
VARIANTS=${VARIANTS:-"default eps500k_end01 eps1m_end01 eps500k_end02 cap500k_eps500k_end01"}

mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_ROOT}"

RUN_DIRS=()

variant_overrides() {
  local variant=$1
  OVERRIDES=()
  case "${variant}" in
    default)
      ;;
    eps500k_end01)
      OVERRIDES+=("agent.epsilon_decay_steps=500000")
      OVERRIDES+=("agent.epsilon_end=0.1")
      ;;
    eps1m_end01)
      OVERRIDES+=("agent.epsilon_decay_steps=1000000")
      OVERRIDES+=("agent.epsilon_end=0.1")
      ;;
    eps500k_end02)
      OVERRIDES+=("agent.epsilon_decay_steps=500000")
      OVERRIDES+=("agent.epsilon_end=0.2")
      ;;
    cap500k_eps500k_end01)
      OVERRIDES+=("agent.replay_capacity=500000")
      OVERRIDES+=("agent.epsilon_decay_steps=500000")
      OVERRIDES+=("agent.epsilon_end=0.1")
      ;;
    *)
      echo "Unknown diagnostic variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    variant_overrides "${variant}"
    run_dir="${RUN_ROOT}/${ENV_NAME}/dqn_nstep/${variant}/seed${seed}"
    log_path="${LOG_DIR}/${ENV_NAME}_dqn_nstep_${variant}_seed${seed}.log"

    echo "===== ${ENV_NAME} dqn_nstep ${variant} seed=${seed} steps=${STEPS} ====="
    python -m rl_project.train \
      algorithm=dqn_nstep \
      package=minimum \
      "env=${ENV_NAME}" \
      "seed=${seed}" \
      "device=${DEVICE}" \
      "package.training_steps.${ENV_NAME}=${STEPS}" \
      "package.eval_interval.${ENV_NAME}=${EVAL_INTERVAL}" \
      "package.eval_episodes.${ENV_NAME}=${EVAL_EPISODES}" \
      "logging.save_interval=${SAVE_INTERVAL}" \
      logging.progress=true \
      "hydra.run.dir=${run_dir}" \
      "${OVERRIDES[@]}" \
      2>&1 | tee "${log_path}"

    RUN_DIRS+=("${run_dir}")
  done
done

python -m rl_project.analyze "${RUN_DIRS[@]}" --out "${ANALYSIS_OUT}"
echo "n-step diagnostic runs written under ${RUN_ROOT}"
echo "n-step diagnostic analysis written to ${ANALYSIS_OUT}"
