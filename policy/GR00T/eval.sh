#!/bin/bash
# Convenience wrapper around script/eval_gr00t.py, mirroring the launch convention of the
# other policy/<Name>/eval.sh scripts (Your_Policy, pi0).
#
# Usage:
#   bash policy/GR00T/eval.sh <task> <task_config> <model_path> <instruction> <num_trials> <seed> <gpu_id>
# Example:
#   bash policy/GR00T/eval.sh stack_bowls_three demo_randomized /path/to/checkpoint \
#       "stack the bowls" 20 0 0

task_name=${1}
task_config=${2}
model_path=${3}
instruction=${4}
num_trials=${5}
seed=${6}
gpu_id=${7}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "$(dirname "$0")/../.." # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_gr00t.py \
    --model-path "${model_path}" \
    --task "${task_name}" \
    --task-config "${task_config}" \
    --instruction "${instruction}" \
    --num-trials "${num_trials}" \
    --seed "${seed}" \
    --device cuda:0
