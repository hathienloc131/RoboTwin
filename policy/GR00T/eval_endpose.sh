#!/bin/bash
# Convenience wrapper around script/eval_gr00t_endpose.py, mirroring eval.sh's launch
# convention. Use this one for a checkpoint trained on RoboTwin's native world-frame
# endpose schema (script/lerobot/convert_robotwin_to_lerobot.py's output / Isaac-GR00T's
# "equi_robotwin_2hand_quat" data config) -- use eval.sh instead for a MotionTrans/
# camera-frame/rot6d checkpoint.
#
# Usage:
#   bash policy/GR00T/eval_endpose.sh <task> <task_config> <model_path> <instruction> <num_trials> <seed> <gpu_id>
# Example:
#   bash policy/GR00T/eval_endpose.sh grab_roller demo_randomized /path/to/checkpoint \
#       "Use both arms to grab the roller on the table." 20 0 0
#
# If the gr00t package isn't installed into this environment, export GR00T_REPO_PATH
# pointing at the ego_gr00t repo root before calling this script -- script/eval_gr00t_endpose.py
# picks it up automatically (--gr00t-path defaults to that env var).

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
python script/eval_gr00t_endpose.py \
    --model-path "${model_path}" \
    --task "${task_name}" \
    --task-config "${task_config}" \
    --instruction "${instruction}" \
    --num-trials "${num_trials}" \
    --seed "${seed}" \
    --device cuda:0
