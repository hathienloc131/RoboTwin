#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <MODEL> [TASK1,TASK2,...|all] [CUDA_ID]"
    echo ""
    echo "  MODEL    checkpoint dir name under /home/locht1/vr_checkpoint/ (required)"
    echo "  TASK     comma-separated list from: ${!TASKS[*]}, or 'all' (default: all)"
    echo "  CUDA_ID  CUDA device index (default: 0)"
    exit 1
}

# task_key -> "task|task-config|instruction|output-dir-suffix"
declare -A TASKS=(
    [adjust_bottle]="adjust_bottle|demo_clean|Pick up the bottle from the table head-up with the correct arm.|adjust_bottle_4"
    [blocks_ranking_rgb]="blocks_ranking_rgb|demo_clean|Place the red block, green block, and blue block in order from left to right, placing them in a row.|block_ranking_rgb_4"
    [grab_roller]="grab_roller|demo_clean|Use both arms to grab the roller on the table.|grab_roller_4"
    [move_stapler_pad]="move_stapler_pad|demo_clean|Use the appropriate arm to move the stapler onto the colored mat.|move_stapler_pub_4"
)

[ $# -lt 1 ] && usage

MODEL=$1
TASK_ARG=${2:-all}
CUDA_ID=${3:-0}

if [ "$TASK_ARG" = "all" ]; then
    TASK_LIST=("${!TASKS[@]}")
else
    IFS=',' read -r -a TASK_LIST <<< "$TASK_ARG"
    for key in "${TASK_LIST[@]}"; do
        if [ -z "${TASKS[$key]+x}" ]; then
            echo "Unknown task: $key"
            usage
        fi
    done
fi

export CUDA_VISIBLE_DEVICES=$CUDA_ID

run_task() {
    local key=$1
    IFS='|' read -r task task_config instruction out_suffix <<< "${TASKS[$key]}"
    python script/eval_gr00t_endpose.py \
        --model-path /home/locht1/vr_checkpoint/$MODEL \
        --task "$task" \
        --task-config "$task_config" \
        --instruction "$instruction" \
        --num-trials 50 \
        --seed 0 \
        --n-action-steps 4 --output-dir ~/eval_robotwin/$MODEL/$out_suffix \
        --device cuda:0 --gr00t-path /home/locht1/miniconda3/envs/gr00t/lib/python3.10/site-packages
}

for key in "${TASK_LIST[@]}"; do
    run_task "$key"
done
