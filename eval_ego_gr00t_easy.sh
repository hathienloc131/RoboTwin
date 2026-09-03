#!/usr/bin/env bash
set -euo pipefail

# task_key -> "task|task-config|instruction|output-dir-suffix"
# Parallel arrays instead of `declare -A` for portability (bash 3.2, e.g. macOS
# stock bash, has no associative arrays).
TASK_KEYS=(
    adjust_bottle
    blocks_ranking_rgb
    stack_bowls_three
)
#TASK_DEFS=(
#    "adjust_bottle|demo_randomized|Use the correct arm to pick up the plastic drink bottle.|adjust_bottle_hard_4"
#)

TASK_DEFS=(
    "adjust_bottle|demo_clean_bg16|Use the correct arm to pick up the plastic drink bottle.|adjust_bottle_4"
    "blocks_ranking_rgb|demo_clean_bg16|Place the red block, green block, and blue block in the order of red, green, and blue from left to right, placing in a row.|blocks_ranking_rgb_4"
    "stack_bowls_three|demo_clean_bg16|Stack the three bowls on top of each other.|stack_bowls_three_4"
)

usage() {
    echo "Usage: $0 <MODEL> [TASK1,TASK2,...|all] [CUDA_ID] [EXEC]"
    echo ""
    echo "  MODEL             checkpoint dir name under /home/locht1/vr_checkpoint/ (required)"
    echo "  TASK              comma-separated list from: ${TASK_KEYS[*]}, or 'all' (default: all)"
    echo "  CUDA_ID           CUDA device index (default: 0)"
    echo "  EXEC              n-action-steps (default: 4)"
    echo ""
    echo "  script/eval_gr00t.py has no --denoising-steps flag, so unlike eval.sh /"
    echo "  eval_baseline.sh / eval_thirdview.sh this script does not take a"
    echo "  DENOISING_STEPS argument."
    exit 1
}

# Prints the def string for a task key, or nothing (+ non-zero exit) if unknown.
task_def() {
    local key=$1
    local i
    for i in "${!TASK_KEYS[@]}"; do
        if [ "${TASK_KEYS[$i]}" = "$key" ]; then
            echo "${TASK_DEFS[$i]}"
            return 0
        fi
    done
    return 1
}

[ $# -lt 1 ] && usage

MODEL=$1
TASK_ARG=${2:-all}
CUDA_ID=${3:-0}
EXEC=${4:-8}

if [ "$TASK_ARG" = "all" ]; then
    TASK_LIST=("${TASK_KEYS[@]}")
else
    IFS=',' read -r -a TASK_LIST <<< "$TASK_ARG"
    for key in "${TASK_LIST[@]}"; do
        if ! task_def "$key" > /dev/null; then
            echo "Unknown task: $key"
            usage
        fi
    done
fi

export CUDA_VISIBLE_DEVICES=$CUDA_ID

run_task() {
    local key=$1
    local def
    def=$(task_def "$key")
    IFS='|' read -r task task_config instruction out_suffix <<< "$def"

    python script/eval_gr00t.py \
        --model-path /mnt/data/sftp/data/vla/vr_checkpoints/$MODEL \
        --task "$task" \
        --task-config "$task_config" \
        --instruction "$instruction" \
        --num-trials 100 \
        --seed 0 \
        --n-action-steps $EXEC --output-dir /mnt/data/sftp/data/locht1/eval_robotwin/${MODEL}_exec${EXEC}/$out_suffix \
        --device cuda:0 --gr00t-path /mnt/data/sftp/data/locht1/miniconda/envs/ego_gr00t/lib/python3.10/site-packages
}

for key in "${TASK_LIST[@]}"; do
    run_task "$key"
done
