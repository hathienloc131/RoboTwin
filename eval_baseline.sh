#!/usr/bin/env bash
set -euo pipefail

# task_key -> "task|task-config|instruction|output-dir-suffix"
# Parallel arrays instead of `declare -A` for portability (bash 3.2, e.g. macOS
# stock bash, has no associative arrays).
TASK_KEYS=(
    adjust_bottle
    blocks_ranking_rgb
    grab_roller
    move_stapler_pad
    open_laptop
    place_mouse_pad
    place_object_basket
)
TASK_DEFS=(
    "adjust_bottle|demo_clean|Pick up the bottle from the table head-up with the correct arm.|adjust_bottle_4"
    "blocks_ranking_rgb|demo_clean|Place the red block, green block, and blue block in order from left to right, placing them in a row.|block_ranking_rgb_4"
    "grab_roller|demo_clean|Use both arms to grab the roller on the table.|grab_roller_4"
    "move_stapler_pad|demo_clean|Use the appropriate arm to move the stapler onto the colored mat.|move_stapler_pub_4"
    "open_laptop|demo_clean|Use the appropriate arm to open the laptop.|open_laptop_4"
    "place_mouse_pad|demo_clean|Pick up the mouse and place it on the colored mat using the appropriate arm.|place_mouse_pad_4"
    "place_object_basket|demo_clean|Use one arm to place the object into the basket, then use the other arm to grab the basket and move it slightly away.|place_object_basket_4"
)

usage() {
    echo "Usage: $0 <MODEL> [TASK1,TASK2,...|all] [CUDA_ID] [EXEC] [DENOISING_STEPS]"
    echo ""
    echo "  MODEL             checkpoint dir name under /home/locht1/vr_checkpoint/ (required)"
    echo "  TASK              comma-separated list from: ${TASK_KEYS[*]}, or 'all' (default: all)"
    echo "  CUDA_ID           CUDA device index (default: 0)"
    echo "  EXEC              n-action-steps (default: 4)"
    echo "  DENOISING_STEPS   override the checkpoint's default flow-matching denoising steps (default: unset, use checkpoint default)"
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
EXEC=${4:-4}
DENOISING_STEPS=${5:-}

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
    cd /home/lmaotan/workspace/Isaac-GR00T
    git checkout gr00t_baseline_locht1
    cd /home/lmaotan/Documents/locht1/RoboTwin

    local denoise_args=()
    local out_suffix_full=$out_suffix
    if [ -n "$DENOISING_STEPS" ]; then
        denoise_args=(--denoising-steps "$DENOISING_STEPS")
        out_suffix_full="${out_suffix}_denoise${DENOISING_STEPS}"
    fi

    python script/eval_gr00t_endpose.py \
        --model-path /home/lmaotan/vr_checkpoints/$MODEL \
        --task "$task" \
        --task-config "$task_config" \
        --instruction "$instruction" \
        --num-trials 50 \
        --seed 0 \
        --n-action-steps $EXEC --output-dir ~/eval_robotwin/${MODEL}_exec${EXEC}/$out_suffix_full \
        --device cuda:0 --gr00t-path /home/lmaotan/miniconda3/envs/gr00t/lib/python3.10/site-packages \
        ${denoise_args[@]+"${denoise_args[@]}"}
}

for key in "${TASK_LIST[@]}"; do
    run_task "$key"
done
