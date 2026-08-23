#!/usr/bin/env bash
set -euo pipefail

# Runs eval_ego_gr00t.sh (adjust_bottle only) for all 5 OT-ablation checkpoints
# under /home/lmaotan/vr_checkpoints/, split across 2 GPUs running in parallel
# (~half the checkpoints sequentially on each GPU). Each GPU's output goes to
# its own log file since both run at once.
MODELS=(
    #gr00t_n17_ot_ablation_adjustbottle_a_robot_only_ep25_adjust_bottle_bs64
    gr00t_n17_ot_ablation_adjustbottle_b_propmix_ep25_adjust_bottle_bs64
    gr00t_n17_ot_ablation_adjustbottle_c_propmix_ep25_adjust_bottle_bs64
    #gr00t_n17_ot_ablation_adjustbottle_d_propmix_debias_ep25_adjust_bottle_bs64
    gr00t_n17_ot_ablation_adjustbottle_e_propmix_debias_ep25_adjust_bottle_bs64
)

LOG_DIR=~/eval_robotwin/ablation_logs

usage() {
    echo "Usage: $0 [GPU0] [GPU1] [EXEC]"
    echo ""
    echo "  GPU0   CUDA device index for the first half of checkpoints (default: 0)"
    echo "  GPU1   CUDA device index for the second half of checkpoints (default: GPU0 + 1)"
    echo "  EXEC   n-action-steps (default: 4)"
    echo ""
    echo "  Splits the 5 checkpoints below into 2 groups and runs each group's"
    echo "  checkpoints one after another on its own GPU, both GPUs in parallel:"
    echo "    bash eval_ego_gr00t.sh <MODEL> adjust_bottle \$GPU \$EXEC"
    echo ""
    echo "  Per-GPU output is logged to $LOG_DIR/gpu<ID>.log (tail -f to watch live)."
    for m in "${MODELS[@]}"; do
        echo "    - $m"
    done
    exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
fi

GPU0=${1:-0}
GPU1=${2:-$((GPU0 + 1))}
EXEC=${3:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LOG_DIR"

# First half of MODELS -> GPU0, second half -> GPU1.
mid=$(((${#MODELS[@]} + 1) / 2))
GROUP0=("${MODELS[@]:0:mid}")
GROUP1=("${MODELS[@]:mid}")

run_group() {
    local gpu=$1
    shift
    local model
    for model in "$@"; do
        echo -e "\n\033[36m=== [gpu $gpu] $model ===\033[0m"
        bash "$SCRIPT_DIR/eval_ego_gr00t.sh" "$model" adjust_bottle "$gpu" "$EXEC"
    done
}

echo "GPU $GPU0: ${GROUP0[*]}"
echo "GPU $GPU1: ${GROUP1[*]}"
echo "Logs: $LOG_DIR/gpu${GPU0}.log , $LOG_DIR/gpu${GPU1}.log"

run_group "$GPU0" "${GROUP0[@]}" > "$LOG_DIR/gpu${GPU0}.log" 2>&1 &
pid0=$!
run_group "$GPU1" "${GROUP1[@]}" > "$LOG_DIR/gpu${GPU1}.log" 2>&1 &
pid1=$!

status=0
wait "$pid0" || { echo "GPU $GPU0 group failed (see $LOG_DIR/gpu${GPU0}.log)"; status=1; }
wait "$pid1" || { echo "GPU $GPU1 group failed (see $LOG_DIR/gpu${GPU1}.log)"; status=1; }

exit $status
