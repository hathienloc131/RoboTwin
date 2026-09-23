#!/bin/bash
# Roll out all 50 tasks for each embodiment: aloha-agilex -> ARX-X5 -> franka-panda -> ur5-wsg.
# Output: data/<embodiment>/<task>/ (rgb + depth + mesh/actor segmentation, config task_config/demo_clean_seg_depth.yml)
# Usage: bash collect_all_tasks_cross_embodiment.sh <gpu_id>
# Safe to re-run after a crash / kill: it resumes where it stopped.

gpu_id=${1:-0}

EXTRA_ARGS="--embodiments aloha-agilex ARX-X5 franka-panda ur5-wsg ${EXTRA_ARGS}" \
bash collect_data_cross_embodiment.sh ${gpu_id}
