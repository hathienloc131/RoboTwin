#!/bin/bash
# Resumable cross-embodiment collection (rgb + depth + mesh/actor segmentation).
# Output: data/<embodiment>/<task>/
# Usage: bash collect_data_cross_embodiment.sh <gpu_id> [task1 task2 ... | tasks.txt]   (no task = all tasks in envs/)
# Extra options go through env var, e.g.
#   EXTRA_ARGS="--embodiments aloha-agilex ur5-wsg -n 20 --episode_timeout 1200" bash collect_data_cross_embodiment.sh 0 adjust_bottle
# Re-run the same command after a crash: finished (task, embodiment) pairs and episodes are skipped.

gpu_id=${1}
shift

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

python script/collect_cross_embodiment.py --tasks "$@" ${EXTRA_ARGS}
