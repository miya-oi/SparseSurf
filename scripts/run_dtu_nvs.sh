#!/usr/bin/env bash
set -e

if [ $# -ne 3 ]; then
    echo "Usage: bash scripts/run_dtu_nvs.sh <gpu_id> <scan> <workspace>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
scan=$2
workspace=$3

if [ -z "${FOUNDATION_STEREO_CKPT:-}" ]; then
    echo "Please export FOUNDATION_STEREO_CKPT before running DTU NVS."
    exit 1
fi

iters=${DTU_NVS_ITERATIONS:-7000}

mkdir -p "$workspace"
export PYTHONPATH=.:${PYTHONPATH}

python train.py \
--source_path data/DTU/submission_data/$scan -m $workspace \
--eval -r 4 --n_views 3 \
--iterations $iters \
--total_virtual_num 240 \
--foundation_stereo_ckpt $FOUNDATION_STEREO_CKPT

bash ./scripts/copy_mask_dtu.sh $workspace $scan

python render.py \
--source_path data/DTU/submission_data/$scan -m $workspace \
--iteration $iters --skip_train

python metrics_dtu.py -m $workspace
