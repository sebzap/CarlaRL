#!/usr/bin/env bash
# seeds 44864 26912 94869 88994 24946 34416 73735 65066
GPU=$1
SEED=$2
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=$GPU python -m carla.train_agent --norm_obs --seed $SEED --exp_name gt --hid_context_net 0 --conditioning gt_encode_concat --contextual