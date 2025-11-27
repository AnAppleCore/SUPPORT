#!/bin/bash

PYTHON="/home/yanhongwei/miniconda3/envs/ssr/bin/python"

EXP_NAME="our_noisy"
NOISY_DATA="/data/yanhongwei/SIM/noisy/train"

GPU="0"
N_EPOCHS=21
CHECKPOINT_INTERVAL=5

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m src.train \
    --exp_name $EXP_NAME \
    --is_folder \
    --noisy_data $NOISY_DATA \
    --n_epochs $N_EPOCHS \
    --checkpoint_interval $CHECKPOINT_INTERVAL
