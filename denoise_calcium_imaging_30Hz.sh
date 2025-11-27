#!/bin/bash

PYTHON="/home/yanhongwei/miniconda3/envs/ssr/bin/python"

MODEL_FILE="./results/saved_models/our_noisy/model_20.pth"  # Change to your trained model
INPUT_FOLDER="/data/yanhongwei/SIM/SRDTrans/calcium_imaging_30Hz"
OUTPUT_FOLDER="./results/support_denoised_calcium_imaging_30Hz"

CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.denoise_folder "$MODEL_FILE" "$INPUT_FOLDER" "$OUTPUT_FOLDER"

