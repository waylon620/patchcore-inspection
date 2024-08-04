#!/bin/bash

# Set the path to the dataset
datapath="../datasets/MVTecAD"

# List of datasets
datasets=('bottle')

# Generate dataset flags
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "-d ${dataset}"; done))

# Combine dataset flags into a single string
dataset_flags_str="${dataset_flags[@]}"

# Run the patchcore script with the specified parameters
env PYTHONPATH=src python bin/run_patchcore.py \
--gpu 0 --seed 123 \
--save_segmentation_images \
--save_patchcore_model \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
--log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
--anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset \
dataset --resize 256 --imagesize 224 -d breakfast_box ${dataset_flags_str} mvtec $datapath
