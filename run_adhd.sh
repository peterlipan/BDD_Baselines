#!/bin/bash

# Define the models and seeds
models=("BrainNetCNN" "graphtransformer" "bnt" "fbnetgen" "comtf" "braingnn" "braingb" "dpm" "lstm" "transformer" "gru" "meanMLP")
seeds=(1 42 123 456 789)
atlases=("ho" "cc200")

# Iterate over each model and seed
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        for atlas in "${atlases[@]}"; do
            python3 main_adhd.py --debug --model "$model" --seed "$seed" --atlas "$atlas"
        done
    done
done