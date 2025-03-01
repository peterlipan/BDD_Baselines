#!/bin/bash

# Define the models and seeds
models=("brainnetcnn" "graphtransformer" "bnt" "fbnetgen" "comtf" "braingnn" "braingb" "dpm" "lstm" "transformer" "gru")
seeds=(1 42 123 456 789 1000)

# Iterate over each model and seed
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 main.py --debug --model "$model" --seed "$seed"
    done
done