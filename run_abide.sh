#!/bin/bash

# Define the models and seeds
models=("meanMLP" "BNT" "BrainNetCNN" "ComTF" "DPM")
seeds=(123 456 789)
atlases=("late" "intermediate")

# Iterate over each model and seed
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        for atlas in "${atlases[@]}"; do
            python3 main_abide.py --debug --model "$model" --seed "$seed" --fusion "$atlas"
        done
    done
done