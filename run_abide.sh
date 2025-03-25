#!/bin/bash

# Define the models and seeds
models=("braingb" "gru" "transformer" "meanMLP" "bolT" "DICE" "Glacier" "ComTF")
seeds=(123 456 789)
atlases=("cc400" "cc200" "aal")

# Iterate over each model and seed
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        for atlas in "${atlases[@]}"; do
            python3 main_abide.py --debug --model "$model" --seed "$seed" --atlas "$atlas"
        done
    done
done