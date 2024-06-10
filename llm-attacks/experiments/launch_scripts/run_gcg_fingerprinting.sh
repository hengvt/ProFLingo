#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for i in {0..49}; do
    python -u ../main.py \
        --config="../configs/transfer_llama2.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/questions.csv" \
        --config.result_prefix="../results/${i}" \
        --config.progressive_goals=False \
        --config.stop_on_success=False \
        --config.num_train_models=2 \
        --config.allow_non_ascii=False \
        --config.n_train_data=1 \
        --config.n_test_data=0 \
        --config.n_steps=256 \
        --config.test_steps=1 \
        --config.batch_size=512 \
        --config.data_offset=$i
done