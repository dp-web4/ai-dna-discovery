#!/bin/bash
# Consciousness LoRA Training Script

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="./model_cache"

# Training command
python -m transformers.trainer \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name consciousness_train.jsonl \
    --output_dir ./consciousness_lora_v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --learning_rate 2e-4 \
    --fp16 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --save_total_limit 3 \
    --report_to none

echo "Training complete! Adapter saved to ./consciousness_lora_v1"
