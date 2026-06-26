#!/bin/bash

EXP_NAME="inference_finetuning_mimic_48_fold4"
# PRETRAIN_PATH="best_multitask18_binary_crossentropy_pretrain_renewal2_mlm3030305epoch30_214_valuepred11_unfreeze_lr5e-5cfweight1_cosinescheduler_dropout0.30.3_batch16_warmup0.3_window24_multitaskembed_mean_onelayer_gap_fold0_SEED42.pth"
# PRETRAIN_PATH="best_multitask18_binary_crossentropy_pretrain_renewal2_mlm3030305epoch30_214_valuepred11_unfreeze_lr5e-5cfweight1_cosinescheduler_dropout0530.3_batch16_warmup0.3_window24_multitaskembed_mean_onelayer_gap_fold3_fewshot0.1_SEED42.pth"
PRETRAIN_PATH="best_multitask18_binary_crossentropy_pretrain_renewal2_mlm3030305epoch30_214_valuepred11_unfreeze_lr5e-5cfweight1_cosinescheduler_dropout0530.3_batch16_warmup0.3_window24_multitaskembed_mean_onelayer_gap_fold4_SEED42.pth"

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 inference.py \
  --num_hidden_layers 6 \
  --num_attention_heads 8 \
  --intermediate_size 1024 \
  --embedding_size 512 \
  --batch_size 16 \
  --seed 42 \
  --acc 4 \
  --epochs 30 \
  --mode finetune \
  --task multitask \
  --exp_name "$EXP_NAME" \
  --learning_rate 5e-5 \
  --dropout_prob 0.3 \
  --classifier_weight 1 \
  --loss binary_cross_entropy \
  --clip_interval 20 \
  --patience 7 \
  --pretrain \
  --pretrain_path "$PRETRAIN_PATH" \
  --classifier_dropout 0.3 \
  --num_binary_tasks 11 \
  --num_sofa_tasks 6 \
  --value_mask_ratio 0.05 \
  --ratio 100 \
  --index 4 \
  --task_mode multitask \
  --selected_data final \
  --window 48 \
  --inference_mode \
  
