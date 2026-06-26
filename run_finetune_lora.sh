#!/bin/bash

# EXP_NAME="phenotype_mlm3030305epoch30_lr5e-5cfweight4_cosinescheduler_windowENTIRE_attentionpooling_onelayer_gap_fold4_SEED42"
# EXP_NAME="multitask_mlm3030305epoch30_lr5e-5cfweight4_dropout0.3_cosinescheduler_window24_attentionpooling_onelayer_gap_fold0_SEED42_orderdescriptionablation"
EXP_NAME="mimic_final_24_fullshot_SEED42_lora16"
Load_EXP_NAME="multitask_mlm3030305epoch30_lr5e-5cfweight4_dropout0.3_cosinescheduler_window24_attentionpooling_onelayer_gap_fold0_SEED42"
# EXP_NAME="TEST"
# PRETRAIN_PATH="best_pretrain_model_new_pretrain_mlm_3030305_214_small_524_typewisevaluepred11_renewal_learnablepe.pth"
PRETRAIN_PATH="pretrain_model_best_pretrain_model_new_pretrain_mlm_3030305_small_512_typewisevaluepred11_renewal2_learnablepe_epoch30.pth"
# PRETRAIN_PATH="best_pretrain_model_new_pretrain_mlm_30303015_214_small_512_typewisevaluepred11_renewal_learnablepe.pth"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 finetune_normal_lora.py \
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
  --load_exp_name "$Load_EXP_NAME" \
  --learning_rate 1e-4 \
  --dropout_prob 0.3 \
  --classifier_weight 2 \
  --loss binary_cross_entropy \
  --clip_interval 20 \
  --patience 7 \
  --pretrain \
  --pretrain_path pretrain_model_best_pretrain_model_new_pretrain_mlm_3030305_small_512_typewisevaluepred11_renewal2_learnablepe_epoch30.pth \
  --classifier_dropout 0.3 \
  --num_binary_tasks 11 \
  --num_sofa_tasks 6 \
  --num_multiclass_labels 4 \
  --value_mask_ratio 0.05 \
  --ratio 100 \
  --index 0 \
  --selected_data final \
  --window 24 \
  --use_lora