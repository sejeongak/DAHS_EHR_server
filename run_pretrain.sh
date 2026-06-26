#!/bin/bash

# Pre-training with MLM objective
torchrun --nproc_per_node=2 /home/DAHS3/DAHS_EHR/pretrain_normal.py \
  --num_hidden_layers 6 \
  --num_attention_heads 8 \
  --intermediate_size 1024 \
  --embedding_size 512 \
  --batch_size 16 \
  --acc 4 \
  --exp_name best_pretrain_model_new_pretrain_mlm_3030305_small_512_typewisevaluepred11_renewal2_continuousvalueembedding_learnablepe \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --epochs 100 \
  --dropout_prob 0.3 \
  --mask_mode mlm \
  --mask_ratio 0.3 0.3 0.3 \
  --value_mask_ratio 0.05 \
  --clip_interval 20 \
  --num_workers 0 \
  --value_embedding_type continuous \