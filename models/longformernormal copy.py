from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts
from transformers import LongformerConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.longformer.modeling_longformer import (
    LongformerForMaskedLM,
    LongformerClassificationHead,
    LongformerModel,
    LongformerPreTrainedModel,
    LongformerSelfAttention,
    LongformerForSequenceClassification,
    LongformerLMHead,
    LongformerMaskedLMOutput,
    LongformerLayer,
    LongformerBaseModelOutput,
    LongformerBaseModelOutputWithPooling,
    LongformerSequenceClassifierOutput,
)
from models.embedding import TaskEmbedding

from copy import deepcopy
import torch.nn as nn

from models.embedding import EHREmbedding, EHREmbedding_finetune

import time

class LongformerRegressionHead(nn.Module):
    """Longformer Head for regression tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Regression output: single continuous value per token/sequence
        self.regressor = nn.Linear(config.hidden_size, 1)
    
    def forward(self, features, **kwargs):
        """
        features: [batch_size, seq_length, hidden_size]
        Returns: [batch_size, seq_length, 1]
        """
        x = self.dense(features)
        x = F.gelu(x)  # Activation function
        x = self.layer_norm(x)

        # Regression output
        x = self.regressor(x)

        return x.squeeze(-1)  # [batch_size, seq_length]
    
class Discriminator(nn.Module):
    def __init__(self, embedding_size, hidden_dim=256, dropout_prob=0.1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),  # 첫 번째 선형 변환
            nn.ReLU(),  # 활성화 함수 추가 (비선형성 증가)
            nn.LayerNorm(hidden_dim),  # LayerNorm 추가 (학습 안정화)
            nn.Dropout(dropout_prob),  # Dropout 추가 (과적합 방지)
            nn.Linear(hidden_dim, 1),  # 최종 출력층 (Binary Classification)
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

class LongformerLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        
    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x
    
class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)  # output: scalar
        )

    def forward(self, x):
        return self.value_head(x).squeeze(-1)

    

class LongformerPretrainNormal(LongformerPreTrainedModel):
    def __init__(
        self,
        # idx2label, ############
        # idx2ordername,
        # idx2orderdescription,
        name_size,
        description_size,
        token_type_size,
        vocab_size,
        itemid_size,
        # embedding_tokenizer,
        # embedding_model,
        # embedding_map,
        max_position_embeddings,
        unit_size,
        task_size,
        max_age,
        gender_size,
        embedding_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        learning_rate,
        dropout_prob,
        # mask_mode=None,
        loss_factor,
        use_discriminator,
        use_value_prediction,
        # layer_norm_eps,
        gpu_mixed_precision=True,
        # ablation=None,
    ):
        self.vocab_size = vocab_size
        self.itemid_size = itemid_size
        self.max_position_embeddings = max_position_embeddings
        self.unit_size = unit_size
        self.task_size = task_size
        self.max_age = max_age
        self.gender_size = gender_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        # self.mask_mode = mask_mode
        self.loss_factor = loss_factor
        # # self.layer_norm_eps = layer_norm_eps
        self.gpu_mixed_precision = gpu_mixed_precision
        self.use_discriminator = use_discriminator
        # self.ablation = ablation
        self.train_mlm_precisions = [] 
        self.val_mlm_precisions = []
        
        self.config = LongformerConfig(
            vocab_size = self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.embedding_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout_prob, 
            attention_probs_dropout_prob=self.dropout_prob,
            attention_window=[512] * self.num_hidden_layers,
            # # layer_norm_eps=self.layer_norm_eps
        )
        
        super().__init__(self.config)
        
        
        self.config.vocab_size = self.itemid_size
        
        self.use_value_prediction = use_value_prediction
        
        

        self.embeddings = EHREmbedding(
            config=self.config,
            itemid_size=itemid_size,
            unit_size=unit_size,
            max_age=max_age,
            max_len=max_position_embeddings,
            gender_size=gender_size,
            task_size=task_size,
            # idx2label=idx2label, ###########
            # idx2ordername=idx2ordername,
            # idx2orderdescription=idx2orderdescription,
            name_size=name_size,
            description_size=description_size,
            token_type_size=token_type_size,
            # embedding_tokenizer=embedding_tokenizer,
            # embedding_model=embedding_model,
            # embedding_map=embedding_map,
            use_itemid=True,
            inputs_embeds=None,
            # ablation=ablation,
        )
        
        self.encoder = LongformerModel(config=self.config)
        
        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None
        
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        self.mlm_head = LongformerLMHead(config=self.config)
        
        if self.use_value_prediction:
            self.value_prediction_head = ValuePredictionHead(self.config.hidden_size)
        
        # self.model = LongformerForMaskedLM(config=self.config)
        
        if self.use_discriminator:
            self.discriminator = (
                Discriminator(self.embedding_size, hidden_dim=256, dropout_prob=0.1) if self.use_discriminator else None
            )
            self.criterion_discriminator = nn.BCEWithLogitsLoss(reduction='none') if self.use_discriminator else None
        else:
            self.discrminator = None
            self.criterion_discriminator = None
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -100)
        self.loss_value = nn.MSELoss(reduction='none')

        
        
        self.post_init()
        
    def _init_weights(self, module: torch.nn.Module) -> None:
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Xavier 초기화 사용 (ReLU 계열 활성화 함수와 호환)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            # Normal 분포 초기화
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm은 bias=0, weight=1로 설정
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self):
        """ Initialize weights for all modules """
        self.mlm_head.apply(self._init_weights)  # MLM Head 가중치 초기화
        if self.use_discriminator and self.discriminator is not None:
            self.discriminator.apply(self._init_weights)  # Discriminator 초기화

    
    
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        position_ids,
        token_type_ids,
        ordername_ids,
        orderdescription_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        value_labels=None,
        discriminator_labels=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    ):   
        combined_embed = self.embeddings(
            input_ids = input_ids,
            value_ids = value_ids,
            unit_ids = unit_ids,
            time_ids = time_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            ordername_ids = ordername_ids,
            orderdescription_ids = orderdescription_ids,
            age_ids = age_ids,
            gender_ids = gender_ids,
            task_ids = task_token
        )

        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)

        
        outputs = self.encoder(
            input_ids=None,
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = outputs["last_hidden_state"]
        
        mlm_output = self.mlm_head(last_hidden_state)
        
        if self.use_value_prediction:
            value_output = self.value_prediction_head(last_hidden_state)

        loss_mlm = None
        loss_value = None
        total_loss = None

        if labels is not None:
            logits = mlm_output[:, 3:, :]               # [B, L, V]
            target = labels[:, 3:]                      # [B, L]
            token_types = token_type_ids                # [B, L] 또는 이미 [B, L - 3]이면 그대로

            mlm_mask = (target != -100).float()         # 1.0 for valid tokens, 0.0 for padding

            # gather-safe index: replace -100 with 0 temporarily (won't be used in loss due to mask)
            safe_target = target.clone()
            safe_target[target == -100] = 0

            weight_map = torch.tensor([2.0, 1.0, 4.0], device=logits.device)
            # weight_map = torch.tensor([2.0, 1.0, 3.0], device=logits.device)
            # weight_map = torch.tensor([1.0, 1.0, 1.0], device=logits.device)
            
            type_weights = weight_map[token_types]      # [B, L]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss_per_token = -torch.gather(log_probs, dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)

            weighted_loss = (loss_per_token * type_weights * mlm_mask).sum() / (mlm_mask * type_weights).sum()
            loss_mlm = weighted_loss
            
            
        if value_labels is not None and self.use_value_prediction:
            mask = (value_labels != -100).float()
            pred_values = value_output[:, 3:]
            true_values = value_labels[:, 3:].float()
            raw_mse = self.loss_value(pred_values, true_values)
            if mask.sum() > 0:
                loss_value = (raw_mse * mask[:, 3:]).sum() / mask[:, 3:].sum()

            
        total_loss = 0
        if loss_mlm is not None:
            total_loss += loss_mlm
        if loss_value is not None and self.use_value_prediction:
            total_loss += loss_value * 0.001
        else:
            loss_value = torch.tensor(0.0)
            
        # print(f"total time: {(time.time() - start_time):.4f}")
        return {
            "loss": total_loss,
            "mlm_loss": loss_mlm,
            "value_pred_loss": loss_value,
            # "discriminator_loss": loss_discriminator,
            "mlm_logits": mlm_output,
            # "value_logits": value_output,
            # "discriminator_logits": discriminator_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


    
# class LongformerFinetune(LongformerPretrainNormal):
    
#     def __init__(
#         self,
#         pretrained_model: LongformerPretrainNormal,
#         idx2label,
#         problem_type: str = "single_label_classification",
#         num_labels: int = 2,
#         learning_rate: float = 1e-6,
#         classifier_dropout: float = 0.1,
#         use_lora: bool = True,
#         freeze: bool = False,
#     ):
#         super().__init__(
#             idx2label=idx2label,
#             name_size=pretrained_model.embeddings.name_size,
#             description_size=pretrained_model.embeddings.description_size,
#             token_type_size=pretrained_model.embeddings.token_type_size,
#             vocab_size=pretrained_model.vocab_size,
#             itemid_size=pretrained_model.itemid_size,
#             max_position_embeddings=pretrained_model.max_position_embeddings,
#             unit_size=pretrained_model.unit_size,
#             task_size=pretrained_model.task_size,
#             max_age=pretrained_model.max_age,
#             gender_size=pretrained_model.gender_size,
#             embedding_size=pretrained_model.embedding_size,
#             num_hidden_layers=pretrained_model.num_hidden_layers,
#             num_attention_heads=pretrained_model.num_attention_heads,
#             intermediate_size=pretrained_model.intermediate_size,
#             learning_rate=learning_rate,   
#             dropout_prob=pretrained_model.dropout_prob,
#             loss_factor=pretrained_model.loss_factor,
#             use_discriminator=pretrained_model.use_discriminator,
#             gpu_mixed_precision=pretrained_model.gpu_mixed_precision
#         )
#         self.num_labels = num_labels
#         self.learning_rate = learning_rate
#         self.classifier_dropout = classifier_dropout
#         self.test_outputs = []
#         self.config = pretrained_model.config
#         self.config.num_labels = self.num_labels
#         self.config.classifier_dropout = self.classifier_dropout
#         self.config.problem_type = problem_type
#         self.config.learning_rate = self.learning_rate
#         self.freeze = freeze
        
#         self.embeddings = pretrained_model.embeddings
#         if use_lora:
#             self.model = pretrained_model.model.model.longformer # LoRA
#         else:
#             self.model = pretrained_model.model.longformer 
        
#         self.classifier = LongformerClassificationHead(self.config)
        
        

#         self.classifier.apply(self._init_weights)
        
#         if self.freeze:
#             self._freeze_pretrained_weights()
#             print("pretrained model freeze!")
        
#     def _init_weights(self, module: torch.nn.Module) -> None:
#         """ Initialize the weights """
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
            
#     def _freeze_pretrained_weights(self):
#         """ Freeze all pretrained weights except for the classifier """
#         for name, param in self.named_parameters():
#             if "classifier" not in name:  # Freeze everything except the classifier
#                 param.requires_grad = False
    

#     # def post_init(self):
#     #     self.classifier.apply(self._init_weights)
#     def pretrained_parameters(self):
#         # Return parameters from the pretrained model only (excluding LoRA and classifier)
#         return [param for name, param in self.named_parameters() if 'lora' not in name and 'classifier' not in name]

#     def lora_parameters(self):
#         # Return parameters related to LoRA
#         return [param for name, param in self.named_parameters() if 'lora' in name]

#     def classifier_parameters(self):
#         # Return parameters of the classifier
#         return [param for name, param in self.named_parameters() if 'classifier' in name]
        
#     def forward(
#         self,
#         input_ids,
#         value_ids,
#         unit_ids,
#         time_ids,
#         position_ids,
#         token_type_ids,
#         ordername_ids,
#         orderdescription_ids,      
#         age_ids,
#         gender_ids,
#         task_token,
#         attention_mask=None,
#         global_attention_mask=None,
#         # labels=None,
#         output_attentions=False,
#         output_hidden_states=False,
#         return_dict=True,
#         # criterion=None,
#     ):
        
#         combined_embed = self.embeddings(
#             input_ids = input_ids,
#             value_ids = value_ids,
#             unit_ids = unit_ids,
#             time_ids = time_ids,
#             position_ids = position_ids,
#             token_type_ids = token_type_ids,
#             ordername_ids = ordername_ids,
#             orderdescription_ids = orderdescription_ids,
#             age_ids = age_ids,
#             gender_ids = gender_ids,
#             task_ids = task_token
#         )
        
#         if global_attention_mask is None:
#             global_attention = torch.zeros_like(attention_mask)
#             global_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
#             global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         else:
#             attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
#             attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            
#         outputs = self.model(
#             inputs_embeds=combined_embed,
#             # position_ids=position_ids,
#             # token_type_ids=type_ids,
#             attention_mask=attention_mask, 
#             global_attention_mask=global_attention_mask,
#             # labels=labels,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         sequence_output = outputs[0]
#         logits = self.classifier(sequence_output)
        
        
#         # loss=None
#         # if criterion:
#         #     # loss = criterion(logits, labels.float())
#         #     if self.num_labels > 1:
#         #         loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
#         #     else:
#         #         loss = criterion(logits, labels.float())
            
            
                
#         # loss_fct = nn.CrossEntropyLoss()
#         # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         return {
#                 # "loss": loss,
#                 "logits": logits,
#                 # "sequence_output": sequence_output,
#                 }


        
class Classifier(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.BatchNorm = nn.BatchNorm1d(config.hidden_size, affine=True, eps=1e-5)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.dense.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
    
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any(): 
            print("?? [ERROR] input NaN/Inf", x)
        x = self.BatchNorm(x)  
        x = F.leaky_relu(self.dense(x), negative_slope=0.01)  
        x = self.dropout(x)
        logits = self.out_proj(x)

        if torch.isnan(logits).any() or torch.isinf(logits).any(): 
            print("?? [ERROR] output NaN/Inf", logits)

        return logits


        
        

class LongformerFinetune(nn.Module):
    def __init__(self, 
                 pretrained_model, 
                 num_labels, 
                 classifier_dropout=0.1, 
                 freeze_pretrained=True, 
                 freeze_layers=0):
        super(LongformerFinetune, self).__init__()

        # 사전 학습된 모델 (pretrained Longformer)
        self.embedding = pretrained_model.embeddings
        self.encoder = pretrained_model.encoder
        self.num_labels = num_labels
        
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None
        
        self.freeze_pretrained = freeze_pretrained

        if self.freeze_pretrained:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            if freeze_layers > 0:
                for name, param in self.encoder.named_parameters():
                    for i in range(freeze_layers):
                        if f"encoder.layer.{i}." in name:
                            param.requires_grad = False
        
        self.classifier = Classifier(pretrained_model.config)
        
                
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        position_ids,
        token_type_ids,
        ordername_ids,
        orderdescription_ids,      
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        # labels=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        # criterion=None,
    ):
        combined_embed = self.embedding(
            input_ids=input_ids,
            value_ids=value_ids,
            unit_ids=unit_ids,
            time_ids=time_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            ordername_ids=ordername_ids,
            orderdescription_ids=orderdescription_ids,
            age_ids=age_ids,
            gender_ids=gender_ids,
            task_ids=task_token
        )
        
        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
        
        # outputs = self.backbone(
        #     input_ids=input_ids,
        #     value_ids=value_ids,
        #     unit_ids=unit_ids,
        #     time_ids=time_ids,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     ordername_ids=ordername_ids,
        #     orderdescription_ids=orderdescription_ids,
        #     age_ids=age_ids,
        #     gender_ids=gender_ids,
        #     task_token=task_token,
        #     attention_mask=attention_mask,
        #     global_attention_mask=global_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        outputs = self.encoder(
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Check if gradients are being computed for Layer 5 parameters
        # for name, param in self.backbone.encoder.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.mean()}")
        #     else:
        #         print(f"No gradient for {name}")
        

        # [CLS] 토큰의 출력만 사용
        last_hidden_state = outputs["last_hidden_state"]
        cls_output = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # print(f"cls_output Min: {cls_output.min()}, Max: {cls_output.max()}")
        
        logits = self.classifier(cls_output) # (batch_size, num_labels)
        
        # print(f"Logits Min: {logits.min()}, Max: {logits.max()}")
        
        return logits
    
class SimpleClassifier(nn.Module):
    def __init__(self, config):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.classifier_dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class MidClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size//2, config.num_labels)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        return self.classifier(x)

    
    
# class MultitaskClassifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         hidden_size = config.hidden_size
#         dropout = config.classifier_dropout
#         num_labels = config.num_labels

#         self.linear = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, num_labels)
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         return self.linear(x)
class OptimizedMultitaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        dropout = config.classifier_dropout

        self.dense1 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size * 2)

        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.out_proj = nn.Linear(hidden_size, num_labels)

        self.activation = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for layer in [self.dense1, self.dense2, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.norm1(x)

        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.norm2(x)

        logits = self.out_proj(x)
        return logits

    
class MultitaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        dropout = config.classifier_dropout
        
        self.dense1 = nn.Linear(hidden_size, hidden_size * 2)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.dense1.weight)  
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)

        nn.init.xavier_uniform_(self.dense2.weight)
        if self.dense2.bias is not None:
            nn.init.zeros_(self.dense2.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)  

    def forward(self, x):
        x = self.dense1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        logits = self.out_proj(x)
        return logits
    
class SimpleMultitaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        dropout = config.classifier_dropout

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(x)
    
class DeepResidualMultitaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        dropout = config.classifier_dropout

        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        self.activation = nn.GELU()  # 또는 F.leaky_relu

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.zeros_(self.dense2.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        residual = x
        x = self.input_norm(x)

        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = x + residual  # Residual connection
        return self.out_proj(x)
    
class Multitask_Residual_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        dropout = config.classifier_dropout

        self.input_norm = nn.LayerNorm(hidden_size)

        self.dense1 = nn.Linear(hidden_size, hidden_size * 2)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.activation = nn.GELU()  # 또는 nn.LeakyReLU()

        self._init_weights()

    def _init_weights(self):
        for layer in [self.dense1, self.dense2, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        residual = x  # for residual connection
        x = self.input_norm(x)

        x = self.dense1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = x + residual  # residual connection
        return self.out_proj(x)

class SharedMultitaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        dropout = config.classifier_dropout

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Binary classification
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (batch, num_tasks, hidden_size)
        logits = self.classifier(x).squeeze(-1)  # (batch, num_tasks)
        return logits




# class MultitaskClassifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         hidden_size = config.hidden_size
#         num_labels = config.num_labels
#         dropout = config.classifier_dropout

#         self.layer1 = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.GELU(),  
#             nn.Dropout(dropout)
#         )

#         self.layer2 = nn.Sequential(
#             nn.LayerNorm(hidden_size * 2),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )

#         self.residual_adapter = nn.Linear(hidden_size, hidden_size)  # for residual scaling
#         self.out_proj = nn.Linear(hidden_size, num_labels)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         residual = self.residual_adapter(x)  # optional: can use identity if dim match
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x + residual  # residual connection
#         logits = self.out_proj(x)
#         return logits

# class onelayer_classifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dropout = nn.Dropout(config.classifier_dropout)
#         self.dense = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, x):
#         x = self.dropout(x)
#         logits = self.dense(x)
#         return logits


    
# class LongformerFinetuneforMultiTask(nn.Module):
#     def __init__(self, 
#                  pretrained_model, 
#                  num_labels, 
#                  num_tasks=10,
#                  classifier_dropout=0.1, 
#                  freeze_pretrained=True, 
#                  freeze_layers=0,
#                  ablation=None,
#                  args=None):
#         super(LongformerFinetuneforMultiTask, self).__init__()
#         self.config = pretrained_model.config

#         # 사전 학습된 모델 (pretrained Longformer)
#         self.embedding = EHREmbedding_finetune(
#             config=self.config,
#             itemid_size=args.itemid_size,
#             unit_size=args.unit_size,
#             max_age=args.max_age,
#             max_len=args.max_position_embeddings,
#             gender_size=args.gender_size,
#             task_size=args.task_size,
#             # idx2label=args.idx2label, ###########
#             # idx2ordername=idx2ordername,
#             # idx2orderdescription=idx2orderdescription,
#             name_size=args.name_size,
#             description_size=args.description_size,
#             token_type_size=args.token_type_size,
#             ablation=ablation,
#             # embedding_tokenizer=embedding_tokenizer,
#             # embedding_model=embedding_model,
#             # embedding_map=embedding_map,
#             use_itemid=True,
#             inputs_embeds=None,
#         )
#         self.encoder = pretrained_model.encoder
#         self.num_labels = num_labels
#         self.num_tasks = num_tasks
        
#         for param in self.encoder.embeddings.parameters():
#             param.requires_grad = False
        
#         if hasattr(self.encoder, "pooler"):
#             self.encoder.pooler = None
    

#         if freeze_pretrained:
#             for name, param in self.embedding.named_parameters():
#                 param.requires_grad = False
#             if freeze_layers > 0:
#                 for i in range(freeze_layers):
#                     for name, param in self.encoder.named_parameters():
#                         if f"encoder.layer.{i}." in name:
#                             param.requires_grad = False
        
#         self.classifiers = nn.ModuleList([
#             onelayer_classifier(pretrained_model.config) for _ in range(self.num_tasks)
#         ])
        
#         multilabel_config = deepcopy(pretrained_model.config)
#         multilabel_config.num_labels = 25
#         multilabel_config.classifier_dropout = classifier_dropout
#         self.multilabel_classifier = onelayer_classifier(multilabel_config)
        
#         # self.task_uncertainties = nn.Parameter(torch.ones(self.num_tasks))
#         # self.phenotype_uncertainties = nn.Parameter(torch.ones(1))

#         # self.task_weights = nn.Parameter(torch.ones(self.num_tasks))
        
                
#     def forward(
#         self,
#         input_ids,
#         value_ids,
#         unit_ids,
#         time_ids,
#         position_ids,
#         token_type_ids,
#         ordername_ids,
#         orderdescription_ids,      
#         age_ids,
#         gender_ids,
#         task_token,
#         attention_mask=None,
#         global_attention_mask=None,
#         # labels=None,
#         output_attentions=False,
#         output_hidden_states=True,
#         return_dict=True,
#         # criterion=None,
#     ):
#         combined_embed = self.embedding(
#             input_ids=input_ids,
#             value_ids=value_ids,
#             unit_ids=unit_ids,
#             time_ids=time_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             ordername_ids=ordername_ids,
#             orderdescription_ids=orderdescription_ids,
#             age_ids=age_ids,
#             gender_ids=gender_ids,
#             task_ids=task_token
#         )
        
#         if global_attention_mask is None:
#             global_attention = torch.zeros_like(attention_mask)
#             global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         else:
#             attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
        
#         # outputs = self.backbone(
#         #     input_ids=input_ids,
#         #     value_ids=value_ids,
#         #     unit_ids=unit_ids,
#         #     time_ids=time_ids,
#         #     position_ids=position_ids,
#         #     token_type_ids=token_type_ids,
#         #     ordername_ids=ordername_ids,
#         #     orderdescription_ids=orderdescription_ids,
#         #     age_ids=age_ids,
#         #     gender_ids=gender_ids,
#         #     task_token=task_token,
#         #     attention_mask=attention_mask,
#         #     global_attention_mask=global_attention_mask,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         #     return_dict=return_dict,
#         # )
#         outputs = self.encoder(
#             inputs_embeds=combined_embed,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         # Check if gradients are being computed for Layer 5 parameters
#         # for name, param in self.backbone.encoder.named_parameters():
#         #     if param.grad is not None:
#         #         print(f"Gradient for {name}: {param.grad.mean()}")
#         #     else:
#         #         print(f"No gradient for {name}")
        

#         # [CLS] 토큰의 출력만 사용
#         last_hidden_state = outputs["last_hidden_state"]
#         attention_mask = attention_mask.float()
#         masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
#         sum_hidden = masked_hidden.sum(dim=1)
#         valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1)
#         valid_token_counts = valid_token_counts.clamp(min=1)
        
#         # cls_output = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
#         cls_output = sum_hidden / valid_token_counts  # (batch_size, hidden_size) 
#         # task_embeds = self.task_embedding_layer(torch.arange(1, self.num_tasks + 1, device=cls_output.device))
#         # task_embeds = task_embeds.unsqueeze(0).expand(cls_output.size(0), -1, -1)  # (batch_size, num_tasks, hidden_size)
#         # cls_output_expanded = cls_output.unsqueeze(1).expand(-1, self.num_tasks, -1)
#         # combined_cls_task = cls_output_expanded + task_embeds
        
        
#         # print(f"cls_output Min: {cls_output.min()}, Max: {cls_output.max()}")
#         # logits = torch.cat([
#         #     self.classifiers[i](combined_cls_task[:, i]) for i in range(self.num_tasks)
#         # ], dim=-1)
#         # logits = self.classifier(cls_output) # (batch_size, num_labels)
#         logits = torch.cat([classifier(cls_output).unsqueeze(-1) for classifier in self.classifiers], dim=-1)
#         # logits = torch.cat([
#         #     self.classifiers[i](last_hidden_state[:, i, :]).unsqueeze(-1) for i in range(self.num_tasks)
#         # ], dim=-1)
        
#         # phenotype_task_embed = self.task_embedding_layer(torch.tensor([self.num_tasks + 1], device=cls_output.device))
#         # cls_output_for_multilabel = cls_output + phenotype_task_embed.squeeze(0)
        
        
#         # multilabel_logits = self.multilabel_classifier(last_hidden_state[:, self.num_tasks, :])
#         multilabel_logits = self.multilabel_classifier(cls_output)
#         # print(f"Logits Min: {logits.min()}, Max: {logits.max()}")
        
#         return {
#             "logits": logits,
#             "multilabel_logits": multilabel_logits,
#             "hidden_states": outputs.hidden_states,
#         }
        

# class Adapter(nn.Module):
#     def __init__(self, hidden_size, bottleneck_size=64):
#         super().__init__()
#         self.down_proj = nn.Linear(hidden_size, bottleneck_size)
#         self.activation = nn.ReLU()
#         self.up_proj = nn.Linear(bottleneck_size, hidden_size)

#     def forward(self, x):
#         return x + self.up_proj(self.activation(self.down_proj(x)))

class onelayer_classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.dense(x)
        return logits
    
    
class two_layer_classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(config.hidden_size // 2, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.act(x)
        logits = self.dense2(x)
        return logits

    
# class Adapter(nn.Module):
#     def __init__(self, hidden_size, bottleneck_size=32):
#         super().__init__()
#         self.down = nn.Linear(hidden_size, bottleneck_size)
#         self.act = nn.GELU()
#         self.up = nn.Linear(bottleneck_size, hidden_size)

#     def forward(self, x):
#         return x + self.up(self.act(self.down(x)))

class Adapter(nn.Module):
    def __init__(self, hidden_size=512, bottleneck_size=32):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.norm = nn.LayerNorm(bottleneck_size)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.scale = nn.Parameter(torch.tensor(1e-3))  

    def forward(self, x):
        z = self.down(x)
        z = self.norm(z)  
        z = self.act(z)
        z = self.up(z)
        return x + self.scale * z

class adapter_classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = Adapter(config.hidden_size, bottleneck_size=32)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.adapter(x)
        x = self.dropout(x)
        return self.classifier(x)
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        # Compute raw attention scores (batch_size, seq_len)
        attn_scores = (last_hidden_state @ self.attention_vector)  # Linear projection

        # Mask padding tokens by large negative number
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)

        # Softmax over seq_len
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len)

        # Weighted sum
        attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = (last_hidden_state * attn_weights).sum(dim=1)  # (batch_size, hidden_size)

        return pooled
    
class ScaledAttentionPooling(nn.Module):
    def __init__(self, hidden_size, temperature=None, attn_dropout=0.1, use_layernorm=True):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.temperature = temperature or hidden_size ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.ln = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: [B, S, H], attention_mask: [B, S] (0/1 또는 bool)
        scores = torch.einsum("bsh,h->bs", last_hidden_state, self.query) / self.temperature
        mask = attention_mask.bool()
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, S, 1]
        weights = self.dropout(weights)
        pooled = (last_hidden_state * weights).sum(dim=1)     # [B, H]
        return self.ln(pooled)


class LongformerFinetuneforMultiTask(nn.Module):
    def __init__(self, pretrained_model, num_labels, num_binary_tasks=10, num_sofa_tasks=5,
                 classifier_dropout=0.1, freeze_pretrained=True,
                 freeze_layers=0, ablation=None, args=None):
        super().__init__()
        self.config = pretrained_model.config
        self.num_labels = num_labels
        self.num_binary_tasks = num_binary_tasks
        self.num_sofa_tasks = num_sofa_tasks
        self.args = args
        
        self.task_idx = getattr(args, 'task_idx', None)

        self.embedding = EHREmbedding_finetune(
            config=self.config,
            itemid_size=args.itemid_size,
            unit_size=args.unit_size,
            max_age=args.max_age,
            max_len=args.max_position_embeddings,
            gender_size=args.gender_size,
            task_size=args.task_size,
            name_size=args.name_size,
            description_size=args.description_size,
            token_type_size=args.token_type_size,
            ablation=ablation,
            use_itemid=True,
            inputs_embeds=None,
            args=args,
        )

        self.encoder = pretrained_model.encoder
        # self.attention_pooling = AttentionPooling(hidden_size=self.config.hidden_size)  

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None

        if freeze_pretrained:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            if freeze_layers > 0:
                for i in range(freeze_layers):
                    for name, param in self.encoder.named_parameters():
                        if f"encoder.layer.{i}." in name:
                            param.requires_grad = False
                            # print(f"Freezing layer {i}: {name}")


            
        # self.adapters = nn.ModuleList([
        #     Adapter(self.config.hidden_size) for _ in range(self.num_tasks)
        # ])
        
   
        self.binary_classifiers = nn.ModuleList([
            onelayer_classifier(pretrained_model.config) for _ in range(self.num_binary_tasks)
        ])
        
        sofa_config = deepcopy(pretrained_model.config)
        sofa_config.num_labels = args.num_multiclass_labels
        sofa_config.classifier_dropout = classifier_dropout
        
        self.sofa_classifiers = nn.ModuleList([
            onelayer_classifier(sofa_config) for _ in range(self.num_sofa_tasks)
        ])
        
        phenotype_config = deepcopy(pretrained_model.config)
        phenotype_config.num_labels = 25
        phenotype_config.classifier_dropout = classifier_dropout
        self.phenotype_classifier = onelayer_classifier(phenotype_config)
        
        # --- task attention pooling module ---
        # self.binary_pools = nn.ModuleList([
        #     ScaledAttentionPooling(self.config.hidden_size, attn_dropout=classifier_dropout)
        #     for _ in range(self.num_binary_tasks)
        # ])
        # self.sofa_pools = nn.ModuleList([
        #     ScaledAttentionPooling(self.config.hidden_size, attn_dropout=classifier_dropout)
        #     for _ in range(self.num_sofa_tasks)
        # ])
        # self.phenotype_pool = ScaledAttentionPooling(self.config.hidden_size, attn_dropout=classifier_dropout)

        self.shared_pool = ScaledAttentionPooling(
                self.config.hidden_size, attn_dropout=classifier_dropout
            )
        
        # if getattr(args, "task", None) == "phenotype":
        #     for i, clf in enumerate(self.binary_classifiers):
        #         for p in clf.parameters():
        #             p.requires_grad = False

        #     for clf in self.sofa_classifiers:
        #         for p in clf.parameters():
        #             p.requires_grad = False
        
        # elif getattr(args, "window", None) == 0:
        #     allowed = [0, 6]
        #     for i, clf in enumerate(self.binary_classifiers):
        #         if i not in allowed:
        #             for p in clf.parameters():
        #                 p.requires_grad = False

        #     for i, clf in enumerate(self.sofa_classifiers):
        #         for p in clf.parameters():
        #             p.requires_grad = False

        # elif getattr(args, "selected_data", None) == "hirid":
        #     allowed = [2, 4, 5, 7, 8, 9, 10]
        #     for i, clf in enumerate(self.binary_classifiers):
        #         if i not in allowed:
        #             for p in clf.parameters():
        #                 p.requires_grad = False

        #     for p in self.phenotype_classifier.parameters():
        #         p.requires_grad = False
                
        if getattr(args, "task", None) == "phenotype":
            # binary/sofa classifier + pool 전부 freeze
            for i, clf in enumerate(self.binary_classifiers):
                for p in clf.parameters():
                    p.requires_grad = False


            for clf in self.sofa_classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
    

        elif getattr(args, "window", None) == 0:
            allowed = [0, 6]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
   

            # SOFA 전부 freeze
            for i, clf in enumerate(self.sofa_classifiers):
                for p in clf.parameters():
                    p.requires_grad = False


            # phenotype 활성 (classifier/풀 둘 다 학습)

        elif getattr(args, "selected_data", None) == "hirid":
            allowed = [2, 4, 5, 7, 8, 9, 10]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False

            # phenotype 전부 freeze
            for p in self.phenotype_classifier.parameters():
                p.requires_grad = False
                
        elif getattr(args, "selected_data", None) == 'P12':
            if args.window == 24:
                allowed = [1, 4, 5, 9]
            elif args.window == 48:
                allowed = [1, 4, 5]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
            for i, clf in enumerate(self.sofa_classifiers):
                for p in clf.parameters():
                    p.requires_grad = False
            for p in self.phenotype_classifier.parameters():
                p.requires_grad = False
                
        elif getattr(args, "selected_data", None) == 'eicu':
            allowed = [i for i in range(2, 11)]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
                        
            for p in self.phenotype_classifier.parameters():
                p.requires_grad = False
  
        
    def forward(self, input_ids, value_ids, unit_ids, time_ids, position_ids,
                token_type_ids, ordername_ids, orderdescription_ids, age_ids,
                gender_ids, task_token, attention_mask=None,
                global_attention_mask=None, output_attentions=False,
                output_hidden_states=True, return_dict=True):

        combined_embed = self.embedding(
            input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids,
            time_ids=time_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, ordername_ids=ordername_ids,
            orderdescription_ids=orderdescription_ids, age_ids=age_ids,
            gender_ids=gender_ids, task_ids=task_token
        )

        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
        
        outputs = self.encoder(
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"]
        bool_mask = attention_mask.bool()

        # --- shared pooling representation ---
        cls_output = self.shared_pool(last_hidden_state, bool_mask)

        binary_logits = sofa_logits = phenotype_logits = None


        if getattr(self.args, "window", None) == 0:
            binary_logits = torch.cat([
                self.binary_classifiers[0](cls_output).unsqueeze(-1),
                self.binary_classifiers[6](cls_output).unsqueeze(-1)
            ], dim=-1)

            sofa_logits = None

            phenotype_logits = self.phenotype_classifier(cls_output)
            
        elif getattr(self.args, "selected_data", None) == "hirid":
            allowed = [2, 4, 5, 7, 8, 9, 10]
            binary_logits = torch.cat([
                self.binary_classifiers[i](cls_output).unsqueeze(-1) for i in allowed
            ], dim=-1)
            sofa_logits = torch.stack([
                clf(cls_output) for clf in self.sofa_classifiers
            ], dim=1)
            phenotype_logits = None  
            
        elif getattr(self.args, "selected_data", None) == 'P12':
            allowed = [1, 4, 5, 9]
            binary_logits = torch.cat([
                self.binary_classifiers[i](cls_output).unsqueeze(-1) for i in allowed
            ], dim=-1)
            sofa_logits = None
            phenotype_logits = None
            
        elif getattr(self.args, "selected_data", None) == 'eicu':
            allowed = [i for i in range(2, 11)]
            binary_logits = torch.cat([
                self.binary_classifiers[i](cls_output).unsqueeze(-1) for i in allowed
            ], dim=-1)
            sofa_logits = torch.stack([
                clf(cls_output) for clf in self.sofa_classifiers
            ], dim=1)
            phenotype_logits = None
        
        else:
            binary_logits = torch.cat([
                clf(cls_output).unsqueeze(-1) for clf in self.binary_classifiers
            ], dim=-1)
            sofa_logits = torch.stack([
                clf(cls_output) for clf in self.sofa_classifiers
            ], dim=1)
            phenotype_logits = self.phenotype_classifier(cls_output)
            

        return {
            "binary_logits": binary_logits,
            "sofa_logits": sofa_logits,
            "phenotype_logits": phenotype_logits,
            "hidden_states": outputs.hidden_states,
            "cls_output": cls_output
        }
        
class LongformerFinetuneforMultiTask_lora(nn.Module):
    def __init__(
        self,
        pretrained_model,
        num_labels,
        num_binary_tasks=10,
        num_sofa_tasks=5,
        classifier_dropout=0.1,
        freeze_pretrained=True,
        freeze_layers=0,
        ablation=None,
        args=None,
    ):
        super().__init__()
        self.config = pretrained_model.config
        self.num_labels = num_labels
        self.num_binary_tasks = num_binary_tasks
        self.num_sofa_tasks = num_sofa_tasks
        self.args = args or type("Args", (), {})()  

        self.task_idx = getattr(self.args, "task_idx", None)

        self.embedding = EHREmbedding_finetune(
            config=self.config,
            itemid_size=self.args.itemid_size,
            unit_size=self.args.unit_size,
            max_age=self.args.max_age,
            max_len=self.args.max_position_embeddings,
            gender_size=self.args.gender_size,
            task_size=self.args.task_size,
            name_size=self.args.name_size,
            description_size=self.args.description_size,
            token_type_size=self.args.token_type_size,
            ablation=ablation,
            use_itemid=True,
            inputs_embeds=None,
        )


        self.encoder = pretrained_model.encoder
        # self.attention_pooling = AttentionPooling(hidden_size=self.config.hidden_size)  

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None

        for param in self.embedding.parameters():
            param.requires_grad = False

        # --------- LoRA (attention/FFN) ---------
        self.use_lora = bool(getattr(self.args, "use_lora", False))
        if self.use_lora:
            target_modules = [
                "query", "value"
            ]
            lora_cfg = LoraConfig(
                r=getattr(self.args, "lora_r", 8),
                lora_alpha=getattr(self.args, "lora_alpha", 16),
                lora_dropout=getattr(self.args, "lora_dropout", 0.05),
                bias="none",
                target_modules=target_modules,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)


        if freeze_pretrained:
            for p in self.embedding.parameters():
                p.requires_grad = False
            
            for name, p in self.encoder.named_parameters():
                # if "lora_" in name:
                #     p.requires_grad = True
                # else:
                #     p.requires_grad = False
                if "embeddings" not in name:
                    p.requires_grad = True
            print("freeze layers:", freeze_layers)
            if freeze_layers > 0:
                for i in range(freeze_layers):
                    prefix = f"encoder.layer.{i}."
                    for name, p in self.encoder.named_parameters():
                        if prefix in name and "lora_" not in name:
                            p.requires_grad = False
                           
        else:
            pass

        # --------- Classifier/Pooling ---------
        self.shared_pool = ScaledAttentionPooling(self.config.hidden_size, attn_dropout=classifier_dropout)

        # Binary heads
        self.binary_classifiers = nn.ModuleList([
            onelayer_classifier(pretrained_model.config) for _ in range(self.num_binary_tasks)
        ])

        # SOFA heads (multiclass)
        sofa_config = deepcopy(pretrained_model.config)
        sofa_config.num_labels = self.args.num_multiclass_labels
        sofa_config.classifier_dropout = classifier_dropout
        self.sofa_classifiers = nn.ModuleList([
            onelayer_classifier(sofa_config) for _ in range(self.num_sofa_tasks)
        ])

        # Phenotype (25-way multi-label or multi-class head)
        phenotype_config = deepcopy(pretrained_model.config)
        phenotype_config.num_labels = 25
        phenotype_config.classifier_dropout = classifier_dropout
        self.phenotype_classifier = onelayer_classifier(phenotype_config)

        # --------- Task gating: classifier ---------
        task_name = getattr(self.args, "task", None)
        if task_name == "phenotype":
            # binary / sofa 헤드 모두 동결
            for clf in self.binary_classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
            for clf in self.sofa_classifiers:
                for p in clf.parameters():
                    p.requires_grad = False

        elif getattr(self.args, "window", None) == 0:
            allowed = [0, 6]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
            for clf in self.sofa_classifiers:
                for p in clf.parameters():
                    p.requires_grad = False

        elif getattr(self.args, "selected_data", None) == "hirid":
            allowed = [2, 4, 5, 7, 8, 9, 10]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
            for p in self.phenotype_classifier.parameters():
                p.requires_grad = False
                
        elif getattr(self.args, "selected_data", None) == 'P12':
            allowed = [1, 4, 5, 9]
            for i, clf in enumerate(self.binary_classifiers):
                if i not in allowed:
                    for p in clf.parameters():
                        p.requires_grad = False
            for clf in self.sofa_classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
            for p in self.phenotype_classifier.parameters():
                p.requires_grad = False

    # ---------- Utility ----------
    def print_trainable_parameters(self, prefix="[LoRA]"):
        total = 0
        trainable = 0
        for n, p in self.named_parameters():
            params = p.numel()
            total += params
            if p.requires_grad:
                trainable += params
        pct = (trainable / total * 100) if total > 0 else 0.0
        print(f"{prefix} trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


    # ---------- Forward ----------
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        position_ids,
        token_type_ids,
        ordername_ids,
        orderdescription_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        combined_embed = self.embedding(
            input_ids=input_ids,
            value_ids=value_ids,
            unit_ids=unit_ids,
            time_ids=time_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            ordername_ids=ordername_ids,
            orderdescription_ids=orderdescription_ids,
            age_ids=age_ids,
            gender_ids=gender_ids,
            task_ids=task_token,
        )

        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            

        outputs = self.encoder(
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"] if return_dict else outputs[0]
        bool_mask = attention_mask.bool()

        # 4) Shared pooling
        cls_output = self.shared_pool(last_hidden_state, bool_mask)

        # 5) Heads & gating
        binary_logits = sofa_logits = phenotype_logits = None

        if getattr(self.args, "window", None) == 0:
            # binary: [0, 6]
            b0 = self.binary_classifiers[0](cls_output).unsqueeze(-1)
            b6 = self.binary_classifiers[6](cls_output).unsqueeze(-1)
            binary_logits = torch.cat([b0, b6], dim=-1)
            # sofa 
            sofa_logits = None
            # phenotype 
            phenotype_logits = self.phenotype_classifier(cls_output)

        elif getattr(self.args, "selected_data", None) == "hirid":
            allowed = [2, 4, 5, 7, 8, 9, 10]
            binary_logits = torch.cat(
                [self.binary_classifiers[i](cls_output).unsqueeze(-1) for i in allowed],
                dim=-1,
            )
            # sofa 
            sofa_logits = torch.stack([clf(cls_output) for clf in self.sofa_classifiers], dim=1)
            # phenotype
            phenotype_logits = None

        else:
            #  binary/sofa/phenotype 
            binary_logits = torch.cat(
                [clf(cls_output).unsqueeze(-1) for clf in self.binary_classifiers],
                dim=-1,
            )
            sofa_logits = torch.stack([clf(cls_output) for clf in self.sofa_classifiers], dim=1)
            phenotype_logits = self.phenotype_classifier(cls_output)

        return {
            "binary_logits": binary_logits,
            "sofa_logits": sofa_logits,
            "phenotype_logits": phenotype_logits,
            "hidden_states": outputs.hidden_states if return_dict else None,
            "cls_output": cls_output,
        }

class LongformerFinetuneforSingleTask(nn.Module):
    def __init__(self, pretrained_model, num_labels=1, classifier_dropout=0.1, 
                 freeze_pretrained=False, freeze_layers=0, ablation=None, args=None):
        super().__init__()
        self.config = pretrained_model.config
        self.num_labels = num_labels
        self.args = args
        
        self.task_idx = getattr(args, 'task_idx', None)

        self.embedding = EHREmbedding_finetune(
            config=self.config,
            itemid_size=args.itemid_size,
            unit_size=args.unit_size,
            max_age=args.max_age,
            max_len=args.max_position_embeddings,
            gender_size=args.gender_size,
            task_size=args.task_size,
            name_size=args.name_size,
            description_size=args.description_size,
            token_type_size=args.token_type_size,
            ablation=ablation,
            use_itemid=True,
            inputs_embeds=None,
        )

        self.encoder = pretrained_model.encoder
        
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None

        if freeze_pretrained:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            if freeze_layers > 0:
                for i in range(freeze_layers):
                    for name, param in self.encoder.named_parameters():
                        if f"encoder.layer.{i}." in name:
                            param.requires_grad = False
                            
        phenotype_config = deepcopy(pretrained_model.config)
        phenotype_config.num_labels = 25
        phenotype_config.classifier_dropout = classifier_dropout
        self.phenotype_classifier = onelayer_classifier(phenotype_config)

    def forward(self, input_ids, value_ids, unit_ids, time_ids, position_ids,
                token_type_ids, ordername_ids, orderdescription_ids, age_ids,
                gender_ids, task_token, attention_mask=None,
                global_attention_mask=None, output_attentions=False,
                output_hidden_states=True, return_dict=True):

        combined_embed = self.embedding(
            input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids,
            time_ids=time_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, ordername_ids=ordername_ids,
            orderdescription_ids=orderdescription_ids, age_ids=age_ids,
            gender_ids=gender_ids, task_ids=task_token
        )

        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)

        outputs = self.encoder(
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"]
        attention_mask = attention_mask.float()
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        cls_output = sum_hidden / valid_token_counts
        
        phenotype_logits = self.phenotype_classifier(cls_output)

        return {
            "binary_logits": None,
            "sofa_logits": None,
            "phenotype_logits": phenotype_logits,
            "hidden_states": outputs.hidden_states,
        }
        
class LongformerFinetuneforPhenotype(nn.Module):
    def __init__(self, pretrained_model, num_labels, classifier_dropout=0.1, freeze_pretrained=True,
                 freeze_layers=0, ablation=None, args=None):
        super().__init__()
        self.config = pretrained_model.config
        self.num_labels = num_labels

        self.embedding = EHREmbedding_finetune(
            config=self.config,
            itemid_size=args.itemid_size,
            unit_size=args.unit_size,
            max_age=args.max_age,
            max_len=args.max_position_embeddings,
            gender_size=args.gender_size,
            task_size=args.task_size,
            name_size=args.name_size,
            description_size=args.description_size,
            token_type_size=args.token_type_size,
            ablation=ablation,
            use_itemid=True,
            inputs_embeds=None,
        )

        self.encoder = pretrained_model.encoder

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None

        if freeze_pretrained:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            if freeze_layers > 0:
                for i in range(freeze_layers):
                    for name, param in self.encoder.named_parameters():
                        if f"encoder.layer.{i}." in name:
                            param.requires_grad = False
                            print(f"Freezing layer {i}: {name}")
                            
                            
        phenotype_config = deepcopy(pretrained_model.config)
        phenotype_config.num_labels = 25
        phenotype_config.classifier_dropout = classifier_dropout
        self.phenotype_classifier = adapter_classifier(phenotype_config)
        
    def forward(self, input_ids, value_ids, unit_ids, time_ids, position_ids,
                token_type_ids, ordername_ids, orderdescription_ids, age_ids,
                gender_ids, task_token, attention_mask=None,
                global_attention_mask=None, output_attentions=False,
                output_hidden_states=True, return_dict=True):
        
        combined_embed = self.embedding(
            input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids,
            time_ids=time_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, ordername_ids=ordername_ids,
            orderdescription_ids=orderdescription_ids, age_ids=age_ids,
            gender_ids=gender_ids, task_ids=task_token
        )

        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)

        outputs = self.encoder(
            inputs_embeds=combined_embed,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"]
        attention_mask = attention_mask.float()
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        cls_output = sum_hidden / valid_token_counts
        
        
        phenotype_logits = self.phenotype_classifier(cls_output)
        
        return {
            "binary_logits": None,
            "sofa_logits": None,
            "phenotype_logits": phenotype_logits,
            "hidden_states": outputs.hidden_states,
        }
        
        
        
        
        
        
        
# class LongformerFinetuneforMultiTask(nn.Module):
#     def __init__(self, pretrained_model, num_labels, num_binary_tasks=10, num_sofa_tasks=5,
#                  classifier_dropout=0.1, freeze_pretrained=True,
#                  freeze_layers=0, ablation=None, args=None):
#         super().__init__()
#         self.config = pretrained_model.config
#         self.num_labels = num_labels
#         self.num_binary_tasks = num_binary_tasks
#         self.num_sofa_tasks = num_sofa_tasks
#         self.num_basetask_tasks = args.num_basetask_tasks
#         self.num_intervention_tasks = args.num_intervention_tasks

#         self.embedding = EHREmbedding_finetune(
#             config=self.config,
#             itemid_size=args.itemid_size,
#             unit_size=args.unit_size,
#             max_age=args.max_age,
#             max_len=args.max_position_embeddings,
#             gender_size=args.gender_size,
#             task_size=args.task_size,
#             name_size=args.name_size,
#             description_size=args.description_size,
#             token_type_size=args.token_type_size,
#             ablation=ablation,
#             use_itemid=True,
#             inputs_embeds=None,
#         )

#         self.encoder = pretrained_model.encoder

#         for param in self.encoder.embeddings.parameters():
#             param.requires_grad = False

#         if hasattr(self.encoder, "pooler"):
#             self.encoder.pooler = None

#         if freeze_pretrained:
#             for name, param in self.embedding.named_parameters():
#                 param.requires_grad = False
#             if freeze_layers > 0:
#                 for i in range(freeze_layers):
#                     for name, param in self.encoder.named_parameters():
#                         if f"encoder.layer.{i}." in name:
#                             param.requires_grad = False

#         # self.adapters = nn.ModuleList([
#         #     Adapter(self.config.hidden_size) for _ in range(self.num_tasks)
#         # ])
        
#         self.task_group = args.task_group
        
#         if self.task_group == "basetask":
#             self.classifier = nn.ModuleList([
#                 onelayer_classifier(pretrained_model.config) for _ in range(self.num_basetask_tasks)
#             ])

        
#         elif self.task_group == "intervention":
#             self.classifier = nn.ModuleList([
#                 onelayer_classifier(pretrained_model.config) for _ in range(self.num_intervention_tasks)
#             ])
            
#         elif self.task_group == "sofa+shock":
            
#             sofa_config = deepcopy(pretrained_model.config)
#             sofa_config.num_labels = args.num_multiclass_labels
#             sofa_config.classifier_dropout = classifier_dropout
            
#             self.sofa_classifier = nn.ModuleList([
#                 onelayer_classifier(sofa_config) for _ in range(self.num_sofa_tasks)
#             ])
            
#             self.shock_classifier = onelayer_classifier(pretrained_model.config)
        
#         elif self.task_group == "phenotype":
#             phenotype_config = deepcopy(pretrained_model.config)
#             phenotype_config.num_labels = 25
#             phenotype_config.classifier_dropout = classifier_dropout
#             self.phenotype_classifier = onelayer_classifier(phenotype_config)
        
#         elif self.task_group == "multitask":
#             self.binary_classifiers = nn.ModuleList([
#                 onelayer_classifier(pretrained_model.config) for _ in range(self.num_binary_tasks)
#             ])
            
#             sofa_config = deepcopy(pretrained_model.config)
#             sofa_config.num_labels = args.num_multiclass_labels
#             sofa_config.classifier_dropout = classifier_dropout
            
#             self.sofa_classifiers = nn.ModuleList([
#                 onelayer_classifier(sofa_config) for _ in range(self.num_sofa_tasks)
#             ])

#             phenotype_config = deepcopy(pretrained_model.config)
#             phenotype_config.num_labels = 25
#             phenotype_config.classifier_dropout = classifier_dropout
#             self.phenotype_classifier = onelayer_classifier(phenotype_config)

#     def forward(self, input_ids, value_ids, unit_ids, time_ids, position_ids,
#                 token_type_ids, ordername_ids, orderdescription_ids, age_ids,
#                 gender_ids, task_token, attention_mask=None,
#                 global_attention_mask=None, output_attentions=False,
#                 output_hidden_states=True, return_dict=True):

#         combined_embed = self.embedding(
#             input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids,
#             time_ids=time_ids, position_ids=position_ids,
#             token_type_ids=token_type_ids, ordername_ids=ordername_ids,
#             orderdescription_ids=orderdescription_ids, age_ids=age_ids,
#             gender_ids=gender_ids, task_ids=task_token
#         )

#         if global_attention_mask is None:
#             global_attention = torch.zeros_like(attention_mask)
#             global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         else:
#             attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)

#         outputs = self.encoder(
#             inputs_embeds=combined_embed,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         last_hidden_state = outputs["last_hidden_state"]
#         attention_mask = attention_mask.float()
#         masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
#         sum_hidden = masked_hidden.sum(dim=1)
#         valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
#         cls_output = sum_hidden / valid_token_counts
        
#         if self.task_group == "basetask":
#             logits = torch.cat([
#                 self.classifier[i](cls_output).unsqueeze(-1)
#                 for i in range(self.num_basetask_tasks)
#             ], dim=-1)
#             return {
#                 "binary_logits": logits,
#                 "hidden_states": outputs.hidden_states,
#             }
#         elif self.task_group == "intervention":
#             logits = torch.cat([
#                 self.classifier[i](cls_output).unsqueeze(-1)
#                 for i in range(self.num_intervention_tasks)
#             ], dim=-1)
#             return {
#                     "binary_logits": logits,
#                     "hidden_states": outputs.hidden_states,
#                     }
            

#         elif self.task_group == "sofa+shock":
#             sofa_logits = torch.stack([
#             clf(cls_output) for clf in self.sofa_classifier
#         ], dim=1)
#             shock_logits = self.shock_classifier(cls_output).unsqueeze(-1)
#             return {
#                 "binary_logits": shock_logits,
#                 "sofa_logits": sofa_logits,
#                 "hidden_states": outputs.hidden_states,
#             }
#         elif self.task_group == "phenotype":
#             logits = self.phenotype_classifier(cls_output).unsqueeze(-1)
#             return {
#                 "phenotype_logits": logits,
#                 "hidden_states": outputs.hidden_states,
#             }
#         elif self.task_group == "multitask":
#             binary_logits = torch.cat([
#                 self.binary_classifiers[i](cls_output).unsqueeze(-1)
#                 for i in range(self.num_binary_tasks)
#             ], dim=-1)

#             sofa_logits = torch.stack([
#                 clf(cls_output) for clf in self.sofa_classifiers
#             ], dim=1)
        
#             phenotype_logits = self.phenotype_classifier(cls_output)

        
#             return {
#                     "binary_logits": binary_logits,
#                     "sofa_logits": sofa_logits,
#                     "phenotype_logits": phenotype_logits,
#                     "hidden_states": outputs.hidden_states,
#                 }        

# class Adapter(nn.Module):
#     def __init__(self, hidden_size, bottleneck_size=128):
#         super().__init__()
#         self.down = nn.Linear(hidden_size, bottleneck_size)
#         self.act = nn.GELU()
#         self.up = nn.Linear(bottleneck_size, hidden_size)

#     def forward(self, x):
#         return x + self.up(self.act(self.down(x)))

# class AdapterClassifier(nn.Module):
#     def __init__(self, adapter, classifier):
#         super().__init__()
#         self.adapter = adapter
#         self.classifier = classifier

#     def forward(self, x):
#         x = self.adapter(x)
#         return self.classifier(x)

# class two_layer_classifier(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dropout = nn.Dropout(config.classifier_dropout)
#         self.dense1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
#         self.act = nn.GELU()
#         self.dense2 = nn.Linear(config.hidden_size // 2, config.num_labels)

#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.dense1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         return self.dense2(x)

# class LongformerFinetuneforMultiTask(nn.Module):
#     def __init__(self, pretrained_model, num_labels, num_binary_tasks=10, num_sofa_tasks=5,
#                  classifier_dropout=0.1, freeze_pretrained=True, freeze_layers=0, ablation=None, args=None):
#         super().__init__()
#         self.config = pretrained_model.config
#         self.num_labels = num_labels
#         self.num_binary_tasks = num_binary_tasks
#         self.num_sofa_tasks = num_sofa_tasks

#         self.embedding = EHREmbedding_finetune(
#             config=self.config,
#             itemid_size=args.itemid_size,
#             unit_size=args.unit_size,
#             max_age=args.max_age,
#             max_len=args.max_position_embeddings,
#             gender_size=args.gender_size,
#             task_size=args.task_size,
#             name_size=args.name_size,
#             description_size=args.description_size,
#             token_type_size=args.token_type_size,
#             ablation=ablation,
#             use_itemid=True,
#             inputs_embeds=None,
#         )

#         self.encoder = pretrained_model.encoder

#         for param in self.encoder.embeddings.parameters():
#             param.requires_grad = False

#         if hasattr(self.encoder, "pooler"):
#             self.encoder.pooler = None

#         if freeze_pretrained:
#             for name, param in self.embedding.named_parameters():
#                 param.requires_grad = False
#             if freeze_layers > 0:
#                 for i in range(freeze_layers):
#                     for name, param in self.encoder.named_parameters():
#                         if f"encoder.layer.{i}." in name:
#                             param.requires_grad = False

#         # 어댑터 그룹 정의
#         self.adapters = nn.ModuleDict({
#             "mortality": Adapter(self.config.hidden_size),
#             "intervention": Adapter(self.config.hidden_size),
#             "shock": Adapter(self.config.hidden_size),
#             "los": Adapter(self.config.hidden_size),
#             "sofa": Adapter(self.config.hidden_size),
#             "phenotype": Adapter(self.config.hidden_size),
#         })

#         mortality_ids = [0, 1, 2, 3] 
#         los_ids = [4, 5]
#         readmit_id = [6]
#         intervention_ids = [7, 8, 9]
#         shock_id = [10]

#         self.binary_classifiers = nn.ModuleList()
#         for i in range(self.num_binary_tasks):
#             if i in mortality_ids + readmit_id:
#                 group = "mortality"
#             elif i in los_ids:
#                 group = "los"
#             elif i in intervention_ids:
#                 group = "intervention"
#             elif i in shock_id:
#                 group = "shock"
#             else:
#                 raise ValueError("Unknown binary task ID")

#             classifier = two_layer_classifier(self.config)
#             self.binary_classifiers.append(AdapterClassifier(self.adapters[group], classifier))

#         # SOFA
#         sofa_config = deepcopy(self.config)
#         sofa_config.num_labels = args.num_multiclass_labels
#         sofa_config.classifier_dropout = classifier_dropout
#         self.sofa_classifiers = nn.ModuleList([
#             AdapterClassifier(self.adapters["sofa"], two_layer_classifier(sofa_config))
#             for _ in range(self.num_sofa_tasks)
#         ])

#         # Phenotype
#         phenotype_config = deepcopy(self.config)
#         phenotype_config.num_labels = 25
#         phenotype_config.classifier_dropout = classifier_dropout
#         self.phenotype_classifier = AdapterClassifier(self.adapters["phenotype"], two_layer_classifier(phenotype_config))

#     def forward(self, input_ids, value_ids, unit_ids, time_ids, position_ids,
#                 token_type_ids, ordername_ids, orderdescription_ids, age_ids,
#                 gender_ids, task_token, attention_mask=None,
#                 global_attention_mask=None, output_attentions=False,
#                 output_hidden_states=True, return_dict=True):

#         combined_embed = self.embedding(
#             input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids,
#             time_ids=time_ids, position_ids=position_ids,
#             token_type_ids=token_type_ids, ordername_ids=ordername_ids,
#             orderdescription_ids=orderdescription_ids, age_ids=age_ids,
#             gender_ids=gender_ids, task_ids=task_token
#         )

#         if global_attention_mask is None:
#             global_attention = torch.zeros_like(attention_mask)
#             global_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         else:
#             attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(combined_embed.device)
#             attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)

#         outputs = self.encoder(
#             inputs_embeds=combined_embed,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         last_hidden_state = outputs["last_hidden_state"]
#         cls_output = last_hidden_state[:, 0, :]  # or use mean pooling if desired

#         binary_logits = torch.cat([
#             clf(cls_output).unsqueeze(-1)
#             for clf in self.binary_classifiers
#         ], dim=-1)

#         sofa_logits = torch.stack([
#             clf(cls_output) for clf in self.sofa_classifiers
#         ], dim=1)

#         phenotype_logits = self.phenotype_classifier(cls_output)

#         return {
#             "binary_logits": binary_logits,
#             "sofa_logits": sofa_logits,
#             "phenotype_logits": phenotype_logits,
#             "hidden_states": outputs.hidden_states,
#         }
