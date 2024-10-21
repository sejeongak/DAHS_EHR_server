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

from models.embedding import EHREmbedding



class LongformerPretrainNormal(LongformerPreTrainedModel):
    def __init__(
        self,
        vocab_size,
        itemid_size,
        max_position_embeddings,
        unit_size,
        continuous_size,
        task_size,
        max_age,
        gender_size,
        embedding_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        learning_rate,
        dropout_prob,
        # layer_norm_eps,
        gpu_mixed_precision=True,
    ):
        self.vocab_size = vocab_size
        self.itemid_size = itemid_size
        self.max_position_embeddings = max_position_embeddings
        self.unit_size = unit_size
        self.continuous_size = continuous_size
        self.task_size = task_size
        self.max_age = max_age
        self.gender_size = gender_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        # # self.layer_norm_eps = layer_norm_eps
        self.gpu_mixed_precision = gpu_mixed_precision
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
        
        

        self.embeddings = EHREmbedding(
            config=self.config,
            itemid_size=itemid_size,
            unit_size=unit_size,
            max_age=max_age,
            max_len=max_position_embeddings,
            continuous_size=continuous_size,
            gender_size=gender_size,
            task_size=task_size,
            use_itemid=True,
            inputs_embeds=None,
        )
        
        self.model = LongformerForMaskedLM(config=self.config)
        
        self.post_init()
        
    def _init_weights(self, module: torch.nn.Module) -> None:
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def post_init(self):
        self.model.apply(self._init_weights)
    
    
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        continuous_ids,
        position_ids,
        token_type_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):   
        combined_embed = self.embeddings(
            input_ids = input_ids,
            value_ids = value_ids,
            unit_ids = unit_ids,
            time_ids = time_ids,
            continuous_ids = continuous_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            age_ids = age_ids,
            gender_ids = gender_ids,
            task_ids = task_token
        )
        
        if torch.isnan(combined_embed).any():
            print("embedding nan ok")
        
        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            
        return self.model(
            inputs_embeds=combined_embed,
            # position_ids=position_ids,
            # token_type_ids=type_ids,
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
    
class LongformerFinetune(LongformerPretrainNormal):
    
    def __init__(
        self,
        pretrained_model: LongformerPretrainNormal,
        problem_type: str = "single_label_classification",
        num_labels: int = 2,
        learning_rate: float = 1e-6,
        classifier_dropout: float = 0.1,
        use_lora: bool = True,
    ):
        super().__init__(
            vocab_size=pretrained_model.vocab_size,
            itemid_size=pretrained_model.itemid_size,
            max_position_embeddings=pretrained_model.max_position_embeddings,
            unit_size=pretrained_model.unit_size,
            continuous_size=pretrained_model.continuous_size,
            task_size=pretrained_model.task_size,
            max_age=pretrained_model.max_age,
            gender_size=pretrained_model.gender_size,
            embedding_size=pretrained_model.embedding_size,
            num_hidden_layers=pretrained_model.num_hidden_layers,
            num_attention_heads=pretrained_model.num_attention_heads,
            intermediate_size=pretrained_model.intermediate_size,
            learning_rate=learning_rate,   
            dropout_prob=pretrained_model.dropout_prob,
            gpu_mixed_precision=pretrained_model.gpu_mixed_precision
        )
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.test_outputs = []
        self.config = pretrained_model.config
        self.config.num_labels = self.num_labels
        self.config.classifier_dropout = self.classifier_dropout
        self.config.problem_type = problem_type
        self.config.learning_rate = self.learning_rate
        
        
        self.embeddings = pretrained_model.embeddings
        if use_lora:
            self.model = pretrained_model.model.model.longformer # LoRA
        else:
            self.model = pretrained_model.model.longformer 
        
        self.classifier = LongformerClassificationHead(self.config)
        
        

        self.classifier.apply(self._init_weights)
        
    def _init_weights(self, module: torch.nn.Module) -> None:
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    # def post_init(self):
    #     self.classifier.apply(self._init_weights)
    def pretrained_parameters(self):
        # Return parameters from the pretrained model only (excluding LoRA and classifier)
        return [param for name, param in self.named_parameters() if 'lora' not in name and 'classifier' not in name]

    def lora_parameters(self):
        # Return parameters related to LoRA
        return [param for name, param in self.named_parameters() if 'lora' in name]

    def classifier_parameters(self):
        # Return parameters of the classifier
        return [param for name, param in self.named_parameters() if 'classifier' in name]
        
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        continuous_ids,
        position_ids,
        token_type_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        
        combined_embed = self.embeddings(
            input_ids = input_ids,
            value_ids = value_ids,
            unit_ids = unit_ids,
            time_ids = time_ids,
            continuous_ids = continuous_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            age_ids = age_ids,
            gender_ids = gender_ids,
            task_ids = task_token
        )
        
        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            
        outputs = self.model(
            inputs_embeds=combined_embed,
            # position_ids=position_ids,
            # token_type_ids=type_ids,
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask,
            # labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
                
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits
        
        