from typing import Optional, Tuple, Union
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention, LongformerPreTrainedModel, LongformerLMHead, LongformerMaskedLMOutput, LongformerLayer, LongformerBaseModelOutput, LongformerBaseModelOutputWithPooling, LongformerForMaskedLM, LongformerSequenceClassifierOutput
import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts

class CustomLongformerConfig(LongformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_vocab_size = kwargs.get('on_vocab_size', 2)
        self.use_itemid = kwargs.get('use_itemid', True)
        self.itemid_size = kwargs.get('itemid_size', 50265)
        self.device = kwargs.get('device', 'cuda:0')
        self.num_attention_heads = kwargs.get('num_attention_heads', 1)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 1)
        self.attention_window = kwargs.get('attention_window', [512] * self.num_hidden_layers)
        self.batch_size=2
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 4096)
        self.finetuning_hidden_size1 = kwargs.get('finetuning_hidden_size1', 384)
        self.finetuning_hidden_size2 = kwargs.get('finetuning_hidden_size2', 192)
        self.problem_type = kwargs.get('problem_type', 'single_label_classification')
        self.num_labels = kwargs.get('num_labels', 1)
        # self.intermediate_size = kwargs.get('intermediate_size', 1536)
        
class NewLongformerConfig(LongformerConfig):
    def __init__(self, on_vocab_size=2, use_itemid=True, itemid_size=600, device=None, finetuning_hidden_size1=384, finetuning_hidden_size2=192, **kwargs):
        super().__init__(**kwargs)
        self.on_vocab_size = on_vocab_size
        self.use_itemid = use_itemid
        self.itemid_size = itemid_size
        self.device = device
        self.finetuning_hidden_size1 = finetuning_hidden_size1
        self.finetuning_hidden_size2 = finetuning_hidden_size2



class Time2Vec(nn.Module):
    def __init__(self, kernel_size):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, kernel_size - 1)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.sin(self.periodic(x))
        return torch.cat([v1, v2], dim=-1)



class EHREmbedding(nn.Module):
    def __init__(self, config):
        super(EHREmbedding, self).__init__()
        
        self.config = config
        
        if not config.use_itemid:
            self.label_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.procedure_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.medication_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.chart_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.label_embeddings = nn.Embedding(config.itemid_size, config.hidden_size)
            self.procedure_embeddings = nn.Embedding(config.itemid_size, config.hidden_size)
            self.medication_embeddings = nn.Embedding(config.itemid_size, config.hidden_size)
            self.chart_embeddings = nn.Embedding(config.itemid_size, config.hidden_size)
        
        # value를 위한 MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # time을 위한 Time2Vec
        self.time_embeddings = Time2Vec(config.hidden_size)
        
        self.on_embeddings = nn.Embedding(config.on_vocab_size, config.hidden_size)
         
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=0
        ).from_pretrained(self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))
        
       
        
        self.age_embeddings = nn.Embedding(100, config.hidden_size)
        self.gender_embeddings = nn.Embedding(2, config.hidden_size)
        self.task_embeddings = nn.Embedding(1, config.hidden_size)
        
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.to(config.device)
        
    def forward(self, label_ids, value_ids, time_ids, on_ids, position_ids, token_type, age_ids, gender_ids, task_token):
        
        if not self.config.use_itemid:
            batch_size, seq_length = label_ids.size()[:2]
            label_embed = torch.zeros(batch_size, seq_length, self.procedure_embeddings.embedding_dim, dtype=torch.float32, device=label_ids.device)
        
        else:
            batch_size, seq_length = label_ids.size()
            label_embed = torch.zeros(batch_size, seq_length, self.procedure_embeddings.embedding_dim, dtype=torch.float32, device=label_ids.device)
        
        procedure_mask = token_type == 1
        medication_mask = token_type == 2
        chart_mask = token_type == 3
        
        
        if not self.config.use_itemid:
            if procedure_mask.any():
                procedure_ids = label_ids[procedure_mask]
                procedure_embeds = self.procedure_embeddings(procedure_ids)
                label_embed[procedure_mask] = procedure_embeds.mean(dim=1)
            
            if medication_mask.any():
                medication_ids = label_ids[medication_mask]
                medication_embeds = self.medication_embeddings(medication_ids)
                label_embed[medication_mask] = medication_embeds.mean(dim=1)
            
            if chart_mask.any():
                chart_ids = label_ids[chart_mask]
                chart_embeds = self.chart_embeddings(chart_ids)
                label_embed[chart_mask] = chart_embeds.mean(dim=1)
        else:
            if procedure_mask.any():
                procedure_ids = label_ids[procedure_mask]
                # print(procedure_ids.shape)
                procedure_embeds = self.procedure_embeddings(procedure_ids)
                label_embed[procedure_mask] = procedure_embeds
            
            if medication_mask.any():
                medication_ids = label_ids[medication_mask]
                medication_embeds = self.medication_embeddings(medication_ids)
                label_embed[medication_mask] = medication_embeds
            
            if chart_mask.any():
                chart_ids = label_ids[chart_mask]
                chart_embeds = self.chart_embeddings(chart_ids)
                label_embed[chart_mask] = chart_embeds
        
        
        # label_embed = label_embed.mean(dim=2)
        with torch.cuda.amp.autocast(enabled=False):
            value_embed = self.value_mlp(value_ids.unsqueeze(-1).float())
        value_embed = value_embed.to(value_ids.dtype)
        # value_embed = self.value_mlp(value_ids.unsqueeze(-1))
        time_embed = self.time_embeddings(time_ids.unsqueeze(-1))
        on_embed = self.on_embeddings(on_ids)
        position_embed = self.position_embeddings(position_ids)
        
        age_embed = self.age_embeddings(age_ids)
        gender_embed = self.gender_embeddings(gender_ids)
        task_embed = self.task_embeddings(task_token)
        
        procedure_or_medication_mask = procedure_mask | medication_mask
        medication_or_chart_mask = medication_mask | chart_mask
        
        embeddings = label_embed + value_embed + time_embed + position_embed
        if procedure_or_medication_mask.any():
            embeddings[procedure_or_medication_mask] += on_embed[procedure_or_medication_mask]
            
        if medication_or_chart_mask.any():
            embeddings[medication_or_chart_mask] += value_embed[medication_or_chart_mask]
        
        embeddings = label_embed + value_embed + time_embed + on_embed + position_embed
     
        combined_embeddings = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)
        
        combined_embeddings = self.LayerNorm(combined_embeddings)
        combined_embeddings = self.dropout(combined_embeddings)
        return combined_embeddings
    
    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))
        
        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))
        
        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)
        
        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
            
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)
                
        return torch.tensor(lookup_table)
    

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    ):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        
        is_global_attn = is_index_global_attn.flatten().any().item()
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions else None
        
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but is is for {head_mask.size()[0]}."
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    is_index_masked,
                    is_index_global_attn,
                    is_global_attn,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
                    
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # undo padding if necessary
        # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
        hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
        if output_hidden_states:
            all_hidden_states = tuple([state[:, : state.shape[1] - padding_len] for state in all_hidden_states])

        if output_attentions:
            all_attentions = tuple([state[:, :, : state.shape[2] - padding_len, :] for state in all_attentions])

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
            )
        return LongformerBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )
        
class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)  
        return pooled_output


class LongformerModeling(LongformerPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super(LongformerModeling, self).__init__(config)
        self.ehrembedding = EHREmbedding(config).to(config.device)
        self.encoder = Encoder(config).to(config.device)
        self.pooler = Pooler(config) if add_pooling_layer else None
        self.post_init()
        
    def get_input_embeddings(self):
        return self.ehrembedding.label_embeddings
        
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask
        
    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, value_ids=None, offset_ids=None, on_ids=None, position_ids=None, token_type=None, age_ids=None, gender_ids=None, task_token=None, output_hidden_states=None, output_attentions=None, return_dict=None):    
        if attention_mask is None:
            attention_mask = torch.ones_like(value_ids)
            
            
        embedding_output = self.ehrembedding(input_ids, value_ids, offset_ids, on_ids, position_ids, token_type, age_ids, gender_ids, task_token)
        
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(attention_mask)
        
        # Compute extended attention mask
        # global_token = torch.full((config.batch_size, 3), 1)
        global_token = torch.full((input_ids.size(0), 3), 1).to(input_ids.device)
        
        attention_mask = torch.cat([global_token, attention_mask], dim=1)
        global_attention_mask = torch.cat([global_token, global_attention_mask], dim=1)
        
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        
        # extended_attention_mask = self.longformer.get_extended_attention_mask(attention_mask, embedding_output.size(), input_ids.device)
        # extended_global_attention_mask = self.longformer.get_extended_attention_mask(global_attention_mask, embedding_output.size(), input_ids.device)
        # extended_attention_mask = self.longformer.get_extended_attention_mask(attention_mask, (input_ids.size(0), input_ids.size(1) + 3), input_ids.device)
    
        # embedding_output = embedding_output.view(
        #     embedding_output.size(0), -1, config.hidden_size
        # )
        
        # print(extended_attention_mask.shape)
        
        outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            # global_attention_mask=extended_global_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
            
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomLongformerForMaskedLM(LongformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder"]
    
    def __init__(self, config):
        super().__init__(config)
        
        self.longformer = LongformerModeling(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        value_ids: Optional[torch.Tensor] = None,
        offset_ids: Optional[torch.Tensor] = None,
        on_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        task_token: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,   
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerMaskedLMOutput]:
        
        
        outputs = self.longformer(
            input_ids,
            attention_mask,
            global_attention_mask,
            value_ids,
            offset_ids,
            on_ids,
            position_ids,
            token_type_ids,
            age_ids,
            gender_ids,
            task_token,
            output_hidden_states,
            output_attentions
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)[:, 3:, :]

        
        masked_lm_loss = None
        if labels is not None:
            # loss_fct = FocalLoss()
            loss_fct = nn.CrossEntropyLoss()
            
            labels = labels.to(prediction_scores.device)
            # print(attention_mask.shape)
            # Make sure the sizes match
            active_loss = (attention_mask.contiguous().view(-1) == 1) & (labels.contiguous().view(-1) != -100)
            
            # print(active_loss.shape)
            # print(prediction_scores.shape)
            active_logits = prediction_scores.contiguous().reshape(-1, self.config.itemid_size)[active_loss]
            active_labels = labels.contiguous().reshape(-1)[active_loss]
            
            masked_lm_loss = loss_fct(active_logits, active_labels)
        
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions
        )
        
class LongformerClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # print("Config 객체 내용:", config.__dict__)
        self.dense1 = nn.Linear(config.hidden_size, config.finetuning_hidden_size1)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = nn.Linear(config.finetuning_hidden_size1, config.finetuning_hidden_size2)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.dense3 = nn.Linear(config.finetuning_hidden_size2, config.finetuning_hidden_size2)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.finetuning_hidden_size2, config.num_labels)
        
    def forward(self, hidden_states, **kwags):
        hidden_states = hidden_states[:, 0, :] # take <s> token (equiv. to [CLS])
        
        
        hidden_states = self.dense1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        
        
        hidden_states = self.dense2(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        
        hidden_states = self.dense3(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout3(hidden_states)
        
        
        output = self.out_proj(hidden_states)
        return output
        
class CustomLongformerForPredictionTask(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.longformer = LongformerModeling(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        value_ids: Optional[torch.Tensor] = None,
        offset_ids: Optional[torch.Tensor] = None,
        on_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        gender_ids: Optional[torch.Tensor] = None,
        task_token: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,   
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        
        outputs = self.longformer(
            input_ids,
            attention_mask,
            global_attention_mask,
            value_ids,
            offset_ids,
            on_ids,
            position_ids,
            token_type_ids,
            age_ids,
            gender_ids,
            task_token,
            output_hidden_states,
            output_attentions
        )
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            
            elif self.config.problem_type == "single_label_classification":
                # loss_fct = FocalLoss()
                # class_weights = torch.tensor([1.0, 4.0]).to(self.config.device)
                # loss_fct = nn.CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits, sequence_output) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions
        )
        
        # return loss, logits, sequence_output
                
                
    