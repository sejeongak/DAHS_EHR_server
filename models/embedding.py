import math
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from transformers import LongformerConfig, BigBirdConfig

# class TimeEmbedding(nn.Module):
#     """ Embedding layer for time features """
    
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.linear = nn.Linear(1, 1)
#         self.periodic = nn.ModuleList([nn.Linear(1, 1) for _ in range(kernel_size - 1)])

#     def forward(self, x):
#         v1 = self.linear(x)
#         v2 = torch.cat([torch.sin(periodic(x)) for periodic in self.periodic], dim=-1)
#         return torch.cat([v1, v2], dim=-1)

class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.w = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(torch.randn(output_dim))
        self.freqs = nn.Parameter(torch.randn(output_dim))
        
        self.projection = nn.Linear(2*output_dim, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        w = self.w.unsqueeze(0)
        b = self.b.unsqueeze(0).unsqueeze(0)
        feqs = self.freqs.unsqueeze(0).unsqueeze(0)
        
        linear = torch.matmul(x, w) + b
        linear = linear.squeeze(-2)
        
        periodic = torch.sin(torch.matmul(x, feqs) + b)
        periodic = periodic.squeeze(-2)
        
        combined = torch.cat([linear, periodic], dim=-1)
        
        return self.projection(combined)
    
        
class ValueEmbedding(nn.Module):
    """ Embedding layer for value features """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.value_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.value_embedding(x)
    
class UnitEmbedding(nn.Module):
    def __init__(self, unit_size, embedding_size):
        super().__init__()
        self.unit_embedding = nn.Embedding(unit_size, embedding_size)
    
    def forward(self, x):
        return self.unit_embedding(x)
    
class PositionalEmbedding(nn.Module):
    """ Embedding layer for position features """
    
    def __init__(self, max_len:int, embedding_size:int):
        super().__init__()
        
        self.position_embeddings = nn.Embedding(
            max_len, embedding_size, padding_idx=0
        ).from_pretrained(self._init_posi_embedding(max_len, embedding_size))
    
    def _init_posi_embedding(self, max_len, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))
        
        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))
        
        # initialize position embedding table
        lookup_table = np.zeros((max_len, hidden_size), dtype=np.float32)
        
        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_len):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
            
        # set odd dimension
        for pos in range(max_len):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)
                
        return torch.tensor(lookup_table)
        
    def forward(self, x):
        """ Apply positional embedding"""
        return self.position_embeddings(x)
    
class ContinuousEmbedding(nn.Module):
    def __init__(self, continuous_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(continuous_size, embedding_size)
        
    def forward(self, x):
        return self.embedding(x)
    

class AgeEmbedding(nn.Module):
    def __init__(self, max_age, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.age_embedding = nn.Embedding(max_age, embedding_size)
    
    def forward(self, x):
        return self.age_embedding(x)
    
class GenderEmbedding(nn.Module):
    def __init__(self, gender_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.gender_embedding = nn.Embedding(gender_size, embedding_size)
    
    def forward(self, x):
        return self.gender_embedding(x)
    
class TaskEmbedding(nn.Module):
    def __init__(self, task_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.task_embedding = nn.Embedding(task_size, embedding_size)
    
    def forward(self, x):
        return self.task_embedding(x)
    
class ConceptEmbedding(nn.Module):
    def __init__(self, vocab_size, itemid_size, embedding_size, padding_idx=1, use_itemid=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.use_itemid = use_itemid
        self.padding_idx = padding_idx

        if use_itemid:  
            self.procedure_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
            self.medication_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
            self.chart_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
        else:
            self.procedure_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
            self.medication_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
            self.chart_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
        
        
    
    def forward(self, concept, token_type):
        
        if not self.use_itemid:
            batch_size, seq_length = concept.size()[:2]
            label_embed = torch.zeros(batch_size, seq_length, self.procedure_embedding.embedding_dim, dtype=torch.float32, device=concept.device)
        
        else:
            batch_size, seq_length = concept.size()
            label_embed = torch.zeros(batch_size, seq_length, self.procedure_embedding.embedding_dim, dtype=torch.float32, device=concept.device)
        
        procedure_mask = token_type == 1
        medication_mask = token_type == 2
        chart_mask = token_type == 3
        
        if not self.use_itemid:
            if procedure_mask.any():
                procedure_ids = concept[procedure_mask]
                procedure_embeds = self.procedure_embedding(procedure_ids)
                label_embed[procedure_mask] = procedure_embeds.mean(dim=1)

            if medication_mask.any():
                medication_ids = concept[medication_mask]
                medication_embeds = self.medication_embedding(medication_ids)
                label_embed[medication_mask] = medication_embeds.mean(dim=1)
            
            if chart_mask.any():
                chart_ids = concept[chart_mask]
                chart_embeds = self.chart_embedding(chart_ids)
                label_embed[chart_mask] = chart_embeds.mean(dim=1)  
                
        else:
            if procedure_mask.any():
                procedure_indices = procedure_mask.nonzero(as_tuple=False)
                procedure_ids = concept[procedure_mask]
                procedure_embeds = self.procedure_embedding(procedure_ids)
                
                for idx, (batch_idx, seq_idx) in enumerate(procedure_indices):
                    label_embed[batch_idx, seq_idx] = procedure_embeds[idx]

            if medication_mask.any():
                medication_indices = medication_mask.nonzero(as_tuple=False)
                medication_ids = concept[medication_mask]
                medication_embeds = self.medication_embedding(medication_ids)
                
                for idx, (batch_idx, seq_idx) in enumerate(medication_indices):
                    label_embed[batch_idx, seq_idx] = medication_embeds[idx]
            
            if chart_mask.any():
                chart_indices = chart_mask.nonzero(as_tuple=False)
                chart_ids = concept[chart_mask]
                chart_embeds = self.chart_embedding(chart_ids)
                
                for idx, (batch_idx, seq_idx) in enumerate(chart_indices):
                    label_embed[batch_idx, seq_idx] = chart_embeds[idx]
                
        return label_embed
    

class EHREmbedding(nn.Module):
    def __init__(self, config: LongformerConfig, itemid_size, unit_size, max_age, max_len, continuous_size, gender_size, task_size, padding_idx=0, use_itemid=True, inputs_embeds=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.itemid_size = itemid_size
        self.unit_size = unit_size
        self.max_age = max_age
        self.hidden_size = config.hidden_size
        self.continuous_size = continuous_size
        self.gender_size = gender_size
        self.task_size = task_size
        self.padding_idx = padding_idx
        self.use_itemid = use_itemid
        self.max_position_embeddings = max_len
        self.inputs_embeds = inputs_embeds 
        
        self.concept_embedding = ConceptEmbedding(self.vocab_size, self.itemid_size, self.hidden_size, padding_idx, use_itemid)
        self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
        self.time_embedding = TimeEmbedding(1, self.hidden_size)
        self.value_embedding = ValueEmbedding(self.hidden_size)
        self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
        self.continuous_embedding = ContinuousEmbedding(self.continuous_size, self.hidden_size)
        self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
        self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
        self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
        
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    # def cache_input(
    #     self,
    #     value_ids: torch.Tensor,
    #     unit_ids: torch.Tensor,
    #     time_ids: torch.Tensor,
    #     continuous_ids: torch.Tensor,
    #     age_ids: torch.Tensor,
    #     gender_ids: torch.Tensor,
    #     task_token: torch.Tensor,
    # ):
    #     self.value_ids = value_ids
    #     self.unit_ids = unit_ids
    #     self.time_ids = time_ids
    #     self.continuous_ids = continuous_ids
    #     self.age_ids = age_ids
    #     self.gender_ids = gender_ids
    #     self.task_ids = task_token
    
    # def clear_cache(self):
    #     del self.value_ids, self.unit_ids, self.time_ids, self.continuous_ids, self.age_ids, self.gender_ids, self.task_ids
    
    
    def forward(self, 
                input_ids: torch.Tensor,
                value_ids: torch.Tensor,
                unit_ids: torch.Tensor,
                time_ids: torch.Tensor,
                continuous_ids: torch.Tensor,
                position_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                age_ids: torch.Tensor,
                gender_ids: torch.Tensor,
                task_ids: torch.Tensor,
                inputs_embeds=None,
                ):
        
        if inputs_embeds is not None:
            return inputs_embeds

        concept_embed = self.concept_embedding(input_ids, token_type_ids)
        time_embed = self.time_embedding(time_ids)
        age_embed = self.age_embedding(age_ids)
        gender_embed = self.gender_embedding(gender_ids)
        positional_embed = self.position_embedding(position_ids)
   
        
        value_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
        unit_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
        continuous_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
        
        value_mask = (token_type_ids == 2) | (token_type_ids == 3)
        continuous_mask = (token_type_ids == 1) | (token_type_ids == 2)
        
    
        
        if value_mask.any():
            value_indices = value_mask.nonzero(as_tuple=False)
            value_ids = value_ids[value_mask]
            value_embeds = self.value_embedding(value_ids)
            unit_ids = unit_ids[value_mask]
            unit_embeds = self.unit_embedding(unit_ids)
            
            for idx, (batch_idx, seq_idx) in enumerate(value_indices):
                value_embed[batch_idx, seq_idx] = value_embeds[idx]
                unit_embed[batch_idx, seq_idx] = unit_embeds[idx]
        
        if continuous_mask.any():
            continuous_indices = continuous_mask.nonzero(as_tuple=False)
            continuous_ids = continuous_ids[continuous_mask]
            continuous_embeds = self.continuous_embedding(continuous_ids)
            
            for idx, (batch_idx, seq_idx) in enumerate(continuous_indices):
                continuous_embed[batch_idx, seq_idx] = continuous_embeds[idx]

        
        task_embed = self.task_embedding(task_ids)
        
        # if torch.isinf(concept_embed).any():
        #     print("concept embed inf ok")
        # if torch.isinf(time_embed).any():
        #     print("time embed inf ok")
        # if torch.isinf(positional_embed).any():
        #     print("position embed inf ok")
        if torch.isinf(value_embed).any():
            print("value_embed inf ok")
        # if torch.isinf(unit_embed).any():
        #     print("unit embed inf ok")
        # if torch.isinf(continuous_embed).any():
        #     print("continuous embed inf ok")
        
        embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed + continuous_embed
        
        combined_embed = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)
        # if torch.isinf(combined_embed).any():
        #     print("combined embed inf ok")
        #     print(torch.max(combined_embed), torch.min(combined_embed))
        combined_embed = self.LayerNorm(combined_embed)
        # if torch.isnan(combined_embed).any():
        #     print("embedding layernorm nan ok")
        #     print(torch.max(combined_embed), torch.min(combined_embed))
        combined_embed = self.dropout(combined_embed)
        # if torch.isnan(combined_embed).any():
        #     print("embedding dropout nan ok")
        
        # self.clear_cache()
        
        # print(combined_embed.shape)
        
        return combined_embed
    

        
                
                