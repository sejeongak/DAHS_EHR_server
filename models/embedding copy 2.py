import math
from typing import Any, Optional
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
from torch import nn
from transformers import LongformerConfig, BigBirdConfig
import torch.nn.init as init
import time
from torch.cuda.amp import autocast
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

# class SwiGLU(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.proj = nn.Linear(hidden_size, hidden_size * 2)
#         self.swish = lambda x: x * torch.sigmoid(x)  # Swish 활성화 함수

#     def forward(self, x):
#         x_proj = self.proj(x)
#         x, gate = x_proj.chunk(2, dim=-1)
#         return x * self.swish(gate)

    

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
    
    def __init__(self, token_type_size, embedding_size):
        super().__init__()
        
        self.token_type_embedding = nn.Embedding(token_type_size, embedding_size)
        
        
        self.linear1 = nn.Linear(1, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        # self.norm2 = nn.LayerNorm(embedding_size)
        # self.gelu = nn.LeakyReLU(0.01)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        
        self.projection = nn.Linear(2*embedding_size, embedding_size)
        
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.projection.weight)
                
    def forward(self, x, token_type_ids):
        token_type_embed = self.token_type_embedding(token_type_ids)
        # token_type_embed = self.norm2(token_type_embed)
        
        x = x.unsqueeze(-1)
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.linear2(x)
                
        combined = torch.cat([x, token_type_embed], dim=-1)
        output = self.projection(combined)
        
        return output
    


    
class UnitEmbedding(nn.Module):
    def __init__(self, unit_size, embedding_size):
        super().__init__()
        self.unit_embedding = nn.Embedding(unit_size, embedding_size)
    
    def forward(self, x):
        return self.unit_embedding(x)
    
# class PositionalEmbedding(nn.Module):
#     """ Embedding layer for position features """
    
#     def __init__(self, max_len:int, embedding_size:int):
#         super().__init__()
        
#         self.position_embeddings = nn.Embedding(
#             max_len, embedding_size, padding_idx=0
#         ).from_pretrained(self._init_posi_embedding(max_len, embedding_size))
    
#     def _init_posi_embedding(self, max_len, hidden_size):
#         def even_code(pos, idx):
#             return np.sin(pos / (10000 ** (2 * idx / hidden_size)))
        
#         def odd_code(pos, idx):
#             return np.cos(pos / (10000 ** (2 * idx / hidden_size)))
        
#         # initialize position embedding table
#         lookup_table = np.zeros((max_len, hidden_size), dtype=np.float32)
        
#         # reset table parameters with hard encoding
#         # set even dimension
#         for pos in range(max_len):
#             for idx in np.arange(0, hidden_size, step=2):
#                 lookup_table[pos, idx] = even_code(pos, idx)
            
#         # set odd dimension
#         for pos in range(max_len):
#             for idx in np.arange(1, hidden_size, step=2):
#                 lookup_table[pos, idx] = odd_code(pos, idx)
                
#         return torch.tensor(lookup_table)
        
#     def forward(self, x):
#         """ Apply positional embedding"""
#         return self.position_embeddings(x)

class PositionalEmbedding(nn.Module):
    
    def __init__(self, max_len: int, embedding_size: int):
        super().__init__()

        self.position_embeddings = nn.Embedding(max_len, embedding_size, padding_idx=0)

        self.register_buffer("sinusoidal_embedding", self._init_sinusoidal_embedding(max_len, embedding_size))

        self.projection = nn.Linear(2 * embedding_size, embedding_size)
        
    def _init_sinusoidal_embedding(self, max_len, embedding_size):
        def even_code(pos, i, embedding_size):
            return np.sin(pos / (10000 ** ((2 * i) / embedding_size)))

        def odd_code(pos, i, embedding_size):
            return np.cos(pos / (10000 ** ((2 * i) / embedding_size)))

        lookup_table = np.zeros((max_len, embedding_size), dtype=np.float32)
        
        for pos in range(max_len):
            for i in range(embedding_size // 2):  
                lookup_table[pos, 2 * i] = even_code(pos, i, embedding_size)
                lookup_table[pos, 2 * i + 1] = odd_code(pos, i, embedding_size)
        
        return torch.tensor(lookup_table, dtype=torch.float32)

    def forward(self, x):
        trainable_embed = self.position_embeddings(x)    # Trainable Embedding
        fixed_embed = self.sinusoidal_embedding[x]       # Sinusoidal Encoding
        
        combined = torch.cat([trainable_embed, fixed_embed], dim=-1)  
        return self.projection(combined) 

# class PositionalEmbedding(nn.Module):

#     def __init__(self, max_len: int, embedding_size: int):
#         super(PositionalEmbedding, self).__init__()
        
#         # 학습 가능한 nn.Embedding 레이어 사용
#         self.position_embeddings = nn.Embedding(max_len, embedding_size)

#     def forward(self, x):
#         return self.position_embeddings(x.long())

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_relative_position, embedding_size):
        super(RelativePositionalEmbedding, self).__init__()
        self.max_relative_position = max_relative_position
        self.embedding_size = embedding_size
        self.relative_embeddings = nn.Embedding(2 * max_relative_position + 1, embedding_size)

    def forward(self, seq_length):
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.relative_embeddings.weight.device)
        relative_position_matrix = position_ids[:, None] - position_ids[None, :]
        relative_position_matrix = torch.clamp(relative_position_matrix, -self.max_relative_position, self.max_relative_position)
        relative_position_matrix = relative_position_matrix + self.max_relative_position 

        return self.relative_embeddings(relative_position_matrix)
    
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
    
# class ConceptEmbedding(nn.Module):
#     def __init__(self, vocab_size, itemid_size, embedding_size, padding_idx=1, use_itemid=True):
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.use_itemid = use_itemid
#         self.padding_idx = padding_idx

#         if use_itemid:  
#             self.procedure_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
#             self.medication_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
#             self.chart_embedding = nn.Embedding(itemid_size, embedding_size, padding_idx=self.padding_idx)
#         else:
#             self.procedure_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
#             self.medication_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
#             self.chart_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=self.padding_idx)
        
        
    
#     def forward(self, concept, token_type):
        
#         if not self.use_itemid:
#             batch_size, seq_length = concept.size()[:2]
#             label_embed = torch.zeros(batch_size, seq_length, self.procedure_embedding.embedding_dim, dtype=torch.float32, device=concept.device)
        
#         else:
#             batch_size, seq_length = concept.size()
#             label_embed = torch.zeros(batch_size, seq_length, self.procedure_embedding.embedding_dim, dtype=torch.float32, device=concept.device)
        
#         procedure_mask = token_type == 1
#         medication_mask = token_type == 2
#         chart_mask = token_type == 3
        
#         if not self.use_itemid:
#             if procedure_mask.any():
#                 procedure_ids = concept[procedure_mask]
#                 procedure_embeds = self.procedure_embedding(procedure_ids)
#                 label_embed[procedure_mask] = procedure_embeds.mean(dim=1)

#             if medication_mask.any():
#                 medication_ids = concept[medication_mask]
#                 medication_embeds = self.medication_embedding(medication_ids)
#                 label_embed[medication_mask] = medication_embeds.mean(dim=1)
            
#             if chart_mask.any():
#                 chart_ids = concept[chart_mask]
#                 chart_embeds = self.chart_embedding(chart_ids)
#                 label_embed[chart_mask] = chart_embeds.mean(dim=1)  
                
#         else:
#             if procedure_mask.any():
#                 procedure_indices = procedure_mask.nonzero(as_tuple=False)
#                 procedure_ids = concept[procedure_mask]
#                 procedure_embeds = self.procedure_embedding(procedure_ids)
                
#                 for idx, (batch_idx, seq_idx) in enumerate(procedure_indices):
#                     label_embed[batch_idx, seq_idx] = procedure_embeds[idx]

#             if medication_mask.any():
#                 medication_indices = medication_mask.nonzero(as_tuple=False)
#                 medication_ids = concept[medication_mask]
#                 medication_embeds = self.medication_embedding(medication_ids)
                
#                 for idx, (batch_idx, seq_idx) in enumerate(medication_indices):
#                     label_embed[batch_idx, seq_idx] = medication_embeds[idx]
            
#             if chart_mask.any():
#                 chart_indices = chart_mask.nonzero(as_tuple=False)
#                 chart_ids = concept[chart_mask]
#                 chart_embeds = self.chart_embedding(chart_ids)
                
#                 for idx, (batch_idx, seq_idx) in enumerate(chart_indices):
#                     label_embed[batch_idx, seq_idx] = chart_embeds[idx]
                
#         return label_embed

# class ClinicalBERTEmbedding(nn.Module):
#     def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
#         super().__init__()
#         self.embedding_model = AutoModel.from_pretrained(model_name).to("cuda")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         for param in self.embedding_model.parameters():
#             param.requires_grad = False
            
#     def get_embedding_batch(self, text):
#         tokens = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.embedding_model.device)
        
#         outputs = self.embedding_model(**tokens)
#         embeddings = outputs.last_hidden_state[:, 0, :]
#         # embeddings = outputs.hidden_state[:, 1:-1, :].mean(dim=1) 
#         # embeddings = embeddings.cpu().clone().detach()

#         return embeddings
    
# class ConceptEmbeddingwithClinicalBert(nn.Module):
#     def __init__(self, idx2label, embedding_model):
#         super().__init__()
#         self.idx2label = idx2label
#         self.embedding_model = embedding_model
        

#     def forward(self, concept):
#         concept = concept.to("cuda")
#         input2label = [[self.idx2label[ids.item()] for ids in seq] for seq in concept.cpu()]
#         label_embeddings = []
#         for labels in input2label:
#             with torch.no_grad():
#                 embeddings =self.embedding_model.get_embedding_batch(labels)
#             label_embeddings.append(embeddings)
#         return torch.stack(label_embeddings).to(self.embedding_model.embedding_model.device)
    
class ConceptEmbeddingwithClinicalBert(nn.Module):
    def __init__(self, idx2label, embedding_tokenizer, embedding_model, embedding_map=None):
        super().__init__()
        self.idx2label = idx2label
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model
        
        if embedding_map is not None:
            self.embedding_map = embedding_map
        else:
            self.embedding_map = {}
            
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
    def get_embedding_batch(self, text):
        tokens = self.embedding_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.embedding_model.device)
        outputs = self.embedding_model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        # embeddings = outputs.hidden_state[:, 1:-1, :].mean(dim=1) 
        # embeddings = embeddings.cpu().clone().detach()

        return embeddings
    
    def forward(self, concept):
        concept = concept.to("cuda")
        input2label = [[self.idx2label[ids.item()] for ids in seq] for seq in concept.cpu()]
        label_embeddings = []
        for labels in input2label:
            # 각 레이블에 대해 임베딩을 계산하거나 캐시된 임베딩을 불러옵니다.
            batch_embeddings = []
            for label in labels:
                if label in self.embedding_map:
                    # 임베딩 맵에 이미 존재하면 해당 임베딩을 사용
                    batch_embeddings.append(torch.tensor(self.embedding_map[label]).to(self.embedding_model.device))
                else:
                    # 임베딩 맵에 없으면 모델을 통해 임베딩 계산
                    with torch.no_grad():
                        embedding = self.get_embedding_batch([label])
                    # 임베딩을 맵에 저장
                    self.embedding_map[label] = embedding
                    batch_embeddings.append(torch.tensor(embedding).to(self.embedding_model.device))
            
            # 레이블들의 임베딩을 하나의 텐서로 결합
            label_embeddings.append(torch.stack(batch_embeddings))

        # 모든 레이블에 대한 임베딩을 결합
        return torch.stack(label_embeddings).to(self.embedding_model.device)
                
# class OrderCategoryNameEmbedding(nn.Module):
#     def __init__(self, idx2label, embedding_model):
#         super().__init__()
#         self.idx2label = idx2label
#         self.embedding_model = embedding_model
    
#     def forward(self, concept):
#         concept = concept.to("cuda")
#         input2label = [[self.idx2label[ids.item()] for ids in seq] for seq in concept.cpu()]
#         label_embeddings = []
#         for labels in input2label:
#             with torch.no_grad():
#                 embeddings =self.embedding_model.get_embedding_batch(labels)
#             label_embeddings.append(embeddings)
#         return torch.stack(label_embeddings).to(self.embedding_model.embedding_model.device)
    
# class OrderCategoryDescriptionEmbedding(nn.Module):
#     def __init__(self, idx2label, embedding_model):
#         super().__init__()
#         self.idx2label = idx2label
#         self.embedding_model = embedding_model
    
#     def forward(self, concept):
#         concept = concept.to("cuda")
#         print(f"idx2label length: {len(self.idx2label)}")
#         print(f"concept min: {concept.min().item()}, concept max: {concept.max().item()}")

#         input2label = [[self.idx2label[ids.item()] for ids in seq] for seq in concept.cpu()]
#         label_embeddings = []
#         for labels in input2label:
#             with torch.no_grad():
#                 embeddings =self.embedding_model.get_embedding_batch(labels)
#             label_embeddings.append(embeddings)
#         return torch.stack(label_embeddings).to(self.embedding_model.embedding_model.device)
    
    
class OrderCategoryNameEmbedding(nn.Module):
    def __init__(self, name_size, embedding_size):
        super().__init__()
        self.ordername_embedding = nn.Embedding(name_size, embedding_size)
    
    def forward(self, x):
        return self.ordername_embedding(x)
            
class OrderCategoryDescriptionEmbedding(nn.Module):
    def __init__(self, description_size, embedding_size):
        super().__init__()
        self.orderdescription_embedding = nn.Embedding(description_size, embedding_size)
        
    def forward(self, x):
        return self.orderdescription_embedding(x)
        
    

# class EHREmbedding(nn.Module):
#     def __init__(self, 
#                  config: LongformerConfig, 
#                  itemid_size, 
#                  unit_size,
#                  max_age, 
#                  max_len, 
#                  continuous_size, 
#                  gender_size, 
#                  task_size, 
#                  idx2label, #########
#                  padding_idx=0, 
#                  use_itemid=True, 
#                  inputs_embeds=None):
#         super().__init__()
#         self.vocab_size = config.vocab_size
#         self.itemid_size = itemid_size
#         self.unit_size = unit_size
#         self.max_age = max_age
#         self.hidden_size = config.hidden_size
#         self.continuous_size = continuous_size
#         self.gender_size = gender_size
#         self.task_size = task_size
#         self.padding_idx = padding_idx
#         self.use_itemid = use_itemid
#         self.max_position_embeddings = max_len
#         self.inputs_embeds = inputs_embeds 
#         self.idx2label = idx2label #####
        
#         # self.concept_embedding = ConceptEmbedding(self.vocab_size, self.itemid_size, self.hidden_size, padding_idx, use_itemid) ####
#         self.concept_embedding = ConceptEmbeddingwithClinicalBert(self.idx2label)
#         self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
#         self.time_embedding = TimeEmbedding(1, self.hidden_size)
#         self.value_embedding = ValueEmbedding(self.hidden_size)
#         self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
#         self.continuous_embedding = ContinuousEmbedding(self.continuous_size, self.hidden_size)
#         self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
#         self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
#         self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
        
#         self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#     # def cache_input(
#     #     self,
#     #     value_ids: torch.Tensor,
#     #     unit_ids: torch.Tensor,
#     #     time_ids: torch.Tensor,
#     #     continuous_ids: torch.Tensor,
#     #     age_ids: torch.Tensor,
#     #     gender_ids: torch.Tensor,
#     #     task_token: torch.Tensor,
#     # ):
#     #     self.value_ids = value_ids
#     #     self.unit_ids = unit_ids
#     #     self.time_ids = time_ids
#     #     self.continuous_ids = continuous_ids
#     #     self.age_ids = age_ids
#     #     self.gender_ids = gender_ids
#     #     self.task_ids = task_token
    
#     # def clear_cache(self):
#     #     del self.value_ids, self.unit_ids, self.time_ids, self.continuous_ids, self.age_ids, self.gender_ids, self.task_ids
    
    
#     def forward(self, 
#                 input_ids: torch.Tensor,
#                 value_ids: torch.Tensor,
#                 unit_ids: torch.Tensor,
#                 time_ids: torch.Tensor,
#                 continuous_ids: torch.Tensor,
#                 position_ids: torch.Tensor,
#                 token_type_ids: torch.Tensor,
#                 age_ids: torch.Tensor,
#                 gender_ids: torch.Tensor,
#                 task_ids: torch.Tensor,
#                 inputs_embeds=None,
#                 ):
        
#         if inputs_embeds is not None:
#             return inputs_embeds

#         # concept_embed = self.concept_embedding(input_ids, token_type_ids)
#         concept_embed = self.concept_embedding(input_ids)
#         time_embed = self.time_embedding(time_ids)
#         age_embed = self.age_embedding(age_ids)
#         gender_embed = self.gender_embedding(gender_ids)
#         positional_embed = self.position_embedding(position_ids)
   
        
#         value_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
#         unit_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
#         continuous_embed = torch.zeros(concept_embed.size(), dtype=torch.float32, device=input_ids.device)
#         # continuous_embed = self.continuous_embedding(continuous_ids)
        
#         value_mask = (token_type_ids == 2) | (token_type_ids == 3)
#         continuous_mask = (token_type_ids == 1) | (token_type_ids == 2)
        
    
        
#         # if value_mask.any():
#         #     value_indices = value_mask.nonzero(as_tuple=False)
#         #     value_ids = value_ids[value_mask]
#         #     value_embeds = self.value_embedding(value_ids)
#         #     unit_ids = unit_ids[value_mask]
#         #     unit_embeds = self.unit_embedding(unit_ids)
            
#         #     for idx, (batch_idx, seq_idx) in enumerate(value_indices):
#         #         value_embed[batch_idx, seq_idx] = value_embeds[idx]
#         #         unit_embed[batch_idx, seq_idx] = unit_embeds[idx]
        
#         # if continuous_mask.any():
#         #     continuous_indices = continuous_mask.nonzero(as_tuple=False)
#         #     continuous_ids = continuous_ids[continuous_mask]
#         #     continuous_embeds = self.continuous_embedding(continuous_ids)
            
#         #     for idx, (batch_idx, seq_idx) in enumerate(continuous_indices):
#         #         continuous_embed[batch_idx, seq_idx] = continuous_embeds[idx]

#         if value_mask.any():
#             # value_indices = value_mask.nonzero(as_tuple=False)
#             # value_ids = value_ids[value_mask]
#             # value_embeds = self.value_embedding(value_ids)
#             # unit_ids = unit_ids[value_mask]
#             # unit_embeds = self.unit_embedding(unit_ids)
            
#             # for idx, (batch_idx, seq_idx) in enumerate(value_indices):
#             #     value_embed[batch_idx, seq_idx] = value_embeds[idx]
#             #     unit_embed[batch_idx, seq_idx] = unit_embeds[idx]
#             value_embeds = self.value_embedding(value_ids[value_mask])
#             unit_embeds = self.unit_embedding(unit_ids[value_mask])
            
#             # Use scatter for efficient assignment
#             # value_embed = value_embed.masked_scatter_(value_mask.unsqueeze(-1), value_embeds)
#             value_embed = value_embed.masked_scatter_(value_mask.unsqueeze(-1), value_embeds.to(value_embed.dtype))
#             # unit_embed = unit_embed.masked_scatter_(value_mask.unsqueeze(-1), unit_embeds)
#             unit_embed = unit_embed.masked_scatter_(value_mask.unsqueeze(-1), unit_embeds.to(unit_embed.dtype))
                
#         if continuous_mask.any():
#             # continuous_indices = continuous_mask.nonzero(as_tuple=False)
#             # continuous_ids = continuous_ids[continuous_mask]
#             # continuous_embeds = self.continuous_embedding(continuous_ids)
            
#             # for idx, (batch_idx, seq_idx) in enumerate(continuous_indices):
#             #     continuous_embed[batch_idx, seq_idx] = continuous_embeds[idx]
#             continuous_embeds = self.continuous_embedding(continuous_ids[continuous_mask])
    
#             # Use scatter for efficient assignment
#             # continuous_embed = continuous_embed.masked_scatter_(continuous_mask.unsqueeze(-1), continuous_embeds)
#             continuous_embed = continuous_embed.masked_scatter_(continuous_mask.unsqueeze(-1), continuous_embeds.to(continuous_embed.dtype))
        
        
#         task_embed = self.task_embedding(task_ids)
        
#         # if torch.isinf(concept_embed).any():
#         #     print("concept embed inf ok")
#         # if torch.isinf(time_embed).any():
#         #     print("time embed inf ok")
#         # if torch.isinf(positional_embed).any():
#         #     print("position embed inf ok")
#         if torch.isinf(value_embed).any():
#             print("value_embed inf ok")
#         # if torch.isinf(unit_embed).any():
#         #     print("unit embed inf ok")
#         # if torch.isinf(continuous_embed).any():
#         #     print("continuous embed inf ok")
        
#         embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed + continuous_embed
        
#         combined_embed = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)
#         # if torch.isinf(combined_embed).any():
#         #     print("combined embed inf ok")
#         #     print(torch.max(combined_embed), torch.min(combined_embed))
#         combined_embed = self.LayerNorm(combined_embed)
#         # if torch.isnan(combined_embed).any():
#         #     print("embedding layernorm nan ok")
#         #     print(torch.max(combined_embed), torch.min(combined_embed))
#         combined_embed = self.dropout(combined_embed)
#         # if torch.isnan(combined_embed).any():
#         #     print("embedding dropout nan ok")
        
#         # self.clear_cache()
        
#         # print(combined_embed.shape)
        
#         return combined_embed
    

        
class EHREmbedding(nn.Module):
    def __init__(self, 
                 config: LongformerConfig, 
                 itemid_size, 
                 unit_size,
                 max_age, 
                 max_len, 
                 gender_size, 
                 task_size, 
                 idx2label, #########
                #  idx2ordername,
                #  idx2orderdescription,
                 name_size,
                 description_size,
                 token_type_size,
                 embedding_tokenizer,
                 embedding_model,
                 embedding_map,
                 padding_idx=0, 
                 use_itemid=True, 
                 inputs_embeds=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.itemid_size = itemid_size
        self.unit_size = unit_size
        self.max_age = max_age
        self.hidden_size = config.hidden_size
        self.gender_size = gender_size
        self.task_size = task_size
        self.padding_idx = padding_idx
        self.use_itemid = use_itemid
        self.max_position_embeddings = max_len
        self.inputs_embeds = inputs_embeds 
        self.idx2label = idx2label #####
        # self.idx2ordername = idx2ordername
        # self.idx2orderdescription = idx2orderdescription
        self.name_size = name_size
        self.description_size = description_size
        self.token_type_size = token_type_size
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model
        self.embedding_map = embedding_map
        
        # self.concept_embedding = ConceptEmbedding(self.vocab_size, self.itemid_size, self.hidden_size, padding_idx, use_itemid) ####
        self.concept_embedding = ConceptEmbeddingwithClinicalBert(self.idx2label, self.embedding_tokenizer, self.embedding_model, self.embedding_map)
        if self.concept_embedding.embedding_model.config.hidden_size != self.hidden_size:
            self.concept_projection = nn.Linear(self.concept_embedding.embedding_model.config.hidden_size, self.hidden_size)
        self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
        # self.position_embedding = RelativePositionalEmbedding(128, self.hidden_size)
        self.time_embedding = TimeEmbedding(1, self.hidden_size)
        self.value_embedding = ValueEmbedding(self.token_type_size, self.hidden_size)
        self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
        # self.continuous_embedding = ContinuousEmbedding(self.continuous_size, self.hidden_size)
        self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
        self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
        self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
        # self.ordername_embedding = OrderCategoryNameEmbedding(self.idx2ordername)
        # self.orderdescription_embedding = OrderCategoryDescriptionEmbedding(self.idx2orderdescription)
        self.ordername_embedding = OrderCategoryNameEmbedding(self.name_size, self.hidden_size)
        self.orderdescription_embedding = OrderCategoryDescriptionEmbedding(self.description_size, self.hidden_size)
        
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.LayerNorm_concept = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.LayerNorm_time = torch.nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=True)
        # self.LayerNorm_position = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.LayerNorm_value = torch.nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=True)
        # self.LayerNorm_unit = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.LayerNorm_ordername = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.LayerNorm_orderdescription = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

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
                # continuous_ids: torch.Tensor,
                position_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                ordername_ids: torch.Tensor,
                orderdescription_ids: torch.Tensor,
                age_ids: torch.Tensor,
                gender_ids: torch.Tensor,
                task_ids: torch.Tensor,
                inputs_embeds=None,
                ):
        
        if inputs_embeds is not None:
            return inputs_embeds
        # concept_embeddiing_start = time.time()
        # concept_embed = self.concept_embedding(input_ids, token_type_ids)
        if input_ids is not None:
            concept_embed = self.concept_embedding(input_ids)
            if self.concept_embedding.embedding_model.config.hidden_size != self.hidden_size:
                concept_embed = self.concept_projection(concept_embed)
            # concept_embed = self.concept_projection(concept_embed)
            # print(concept_embed.shape)
            # concept_embed = self.LayerNorm_concept(concept_embed)
        else:
            concept_embed = torch.zeros((batch_size, seq_len, self.hidden_size), device=self.device)
        # print(f"concept embedding time: {(time.time() - concept_embeddiing_start):.4f}")

        # time_embedding_start = time.time()
        if time_ids is not None:
            time_embed = self.time_embedding(time_ids)
            # time_embed = self.LayerNorm_time(time_embed)
        else:
            time_embed = torch.full_like(concept_embed, -500)
        # print(f"time embedding time: {(time.time() - time_embedding_start):.4f}")

        # position_embedding_start = time.time()
        if position_ids is not None:
            positional_embed = self.position_embedding(position_ids)
            # positional_embed = self.LayerNorm_position(positional_embed)
            # seq_len = positional_embed.size(1)
            # positional_embed = self.position_embedding(seq_len)
        else:
            positional_embed = torch.zeros_like(concept_embed)
        # print(f"position embedding time: {(time.time() - position_embedding_start):.4f}")

        # value_embedding_start = time.time()
        if value_ids is not None and token_type_ids is not None:
            value_embed = self.value_embedding(value_ids, token_type_ids)
            # value_embed = self.LayerNorm_value(value_embed)
        else:
            value_embed = torch.full_like(concept_embed, -500)
        # print(f"value embedding time: {(time.time() - value_embedding_start):.4f}")
        
        # unit_embedding_start = time.time()
        if unit_ids is not None:
            unit_embed = self.unit_embedding(unit_ids)
            # unit_embed = self.LayerNorm_unit(unit_embed)
        else:
            unit_embed = torch.full_like(concept_embed, 4)
        # print(f"unit embedding time: {(time.time() - unit_embedding_start):.4f}")

        # ordername_embedding_start = time.time()
        if ordername_ids is not None:
            ordername_embed = self.ordername_embedding(ordername_ids)
            # ordername_embed = self.LayerNorm_ordername(ordername_embed)
        else:
            ordername_embed = torch.full_like(concept_embed, 4)
        # print(f"ordername embedding time: {(time.time() - ordername_embedding_start):.4f}")

        # orderdescription_embedding_start = time.time()
        if orderdescription_ids is not None:
            orderdescription_embed = self.orderdescription_embedding(orderdescription_ids)
            # orderdescription_embed = self.LayerNorm_orderdescription(orderdescription_embed)
        else:
            orderdescription_embed = torch.full_like(concept_embed, 4)
        # print(f"orderdescription embedding time: {(time.time() - orderdescription_embedding_start):.4f}")
        
        # age_gender_task_embedding_start = time.time()
        age_embed = self.age_embedding(age_ids)
        gender_embed = self.gender_embedding(gender_ids)
        # concept_embed = self.concept_embedding(input_ids)
        # time_embed = self.time_embedding(time_ids)
        # positional_embed = self.position_embedding(position_ids)
        # value_embed = self.value_embedding(value_ids, token_type_ids)
        # unit_embed = self.unit_embedding(unit_ids)
        # ordername_embed = self.ordername_embedding(ordername_ids)
        # orderdescription_embed = self.orderdescription_embedding(orderdescription_ids)
               
        task_embed = self.task_embedding(task_ids)
        # print(f"age gender task embedding time: {(time.time() - age_gender_task_embedding_start):.4f}")
        
        # combeined_embedding_start = time.time()
        embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed + ordername_embed + orderdescription_embed
        
        combined_embed = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)
        # if torch.isinf(combined_embed).any():
        #     print("combined embed inf ok")
        #     print(torch.max(combined_embed), torch.min(combined_embed))
        combined_embed = self.LayerNorm(combined_embed)
        # if torch.isnan(combined_embed).any():
        #     print("embedding layernorm nan ok")
        #     print(torch.max(combined_embed), torch.min(combined_embed))
        # if torch.isnan(combined_embed).any() or torch.isinf(combined_embed).any():
        #     print("NaN or Inf detected in combined_embed!")
        #     print(combined_embed)

        combined_embed = self.dropout(combined_embed)
        # print(f"combined embedding time: {(time.time() - combeined_embedding_start):.4f}")
        # if torch.isnan(combined_embed).any():
        #     print("embedding dropout nan ok")
        
        # self.clear_cache()
        
        # print(combined_embed.shape)
        return combined_embed
    

        
                
                