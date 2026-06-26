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


class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_frequencies=64):
        super(TimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        
        self.w = nn.Parameter(torch.randn(num_frequencies))
        self.b = nn.Parameter(torch.randn(num_frequencies))
        
        self.projection = nn.Linear(num_frequencies + 1, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        
        linear_out = self.w0 * x + self.b0
        
        periodic_out = torch.sin(self.w * x + self.b)
        
        combined = torch.cat([linear_out, periodic_out], dim=-1)
        
        output = self.projection(combined)
        
        return output
    
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

    

class ContinuousValueEmbedding(nn.Module):
    def __init__(self, embedding_size, device='cuda'):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.device = device
        
        self.value_transform = nn.Sequential(
            nn.Linear(1, embedding_size * 2),  
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
        ).to(device)
        
        self.raw_value_projection = nn.Linear(1, embedding_size).to(device)
        
        self.log_scale = nn.Parameter(torch.ones(1, device=device))
        self.log_offset = nn.Parameter(torch.ones(1, device=device) * 1e-8)
        self.log_projection = nn.Linear(1, embedding_size).to(device)
        
        self.gate = nn.Linear(embedding_size * 3, 3).to(device)
             
        self.final_projection = nn.Linear(embedding_size, embedding_size).to(device)
        
        self.LayerNorm = torch.nn.LayerNorm(embedding_size).to(device)

        
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.value_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.raw_value_projection.weight)
        nn.init.zeros_(self.raw_value_projection.bias)
        
        nn.init.xavier_uniform_(self.log_projection.weight)
        nn.init.zeros_(self.log_projection.bias)
        
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
    def forward(self, values):
        
        values = values.to(torch.float32).to(self.device).unsqueeze(-1) 
        
        raw_value_embeds = self.raw_value_projection(values)
        
        transformed_values = self.value_transform(values)     
        
        log_values = torch.sign(values) * torch.log1p(torch.abs(values) + self.log_offset) * self.log_scale
        log_embeds = self.log_projection(log_values)
        
        combined_embeds = torch.cat([transformed_values, raw_value_embeds, log_embeds], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined_embeds))
        
        gated_output = (
            gate_weights[:, :, 0:1] * transformed_values +
            gate_weights[:, :, 1:2] * raw_value_embeds +
            gate_weights[:, :, 2:3] * log_embeds
        )
        
        output = self.final_projection(gated_output)
        output = output.to(torch.float32).to(self.device)
        return self.LayerNorm(output)
    

class UnitEmbedding(nn.Module):
    
    def __init__(self, unit_size, embedding_size):
        super().__init__()
        self.unit_embedding = nn.Embedding(unit_size, embedding_size)
    
    def forward(self, x):
        return self.unit_embedding(x)
    

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, embedding_size: int):
        super(PositionalEmbedding, self).__init__()
        
        # 학습 가능한 nn.Embedding 레이어 사용
        self.position_embeddings = nn.Embedding(max_len, embedding_size)

    def forward(self, x):
        return self.position_embeddings(x.long())


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
    
class EHRtokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
    def forward(self, x):
        return self.embedding(x)
    
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
        

    

        
class EHREmbedding(nn.Module):
    def __init__(self, 
                 config: LongformerConfig, 
                 itemid_size, 
                 unit_size,
                 max_age, 
                 max_len, 
                 gender_size, 
                 task_size, 
                #  idx2label, #########
                 name_size,
                 description_size,
                 token_type_size,
                #  ablation=None,
                 padding_idx=0, 
                 use_itemid=True, 
                 inputs_embeds=None,
                 args=None,):
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
        # self.idx2label = idx2label #####
        self.name_size = name_size
        self.description_size = description_size
        self.token_type_size = token_type_size
        self.value_embedding_type = args.value_embedding_type
        # self.ablation = ablation

        
        self.concept_embedding = EHRtokenEmbedding(self.itemid_size, self.hidden_size) ####
        self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
        self.time_embedding = TimeEmbedding(1, self.hidden_size)
        if self.value_embedding_type == "simple":
            self.value_embedding = ValueEmbedding(self.hidden_size)
        else:
            self.value_embedding = ContinuousValueEmbedding(self.hidden_size)
        self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
        self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
        self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
        self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
        self.ordername_embedding = OrderCategoryNameEmbedding(self.name_size, self.hidden_size)
        self.orderdescription_embedding = OrderCategoryDescriptionEmbedding(self.description_size, self.hidden_size)
        
    
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

        if input_ids is not None:
            concept_embed = self.concept_embedding(input_ids)
        else:
            concept_embed = torch.zeros((batch_size, seq_len, self.hidden_size), device=self.device)

        if time_ids is not None:
            time_embed = self.time_embedding(time_ids)
            # time_embed = self.LayerNorm_time(time_embed)
        else:
            time_embed = torch.full_like(concept_embed, -150)

        if position_ids is not None:
            positional_embed = self.position_embedding(position_ids)

        else:
            positional_embed = torch.zeros_like(concept_embed)

        if value_ids is not None:
            value_embed = self.value_embedding(value_ids)
        else:
            value_embed = torch.full_like(concept_embed, -150)

        if unit_ids is not None:
            unit_embed = self.unit_embedding(unit_ids)

        else:
            unit_embed = torch.full_like(concept_embed, 4)
   
        if ordername_ids is not None:
            ordername_embed = self.ordername_embedding(ordername_ids)

        else:
            ordername_embed = torch.full_like(concept_embed, 4)

        if orderdescription_ids is not None:
            orderdescription_embed = self.orderdescription_embedding(orderdescription_ids)
        else:
            orderdescription_embed = torch.full_like(concept_embed, 4)

        age_embed = self.age_embedding(age_ids)
        gender_embed = self.gender_embedding(gender_ids)

               
        task_embed = self.task_embedding(task_ids)

        embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed + ordername_embed + orderdescription_embed

        combined_embed = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)


        return combined_embed
    

        


        
class EHREmbedding_finetune(nn.Module):
    def __init__(self, 
                 config: LongformerConfig, 
                 itemid_size, 
                 unit_size,
                 max_age, 
                 max_len, 
                 gender_size, 
                 task_size, 
                #  idx2label, #########
                 name_size,
                 description_size,
                 token_type_size,
                 ablation=None,
                 padding_idx=0, 
                 use_itemid=True, 
                 inputs_embeds=None,
                 args=None):
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
        # self.idx2label = idx2label #####
        self.name_size = name_size
        self.description_size = description_size
        self.token_type_size = token_type_size
        self.args = args
        if ablation is None:
            self.ablation = []
        elif isinstance(ablation, str):
            self.ablation = ablation.split('+')
        elif isinstance(ablation, list):
            self.ablation = ablation
        else:
            raise ValueError("ablation must be None, a string, or a list of strings.")


        
        self.concept_embedding = EHRtokenEmbedding(self.itemid_size, self.hidden_size) ####
        
        if self.ablation:
            if 'position' not in self.ablation:
                self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
            if 'time' not in self.ablation:
                self.time_embedding = TimeEmbedding(1, self.hidden_size)
            if 'value' not in self.ablation:
                if self.args.value_embedding_type == "simple":
                    print("Using Simple Value Embedding")
                    self.value_embedding = ValueEmbedding(self.hidden_size)
                else:
                    print("Using Continuous Value Embedding")
                    self.value_embedding = ContinuousValueEmbedding(self.hidden_size)
            if 'unit' not in self.ablation:
                self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
            self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
            self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
            self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
            if 'ordername' not in self.ablation:
                self.ordername_embedding = OrderCategoryNameEmbedding(self.name_size, self.hidden_size)
            if 'orderdescription' not in self.ablation:
                self.orderdescription_embedding = OrderCategoryDescriptionEmbedding(self.description_size, self.hidden_size)
            
        elif self.args.selected_data == "benchmark" or self.args.selected_data == "final":
            self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
            self.time_embedding = TimeEmbedding(1, self.hidden_size)
            if self.args.value_embedding_type == "simple":
                print("Using Simple Value Embedding")
                self.value_embedding = ValueEmbedding(self.hidden_size)
            else:
                print("Using Continuous Value Embedding")
                self.value_embedding = ContinuousValueEmbedding(self.hidden_size)
            self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
            self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
            self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
            self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
            self.ordername_embedding = OrderCategoryNameEmbedding(self.name_size, self.hidden_size)
            self.orderdescription_embedding = OrderCategoryDescriptionEmbedding(self.description_size, self.hidden_size)
        else:
            self.position_embedding = PositionalEmbedding(self.max_position_embeddings, self.hidden_size)
            self.time_embedding = TimeEmbedding(1, self.hidden_size)
            if self.args.value_embedding_type == "simple":
                print("Using Simple Value Embedding")
                self.value_embedding = ValueEmbedding(self.hidden_size)
            else:
                print("Using Continuous Value Embedding")
                self.value_embedding = ContinuousValueEmbedding(self.hidden_size)
            self.unit_embedding = UnitEmbedding(self.unit_size, self.hidden_size)
            self.age_embedding = AgeEmbedding(self.max_age, self.hidden_size)
            self.gender_embedding = GenderEmbedding(self.gender_size, self.hidden_size)
            self.task_embedding = TaskEmbedding(self.task_size, self.hidden_size)
        
        
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

        if input_ids is not None:
            concept_embed = self.concept_embedding(input_ids)
        else:
            concept_embed = torch.zeros((batch_size, seq_len, self.hidden_size), device=self.device)

        # if time_ids is not None:
        #     time_embed = self.time_embedding(time_ids)
        #     # time_embed = self.LayerNorm_time(time_embed)
        # else:
        #     time_embed = torch.full_like(concept_embed, -150)

        # if position_ids is not None:
        #     positional_embed = self.position_embedding(position_ids)

        # else:
        #     positional_embed = torch.zeros_like(concept_embed)

        # if value_ids is not None:
        #     value_embed = self.value_embedding(value_ids)
        # else:
        #     value_embed = torch.full_like(concept_embed, -150)

        # if unit_ids is not None:
        #     unit_embed = self.unit_embedding(unit_ids)

        # else:
        #     unit_embed = torch.full_like(concept_embed, 4)
   
        # if ordername_ids is not None:
        #     ordername_embed = self.ordername_embedding(ordername_ids)

        # else:
        #     ordername_embed = torch.full_like(concept_embed, 4)

        # if orderdescription_ids is not None:
        #     orderdescription_embed = self.orderdescription_embedding(orderdescription_ids)
        # else:
        #     orderdescription_embed = torch.full_like(concept_embed, 4)
        
        embeddings_dict = {}
        
        if self.ablation:

            if 'position' not in self.ablation:
                embeddings_dict['position'] = self.position_embedding(position_ids)
            if 'time' not in self.ablation:
                embeddings_dict['time'] = self.time_embedding(time_ids)
            if 'value' not in self.ablation:
                embeddings_dict['value'] = self.value_embedding(value_ids)
            if 'unit' not in self.ablation:
                embeddings_dict['unit'] = self.unit_embedding(unit_ids)
            if 'ordername' not in self.ablation:
                embeddings_dict['ordername'] = self.ordername_embedding(ordername_ids)
            if 'orderdescription' not in self.ablation:
                embeddings_dict['orderdescription'] = self.orderdescription_embedding(orderdescription_ids)
            
            age_embed = self.age_embedding(age_ids)
            gender_embed = self.gender_embedding(gender_ids)

                
            task_embed = self.task_embedding(task_ids)

            embeddings = concept_embed
            for key in ['position', 'time', 'value', 'unit', 'ordername', 'orderdescription']:
                if key in embeddings_dict:
                    embeddings += embeddings_dict[key]
                    
        elif self.args.selected_data == "benchmark" or self.args.selected_data == "final":
            time_embed = self.time_embedding(time_ids)
            positional_embed = self.position_embedding(position_ids)
            value_embed = self.value_embedding(value_ids)
            unit_embed = self.unit_embedding(unit_ids)
            ordername_embed = self.ordername_embedding(ordername_ids)
            orderdescription_embed = self.orderdescription_embedding(orderdescription_ids)
            age_embed = self.age_embedding(age_ids)
            gender_embed = self.gender_embedding(gender_ids)
            task_embed = self.task_embedding(task_ids)
            embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed + ordername_embed + orderdescription_embed
        else:
            time_embed = self.time_embedding(time_ids)
            positional_embed = self.position_embedding(position_ids)
            value_embed = self.value_embedding(value_ids)
            unit_embed = self.unit_embedding(unit_ids)
            age_embed = self.age_embedding(age_ids)
            gender_embed = self.gender_embedding(gender_ids)
            task_embed = self.task_embedding(task_ids)
            embeddings = concept_embed + time_embed + positional_embed + value_embed + unit_embed

        combined_embed = torch.cat((task_embed, age_embed, gender_embed, embeddings), dim=1)


        return combined_embed
    

        
                
                