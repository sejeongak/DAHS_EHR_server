# Pure python

import os
from pathlib import Path

# Local

# PyTorch
import torch
from torch.utils.data import Dataset

# ETC
import numpy as np
import pandas as pd
import random
import time


# class EHRDataset(Dataset):
#     def __init__(self, data_path: Path, split: str, token2code, age2idx, offset2idx, max_len):
#         pickle_path = "datasets/train_data.pkl" if split == "train" else "datasets/valid_data.pkl"
#         # pickle_path = data_path / pickle_path
#         self.df = pd.read_pickle(pickle_path)
#         self.vocab = token2code
#         self.age2idx = age2idx
#         self.offset2idx = offset2idx
#         self.max_len = max_len
         
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         data = self.df[idx]
#         stayid = data['stay_id']
#         state = data['state']
#         offset = data['offset_data']
#         code = data['code_token']
#         age = data['age']
#         gender = data['gender']


#         code = np.append(np.array(['CLS']), code)
#         age = np.append(np.array(age[0]), age)
#         offset = np.append(np.array(offset[0]), offset)
#         gender = np.append(np.array(gender[0]), gender)
  
        
#         # mask 0: len(code) to 1, padding to be 0
#         mask = np.ones(self.max_len)
#         mask[len(code):] = 0
        
#         # pad age sequence and code sequence
#         age = seq_padding(age, self.max_len, token2code = self.age2idx)
#         offset = seq_padding(offset, self.max_len, token2code = self.offset2idx)
        
        
        
        
#         tokens, code, label = random_mask(code, self.vocab)
#         # get segment code
#         tokens = seq_padding(tokens, self.max_len, token2code=self.vocab)
#         position = position_idx(tokens)
#         segment = index_seg(offset)
#         gender = gender_idx(gender, self.max_len)
        
#         # pad code and label
#         code = seq_padding(code, self.max_len, symbol = self.vocab['PAD'])
#         label = seq_padding(label, self.max_len, symbol = -1)
      
#         # return len(age), len(code), len(offset), len(segment), len(mask), len(label), len([state]), len(gender), len(position)
        
#         return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(offset), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor([state]), torch.LongTensor(gender), torch.LongTensor(position)

TASK_TO_INDEX = {
    "pretrain": 0,
    "mortality_30days":1,
    "mortality_inhospital":2,
    "mortality_icu": 3,
    "mortality48hr": 4,
    "los_3days": 5,
    "los_7days": 6,
    "readmission_30": 7,
    "transfusion_12hr": 8,
    "vasopressor_need_12hr": 9,
    "ventilation_need_12hr": 10,
    "shock_8hr": 11,
    'sofa_centralnervous_24hr': 12,
    'sofa_cardiovascular_24hr': 13,
    'sofa_respiratory_24hr': 14,
    'sofa_coagulation_24hr': 15,
    'sofa_liver_24hr': 16,
    'sofa_renal_24hr': 17,
    "phenotype": 18,
    "multitask": 19
}


class EHR_Longformer_Dataset(Dataset):
    def __init__(self, data_path: Path, split: str, tokenizer, itemid2idx, unit2idx, vocab_size, block_size=17, max_legnth = 4093, use_itemid=False, mask_token=1, mode="pretrain", mask_mode='mlm', mask_ratio=[0.3, 0.15, 0.3], value_mask_ratio=0.15, task=None, seed=None, ablation=None, window=None, index=0, no_gap=False, ratio=None, selected_data=None, locate=None):
      
        # if mode == 'pretrain':
        #     pickle_path = f"{mode}_train_token.pkl" if split == "train" else f"{mode}_valid_token.pkl"
        #     pickle_path = f"{mode}_test_token.pkl" if split == "test" else pickle_path
        #     pickle_path = f"{data_path}/{pickle_path}"
        #     self.df = pd.read_pickle(pickle_path)
        # else:
        #     if split == 'train':
        #         pickle_path = f"{data_path}/{mode}_train_token.pkl" 
        #         self.df = pd.read_pickle(pickle_path)
        #     else:
        #         valid_pickle_path = f"{data_path}/{mode}_valid_token.pkl"
        #         test_pickle_path = f"{data_path}/{mode}_test_token.pkl"
        #         valid_df = pd.read_pickle(valid_pickle_path)
        #         test_df = pd.read_pickle(test_pickle_path)
        #         combined_df = {**valid_df, **test_df}
        #         self.df = combined_df
        
        if mode == "pretrain":
            pickle_path = f"{mode}_train.pkl" if split == "train" else f"{mode}_test.pkl"
            suffix = ""
        else:
            # suffix = "selected" if selected_data == "selected" else ""
            # suffix = "final" if selected_data == "final" else suffix
            # suffix = "hirid" if selected_data == "hirid" else suffix
            suffix = selected_data if selected_data is not None else ""
            locate_suffix = f"_{locate}" if locate is not None else ""
            
            
            
            if window is not None:
                if split == "train":
                    if ratio is not None:
                        if ratio != 100:
                            pickle_path = f"{mode}_train_{window}_{ratio}_fold{index}_{suffix}{locate_suffix}.pkl"
                        else:
                            pickle_path = f"{mode}_train_{window}_fold{index}_{suffix}{locate_suffix}.pkl"
                    else:
                        pickle_path = f"{mode}_train_{window}_fold{index}_{suffix}{locate_suffix}.pkl"
                elif split == "valid":
                    if ratio is not None:
                        if ratio != 100:
                            pickle_path = f"{mode}_val_{window}_{ratio}_{suffix}{locate_suffix}.pkl"
                        else:
                            pickle_path = f"{mode}_val_{window}_{suffix}{locate_suffix}.pkl"
                    pickle_path = f"{mode}_val_{window}_fold{index}_{suffix}{locate_suffix}.pkl"
                elif split == "test":
                    pickle_path = f"{mode}_test_{window}_{suffix}{locate_suffix}.pkl"
                    
            # elif suffix == "benchmark":
            #     if split == "train":
            #         pickle_path = f"{mode}_train_{window}_{suffix}{locate_suffix}.pkl"
            #     elif split == "valid":
            #         pickle_path = f"{mode}_val_{window}_{suffix}{locate_suffix}.pkl"
            #     elif split == "test":
            #         pickle_path = f"{mode}_test_{window}_{suffix}{locate_suffix}.pkl"
            else:
                if split == "train":
                    pickle_path = f"{mode}_train_{window}_fold{index}_{suffix}{locate_suffix}.pkl"
                elif split == "valid":
                    pickle_path = f"{mode}_val_{window}_fold{index}_{suffix}{locate_suffix}.pkl"
                elif split == "test":
                    pickle_path = f"{mode}_test_{window}_{suffix}{locate_suffix}.pkl"

            print(f"window: {window}, ratio: {ratio}, fold: {index}, suffix: {suffix}, locate: {locate}")
        
        pickle_path = f"{data_path}/{pickle_path}"
        self.df = pd.read_pickle(pickle_path)   
        
        # if mode == "finetune":
        #     if 33006228 in self.df:
        #         del self.df[33006228]
        #         print("Removed itemid 33006228 from the dataset.")
            
            
        self.tokenizer = tokenizer
        self.itemid2idx = itemid2idx
        self.unit2idx = unit2idx
        self.block_size = block_size
        self.keys = list(self.df.keys())
        self.max_length = max_legnth
        self.mask_ratio = mask_ratio
        self.value_mask_ratio = value_mask_ratio
        self.use_itemid = use_itemid
        self.mask_token = mask_token
        self.mode = mode
        self.vocab_size = vocab_size
        self.mask_mode = mask_mode
        self.task = task
        self.ablation = ablation
        self.window = window
        self.no_gap = no_gap
        self.suffix = suffix

        # CLS 토큰 ID 가져오기
        # self.cls_token_id = self.tokenizer.cls_token_id
        if mode == "pretrain":
            self.cls_token_id = TASK_TO_INDEX[mode]
        else:
            # self.cls_token_id = TASK_TO_INDEX[task]
            self.cls_token_id = TASK_TO_INDEX[task]
            
        # if split == "train" and mode == "mortality" and undersample:
        #     self.keys = self.undersample_keys()
        
        # self.masking_function = {
        #     "mlm": self.mask_tokens,
        #     "mlm+discriminator": self.mask_tokens_with_discriminator,
        #     "span_mlm": self.spanmask_tokens,
        #     "span_mlm+discriminator": self.spanmask_tokens_with_discriminator
        # }
            
    # def mask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, attention_mask, mask_ratio, mask_token=-150, mask_label_token=4, ignore_token=-100):
    # # def mask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type,  mask_ratio, mask_token=-351):
    #     token_length = attention_mask.sum().item()
    #     num_tokens_to_mask = int(mask_ratio * token_length)
        
    #     valid_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
        
    #     num_tokens_to_mask = min(num_tokens_to_mask, len(valid_indices))
    #     mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_tokens_to_mask]]
        
    #     tokenized_token = tokenized_token.clone()
    #     tokenized_units = tokenized_units.clone()
    #     tokenized_values = tokenized_values.clone()
    #     tokenized_offsets = tokenized_offsets.clone()
    #     tokenized_token_type = tokenized_token_type.clone()
    #     tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
    #     tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()
        
    #     labels = tokenized_token.clone()
    #     mask_labels = torch.full_like(tokenized_token, ignore_token)
    #     rand_probs = torch.rand(len(mask_indices))
        
    #     mask_condition = rand_probs < 0.8
    #     tokenized_token[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_units[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_values[mask_indices[mask_condition]] = mask_token
    #     tokenized_offsets[mask_indices[mask_condition]] = mask_token
    #     tokenized_token_type[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[mask_condition]] = mask_label_token
    #     mask_labels[mask_indices[mask_condition]] = labels[mask_indices[mask_condition]]
                    
    #     random_condition = (rand_probs >= 0.8) & (rand_probs < 0.9)
    #     tokenized_token[mask_indices[random_condition]] = torch.randint(0, self.vocab_size, (random_condition.sum().item(),))
    #     tokenized_units[mask_indices[random_condition]] = mask_label_token
    #     tokenized_values[mask_indices[random_condition]] = mask_token
    #     tokenized_offsets[mask_indices[random_condition]] = mask_token
    #     tokenized_token_type[mask_indices[random_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[random_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[random_condition]] = mask_label_token

    #     return (
    #     tokenized_token,
    #     tokenized_units,
    #     tokenized_values,
    #     tokenized_offsets,
    #     tokenized_token_type,
    #     tokenized_ordercategoryname,
    #     tokenized_ordercategorydescription,
    #     mask_labels,
    # )
        
    
    def mask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets,
                tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription,
                attention_mask, mask_ratio, value_mask_ratio,
                mask_token=-150, mask_label_token=4, ignore_token=-100):

        # Clone tensors to avoid modifying in-place
        tokenized_token = tokenized_token.clone()
        tokenized_units = tokenized_units.clone()
        tokenized_values = tokenized_values.clone()
        tokenized_offsets = tokenized_offsets.clone()
        tokenized_token_type = tokenized_token_type.clone()
        tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
        tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()
        
        labels = tokenized_token.clone()
        mask_labels = torch.full_like(tokenized_token, ignore_token)

        attention_mask = attention_mask[:tokenized_token_type.size(0)]
                
        
        for idx, event_type in enumerate([0, 1, 2]):
            indices = torch.where((tokenized_token_type == event_type) & (attention_mask == 1))[0]
            if len(indices) == 0:
                continue

            num_to_mask = int(mask_ratio[idx] * len(indices))
            selected_indices = indices[torch.randperm(len(indices))[:num_to_mask]]
            
            # Shuffle again for 80/10/10 split
            perm = torch.randperm(len(selected_indices))
            selected_indices = selected_indices[perm]

            n = len(selected_indices)
            num_80 = int(n * 0.8)
            num_10 = int(n * 0.1)

            mask_80 = selected_indices[:num_80]
            rand_10 = selected_indices[num_80:num_80 + num_10]
            keep_10 = selected_indices[num_80 + num_10:]

            # 80% -> mask
            tokenized_token[mask_80] = mask_label_token
            tokenized_units[mask_80] = mask_label_token
            tokenized_values[mask_80] = mask_token
            tokenized_offsets[mask_80] = mask_token
            # tokenized_token_type[mask_80] = mask_label_token
            tokenized_ordercategoryname[mask_80] = mask_label_token
            tokenized_ordercategorydescription[mask_80] = mask_label_token
            mask_labels[mask_80] = labels[mask_80]

            # 10% -> random token
            tokenized_token[rand_10] = torch.randint(0, 3892, (len(rand_10),), device=tokenized_token.device)
            tokenized_units[rand_10] = mask_label_token
            tokenized_values[rand_10] = mask_token
            tokenized_offsets[rand_10] = mask_token
            # tokenized_token_type[rand_10] = mask_label_token
            tokenized_ordercategoryname[rand_10] = mask_label_token
            tokenized_ordercategorydescription[rand_10] = mask_label_token
            mask_labels[rand_10] = labels[rand_10]

            # 10% -> keep (label만 학습)
            mask_labels[keep_10] = labels[keep_10]

        # Value prediction masking (별도 처리)
        value_target_ids = torch.tensor([7, 11, 12, 33, 41, 342, 142, 161, 2208, 2209, 37], device=tokenized_token.device)
        
        value_labels = torch.full_like(tokenized_values, ignore_token, dtype=torch.float32)
        
        if value_mask_ratio != 0:
        
            for item_id in value_target_ids:
                item_mask_candidate = torch.where(
                    (tokenized_token == item_id) &
                    (mask_labels == ignore_token) &
                    (attention_mask == 1)
                )[0]

                num_value_to_mask = int(value_mask_ratio * len(item_mask_candidate))
                if num_value_to_mask > 0:
                    selected_value_indices = item_mask_candidate[torch.randperm(len(item_mask_candidate))[:num_value_to_mask]]
                    value_labels[selected_value_indices] = tokenized_values[selected_value_indices]
                    tokenized_values[selected_value_indices] = mask_token
        

        return (
            tokenized_token,
            tokenized_units,
            tokenized_values,
            tokenized_offsets,
            tokenized_token_type,
            tokenized_ordercategoryname,
            tokenized_ordercategorydescription,
            mask_labels,
            value_labels
        )
        
    # def mask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, 
    #             tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, 
    #             attention_mask, mask_ratio, mask_token=-500, mask_label_token=4, ignore_token=-100):
    
    #     token_length = tokenized_token.shape[0]
    #     print(mask_ratio, token_length)
    #     num_tokens_to_mask = int(mask_ratio * token_length)

 
    #     mask_indices = torch.randperm(token_length, device=tokenized_token.device)[:num_tokens_to_mask]

    #     tokenized_token = tokenized_token.clone()
    #     tokenized_units = tokenized_units.clone()
    #     tokenized_values = tokenized_values.clone()
    #     tokenized_offsets = tokenized_offsets.clone()
    #     tokenized_token_type = tokenized_token_type.clone()
    #     tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
    #     tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()

    #     random_probs = torch.rand(num_tokens_to_mask, device=tokenized_token.device)


    #     masking_condition = random_probs < 0.8
    #     random_replacement_condition = (random_probs >= 0.8) & (random_probs < 0.9)

    #     tokenized_token[mask_indices[masking_condition]] = mask_label_token
    #     tokenized_units[mask_indices[masking_condition]] = mask_label_token
    #     tokenized_values[mask_indices[masking_condition]] = mask_token
    #     tokenized_offsets[mask_indices[masking_condition]] = mask_token
    #     tokenized_token_type[mask_indices[masking_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[masking_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[masking_condition]] = mask_label_token

    #     tokenized_token[mask_indices[random_replacement_condition]] = torch.randint(0, self.vocab_size, (random_replacement_condition.sum(),), device=tokenized_token.device)

    #     mask_labels = torch.full(tokenized_token.shape, ignore_token, device=tokenized_token.device)
    #     mask_labels[mask_indices[masking_condition]] = tokenized_token[mask_indices[masking_condition]]

    #     return tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, mask_labels
    
    # def mask_tokens_with_discriminator(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, attention_mask, mask_ratio, mask_token=-500, mask_label_token=4, ignore_token=-100):
    # # def mask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type,  mask_ratio, mask_token=-351):
    #     token_length = attention_mask.sum().item()
    #     num_tokens_to_mask = int(mask_ratio * token_length)
        
    #     valid_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
        
    #     num_tokens_to_mask = min(num_tokens_to_mask, len(valid_indices))
    #     mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_tokens_to_mask]]
        
    #     tokenized_token = tokenized_token.clone()
    #     tokenized_units = tokenized_units.clone()
    #     tokenized_values = tokenized_values.clone()
    #     tokenized_offsets = tokenized_offsets.clone()
    #     tokenized_token_type = tokenized_token_type.clone()
    #     tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
    #     tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()
        
    #     labels = tokenized_token.clone()
    #     mask_labels = torch.full(tokenized_token.shape, ignore_token)
    #     discriminator_labels = torch.full(tokenized_token.shape, ignore_token)    
        
    #     rand_probs = torch.rand(len(mask_indices))
        
    #     mask_condition = rand_probs < 0.8
    #     tokenized_token[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_units[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_values[mask_indices[mask_condition]] = mask_token
    #     tokenized_offsets[mask_indices[mask_condition]] = mask_token
    #     tokenized_token_type[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[mask_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[mask_condition]] = mask_label_token
    #     mask_labels[mask_indices[mask_condition]] = labels[mask_indices[mask_condition]]
                
    #     random_condition = (rand_probs >= 0.8) & (rand_probs < 0.9)
    #     tokenized_token[mask_indices[random_condition]] = torch.randint(
    #         0, self.vocab_size, (random_condition.sum().item(),)
    #     )
    #     tokenized_units[mask_indices[random_condition]] = mask_label_token
    #     tokenized_values[mask_indices[random_condition]] = mask_token
    #     tokenized_offsets[mask_indices[random_condition]] = mask_token
    #     tokenized_token_type[mask_indices[random_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[random_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[random_condition]] = mask_label_token
    #     discriminator_labels[mask_indices[random_condition]] = 1
        
    #     keep_condition = rand_probs >= 0.9
    #     tokenized_units[mask_indices[keep_condition]] = mask_label_token
    #     tokenized_values[mask_indices[keep_condition]] = mask_token
    #     tokenized_offsets[mask_indices[keep_condition]] = mask_token
    #     tokenized_token_type[mask_indices[keep_condition]] = mask_label_token
    #     tokenized_ordercategoryname[mask_indices[keep_condition]] = mask_label_token
    #     tokenized_ordercategorydescription[mask_indices[keep_condition]] = mask_label_token
    #     discriminator_labels[mask_indices[keep_condition]] = 0
        
    #     return (
    #     tokenized_token,
    #     tokenized_units,
    #     tokenized_values,
    #     tokenized_offsets,
    #     tokenized_token_type,
    #     tokenized_ordercategoryname,
    #     tokenized_ordercategorydescription,
    #     mask_labels,
    #     discriminator_labels,
    # )
    
    # def sample_span_length(self):
    #     return random.randint(1, 3)
    
    # def spanmask_tokens(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, attention_mask, mask_ratio, mask_token=-500, mask_label_token=4, ignore_token=-100):
    #     token_length = attention_mask.sum().item()
    #     num_tokens_to_mask = int(mask_ratio * token_length)
    #     num_span_mask_tokens = int(num_tokens_to_mask * 0.6)
    #     num_remain_tokens = num_tokens_to_mask - num_span_mask_tokens
        
    #     valid_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
        
    #     tokenized_token = tokenized_token.clone()
    #     tokenized_units = tokenized_units.clone()
    #     tokenized_values = tokenized_values.clone()
    #     tokenized_offsets = tokenized_offsets.clone()
    #     tokenized_token_type = tokenized_token_type.clone()
    #     tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
    #     tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()    
              
    #     mask_labels = torch.full_like(tokenized_token, ignore_token)
    #     labels = tokenized_token.clone()
    #     span_mask_indices = torch.tensor([], dtype=torch.long, device=valid_indices.device)
    #     if num_span_mask_tokens > 0 and len(valid_indices) > 0:
    #         span_start_indices = valid_indices[torch.randperm(len(valid_indices))[:num_span_mask_tokens]]
    #         span_lengths = torch.randint(1, 6, (num_span_mask_tokens,), device=tokenized_token.device)

    #         span_indices = []
    #         for i in range(len(span_start_indices)):
    #             start = span_start_indices[i]
    #             length = span_lengths[i]
    #             span = torch.arange(start, start + length, device=tokenized_token.device)
    #             span = span[span < len(tokenized_token)]  # 오버플로우 방지
    #             span_indices.append(span)
            
    #         if span_indices:
    #             span_mask_indices = torch.cat(span_indices)

    #         # 마스킹 적용
    #         tokenized_token[span_mask_indices] = mask_label_token
    #         tokenized_units[span_mask_indices] = mask_label_token
    #         tokenized_values[span_mask_indices] = mask_token
    #         tokenized_offsets[span_mask_indices] = mask_token
    #         tokenized_token_type[span_mask_indices] = mask_label_token
    #         tokenized_ordercategoryname[span_mask_indices] = mask_label_token
    #         tokenized_ordercategorydescription[span_mask_indices] = mask_label_token
    #         mask_labels[span_mask_indices] = labels[span_mask_indices]
            
    #     remaining_indices = valid_indices[~torch.isin(valid_indices, span_mask_indices)]
    #     if num_remain_tokens > 0 and len(remaining_indices) > 0:
    #         remain_mask_indices = remaining_indices[torch.randperm(len(remaining_indices))[:num_remain_tokens]]
    #         rand_probs = torch.rand(len(remain_mask_indices))
            
    #         random_condition = rand_probs < 0.5
    #         tokenized_token[remain_mask_indices[random_condition]] = torch.randint(0, self.vocab_size, (random_condition.sum().item(),))

    #     return (
    #     tokenized_token,
    #     tokenized_units,
    #     tokenized_values,
    #     tokenized_offsets,
    #     tokenized_token_type,
    #     tokenized_ordercategoryname,
    #     tokenized_ordercategorydescription,
    #     mask_labels,
    # )
    
    # def spanmask_tokens_with_discriminator(self, tokenized_token, tokenized_units, tokenized_values, tokenized_offsets, tokenized_token_type, tokenized_ordercategoryname, tokenized_ordercategorydescription, attention_mask, mask_ratio, mask_token=-500, mask_label_token=4, ignore_token=-100):
    #     token_length = attention_mask.sum().item()
    #     num_tokens_to_mask = int(mask_ratio * token_length)
    #     num_span_mask_tokens = int(num_tokens_to_mask * 0.6)
    #     num_discriminator_tokens = num_tokens_to_mask - num_span_mask_tokens
        
    #     valid_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
        
    #     tokenized_token = tokenized_token.clone()
    #     tokenized_units = tokenized_units.clone()
    #     tokenized_values = tokenized_values.clone()
    #     tokenized_offsets = tokenized_offsets.clone()
    #     tokenized_token_type = tokenized_token_type.clone()
    #     tokenized_ordercategoryname = tokenized_ordercategoryname.clone()
    #     tokenized_ordercategorydescription = tokenized_ordercategorydescription.clone()    
        
    #     mask_labels = torch.full_like(tokenized_token, ignore_token)
    #     discriminator_labels = torch.full_like(tokenized_token, ignore_token)
    #     labels = tokenized_token.clone()
        
    #     if num_span_mask_tokens > 0:
    #         span_start_indices = valid_indices[torch.randperm(len(valid_indices))[:num_span_mask_tokens]]
    #         span_lengths = torch.randint(1, 6, (num_span_mask_tokens,))
    #         span_mask_indices = torch.cat([span_start_indices + i  for i in range(span_lengths.max()) if i < len(valid_indices)])
            
    #         span_mask_indices = span_mask_indices[span_mask_indices < len(tokenized_token)]
            
    #         tokenized_token[span_mask_indices] = mask_label_token
    #         tokenized_units[span_mask_indices] = mask_label_token
    #         tokenized_values[span_mask_indices] = mask_token
    #         tokenized_offsets[span_mask_indices] = mask_token
    #         tokenized_token_type[span_mask_indices] = mask_label_token
    #         tokenized_ordercategoryname[span_mask_indices] = mask_label_token
    #         tokenized_ordercategorydescription[span_mask_indices] = mask_label_token
    #         mask_labels[span_mask_indices] = labels[span_mask_indices]
                
    #     remaining_indices = valid_indices[~torch.isin(valid_indices, span_mask_indices)]
    #     if num_discriminator_tokens > 0 and len(remaining_indices) > 0:
    #         disc_indices = remaining_indices[torch.randperm(len(remaining_indices))[:num_discriminator_tokens]]

    #         rand_probs = torch.rand(len(disc_indices))  # 0~1 사이 랜덤 값 생성

    #         # 50%: 랜덤 토큰 변경 (디스크리미네이터에 1 할당)
    #         random_condition = rand_probs < 0.5
    #         tokenized_token[disc_indices[random_condition]] = torch.randint(
    #             0, self.vocab_size, (random_condition.sum().item(),)
    #         )
    #         discriminator_labels[disc_indices[random_condition]] = 1

    #         # 50%: 원본 유지 (디스크리미네이터에 0 할당)
    #         discriminator_labels[disc_indices[~random_condition]] = 0

    #         # 동일한 마스킹 처리
    #         tokenized_units[disc_indices] = mask_label_token
    #         tokenized_values[disc_indices] = mask_token
    #         tokenized_offsets[disc_indices] = mask_token
    #         tokenized_token_type[disc_indices] = mask_label_token
    #         tokenized_ordercategoryname[disc_indices] = mask_label_token
    #         tokenized_ordercategorydescription[disc_indices] = mask_label_token
    #     return (
    #     tokenized_token,
    #     tokenized_units,
    #     tokenized_values,
    #     tokenized_offsets,
    #     tokenized_token_type,
    #     tokenized_ordercategoryname,
    #     tokenized_ordercategorydescription,
    #     mask_labels,
    #     discriminator_labels,
    # )
    
    def __len__(self):
        return len(self.keys)
    
    
    
    def __getitem__(self, idx):
        
        def safe_tensor(seq, dtype, fill_value=3):
            clean_seq = [fill_value if pd.isna(x) else x for x in seq]
            return torch.tensor(clean_seq, dtype=dtype)
        
        key = self.keys[idx]
        data = self.df[key]
        stayid = data['stay_id']
        
        
        # try:
        #     seq_keys = [
        #         "token_type_seq",
        #         "ordercategoryname_seq",
        #         "ordercategorydescription_seq"
        #     ]
            
        #     for seq_key in seq_keys:
        #         seq = data.get(seq_key, None)
        #         if seq is None:
        #             print(f"\n?? {seq_key} missing for stay_id {stayid}, key {key}")
        #             continue
        #         if any(pd.isna(x) for x in seq):
        #             print(f"\n? NA detected in {seq_key} for stay_id {stayid}, key {key}")
        #             na_positions = [i for i, v in enumerate(seq) if pd.isna(v)]
        #             print(f"Positions: {na_positions}")
        #             print(f"Example values: {[seq[i] for i in na_positions[:5]]}")
                    
        #             for i in na_positions:
        #                 print("label_idx:", data['ehr_seq'][i])
                    

        unit = torch.tensor(data['unit_seq'], dtype=torch.int64)
        value = torch.tensor(data['value_seq'], dtype=torch.float32)
        offset = torch.tensor(data['offset_seq'], dtype=torch.float32)
        position = torch.tensor(data['position_seq'])
        token_type = safe_tensor(data['token_type_seq'], torch.int64, fill_value=3)
        ordercategoryname = safe_tensor(data['ordercategoryname_seq'], torch.int64, fill_value=3)
        ordercategorydescription = safe_tensor(data['ordercategorydescription_seq'], torch.int64, fill_value=3)
        
        # except Exception as e:
        #     print("\n? Error occurred in __getitem__:")
        #     print(f"  ? Dataset idx: {idx}")
        #     print(f"  ? stay_id: {stayid}")
        #     print(f"  ? key: {key}")
        #     print(f"  ? Error type: {type(e).__name__}")
        #     print(f"  ? Error message: {e}")
        #     print(f"  ? Problematic data['unit_seq']: {data.get('unit_seq', None)}")
        #     raise
        
        age = torch.tensor(data['age'], dtype=torch.int64).unsqueeze(0)
        gender = 0 if data['gender'] == "F" else 1
        gender = torch.tensor(gender, dtype=torch.int64).unsqueeze(0) 
            
         
        # # unit = [self.unit2idx.get(unit.lower(), 1) if unit is not None else 1 for unit in data['unit_seq']]

        # unit = torch.tensor(data['unit_seq'], dtype=torch.int64)
        # value = torch.tensor(data['value_seq'], dtype=torch.float32)
        # offset = torch.tensor(data['offset_seq'], dtype=torch.float32)
        # # offset = torch.tensor(data['offset_hr_seq'], dtype=torch.float32)
        # position = torch.tensor(data['position_seq'])
        # token_type = torch.tensor(data['token_type_seq'])
        # ordercategoryname = torch.tensor(data['ordercategoryname_seq'])
        # ordercategorydescription = torch.tensor(data['ordercategorydescription_seq'])
        
        
        attention_mask = torch.zeros((self.max_length), dtype=torch.long)
        
        
        if not self.use_itemid:
            ehr = data['label']
            encoded_ehr = self.tokenizer(ehr, return_tensors="pt", padding='max_length', truncation=True, max_length=self.block_size)
            tokenized_token = encoded_ehr['input_ids']
            attention_token = encoded_ehr['attention_mask'][:, 0]
            padded_label = torch.zeros((self.max_length, self.block_size), dtype=torch.long)
            padded_label[:tokenized_token.shape[0], :] = tokenized_token
            attention_mask[:attention_token.shape[0]] = attention_token
        else: ## Masking 추가 0: Padding 1: Masking
            ehr = [3 if pd.isna(x) else int(x) for x in data['ehr_seq']]
            ehr = torch.tensor(ehr, dtype=torch.int64)
            # ehr_idx = [self.itemid2idx[item] for item in ehr]
            itemid_tensor = torch.tensor(ehr, dtype=torch.int64)
            attention_mask[:itemid_tensor.shape[0]] = 1
            if self.mode == "pretrain":
                if self.mask_mode == "mlm":
                    mask_token, mask_unit, mask_value, mask_offset, mask_token_type, mask_ordercategoryname, mask_ordercategorydescription, mask_labels, mask_value_labels = self.mask_tokens(
                        itemid_tensor, unit, value, offset, token_type, ordercategoryname, ordercategorydescription, attention_mask, self.mask_ratio, self.value_mask_ratio
                        )
                # elif self.mask_mode == "mlm+discriminator":
                #     mask_token, mask_unit, mask_value, mask_offset, mask_token_type, mask_ordercategoryname, mask_ordercategorydescription, mask_labels, discriminator_labels = self.mask_tokens_with_discriminator(
                #         itemid_tensor, unit, value, offset, token_type, ordercategoryname, ordercategorydescription, attention_mask, self.mask_ratio
                #     )
                # elif self.mask_mode == "span_mlm":
                #     mask_token, mask_unit, mask_value, mask_offset, mask_token_type, mask_ordercategoryname, mask_ordercategorydescription, mask_labels = self.spanmask_tokens(
                #     itemid_tensor, unit, value, offset, token_type,  ordercategoryname, ordercategorydescription,attention_mask, self.mask_ratio
                # )
                # elif self.mask_mode == "span_mlm+discriminator":
                #     mask_token, mask_unit, mask_value, mask_offset, mask_token_type, mask_ordercategoryname, mask_ordercategorydescription, mask_labels, discriminator_labels = self.spanmask_tokens_with_discriminator(
                #     itemid_tensor, unit, value, offset, token_type, ordercategoryname, ordercategorydescription, attention_mask, self.mask_ratio
                # )
                
                padded_ehr = torch.zeros((self.max_length), dtype=torch.long)
                padded_label = torch.full((self.max_length,), -100, dtype=torch.long)
                padded_value_label = torch.full((self.max_length,), -100, dtype=torch.float32)
                padded_value = torch.zeros((self.max_length), dtype=torch.float32)
                padded_unit = torch.zeros((self.max_length), dtype=torch.long)
                padded_offset = torch.zeros((self.max_length), dtype=torch.float32)
                padded_position = torch.zeros((self.max_length), dtype=torch.long)
                padded_token_type = torch.zeros((self.max_length), dtype=torch.long)
                padded_ordercategoryname = torch.zeros((self.max_length), dtype=torch.long)
                padded_ordercategorydescription = torch.zeros((self.max_length), dtype=torch.long)
                


                padded_ehr[:mask_token.shape[0]] = mask_token
                padded_label[:mask_labels.shape[0]] = mask_labels
                padded_value_label[:mask_value_labels.shape[0]] = mask_value_labels

                padded_value[:mask_value.shape[0]] = mask_value
                padded_unit[:mask_unit.shape[0]] = mask_unit
                padded_offset[:mask_offset.shape[0]] = mask_offset
                padded_position[:position.shape[0]] = position
                padded_token_type[:token_type.shape[0]] = mask_token_type 
                padded_ordercategoryname[:ordercategoryname.shape[0]] = mask_ordercategoryname
                padded_ordercategorydescription[:ordercategorydescription.shape[0]] = mask_ordercategorydescription  
                
                # if self.mask_mode == "mlm+discriminator" or self.mask_mode == "span_mlm+discriminator":    
                #     padded_discriminator_labels = torch.full((4093,), -100, dtype=torch.long)
                #     padded_discriminator_labels[:discriminator_labels.shape[0]] = discriminator_labels
                
            else:
                padded_ehr = torch.zeros((self.max_length), dtype=torch.long)
                padded_ehr[:itemid_tensor.shape[0]] = itemid_tensor
                attention_mask[:itemid_tensor.shape[0]] = 1
                
                padded_unit = torch.zeros((self.max_length), dtype=torch.long)
                padded_value = torch.zeros((self.max_length), dtype=torch.float32)
                padded_offset = torch.zeros((self.max_length), dtype=torch.float32)
                padded_position = torch.zeros((self.max_length), dtype=torch.long)
                padded_token_type = torch.zeros((self.max_length), dtype=torch.long)
                padded_ordercategoryname = torch.zeros((self.max_length), dtype=torch.long)
                padded_ordercategorydescription = torch.zeros((self.max_length), dtype=torch.long)
                
                padded_unit[:unit.shape[0]] = unit
                padded_value[:value.shape[0]] = value
                padded_offset[:offset.shape[0]] = offset
                padded_position[:position.shape[0]] = position
                padded_token_type[:token_type.shape[0]] = token_type
                padded_ordercategoryname[:ordercategoryname.shape[0]] = ordercategoryname
                padded_ordercategorydescription[:ordercategorydescription.shape[0]] = ordercategorydescription
                
                # task_label = data[self.task]
                # task_label = torch.tensor(task_label, dtype=torch.int64).unsqueeze(0)
                
                if self.suffix == "hirid":
                    mortality_icu = data['mortality_inicu']
                    los_3days = data['los_3days']
                    los_7days = data['los_7days']
                    transfusion_12hr = data['transfusion_12hr']
                    shock_8hr = data['shock_8hr']
                    vasopressor_need_12hr = data['vasopressor_need_12hr']
                    ventilator_need_12hr = data['ventilator_need_12hr']
                    sofa_centralnervous_24hr = data['sofa_centralnervous_24hr']
                    sofa_cardiovascular_24hr = data['sofa_cardiovascular_24hr']
                    sofa_respiratory_24hr = data['sofa_respiratory_24hr']
                    sofa_coagulation_24hr = data['sofa_coagulation_24hr']
                    sofa_liver_24hr = data['sofa_liver_24hr']
                    sofa_renal_24hr = data['sofa_renal_24hr']
                    
                    task_label = [mortality_icu, los_3days, los_7days, transfusion_12hr, vasopressor_need_12hr, ventilator_need_12hr, shock_8hr, sofa_centralnervous_24hr, sofa_cardiovascular_24hr, sofa_respiratory_24hr, sofa_coagulation_24hr, sofa_liver_24hr, sofa_renal_24hr]
                    task_label = torch.tensor(task_label, dtype=torch.int64)
                    
                elif self.suffix == "P12":
                    if self.window == 24:
                        mortality_inhospital = data['mortality_inhospital']
                        task_label = [mortality_inhospital]
                        # los_3days = data['los_3days']
                        # los_7days = data['los_7days']
                        # ventilator_need_12hr = data['ventilator_need_12hr']
                        
                        # task_label = [mortality_inhospital, los_3days, los_7days, ventilator_need_12hr]
                    elif self.window == 48:
                        mortality_inhospital = data['mortality_inhospital']
                        task_label = [mortality_inhospital]
                        # los_3days = data['los_3days']
                        # los_7days = data['los_7days']
                        
                        # task_label = [mortality_inhospital, los_3days, los_7days]
                    task_label = torch.tensor(task_label, dtype=torch.int64)
                    
                elif self.suffix == "eicu":
                    mortality_icu = data['mortality_inicu']
                    los_3days = data['los_3days']
                    los_7days = data['los_7days']
                    readmission_30 = data['readmission_30days']
                    mortality48hr = data['mortality_48hr']
                    transfusion_12hr = data['transfusion_12hr']
                    shock_8hr = data['shock_8hr']
                    vasopressor_need_12hr = data['vasopressor_need_12hr']
                    ventilator_need_12hr = data['ventilator_need_12hr']
                    sofa_centralnervous_24hr = data['sofa_centralnervous_24hr']
                    sofa_cardiovascular_24hr = data['sofa_cardiovascular_24hr']
                    sofa_respiratory_24hr = data['sofa_respiratory_24hr']
                    sofa_coagulation_24hr = data['sofa_coagulation_24hr']
                    sofa_liver_24hr = data['sofa_liver_24hr']
                    sofa_renal_24hr = data['sofa_renal_24hr']
                    
                    
                    phenotyping_cols = [
                        'Acute and unspecified renal failure',
                        'Acute cerebrovascular disease',
                        'Acute myocardial infarction',
                        'Cardiac dysrhythmias',
                        'Chronic kidney disease',
                        'Chronic obstructive pulmonary disease and bronchiectasis',
                        'Complications of surgical procedures or medical care',
                        'Conduction disorders',
                        'Congestive heart failure; nonhypertensive',
                        'Coronary atherosclerosis and other heart disease',
                        'Diabetes mellitus with complications',
                        'Diabetes mellitus without complication',
                        'Disorders of lipid metabolism',
                        'Essential hypertension',
                        'Fluid and electrolyte disorders',
                        'Gastrointestinal hemorrhage',
                        'Hypertension with complications and secondary hypertension',
                        'Other liver diseases',
                        'Other lower respiratory disease',
                        'Other upper respiratory disease',
                        'Pleurisy; pneumothorax; pulmonary collapse',
                        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                        'Respiratory failure; insufficiency; arrest (adult)',
                        'Septicemia (except in labor)',
                        'Shock'
                    ]
                    task_label = [mortality_icu, mortality48hr, los_3days, los_7days, readmission_30, transfusion_12hr, vasopressor_need_12hr, ventilator_need_12hr, shock_8hr, sofa_centralnervous_24hr, sofa_cardiovascular_24hr, sofa_respiratory_24hr, sofa_coagulation_24hr, sofa_liver_24hr, sofa_renal_24hr]
                    multilabel_label = torch.tensor([data[col] for col in phenotyping_cols], dtype=torch.float32)        
                    task_label = torch.tensor(task_label, dtype=torch.int64)
                    
                elif self.task == "phenotype":
                    # mortality_30 = data['mortality_30days']
                    # readmission_30 = data['readmission_30days']
         
                    phenotyping_cols = [
                        'Acute and unspecified renal failure',
                        'Acute cerebrovascular disease',
                        'Acute myocardial infarction',
                        'Cardiac dysrhythmias',
                        'Chronic kidney disease',
                        'Chronic obstructive pulmonary disease and bronchiectasis',
                        'Complications of surgical procedures or medical care',
                        'Conduction disorders',
                        'Congestive heart failure; nonhypertensive',
                        'Coronary atherosclerosis and other heart disease',
                        'Diabetes mellitus with complications',
                        'Diabetes mellitus without complication',
                        'Disorders of lipid metabolism',
                        'Essential hypertension',
                        'Fluid and electrolyte disorders',
                        'Gastrointestinal hemorrhage',
                        'Hypertension with complications and secondary hypertension',
                        'Other liver diseases',
                        'Other lower respiratory disease',
                        'Other upper respiratory disease',
                        'Pleurisy; pneumothorax; pulmonary collapse',
                        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                        'Respiratory failure; insufficiency; arrest (adult)',
                        'Septicemia (except in labor)',
                        'Shock'
                    ]
                    # task_label = [mortality_30, readmission_30]
                    multilabel_label = torch.tensor([data[col] for col in phenotyping_cols], dtype=torch.float32) 
                        
                elif self.window == "entire":
                    mortality_30 = data['mortality_30days']
                    readmission_30 = data['readmission_30days']
         
                    phenotyping_cols = [
                        'Acute and unspecified renal failure',
                        'Acute cerebrovascular disease',
                        'Acute myocardial infarction',
                        'Cardiac dysrhythmias',
                        'Chronic kidney disease',
                        'Chronic obstructive pulmonary disease and bronchiectasis',
                        'Complications of surgical procedures or medical care',
                        'Conduction disorders',
                        'Congestive heart failure; nonhypertensive',
                        'Coronary atherosclerosis and other heart disease',
                        'Diabetes mellitus with complications',
                        'Diabetes mellitus without complication',
                        'Disorders of lipid metabolism',
                        'Essential hypertension',
                        'Fluid and electrolyte disorders',
                        'Gastrointestinal hemorrhage',
                        'Hypertension with complications and secondary hypertension',
                        'Other liver diseases',
                        'Other lower respiratory disease',
                        'Other upper respiratory disease',
                        'Pleurisy; pneumothorax; pulmonary collapse',
                        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                        'Respiratory failure; insufficiency; arrest (adult)',
                        'Septicemia (except in labor)',
                        'Shock'
                    ]
                    task_label = [mortality_30, readmission_30]
                    task_label = torch.tensor(task_label, dtype=torch.int64)
                    multilabel_label = torch.tensor([data[col] for col in phenotyping_cols], dtype=torch.float32) 
                        
                
                else:
                    mortality_30 = data['mortality_30days']
                    mortality_inhospital = data['mortality_inhospital']
                    mortality_icu = data['mortality_inicu']
                    los_3days = data['los_3days']
                    los_7days = data['los_7days']
                    readmission_30 = data['readmission_30days']

                    if self.no_gap:
                        mortality48hr = data['mortality_48hr_nogap']
                        transfusion_12hr = data['transfusion_12hr_nogap']
                        shock_8hr = data['shock_8hr_nogap']
                        vasopressor_need_12hr = data['vasopressor_need_12hr_nogap']
                        ventilator_need_12hr = data['ventilator_need_12hr_nogap']
                        sofa_centralnervous_24hr = data['sofa_centralnervous_24hr_nogap']
                        sofa_cardiovascular_24hr = data['sofa_cardiovascular_24hr_nogap']
                        sofa_respiratory_24hr = data['sofa_respiratory_24hr_nogap']
                        sofa_coagulation_24hr = data['sofa_coagulation_24hr_nogap']
                        sofa_liver_24hr = data['sofa_liver_24hr_nogap']
                        sofa_renal_24hr = data['sofa_renal_24hr_nogap']
                    else:
                        mortality48hr = data['mortality_48hr']
                        transfusion_12hr = data['transfusion_12hr']
                        shock_8hr = data['shock_8hr']
                        vasopressor_need_12hr = data['vasopressor_need_12hr']
                        ventilator_need_12hr = data['ventilator_need_12hr']
                        sofa_centralnervous_24hr = data['sofa_centralnervous_24hr']
                        sofa_cardiovascular_24hr = data['sofa_cardiovascular_24hr']
                        sofa_respiratory_24hr = data['sofa_respiratory_24hr']
                        sofa_coagulation_24hr = data['sofa_coagulation_24hr']
                        sofa_liver_24hr = data['sofa_liver_24hr']
                        sofa_renal_24hr = data['sofa_renal_24hr']
                    
                    phenotyping_cols = [
                        'Acute and unspecified renal failure',
                        'Acute cerebrovascular disease',
                        'Acute myocardial infarction',
                        'Cardiac dysrhythmias',
                        'Chronic kidney disease',
                        'Chronic obstructive pulmonary disease and bronchiectasis',
                        'Complications of surgical procedures or medical care',
                        'Conduction disorders',
                        'Congestive heart failure; nonhypertensive',
                        'Coronary atherosclerosis and other heart disease',
                        'Diabetes mellitus with complications',
                        'Diabetes mellitus without complication',
                        'Disorders of lipid metabolism',
                        'Essential hypertension',
                        'Fluid and electrolyte disorders',
                        'Gastrointestinal hemorrhage',
                        'Hypertension with complications and secondary hypertension',
                        'Other liver diseases',
                        'Other lower respiratory disease',
                        'Other upper respiratory disease',
                        'Pleurisy; pneumothorax; pulmonary collapse',
                        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                        'Respiratory failure; insufficiency; arrest (adult)',
                        'Septicemia (except in labor)',
                        'Shock'
                    ]
                    # if self.window == 48:
                    #     task_label = [mortality_30, mortality_inhospital, mortality_icu, mortality48hr, los_7days, readmission_30, transfusion_12hr, vasopressor_need_12hr, ventilator_need_12hr, shock_8hr, sofa_centralnervous_24hr, sofa_cardiovascular_24hr, sofa_respiratory_24hr, sofa_coagulation_24hr, sofa_liver_24hr, sofa_renal_24hr]
                    # else:
                    #     task_label = [mortality_30, mortality_inhospital, mortality_icu, mortality48hr, los_3days, los_7days, readmission_30, transfusion_12hr, vasopressor_need_12hr, ventilator_need_12hr, shock_8hr, sofa_centralnervous_24hr, sofa_cardiovascular_24hr, sofa_respiratory_24hr, sofa_coagulation_24hr, sofa_liver_24hr, sofa_renal_24hr]
                    task_label = [mortality_30, mortality_inhospital, mortality_icu, mortality48hr, los_3days, los_7days, readmission_30, transfusion_12hr, vasopressor_need_12hr, ventilator_need_12hr, shock_8hr, sofa_centralnervous_24hr, sofa_cardiovascular_24hr, sofa_respiratory_24hr, sofa_coagulation_24hr, sofa_liver_24hr, sofa_renal_24hr]
                    multilabel_label = torch.tensor([data[col] for col in phenotyping_cols], dtype=torch.float32)        
                    task_label = torch.tensor(task_label, dtype=torch.int64)
                
        # CLS 토큰 추가
        cls_token_tensor = torch.tensor(self.cls_token_id, dtype=torch.int64).unsqueeze(0) 
        

        if self.mode == "pretrain":
            # if self.mask_mode == 'mlm' or self.mask_mode == 'span_mlm':
            if self.mask_mode == 'mlm':
                return padded_ehr, attention_mask, age, gender, padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, padded_label, padded_value_label
            # elif self.mask_mode == 'mlm+discriminator' or self.mask_mode == 'span_mlm+discriminator':
            #     return padded_ehr, attention_mask, age, gender , padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, padded_label, padded_discriminator_labels
        else:
            # task_cls_token = torch.arange(1, 12, dtype=torch.int64)
            if self.suffix == "hirid" or self.suffix == "P12":
                return padded_ehr, attention_mask, age, gender, padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, task_label
            elif self.task == "phenotype":
                return padded_ehr, attention_mask, age, gender, padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, multilabel_label
            elif self.window == "entire":
                return padded_ehr, attention_mask, age, gender, padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, task_label, multilabel_label
            else:
                return padded_ehr, attention_mask, age, gender, padded_value, padded_unit, padded_offset, padded_position, padded_token_type, padded_ordercategoryname, padded_ordercategorydescription, cls_token_tensor, task_label, multilabel_label