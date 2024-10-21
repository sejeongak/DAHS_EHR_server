import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import wandb
from sklearn.metrics import precision_score
from accelerate import Accelerator
from accelerate import DistributedType
import os
from utils.utils import seed_everything
from transformers import LongformerTokenizer
from datasets import EHR_Longformer_Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from models.longformernormal import LongformerPretrainNormal
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts
from pretrain_train import train
import logging
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.nn.utils.rnn import pad_sequence
import random


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def configure_optimizers(model, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.learning_rate
    
    # n_warmup_steps = int(n_steps * 0.1)
    # n_decay_steps = n_steps - n_warmup_steps
    
    # warmup = LinearLR(optimizer, 
    #                     start_factor=0.01,
    #                     end_factor=1.0,
    #                     total_iters=n_warmup_steps)
    
    # decay = LinearLR(optimizer,
    #                     start_factor=1.0,
    #                     end_factor=0.01,
    #                     total_iters=n_decay_steps)
    
    # scheduler = SequentialLR(optimizer, 
    #                             schedulers=[warmup, decay],
    #                             milestones=[n_warmup_steps])
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    if args.resume:
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=2690,
                                                cycle_mult=1.5,
                                                max_lr=0.0001,
                                                min_lr=0.000001,
                                                warmup_steps=269,
                                                gamma=0.9,
                                                last_epoch=args.resume_epoch
                                                )
    else:
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=2690,
                                                cycle_mult=1.5,
                                                max_lr=0.0001,
                                                min_lr=0.000001,
                                                warmup_steps=269,
                                                gamma=0.9,
                                                )

    return optimizer, {"scheduler": scheduler, "interval": "step"}

# class BucketBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#         self.indices = list(range(len(self.dataset)))
#         self.indices.sort(key=lambda i: self.dataset.get_seq_length(i))

#     def __iter__(self):
#         buckets = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
   
#         random.shuffle(buckets)
        
#         for bucket in buckets:
#             yield bucket

#     def __len__(self):
#         return len(self.indices) // self.batch_size if self.drop_last else (len(self.indices) + self.batch_size - 1) // self.batch_size


# def collate_fn(batch):
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     ehr_tensors, ages, genders, values, units, offsets, ons, positions, token_types, task_token, labels = zip(*batch)
    
#     ehr_tensors_padded = pad_sequence(ehr_tensors, batch_first=True, padding_value=0)
#     attention_masks = (ehr_tensors_padded != 0).long()
#     units_padded = pad_sequence(units, batch_first=True, padding_value=0)
#     values_padded = pad_sequence(values, batch_first=True, padding_value=0)
#     offsets_padded = pad_sequence(offsets, batch_first=True, padding_value=0)
#     ons_padded = pad_sequence(ons, batch_first=True, padding_value=0)
#     positions_padded = pad_sequence(positions, batch_first=True, padding_value=0)
#     token_types_padded = pad_sequence(token_types, batch_first=True, padding_value=0)
#     labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    
#     return ehr_tensors_padded, attention_masks, torch.stack(ages), torch.stack(genders), values_padded, units_padded, offsets_padded, ons_padded, positions_padded, token_types_padded, torch.stack(task_token), labels_padded


def main(args):

    accelerator = Accelerator(mixed_precision="fp16" if args.gpu_mixed_precision else "no")
    print(f"Distributed Type: {accelerator.distributed_type}")

    
    device = accelerator.device

    save_path = Path(args.save_path) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Logger 
    if accelerator.is_local_main_process:
        logging.basicConfig(
            format="[%(asctime)s][%(levelname)s]\t %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            level=logging.INFO,
            filename=os.path.join(save_path, "exp.log"),
            filemode='w',
        )
    
    seed_everything(args.seed)
    
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    itemid2idx = pd.read_pickle("datasets/entire_itemid2idx.pkl")
    unit2idx = pd.read_pickle("datasets/unit2idx.pkl")
    
    
    
    # WandB initialization
    if accelerator.is_local_main_process:
        logging.info("Wandb initialization")
        wandb.init(project="EHR_Pretrain",
                   name=args.exp_name + "_resume" if args.resume else args.exp_name,
                   config=args,
                   )


    # Model
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # logging.info(f"Device: {device}")
    
    logging.info("Setting Model")
    model = LongformerPretrainNormal(
        vocab_size=args.vocab_size,
        itemid_size=args.itemid_size,
        max_position_embeddings=args.max_position_embeddings,
        unit_size=args.unit_size,
        continuous_size=args.continuous_size,
        task_size=args.task_size,
        max_age=args.max_age,
        gender_size=args.gender_size,
        embedding_size=args.embedding_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        gpu_mixed_precision=args.gpu_mixed_precision,
    ).to(device)
    
    
        
    # DataLoader
    logging.info("Setting Dataset")
    
    train_dataset = EHR_Longformer_Dataset(Path("./datasets"), "train", tokenizer, itemid2idx, unit2idx, use_itemid=True)
    valid_dataset = EHR_Longformer_Dataset(Path("./datasets"), "valid", tokenizer, itemid2idx, unit2idx, use_itemid=True)
    
    # train_sampler = BucketBatchSampler(train_dataset, args.batch_size)
    # valid_sampler = BucketBatchSampler(valid_dataset, args.batch_size)
    
    # train_loader = DataLoader(train_dataset,
    #                           batch_sampler=train_sampler,
    #                           collate_fn=collate_fn,
    #                           num_workers=args.num_workers,
    #                           pin_memory=args.pin_memory)
    
    # valid_loader = DataLoader(valid_dataset,
    #                           batch_sampler=valid_sampler,
    #                           collate_fn=collate_fn,
    #                           num_workers=args.num_workers,
    #                           pin_memory=args.pin_memory)
    

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True,  # shuffle should be False if using DistributedSampler
                            # sampler=train_sampler,
                            pin_memory=args.pin_memory, 
                            num_workers=args.num_workers,
                            )

    valid_loader = DataLoader(valid_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,  # Validation should not be shuffled
                            # sampler=valid_sampler,
                            pin_memory=args.pin_memory, 
                            num_workers=args.num_workers,
                            )

    
    # Optimizer, Scheduler, Gradient Scaler
    # n_steps = (len(train_dataset) // args.batch_size) * args.epochs
    optimizer, scheduler = configure_optimizers(model, args,)
    
    if args.resume:
        checkpoint_path = f"./results/best_pretrain_model_after_masking_{args.resume_epoch}epoch.pth"
        # checkpoint_path = f"./results/best_pretrain_model_resume_{args.resume_epoch}epoch.pth" # debugging
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('module.module.'):
                new_state_dict[k[14:]] = v

        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resume model loaded successfully.")
        
        args.start_epoch = args.resume_epoch + 1
        logging.info(f"Resuming from epoch {args.start_epoch}")
        
        lr = scheduler["scheduler"].get_lr()
        for param_group, lr_val in zip(optimizer.param_groups, lr):
            param_group['lr'] = lr_val
            
        logging.info(f"Resumed learning rates: {lr}")
        


    else:
        print("Start from the beggining")
    # scaler = GradScaler(enabled=args.gpu_mixed_precision)
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler['scheduler']
    )
    
    
    train(
        device,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        accelerator.scaler,
        accelerator,
        args.epochs,
        args.start_epoch,
        args.patience,
        args.save_path,
        args
    )
    
      
        
                
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--exp_name", type=str, default="pretrain")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=50265)
    parser.add_argument("--itemid_size", type=int, default=4016)
    parser.add_argument("--unit_size", type=int, default=60)
    parser.add_argument("--gender_size", type=int, default=2)
    parser.add_argument("--continuous_size", type=int, default=3)
    parser.add_argument("--task_size", type=int, default=20)
    parser.add_argument("--max_position_embeddings", type=int, default=5000)
    parser.add_argument("--max_age", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)      
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_mixed_precision", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--clip_interval", type=int, default=200)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_epoch", type=int, default=0)
    
    
    
    args = parser.parse_args()
    args.attention_window = [512] * args.num_hidden_layers

    

    
    
    main(args=args)
    