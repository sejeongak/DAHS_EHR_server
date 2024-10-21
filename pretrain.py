
import argparse
import os
import sys
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
import pandas as pd
from transformers import LongformerTokenizer

from datasets import EHR_Longformer_Dataset
from models.model import LongformerPretrain

from utils.utils import seed_everything

from pathlib import Path
from tqdm import tqdm

def main(args: argparse.Namespace):
    save_path = Path(args.save_path) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    itemid2idx = pd.read_pickle("datasets/entire_itemid2idx.pkl")
    unit2idx = pd.read_pickle("datasets/unit2idx.pkl")
    
    train_dataset = EHR_Longformer_Dataset(Path("./datasets"), "train", tokenizer, itemid2idx, unit2idx, use_itemid=True)
    valid_dataset = EHR_Longformer_Dataset(Path("./datasets"), "valid", tokenizer, itemid2idx, unit2idx, use_itemid=True)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True, 
                              pin_memory=args.pin_memory, 
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              pin_memory=args.pin_memory, 
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)
    
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best",
            save_top_k=1,
            save_last=True,
            verbose=True,
            save_on_train_epoch_end=True, 
            dirpath=args.checkpoint_dir,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    model = LongformerPretrain(
    vocab_size=args.vocab_size,
    itemid_size=args.itemid_size,
    max_position_embeddings=args.max_position_embeddings,
    unit_size=args.unit_size,
    continuous_size=args.continuous_size,
    task_size=args.task_size,
    max_age=args.max_age,
    gender_size=args.gender_size,
    embedding_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.intermediate_size,
    learning_rate=args.learning_rate,
    dropout_prob=args.dropout_prob,
)

    
    wandb_logger = WandbLogger(name=args.exp_name, project="ehr-longformer", save_dir=args.save_path)
    # gpu
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=args.nodes,
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        precision='16-mixed',
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0,
    )
    # cpu
    # trainer = pl.Trainer(
    #     accelerator="cpu",  
    #     devices=1,
    #     precision=32,  
    #     check_val_every_n_epoch=1,
    #     max_epochs=args.max_epochs,
    #     callbacks=callbacks,
    #     deterministic=False,
    #     enable_checkpointing=True,
    #     enable_progress_bar=True,
    #     enable_model_summary=True,
    #     logger=wandb_logger,
    #     log_every_n_steps=args.log_every_n_steps,
    #     accumulate_grad_batches=args.acc,
    #     gradient_clip_val=1.0,
    # )
    
    trainer.fit(
        model = model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.resume_checkpoint,
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
    parser.add_argument("--task_size", type=int, default=4)
    parser.add_argument("--max_position_embeddings", type=int, default=5000)
    parser.add_argument("--max_age", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    
    
    
    args = parser.parse_args()
    args.attention_window = [512] * args.num_hidden_layers

    

    
    
    main(args=args)
    