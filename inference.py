import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from sklearn.metrics import precision_score
from accelerate import Accelerator
from accelerate import DistributedType, DistributedDataParallelKwargs
from peft import get_peft_model, LoraConfig, TaskType
import os
from utils.utils import seed_everything
from transformers import LongformerTokenizer
from datasets import EHR_Longformer_Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from models.longformernormal import LongformerPretrainNormal, LongformerFinetune, LongformerFinetuneforMultiTask
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
# from finetune_train_grouping import train
from finetune_inference import train
import logging
import sys
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch.nn.init as init
import warnings
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
from collections import Counter
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from utils.sampler import RandomOversamplingDistributedSampler
from torch.utils.data import ConcatDataset
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_optimizers(model, args, n_steps, loss_weighter=None):
    base_lr = args.learning_rate  
    classifier_lr = args.learning_rate * args.classifier_weight

    pretrained_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            pretrained_params.append(param)
            
    if loss_weighter is not None:
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': base_lr, 'weight_decay': 0.01},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.0},
            {'params': loss_weighter.parameters(), 'lr': base_lr, 'weight_decay': 0.0}
        ], lr=base_lr, eps=1e-8, betas=(0.9, 0.98))
        
    else:
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': base_lr, 'weight_decay': 0.01},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.0},
        ], lr=base_lr, eps=1e-8, betas=(0.9, 0.98))


    num_warmup_steps = int(n_steps * 0.05)  
    num_training_steps = n_steps 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    

    return optimizer, scheduler

class MultiTaskUncertaintyWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_binary = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_multiclass = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_multilabel = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, binary_loss, multiclass_loss, multilabel_loss):
        loss = (
            torch.exp(-2 * self.log_sigma_binary) * binary_loss +
            torch.exp(-2 * self.log_sigma_multiclass) * multiclass_loss +
            torch.exp(-2 * self.log_sigma_multilabel) * multilabel_loss + 
            self.log_sigma_binary + self.log_sigma_multiclass + self.log_sigma_multilabel
        )
        
        return loss

def main(args):
    warnings.filterwarnings('ignore')
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(mixed_precision="fp16" if args.gpu_mixed_precision else "no",
                              kwargs_handlers=kwargs_handlers)
    print(f"Distributed Type: {accelerator.distributed_type}")
    device = accelerator.device

 
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    seed_everything(args.seed)

    save_path = Path(args.save_path) / args.mode
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

    # Tokenizer & Label
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    itemid2idx = pd.read_pickle("datasets/new_label2idx.pkl")
    unit2idx = pd.read_pickle("datasets/new_unit2idx.pkl")


    multitask_labels = [
        "mortality_30days", "mortality_inhospital", "mortality_icu", "mortality48hr",
        "los_3days", "los_7days", "readmission_30days", "transfusion_12hr",
        "vasopressor_need_12hr", "ventilation_need_12hr", "shock_8hr"
    ]
    sofa_labels = [
        "sofa_centralnervous_24hr", "sofa_cardiovascular_24hr", "sofa_respiratory_24hr",
        "sofa_coagulation_24hr", "sofa_liver_24hr", "sofa_renal_24hr"
    ]
    
    if args.selected_data == "hirid":
        needed_binary_tasks = [2, 4, 5, 7, 8, 9, 10]
        multilabel_labels = []  
        
    else:
    
        needed_binary_tasks = [idx for idx in range(args.num_binary_tasks)]
            
        
            
            
        multilabel_labels = [
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
    

    # WandB
    if not args.debug and accelerator.is_local_main_process:
        logging.info("Wandb initialization")
        wandb.init(project="EHR_" + args.task,
                   name=args.exp_name + "_resume" if args.resume else args.exp_name,
                   config=args)

    logging.info("Setting Model")
    use_value_prediction = args.value_mask_ratio > 0

    pretrained_model = LongformerPretrainNormal(
        name_size=args.name_size,
        description_size=args.description_size,
        token_type_size=args.token_type_size,
        vocab_size=args.vocab_size,
        itemid_size=args.itemid_size,
        max_position_embeddings=args.max_position_embeddings,
        unit_size=args.unit_size,
        task_size=args.task_size,
        max_age=args.max_age,
        gender_size=args.gender_size,
        embedding_size=args.embedding_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        loss_factor=args.loss_factor,
        use_discriminator=args.use_discriminator,
        use_value_prediction=use_value_prediction,
        gpu_mixed_precision=args.gpu_mixed_precision,
    ).to(device)
    pretrained_model.config.classifier_dropout = 0.3
    pretrained_model.config.num_labels = args.num_labels

    model = LongformerFinetuneforMultiTask(
        pretrained_model=pretrained_model,
        num_labels=args.num_labels,
        num_binary_tasks=args.num_binary_tasks,
        num_sofa_tasks=args.num_sofa_tasks,
        classifier_dropout=args.classifier_dropout,
        freeze_pretrained=args.freeze,
        freeze_layers=args.freeze_layers,
        ablation=args.ablation,
        args=args
    ).to(device)

    # === Pretrained Weight Load + Freeze ===
    if args.pretrain:
        pretrain_path = os.path.join("./results/", args.pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.module.'):
                new_state_dict[k[14:]] = v
            elif k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        
        if args.selected_data == "hirid":

            # freeze unused binary classifiers
            for idx, clf in enumerate(model.binary_classifiers):
                if idx not in needed_binary_tasks:
                    for param in clf.parameters():
                        param.requires_grad = False

            # phenotype classifier freeze
            if hasattr(model, 'phenotype_classifier'):
                for param in model.phenotype_classifier.parameters():
                    param.requires_grad = False

        print("Pre-trained model loaded successfully.")
    else:
        print("No pre-trained model loaded.")

    # Dataset
    logging.info("Setting Dataset")
    locate = args.locate if args.locate is not None else None
    print("locater: ", locate)

    train_dataset = EHR_Longformer_Dataset(Path("./datasets/new_data_preparation"), "train",
                                           tokenizer, itemid2idx, unit2idx,
                                           vocab_size=args.vocab_size, use_itemid=True,
                                           mode=args.mode, mask_mode=args.mask_mode, task=args.task,
                                           seed=args.seed, ablation=args.ablation,
                                           window=args.window, index=args.index,
                                           no_gap=args.no_gap, ratio=args.ratio,
                                           selected_data=args.selected_data, locate=locate)
    valid_dataset = EHR_Longformer_Dataset(Path("./datasets/new_data_preparation"), "valid",
                                           tokenizer, itemid2idx, unit2idx,
                                           vocab_size=args.vocab_size, use_itemid=True,
                                           mode=args.mode, mask_mode=args.mask_mode, task=args.task,
                                           seed=args.seed, ablation=args.ablation,
                                           window=args.window, index=args.index,
                                           no_gap=args.no_gap, ratio=args.ratio,
                                           selected_data=args.selected_data, locate=locate)
    test_dataset = EHR_Longformer_Dataset(Path("./datasets/new_data_preparation"), "test",
                                          tokenizer, itemid2idx, unit2idx,
                                          vocab_size=args.vocab_size, use_itemid=True,
                                          mode=args.mode, mask_mode=args.mask_mode, task=args.task,
                                          seed=args.seed, ablation=args.ablation,
                                          window=args.window, index=args.index,
                                          no_gap=args.no_gap, ratio=args.ratio,
                                          selected_data=args.selected_data)

    def custom_collate_fn(batch):
        return tuple(torch.stack(samples, dim=0) for samples in zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=args.pin_memory, num_workers=args.num_workers,
                              persistent_workers=False, drop_last=True,
                              collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=args.pin_memory, num_workers=args.num_workers,
                              persistent_workers=False, drop_last=True,
                              collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             pin_memory=args.pin_memory, num_workers=args.num_workers,
                             persistent_workers=False, drop_last=True,
                             collate_fn=custom_collate_fn)

    # inference 
    if args.inference_mode and args.selected_data == "hirid":
        inference_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        # inference_dataset = ConcatDataset([train_dataset, valid_dataset])
        test_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=args.pin_memory, num_workers=args.num_workers,
                                 persistent_workers=False, drop_last=False,
                                 collate_fn=custom_collate_fn)
        print("inference_dataset: ", len(inference_dataset))
        print("inference_loader: ", len(test_loader))

    print("train_dataset:", len(train_dataset))
    print("valid_dataset:", len(valid_dataset))
    print("test_dataset:", len(test_dataset))

    model, train_loader, valid_loader, test_loader = accelerator.prepare(model, train_loader,
                                                                        valid_loader, test_loader)

    # Loss weighting
    loss_weighter = MultiTaskUncertaintyWeighting().to(device) if args.loss_mode == "weighted" else None

    n_steps = len(train_loader) * args.epochs
    optimizer, scheduler = configure_optimizers(model, args, n_steps, loss_weighter=loss_weighter)
    optimizer = accelerator.prepare(optimizer)

    criterion = nn.BCEWithLogitsLoss()
    multiclass_criterion = nn.CrossEntropyLoss()
    print("loss: binary cross-entropy with logits")

    # === Train ===
    train(device, model, train_loader, valid_loader, test_loader,
          criterion, multiclass_criterion,
          optimizer, scheduler, accelerator.scaler, accelerator,
          multitask_labels, sofa_labels, multilabel_labels,
          loss_weighter, args.epochs, args.start_epoch,
          args.patience, args.save_path, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--exp_name", type=str, default="mortality30")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Model parameters
    parser.add_argument("--mode", type=str, default="finetune")
    parser.add_argument("--task", type=str, default="mortality30")
    parser.add_argument("--vocab_size", type=int, default=50265)
    parser.add_argument("--itemid_size", type=int, default=3892)
    parser.add_argument("--unit_size", type=int, default=81)
    parser.add_argument("--gender_size", type=int, default=2)
    parser.add_argument("--task_size", type=int, default=20)
    parser.add_argument("--token_type_size", type=int, default=5)
    parser.add_argument("--name_size", type=int, default=35)
    parser.add_argument("--description_size", type=int, default=12)
    parser.add_argument("--max_position_embeddings", type=int, default=10000)
    parser.add_argument("--max_age", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    # parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_mixed_precision", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_labels", type=int, default=1)
    # parser.add_argument("--use_lora", action='store_true', default=False)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.99)
    # parser.add_argument('--lora_weight', type=int, default=2)
    parser.add_argument('--classifier_weight', type=int, default=5)
    parser.add_argument('--adapter_weight', type=int, default=1)
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--pretrain", action='store_true', default=False)
    parser.add_argument("--clip_interval", type=int, default=1)
    parser.add_argument("--pretrain_path", type=str, default="best_pretrain_model.pth")
    # parser.add_argument("--lora_layers", type=list, default=[8,9,10,11])
    # parser.add_argument("--lora_r", type=int, default=4)
    # parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--loss_factor", type=float, default=0.5)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--use_discriminator", action="store_true", help="Enable discriminator")
    parser.add_argument("--regression_mode", type=bool, default=False)
    parser.add_argument("--similarity_mode", type=bool, default=False)
    parser.add_argument("--similarity_factor", type=float, default=0.25)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--freeze", action="store_true", default=False)
    # parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--oversampling_ratio", type=float, default=1.0)
    parser.add_argument("--mask_mode", type=str, default="mlm")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=7)
    parser.add_argument("--num_binary_tasks", type=int, default=11)
    parser.add_argument("--num_sofa_tasks", type=int, default=6)
    parser.add_argument("--num_multiclass_labels", type=int, default=4)
    parser.add_argument("--num_binary_tasks_hirid", type=int, default=7)
    parser.add_argument("--num_sofa_tasks_hirid", type=int, default=6)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--no_gap", action='store_true', default=False)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--task_group", type=str, default="multitask")
    parser.add_argument("--num_basetask_tasks", type=int, default=1)
    parser.add_argument("--num_intervention_tasks", type=int, default=1)
    parser.add_argument("--value_mask_ratio", type=float, default=0.0)
    parser.add_argument("--loss_mode", type=str, default="None")
    parser.add_argument("--task_mode", type=str, default="multitask")
    parser.add_argument("--task_idx", type=lambda x: int(x) if x != "None" else None, default=None, help="Index of single binary task to run (None = run all tasks)")
    parser.add_argument("--inference_mode", action='store_true', default=False)
    parser.add_argument("--selected_data", type=str, default="all")
    parser.add_argument("--locate", type=str, default=None)
    args = parser.parse_args()
    args.attention_window = [512] * args.num_hidden_layers
    
    main(args=args)