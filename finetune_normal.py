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
from models.longformernormal import LongformerPretrainNormal, LongformerFinetune
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from finetune_train import train
import logging
import sys
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch.nn.init as init
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Initialize FocalLoss.
        
        Parameters:
        - gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
        - alpha (float or Tensor): Weighting factor for the positive class. Can be a scalar or a tensor of shape (num_classes,).
        - reduction (str): The method used to reduce the loss. Options are 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        # Convert alpha to a tensor if it's a single float value
        if isinstance(self.alpha, (float, list)):
            if isinstance(self.alpha, float):
                self.alpha = [self.alpha, 1 - self.alpha]
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
    def forward(self, inputs, targets):
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Cross-entropy loss per sample
        
        probs = torch.exp(-ce_loss)  # probs = 1 - ce_loss when using log_softmax implicitly in cross_entropy
        
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)  # Ensure alpha is on the same device
            alpha_t = self.alpha.gather(0, targets)  # Gather alpha values based on targets
            focal_loss = alpha_t * focal_loss  # Apply alpha scaling

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Shape: (batch_size,)
        
        # # Compute probabilities for each class
        # probs = F.softmax(inputs, dim=1)  # Shape: (batch_size, num_classes)
        
        # # Extract probabilities for the true class
        # pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
        # focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # if self.alpha is not None:
        #     # self.alpha = self.alpha.to(targets.device)
        #     alpha_t = self.alpha.gather(0, targets)  # Shape: (batch_size,)
        #     focal_loss = alpha_t * focal_loss
        
        
        # # Reduce loss
        # if self.reduction == 'mean':
        #     return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss
        

class CB_Loss(nn.Module):
    def __init__(self, beta, gamma, class_counts):
        super(CB_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.class_counts = class_counts

        # Effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        self.class_weights = (1.0 - self.beta) / effective_num
        self.class_weights = self.class_weights / self.class_weights.sum() * len(self.class_counts)
    
    def forward(self, logits, labels):
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        weights = self.class_weights[labels].unsqueeze(1)
        
        focal_weight = torch.pow(1 - torch.softmax(logits, dim=1), self.gamma)
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = focal_weight * loss
        loss = weights * loss

        return loss.mean()
    
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta, gamma=2.0, alpha=None, loss_type="focal", no_of_classes=None):
        """
        Initialize Class-Balanced Focal Loss.

        Parameters:
        - beta (float): Hyperparameter for Class-Balanced Loss.
        - gamma (float): Focusing parameter for Focal Loss, adjusts rate at which easy examples are down-weighted.
        - alpha (Tensor or None): Optional weight for each class.
        - loss_type (str): The type of loss to compute ("focal", "sigmoid", "softmax").
        - no_of_classes (int): The number of classes.
        """
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.loss_type = loss_type
        self.no_of_classes = no_of_classes

    def focal_loss(self, labels, logits, alpha, gamma):
        """
        Compute the Focal Loss.
        
        Parameters:
        - labels (Tensor): One-hot encoded ground truth labels.
        - logits (Tensor): Model predictions (logits).
        - alpha (Tensor): Class weighting factor.
        - gamma (float): Focal loss focusing parameter.
        
        Returns:
        - Focal loss (Tensor): The computed Focal loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')
        
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-logits)))

        loss = modulator * bce_loss
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels)
        return focal_loss

    def class_balanced_loss(self, labels, logits, samples_per_cls):
        """
        Compute the Class-Balanced Loss with Focal Loss as an option.

        Parameters:
        - labels (Tensor): Ground truth labels of shape [batch].
        - logits (Tensor): Model predictions (logits) of shape [batch, num_classes].
        - samples_per_cls (list): List with the number of samples per class.
        - no_of_classes (int): Total number of classes.
        
        Returns:
        - Class-balanced loss (Tensor).
        """
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        class_weights = (1.0 - self.beta) / np.array(effective_num)
        class_weights = class_weights / np.sum(class_weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(class_weights).float().unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1).unsqueeze(1).repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

        return cb_loss

    def forward(self, labels, logits, samples_per_cls):
        """
        Forward pass for the Class-Balanced Focal Loss.

        Parameters:
        - labels (Tensor): Ground truth labels.
        - logits (Tensor): Model predictions (logits).
        - samples_per_cls (list): List with the number of samples per class.

        Returns:
        - Computed loss (Tensor).
        """
        return self.class_balanced_loss(labels, logits, samples_per_cls)


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

        
# def configure_optimizers(model, args):
    
#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

#     return optimizer, scheduler

def configure_optimizers(model, args, n_steps):
    base_lr = args.learning_rate  
    lora_lr = args.learning_rate * args.lora_weight
    classifier_lr = args.learning_rate * args.classifier_weight
    
    pretrained_params = model.pretrained_parameters()
    lora_params = model.lora_parameters()
    classifier_params = model.classifier_parameters()

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': base_lr, 'weight_decay': 0.01},
        {'params': lora_params, 'lr': lora_lr, 'weight_decay': 0.001},
        {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.001}
    ])


    # CosineAnnealingWarmRestarts 
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                           first_cycle_steps=538,
    #                                           cycle_mult=2,
    #                                           max_lr=base_lr,
    #                                           min_lr=1e-6,
    #                                           warmup_steps=54,
    #                                           gamma=0.5
    #                                           )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=538,  # Total steps per epoch
    #     T_mult=2,  # Multiplier for increasing cycle lengths
    #     eta_min=1e-6  # Minimum learning rate for warm restarts
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    return optimizer, {"scheduler": scheduler, "interval": "step"}

# def configure_optimizers(model, args, n_steps):
#     base_lr = args.learning_rate  
#     lora_lr = args.learning_rate * args.lora_weight
#     classifier_lr = args.learning_rate * args.lora_weight
    
#     pretrained_params = model.pretrained_parameters()
#     lora_params = model.lora_parameters()
#     classifier_params = model.classifier_parameters()

#     optimizer = optim.AdamW([
#         {'params': pretrained_params, 'lr': base_lr},  
#         {'params': lora_params, 'lr': lora_lr},  
#         {'params': classifier_params, 'lr': classifier_lr}, 
#     ])

#     warmup_steps = int(n_steps * 0.1)
#     warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
#     cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

#     scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

#     return optimizer, {"scheduler": scheduler, "interval": "step"}

# def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
        
#         # Cosine Decay start
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress)))

#     return LambdaLR(optimizer, lr_lambda, last_epoch)

# def configure_optimizers(model, args, n_steps):
#     base_lr = args.learning_rate  
#     lora_lr = args.learning_rate * args.lora_weight
#     classifier_lr = args.learning_rate * args.lora_weight
    
#     pretrained_params = model.pretrained_parameters()
#     lora_params = model.lora_parameters()
#     classifier_params = model.classifier_parameters()


#     optimizer = optim.AdamW([
#     {'params': pretrained_params, 'lr': base_lr, 'name': 'pretrained_params'},  
#     {'params': lora_params, 'lr': lora_lr, 'name': 'lora_params'},  
#     {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier_params'}
# ], weight_decay=0.01)

   
#     num_warmup_steps = int(n_steps * 0.1)  
#     num_training_steps = n_steps 

#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps
#     )

#     return optimizer, {"scheduler": scheduler, "interval": "step"}

def calculate_alpha(dataset, num_classes):
    """
    Calculate the alpha values based on the class distribution in the dataset.
    
    Parameters:
    - dataset: The dataset to calculate class distribution from.
    - num_classes: The total number of classes in the dataset.
    
    Returns:
    - alpha: Tensor containing the alpha values for each class.
    """
    # Get all targets in the dataset
    all_targets = [label.item() for *_, label in dataset]

    # Count the occurrences of each class
    class_counts = Counter(all_targets)

    # Calculate the total number of samples
    total_count = sum(class_counts.values())

    # Calculate alpha as the inverse of the class frequency
    alpha = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        alpha[cls] = total_count / (num_classes * count)
    
    # Normalize alpha to sum to 1
    alpha = alpha / alpha.sum()

    return alpha


def calculate_class_weights(dataset):
    labels = [label.item() for *_, label in dataset]
    

    class_counts = Counter(labels)
    total_samples = len(dataset)
      
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    

    weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32)
    
    return weights

def calculate_weights_for_sampler(dataset):
    labels = [label.item() for *_, label in dataset]
    class_counts = Counter(labels)

    weights = [1.0 / class_counts[label] for label in labels]
    
    return weights, class_counts

def normalize_class_weights(class_weights):
    total = sum(class_weights.values())
    return {k: v / total for k, v in class_weights.items()}

class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        

        self.labels = [self.dataset[idx][-1].item() for idx in range(len(self.dataset))]
        
   
        self.class_indices = {label: np.where(np.array(self.labels) == label)[0].tolist() for label in np.unique(self.labels)}
        
 
        self.class_probs = {label: len(self.class_indices[label]) / len(self.dataset) for label in np.unique(self.labels)}
        
    
        self.min_class_1_per_batch = 1
        self.class_1_indices = self.class_indices[1]
        self.other_class_indices_template = {label: self.class_indices[label] for label in self.class_indices if label != 1}

   
        self.indices = []

    def _generate_indices(self):
        indices = []
        num_batches = len(self.dataset) // self.batch_size
        
      
        class_1_per_batch = np.array_split(self.class_1_indices, num_batches)

        other_class_indices = {label: indices_list[:] for label, indices_list in self.other_class_indices_template.items()}
        
        for batch_num in range(num_batches):
            batch_indices = []
            
        
            batch_indices.extend(class_1_per_batch[batch_num])
            
      
            remaining_batch_size = self.batch_size - len(batch_indices)
            other_samples = []
            
            for label, indices_list in other_class_indices.items():
                if len(indices_list) > 0:
                
                    if remaining_batch_size > len(indices_list):
                        selected_indices = indices_list
                    else:
                        selected_indices = np.random.choice(indices_list, remaining_batch_size, replace=False).tolist()
                    
                    other_samples.extend(selected_indices)
                    
                  
                    other_class_indices[label] = [idx for idx in indices_list if idx not in selected_indices]
            
            np.random.shuffle(other_samples)
            batch_indices.extend(other_samples[:remaining_batch_size])
        
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)
        
        return indices
    
    def __iter__(self):
   
        self.indices = self._generate_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def main(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(mixed_precision="fp16" if args.gpu_mixed_precision else "no", kwargs_handlers=kwargs_handlers)
    print(f"Distributed Type: {accelerator.distributed_type}")
    device = accelerator.device

    
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
        
    seed_everything(args.seed)
    
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    itemid2idx = pd.read_pickle("datasets/entire_itemid2idx.pkl")
    unit2idx = pd.read_pickle("datasets/unit2idx.pkl")
    
    # WandB initialization
    if accelerator.is_local_main_process:
        logging.info("Wandb initialization")
        wandb.init(project="EHR_" + args.mode,
                   name=args.exp_name + "_resume" if args.resume else args.exp_name,
                   config=args,
                   )
        
    logging.info("Setting Model")
    pretrained_model = LongformerPretrainNormal(
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
    
    if args.pretrain:
        pretrain_path = os.path.join("./results/", args.pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.module.'):
                new_state_dict[k[14:]] = v  
            elif k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v  
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if 'task_embedding' not in k}

        pretrained_model.load_state_dict(filtered_state_dict, strict=False)
        print("Pre-trained model loaded successfully.")
        def initialize_weights(module):
            if isinstance(module, torch.nn.Embedding):
                init.xavier_uniform_(module.weight.data) 

        pretrained_model.embeddings.task_embedding.apply(initialize_weights)
    else:
        print("No model loaded.")
    
    # Apply LoRA
    
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Assuming this is for sequence classification
            inference_mode=False,  # Set to True if using for inference only
            r=8,  # Rank of the low-rank matrices
            lora_alpha=32,  # Scaling factor for the low-rank matrices
            lora_dropout=args.lora_dropout,  # Dropout probability for LoRA layers
            target_modules=["query", "value"], # Target attention layers
        )
        pretrained_model = get_peft_model(pretrained_model, peft_config)
        print("Applying LoRA")

    
    model = LongformerFinetune(
        pretrained_model=pretrained_model,
        problem_type="single_label_classification",
        num_labels=2,
        learning_rate=args.learning_rate,
        classifier_dropout=args.classifier_dropout,
        use_lora=args.use_lora,
    ).to(device)
        
    print(f"Train Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # DataLoader
    logging.info("Setting Dataset")
    
    train_dataset = EHR_Longformer_Dataset(Path("./datasets"), "train", tokenizer, itemid2idx, unit2idx, use_itemid=True, mode=args.mode)
    valid_dataset = EHR_Longformer_Dataset(Path("./datasets"), "valid", tokenizer, itemid2idx, unit2idx, use_itemid=True, mode=args.mode)
    
    
    # weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float32).to(args.device)
    
    # Custom Sampler
    # sample_weights, class_counts = calculate_weights_for_sampler(train_dataset)
    # ns = int(class_counts[1] /len(train_dataset) * args.batch_size)
    custom_sampler = CustomSampler(train_dataset, batch_size=args.batch_size)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            sampler=custom_sampler,  # shuffle should be False if using DistributedSampler
                            # shuffle=True,
                            pin_memory=args.pin_memory, 
                            num_workers=args.num_workers,
                            )


    # Create WeightedRandomSampler
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # sample_weights, class_counts = calculate_weights_for_sampler(train_dataset)
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # train_loader = DataLoader(train_dataset, 
    #                         batch_size=args.batch_size,
    #                         sampler=sampler,  # shuffle should be False if using DistributedSampler
    #                         # shuffle=True,
    #                         pin_memory=args.pin_memory, 
    #                         num_workers=args.num_workers,
    #                         )

    valid_loader = DataLoader(valid_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,  # Validation should not be shuffled
                            pin_memory=args.pin_memory, 
                            num_workers=args.num_workers,
                            )
    
    class_weights = calculate_alpha(train_dataset, args.num_labels)
    
    print(class_weights)

    # Loss
    
    if args.loss == "focal_loss":
        criterion = FocalLoss(gamma=args.gamma, alpha=class_weights.to(args.device), reduction='mean')
        print("loss: focal loss")
    elif args.loss == "cross_entropy":
        # criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))
        criterion = nn.CrossEntropyLoss()
        print("loss: cross entropy loss")
    elif args.loss == "cb_loss":
        criterion = CB_Loss(beta=args.beta, gamma=args.gamma, class_weights=class_weights.to(args.device))
        print("loss: cb loss")
    elif args.loss == "class_balanced_focal_loss":
        criterion = ClassBalancedFocalLoss(beta=args.beta, gamma=args.gamma, alpha=class_weights.to(args.device), no_of_classes=args.num_labels)
        print("loss: class balanced focal loss")


    # Optimizer, Scheduler, z`radient Scaler
    n_steps = (len(train_dataset) // args.batch_size) * args.epochs
    optimizer, scheduler = configure_optimizers(model, args, n_steps)
    # scaler = GradScaler(enabled=args.gpu_mixed_precision)
    # if accelerator.distributed_type == DistributedType.MULTI_GPU:
    #     model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler['scheduler']
    )
    
    
    train(
        device,
        model,
        train_loader,
        valid_loader,
        criterion,
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
    parser.add_argument("--exp_name", type=str, default="mortality30")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Model parameters
    parser.add_argument("--mode", type=str, default="mortality30")
    parser.add_argument("--vocab_size", type=int, default=50265)
    parser.add_argument("--itemid_size", type=int, default=4016)
    parser.add_argument("--unit_size", type=int, default=60)
    parser.add_argument("--gender_size", type=int, default=2)
    parser.add_argument("--continuous_size", type=int, default=3)
    parser.add_argument("--task_size", type=int, default=20)
    parser.add_argument("--max_position_embeddings", type=int, default=5000)
    parser.add_argument("--max_age", type=int, default=100)
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
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_mixed_precision", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument('--lora_weight', type=int, default=2)
    parser.add_argument('--classifier_weight', type=int, default=5)
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--pretrain", action='store_true', default=False)
    parser.add_argument("--clip_interval", type=int, default=10)
    parser.add_argument("--pretrain_path", type=str, default="best_pretrain_model.pth")

    
    args = parser.parse_args()
    args.attention_window = [512] * args.num_hidden_layers
    
    main(args=args)