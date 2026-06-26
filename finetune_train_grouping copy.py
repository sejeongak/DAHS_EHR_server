import logging
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import torch.nn as nn
import time
import os
import torch
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts
import numpy as np
from torch.cuda.amp import autocast
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
from utils.sampler import RandomOversamplingDistributedSampler
from torch.autograd import grad
from sklearn.preprocessing import label_binarize

def calculate_metrics_test(predictions, labels, thresholds): 
    probs = F.softmax(predictions, dim=1)
    # preds = torch.argmax(probs, dim=1)
    
    positive_probs = probs[:, 1]
    preds = (positive_probs >= thresholds).long()
    
    labels = labels.view(-1) 
    
    preds_np = preds.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    if probs.size(1) > 1:
        probs_np = probs[:, 1].cpu().detach().numpy()
    else:
        probs_np = probs[:, 0].cpu().detach().numpy() 
    
    accuracy = round(accuracy_score(labels_np, preds_np), 4)
    precision = round(precision_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    recall = round(recall_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    f1 = round(f1_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    
    
    if len(np.unique(labels_np)) > 1 and np.sum(labels_np) > 0:
        auroc = round(roc_auc_score(labels_np, probs_np), 4)
        auprc = round(average_precision_score(labels_np, probs_np), 4)
        # if auprc > 1:
        #     print("auprc > 1!!!!!!!!!!!!!!!!!")
    else:
        auroc = "one class"
        auprc = -1 
 
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'auprc': auprc
    }
    return metrics


# def calculate_metrics(predictions, labels): 
#     # positive_class = 1
#     # positive_probs = torch.sigmoid(predictions[:, positive_class])
#     positive_probs = torch.sigmoid(predictions)
#     # preds = torch.argmax(probs, dim=1)
    
    
#     # positive_probs = probs[:, positive_class]
#     # thresholds = 0.4
    
    
#     positive_probs_np = positive_probs.detach().cpu().numpy()
#     labels_np = labels.view(-1).detach().cpu().numpy()
    
#     precisions, recalls, thresholds = precision_recall_curve(labels_np, positive_probs_np)
        
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
#     best_threshold_idx = f1_scores.argmax()
#     best_threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5
    
#     preds = (positive_probs >= best_threshold).long()
#     preds_np = preds.cpu().detach().numpy()
    
#     accuracy = round(accuracy_score(labels_np, preds_np), 4)
#     precision = round(precision_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
#     recall = round(recall_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
#     f1 = round(f1_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    
    
#     if len(np.unique(labels_np)) > 1 and np.sum(labels_np) > 0:
#         auroc = round(roc_auc_score(labels_np, positive_probs_np), 4)
#         auprc = round(average_precision_score(labels_np, positive_probs_np), 4)
#         # if auprc > 1:
#         #     print("auprc > 1!!!!!!!!!!!!!!!!!")
#     else:
#         auroc = None
#         auprc = None
 
#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'auroc': auroc,
#         'auprc': auprc,
#         'best_threshold': round(best_threshold, 4)
#     }
#     return metrics


def calculate_metrics(predictions, labels): 
    positive_probs = torch.sigmoid(predictions)  # For binary classification
    
    # Handle multi-class case: Use sigmoid or softmax as appropriate
    if predictions.ndim > 1 and predictions.size(1) > 1:
        positive_probs = torch.softmax(predictions, dim=1)[:, 1]  # Assume second column is positive class
    
    # Convert tensors to numpy arrays
    positive_probs_np = positive_probs.detach().cpu().numpy().squeeze()
    labels_np = labels.view(-1).detach().cpu().numpy()

    preds = (positive_probs >= 0.5).long()
    preds_np = preds.cpu().numpy().squeeze()


    # Calculate classification metrics
    accuracy = round(accuracy_score(labels_np, preds_np), 4)
    precision = round(precision_score(labels_np, preds_np, zero_division=0), 4)
    recall = round(recall_score(labels_np, preds_np, zero_division=0), 4)
    f1 = round(f1_score(labels_np, preds_np, zero_division=0), 4)
    print("labels: ", labels_np)
    print("preds: ", preds_np)
    # Calculate AUROC and AUPRC
    try:
        auroc = round(roc_auc_score(labels_np, positive_probs_np), 4)
        auprc = round(average_precision_score(labels_np, positive_probs_np), 4)
    except ValueError:
        auroc, auprc = None, None 


    # Return all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'auprc': auprc,
        # 'best_threshold': round(best_threshold, 4)
    }
    return metrics


def calculate_binary_metrics(preds, labels, num_tasks=7):
    task_metrics = []

    for i in range(num_tasks):
        task_preds = preds[:, i]  # 모델 예측값
        task_labels = labels[:, i]  # 실제 정답 레이블

        # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
        if len(np.unique(task_labels)) < 2:  
            auroc = np.nan  # 또는 0.5 (랜덤 분류 성능)
            auprc = np.nan
        else:
            auroc = roc_auc_score(task_labels, task_preds)
            auprc = average_precision_score(task_labels, task_preds)

        task_metrics.append({
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy_score(task_labels, task_preds > 0.5),
            "precision": precision_score(task_labels, task_preds > 0.5, zero_division=0),
            "recall": recall_score(task_labels, task_preds > 0.5, zero_division=0),
            "f1_score": f1_score(task_labels, task_preds > 0.5, zero_division=0),
        })

    return task_metrics

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

def calculate_multiclass_metrics(preds, labels, num_tasks=5, num_classes=5):
    task_metrics = []

    for i in range(num_tasks):
        prob = preds[i]    # shape: (N, num_classes)
        true = labels[i]   # shape: (N,)

        unique_labels = np.unique(true)
        pred_classes = np.argmax(prob, axis=1)

        # 기본 classification metrics
        task_result = {
            "accuracy": accuracy_score(true, pred_classes),
            "precision_macro": precision_score(true, pred_classes, average="macro", zero_division=0),
            "recall_macro": recall_score(true, pred_classes, average="macro", zero_division=0),
            "f1_score_macro": f1_score(true, pred_classes, average="macro", zero_division=0),
            "precision_micro": precision_score(true, pred_classes, average="micro", zero_division=0),
            "recall_micro": recall_score(true, pred_classes, average="micro", zero_division=0),
            "f1_score_micro": f1_score(true, pred_classes, average="micro", zero_division=0),
        }

        if len(unique_labels) < 2:
            task_result.update({
                "macro_auroc": np.nan,
                "macro_auprc": np.nan,
                "micro_auroc": np.nan,
                "micro_auprc": np.nan,
            })
        else:
            try:
                y_true_bin = label_binarize(true, classes=np.arange(num_classes))
                y_true_subset = y_true_bin[:, unique_labels]
                prob_subset = prob[:, unique_labels]

                task_result.update({
                    "macro_auroc": roc_auc_score(y_true_subset, prob_subset, average="macro", multi_class="ovr"),
                    "macro_auprc": average_precision_score(y_true_subset, prob_subset, average="macro"),
                    "micro_auroc": roc_auc_score(y_true_subset, prob_subset, average="micro", multi_class="ovr"),
                    "micro_auprc": average_precision_score(y_true_subset, prob_subset, average="micro"),
                })
            except ValueError:
                task_result.update({
                    "macro_auroc": np.nan,
                    "macro_auprc": np.nan,
                    "micro_auroc": np.nan,
                    "micro_auprc": np.nan,
                })

        task_metrics.append(task_result)

    return task_metrics



def calculate_phenotype_metrics(preds, labels, num_tasks=25):
    # macro auroc, micro auroc
    macro_auroc = []
    macro_auprc = []
    
    for i in range(num_tasks):
        task_preds = preds[:, i]
        task_labels = labels[:, i]

        # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
        if len(np.unique(task_labels)) < 2:
            macro_auroc.append(np.nan)
            macro_auprc.append(np.nan)
        else:
            macro_auroc.append(roc_auc_score(task_labels, task_preds))
            macro_auprc.append(average_precision_score(task_labels, task_preds))
            
    preds_flat = preds.ravel()
    labels_flat = labels.ravel()
    
    try:
        micro_auroc = roc_auc_score(labels_flat, preds_flat)
        micro_auprc = average_precision_score(labels_flat, preds_flat)
    except ValueError:
        micro_auroc = np.nan
        micro_auprc = np.nan
        
    return {
        "macro_auroc": np.nanmean(macro_auroc),
        "macro_auprc": np.nanmean(macro_auprc),
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "per_task_auroc": macro_auroc,
        "per_task_auprc": macro_auprc,
    }
            

# def find_best_threshold(probs_np, labels_np):
#     precision, recall, thresholds = precision_recall_curve(labels_np, probs_np)
    
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

#     best_threshold = thresholds[f1_scores.argmax()]
    
#     return best_threshold, f1_scores.max()

# def calculate_metrics(predictions, labels): 
#     probs = F.softmax(predictions, dim=1)
#     probs_np = probs[:, 1].cpu().detach().numpy()
#     labels_np = labels.view(-1).cpu().detach().numpy()

#     best_threshold, best_f1 = find_best_threshold(probs_np, labels_np)

#     preds_np = (probs_np >= best_threshold).astype(int)
    
#     accuracy = accuracy_score(labels_np, preds_np) 
#     precision = precision_score(labels_np, preds_np, zero_division=0)
#     recall = recall_score(labels_np, preds_np, zero_division=0)
#     f1 = f1_score(labels_np, preds_np, zero_division=0)
    
#     if len(np.unique(labels_np)) > 1:
#         auroc = roc_auc_score(labels_np, probs_np)
#         auprc = average_precision_score(labels_np, probs_np)
#     else:
#         auroc = "one class"
#         auprc = -1

#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'auroc': auroc,
#         'auprc': auprc,
#         'best_threshold': best_threshold,
#         'best_f1': best_f1
#     }
#     return metrics

def get_attr(model, name):
    if hasattr(model, "module"):
        return getattr(model.module, name)
    else:
        return getattr(model, name)
def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    check_patience = 0
    best_auprc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        multiclass_task_losses = [[] for _ in range(args.num_sofa_tasks)]
        all_preds, all_labels = [], []
        all_sofa_preds = [[] for _ in range(args.num_sofa_tasks)]
        all_sofa_labels = [[] for _ in range(args.num_sofa_tasks)]
        all_phenotype_preds, all_phenotype_labels = [], []

        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):
            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                task_ids, labels, multi_labels) = batch

                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        value_ids=value_ids,
                        unit_ids=unit_ids,
                        time_ids=offset_ids,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                        ordername_ids=ordercategoryname_ids,
                        orderdescription_ids=ordercategorydescription_ids,
                        age_ids=age_ids,
                        gender_ids=gender_ids,
                        task_token=task_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                    task_group = args.task_group
                    
                    

                    if task_group == "basetask":
                        logits = outputs['binary_logits'].squeeze(1)
                        binary_labels = labels[:, :args.num_basetask_tasks].float()
                        
                        binary_losses = []
                        for i in range(args.num_basetask_tasks):
                            binary_losses.append(binary_criterion(logits[:, i], binary_labels[:, i]))
                        loss = sum(binary_losses)
                        all_preds.append(torch.sigmoid(logits).detach().cpu())
                        all_labels.append(binary_labels.detach().cpu())
                        
                    elif task_group == "intervention":
                        logits = outputs['binary_logits'].squeeze(1)
                        binary_labels = labels[:, args.num_basetask_tasks:args.num_basetask_tasks+args.num_intervention_tasks].float()
                        
                        binary_losses = []
                        for i in range(args.num_intervention_tasks):
                            binary_losses.append(binary_criterion(logits[:, i], binary_labels[:, i]))
                        loss = sum(binary_losses)
                        all_preds.append(torch.sigmoid(logits).detach().cpu())
                        all_labels.append(binary_labels.detach().cpu())

                    elif task_group == "sofa+shock":
                        sofa_logits = outputs['sofa_logits']
                        shock_logits = outputs['binary_logits'].squeeze(1)
                        sofa_labels = labels[:, args.num_basetask_tasks+args.num_intervention_tasks+1:args.num_basetask_tasks+args.num_intervention_tasks+1+args.num_sofa_tasks].long()
                        shock_labels = labels[:, args.num_basetask_tasks+args.num_intervention_tasks].float()
                        
                        binary_loss = binary_criterion(shock_logits, shock_labels)
                        
                        multiclass_losses = []
                        for i in range(args.num_sofa_tasks):
                            pred_i = sofa_logits[:, i, :]
                            label_i = sofa_labels[:, i]
                            multiclass_losses.append(multiclass_criterion(pred_i, label_i))
                            all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                            all_sofa_labels[i].append(label_i.detach().cpu())
                        loss = sum(multiclass_losses) + binary_loss
                            
                        all_preds.append(torch.sigmoid(shock_logits).detach().cpu().unsqueeze(1))
                        all_labels.append(shock_labels.detach().cpu().unsqueeze(1))

                    elif task_group == "phenotype":
                        phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                        phenotype_labels = multi_labels.float()
                        loss = binary_criterion(phenotype_logits, phenotype_labels)
                        all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                        all_phenotype_labels.append(phenotype_labels.detach().cpu())

                    elif task_group == "multitask":
                        logits = outputs['binary_logits'].squeeze(1)
                        sofa_logits = outputs['sofa_logits']
                        phenotype_logits = outputs['phenotype_logits'].squeeze(1)

                        binary_labels = labels[:, :args.num_binary_tasks].float()
                        sofa_labels = labels[:, args.num_binary_tasks:args.num_binary_tasks + args.num_sofa_tasks].long()
                        phenotype_labels = multi_labels.float()

                        loss = sum([binary_criterion(logits[:, i], binary_labels[:, i]) for i in range(args.num_binary_tasks)])
                        loss += sum([multiclass_criterion(sofa_logits[:, i, :], sofa_labels[:, i]) for i in range(args.num_sofa_tasks)])
                        loss += binary_criterion(phenotype_logits, phenotype_labels)

                        all_preds.append(torch.sigmoid(logits).detach().cpu())
                        all_labels.append(binary_labels.detach().cpu())
                        for i in range(args.num_sofa_tasks):
                            all_sofa_preds[i].append(torch.softmax(sofa_logits[:, i, :], dim=-1).detach().cpu())
                            all_sofa_labels[i].append(sofa_labels[:, i].detach().cpu())
                        all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                        all_phenotype_labels.append(phenotype_labels.detach().cpu())

                accelerator.backward(loss)
                
                if (step + 1) % args.acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad()
                total_loss += loss.item()
                epoch_loss = total_loss / (step + 1)
   
        if accelerator.is_local_main_process:
            log_data = {"Epoch": epoch, "Train Loss": epoch_loss}
            
        if task_group == "basetask":
            binary_preds = torch.cat(all_preds).numpy() if all_preds else None
            binary_labels = torch.cat(all_labels).numpy() if all_labels else None
            
            taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_basetask_tasks) if binary_preds is not None else []

            if accelerator.is_local_main_process:
                for i, metrics in enumerate(taskwise_metrics):
                    log_data.update({
                        f"Train {multitask_labels[i]} AUROC": metrics['auroc'],
                        f"Train {multitask_labels[i]} AUPRC": metrics['auprc'],
                    })
                wandb.log(log_data)
                for k, v in log_data.items():
                    print(f"{k}: {v}")
                    
        elif task_group == "intervention":
            binary_preds = torch.cat(all_preds).numpy() if all_preds else None
            binary_labels = torch.cat(all_labels).numpy() if all_labels else None
            
            taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_intervention_tasks) if binary_preds is not None else []

            if accelerator.is_local_main_process:
                for i, metrics in enumerate(taskwise_metrics):
                    log_data.update({
                        f"Train {multitask_labels[args.num_basetask_tasks:][i]} AUROC": metrics['auroc'],
                        f"Train {multitask_labels[args.num_basetask_tasks:][i]} AUPRC": metrics['auprc'],
                    })
                wandb.log(log_data)
                for k, v in log_data.items():
                    print(f"{k}: {v}")
                    
        elif task_group == "sofa+shock":
            binary_preds = torch.cat(all_preds).numpy() if all_preds else None
            binary_labels = torch.cat(all_labels).numpy() if all_labels else None
            sofa_preds = [torch.cat(p).numpy() for p in all_sofa_preds] if any(all_sofa_preds) else []
            sofa_labels = [torch.cat(l).numpy() for l in all_sofa_labels] if any(all_sofa_labels) else []
            
            taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=1) if binary_preds is not None else []
            sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels, num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels) if sofa_preds else []

            if accelerator.is_local_main_process:
                for i, metrics in enumerate(taskwise_metrics):
                    log_data.update({
                        f"Train {multitask_labels[-1]} AUROC": metrics['auroc'],
                        f"Train {multitask_labels[-1]} AUPRC": metrics['auprc'],
                    })
                for i, metrics in enumerate(sofa_metrics):
                    log_data.update({
                        f"Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                        f"Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                        f"Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                        f"Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                    })
                wandb.log(log_data)
                for k, v in log_data.items():
                    print(f"{k}: {v}")
                    
        elif task_group == "phenotype":
            phenotype_preds = torch.cat(all_phenotype_preds).numpy() if all_phenotype_preds else None
            phenotype_labels = torch.cat(all_phenotype_labels).numpy() if all_phenotype_labels else None

            phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels) if phenotype_preds is not None else {}

            if accelerator.is_local_main_process:
                if phenotype_metrics:
                    log_data.update({
                        "Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                        "Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                        "Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                        "Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                    })
                wandb.log(log_data)
                for k, v in log_data.items():
                    print(f"{k}: {v}")
        
        elif task_group == "multitask":

            binary_preds = torch.cat(all_preds).numpy() if all_preds else None
            binary_labels = torch.cat(all_labels).numpy() if all_labels else None
            sofa_preds = [torch.cat(p).numpy() for p in all_sofa_preds] if any(all_sofa_preds) else []
            sofa_labels = [torch.cat(l).numpy() for l in all_sofa_labels] if any(all_sofa_labels) else []
            phenotype_preds = torch.cat(all_phenotype_preds).numpy() if all_phenotype_preds else None
            phenotype_labels = torch.cat(all_phenotype_labels).numpy() if all_phenotype_labels else None

            taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_binary_tasks) if binary_preds is not None else []
            sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels, num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels) if sofa_preds else []
            phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels) if phenotype_preds is not None else {}

            if accelerator.is_local_main_process:
                for i, metrics in enumerate(taskwise_metrics):
                    log_data.update({
                        f"Train {multitask_labels[i]} AUROC": metrics['auroc'],
                        f"Train {multitask_labels[i]} AUPRC": metrics['auprc'],
                    })
                for i, metrics in enumerate(sofa_metrics):
                    log_data.update({
                        f"Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                        f"Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                        f"Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                        f"Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                    })
                if phenotype_metrics:
                    log_data.update({
                        "Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                        "Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                        "Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                        "Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                    })
                wandb.log(log_data)
                for k, v in log_data.items():
                    print(f"{k}: {v}")
                    
        if task_group == "basetask":
            valid_loss, taskwise_metrics = validation(
                device,
                model,
                val_loader,
                scheduler,
                scaler,
                accelerator,
                multitask_labels,
                multiclass_labels,
                multilabel_labels,
                binary_criterion,
                multiclass_criterion,
                epoch + 1,
                args)
            
            mean_auroc = np.mean([m['auroc'] for m in taskwise_metrics]) if taskwise_metrics else 0
            mean_auprc = np.mean([m['auprc'] for m in taskwise_metrics]) if taskwise_metrics else 0

            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                best_epoch = epoch + 1
                check_patience = 0
                if accelerator.is_local_main_process:
                    wandb.log({
                        "Best Valid Epoch": epoch + 1,
                        "Best Valid Loss": valid_loss,
                        "Best Valid AUROC": mean_auroc,
                        "Best Valid AUPRC": mean_auprc,
                    })
                    output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                    accelerator.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, output_path)
            else:
                check_patience += 1

            if check_patience >= patience:
                print(f"Early Stopping triggered after {epoch + 1} epochs.")
                break
        
        elif task_group == "intervention":
            valid_loss, taskwise_metrics = validation(
                device,
                model,
                val_loader,
                scheduler,
                scaler,
                accelerator,
                multitask_labels,
                multiclass_labels,
                multilabel_labels,
                binary_criterion,
                multiclass_criterion,
                epoch + 1,
                args)
            
            mean_auroc = np.mean([m['auroc'] for m in taskwise_metrics]) if taskwise_metrics else 0
            mean_auprc = np.mean([m['auprc'] for m in taskwise_metrics]) if taskwise_metrics else 0

            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                best_epoch = epoch + 1
                check_patience = 0
                if accelerator.is_local_main_process:
                    wandb.log({
                        "Best Valid Epoch": epoch + 1,
                        "Best Valid Loss": valid_loss,
                        "Best Valid AUROC": mean_auroc,
                        "Best Valid AUPRC": mean_auprc,
                    })
                    output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                    accelerator.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, output_path)
            else:
                check_patience += 1

            if check_patience >= patience:
                print(f"Early Stopping triggered after {epoch + 1} epochs.")
                break
            
        elif task_group == "sofa+shock":
            valid_loss, taskwise_metrics, sofa_metrics = validation(
                device,
                model,
                val_loader,
                scheduler,
                scaler,
                accelerator,
                multitask_labels,
                multiclass_labels,
                multilabel_labels,
                binary_criterion,
                multiclass_criterion,
                epoch + 1,
                args
                )

            mean_auroc = np.nanmean([m['auroc'] for m in taskwise_metrics] + [m['macro_auroc'] for m in sofa_metrics])
            mean_auprc = np.nanmean([m['auprc'] for m in taskwise_metrics] + [m['macro_auprc'] for m in sofa_metrics])
            
            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                best_epoch = epoch + 1
                check_patience = 0
                if accelerator.is_local_main_process:
                    wandb.log({
                        "Best Valid Epoch": epoch + 1,
                        "Best Valid Loss": valid_loss,
                        "Best Valid AUROC": mean_auroc,
                        "Best Valid AUPRC": mean_auprc,
                    })
                    output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                    accelerator.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, output_path)
            else:
                check_patience += 1
            if check_patience >= patience:
                print(f"Early Stopping triggered after {epoch + 1} epochs.")
                break
            
        elif task_group == "phenotype":
            valid_loss, phenotype_metrics = validation(
                device,
                model,
                val_loader,
                scheduler,
                scaler,
                accelerator,
                multitask_labels,
                multiclass_labels,
                multilabel_labels,
                binary_criterion,
                multiclass_criterion,
                epoch + 1,
                args)
            mean_auroc = np.nanmean(phenotype_metrics['per_task_auroc'])
            mean_auprc = np.nanmean(phenotype_metrics['per_task_auprc'])

            if mean_auprc > best_auprc:
                best_auprc = mean_auprc
                best_epoch = epoch + 1
                check_patience = 0
                if accelerator.is_local_main_process:
                    wandb.log({
                        "Best Valid Epoch": epoch + 1,
                        "Best Valid Loss": valid_loss,
                        "Best Valid AUROC": mean_auroc,
                        "Best Valid AUPRC": mean_auprc,
                    })
                    output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                    accelerator.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, output_path)
            else:
                check_patience += 1
            if check_patience >= patience:
                print(f"Early Stopping triggered after {epoch + 1} epochs.")
                break
            
            
            
        elif task_group == "multitask":
            valid_loss, taskwise_metrics, sofa_metrics, phenotype_metrics = validation(
                device,
                model,
                val_loader,
                scheduler,
                scaler,
                accelerator,
                multitask_labels,
                multiclass_labels,
                multilabel_labels,
                binary_criterion,
                multiclass_criterion,
                epoch + 1,
                args)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)

        

# === VALIDATION ===
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    epoch: int,
    args: dict
):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_sofa_preds = [[] for _ in range(args.num_sofa_tasks)]
    all_sofa_labels = [[] for _ in range(args.num_sofa_tasks)]
    all_phenotype_preds, all_phenotype_labels = [], []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
             position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
             task_ids, labels, multi_labels) = batch

            outputs = model(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=offset_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                ordername_ids=ordercategoryname_ids,
                orderdescription_ids=ordercategorydescription_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            task_group = args.task_group

            if task_group == "basetask":
                logits = outputs['binary_logits'].squeeze(1)
                binary_labels = labels[:, :args.num_basetask_tasks].float()
                
                binary_losses = []
                for i in range(args.num_basetask_tasks):
                    binary_losses.append(binary_criterion(logits[:, i], binary_labels[:, i]))
                loss = sum(binary_losses)
                all_preds.append(torch.sigmoid(logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())
                
            elif task_group == "intervention":
                logits = outputs['binary_logits'].squeeze(1)
                binary_labels = labels[:, args.num_basetask_tasks:args.num_basetask_tasks+args.num_intervention_tasks].float()
                
                binary_losses = []
                for i in range(args.num_intervention_tasks):
                    binary_losses.append(binary_criterion(logits[:, i], binary_labels[:, i]))
                loss = sum(binary_losses)
                all_preds.append(torch.sigmoid(logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())

            elif task_group == "sofa+shock":
                sofa_logits = outputs['sofa_logits']
                shock_logits = outputs['binary_logits'].squeeze(1)
                sofa_labels = labels[:, args.num_basetask_tasks+args.num_intervention_tasks+1:args.num_basetask_tasks+args.num_intervention_tasks+1+args.num_sofa_tasks].long()
                shock_labels = labels[:, args.num_basetask_tasks+args.num_intervention_tasks].float()
                
                binary_loss = binary_criterion(shock_logits, shock_labels)
                
                multiclass_losses = []
                for i in range(args.num_sofa_tasks):
                    pred_i = sofa_logits[:, i, :]
                    label_i = sofa_labels[:, i]
                    multiclass_losses.append(multiclass_criterion(pred_i, label_i))
                    all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                    all_sofa_labels[i].append(label_i.detach().cpu())
                loss = sum(multiclass_losses) + binary_loss
                    
                all_preds.append(torch.sigmoid(shock_logits).detach().cpu().unsqueeze(1))
                all_labels.append(shock_labels.detach().cpu().unsqueeze(1))
                for i in range(args.num_sofa_tasks):
                    all_sofa_preds[i].append(torch.softmax(sofa_logits[:, i, :], dim=-1).detach().cpu())
                    all_sofa_labels[i].append(sofa_labels[:, i].detach().cpu())

            elif task_group == "phenotype":
                phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                phenotype_labels = multi_labels.float()
                loss = binary_criterion(phenotype_logits, phenotype_labels)
                all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                all_phenotype_labels.append(phenotype_labels.detach().cpu())

            elif task_group == "multitask":
                logits = outputs['binary_logits'].squeeze(1)
                sofa_logits = outputs['sofa_logits']
                phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                binary_labels = labels[:, :args.num_binary_tasks].float()
                sofa_labels = labels[:, args.num_binary_tasks:args.num_binary_tasks + args.num_sofa_tasks].long()
                phenotype_labels = multi_labels.float()
                loss = sum([binary_criterion(logits[:, i], binary_labels[:, i]) for i in range(args.num_binary_tasks)])
                loss += sum([multiclass_criterion(sofa_logits[:, i, :], sofa_labels[:, i]) for i in range(args.num_sofa_tasks)])
                loss += binary_criterion(phenotype_logits, phenotype_labels)
                all_preds.append(torch.sigmoid(logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())
                for i in range(args.num_sofa_tasks):
                    all_sofa_preds[i].append(torch.softmax(sofa_logits[:, i, :], dim=-1).detach().cpu())
                    all_sofa_labels[i].append(sofa_labels[:, i].detach().cpu())
                all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                all_phenotype_labels.append(phenotype_labels.detach().cpu())
            total_loss += loss.item()

    epoch_loss = total_loss / len(data_loader)
    
    if accelerator.is_local_main_process:
        log_data = {"Epoch": epoch, "Validation Loss": epoch_loss}
    
    if task_group == "basetask":
        binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
        binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_basetask_tasks) if binary_preds is not None else []

        if accelerator.is_local_main_process:
            for i, metrics in enumerate(taskwise_metrics):
                log_data.update({
                    f"Validation {multitask_labels[i]} AUROC": metrics['auroc'],
                    f"Validation {multitask_labels[i]} AUPRC": metrics['auprc'],
                })
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")
            
        return epoch_loss, taskwise_metrics
    
    elif task_group == "intervention":
        binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
        binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_intervention_tasks) if binary_preds is not None else []

        if accelerator.is_local_main_process:
            for i, metrics in enumerate(taskwise_metrics):
                log_data.update({
                    f"Validation {multitask_labels[args.num_basetask_tasks:][i]} AUROC": metrics['auroc'],
                    f"Validation {multitask_labels[args.num_basetask_tasks:][i]} AUPRC": metrics['auprc'],
                })
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")
            
        return epoch_loss, taskwise_metrics
            
    elif task_group == "sofa+shock":
        binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
        binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
        sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
        sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
        
        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=1) if binary_preds is not None else []
        sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels, num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels) if sofa_preds else []

        if accelerator.is_local_main_process:
            for i, metrics in enumerate(taskwise_metrics):
                log_data.update({
                    f"Validation {multitask_labels[-1]} AUROC": metrics['auroc'],
                    f"Validation {multitask_labels[-1]} AUPRC": metrics['auprc'],
                })
            for i, metrics in enumerate(sofa_metrics):
                log_data.update({
                    f"Validation {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                    f"Validation {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                    f"Validation {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                    f"Validation {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                })
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")
            
        return epoch_loss, taskwise_metrics, sofa_metrics

    elif task_group == "phenotype":
        phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
        phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels) if phenotype_preds is not None else {}

        if accelerator.is_local_main_process:
            log_data.update({
                "Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
            })
            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                label = multilabel_labels[i] if i < len(multilabel_labels) else f"phenotype_task_{i}"
                log_data[f"Validation Phenotype {label} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                label = multilabel_labels[i] if i < len(multilabel_labels) else f"phenotype_task_{i}"
                log_data[f"Validation Phenotype {label} AUPRC"] = auprc
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")

        return epoch_loss, phenotype_metrics
    
    elif task_group == "multitask":
        binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
        binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
        sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
        sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
        phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()

        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=args.num_binary_tasks) if binary_preds is not None else []
        sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels, num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels) if sofa_preds else []
        phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels) if phenotype_preds is not None else {}

        if accelerator.is_local_main_process:
            for i, metrics in enumerate(taskwise_metrics):
                label = multitask_labels[i] if i < len(multitask_labels) else f"binary_task_{i}"
                log_data.update({
                    f"Validation {label} AUROC": metrics['auroc'],
                    f"Validation {label} AUPRC": metrics['auprc'],
                })
            for i, metrics in enumerate(sofa_metrics):
                label = multiclass_labels[i] if i < len(multiclass_labels) else f"sofa_task_{i}"
                log_data.update({
                    f"Validation {label} Macro AUROC": metrics['macro_auroc'],
                    f"Validation {label} Macro AUPRC": metrics['macro_auprc'],
                    f"Validation {label} Micro AUROC": metrics['micro_auroc'],
                    f"Validation {label} Micro AUPRC": metrics['micro_auprc'],
                })
            if phenotype_metrics:
                log_data.update({
                    "Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                    "Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                    "Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                    "Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                })
                for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                    label = multilabel_labels[i] if i < len(multilabel_labels) else f"phenotype_task_{i}"
                    log_data[f"Validation Phenotype {label} AUROC"] = auroc
                for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                    label = multilabel_labels[i] if i < len(multilabel_labels) else f"phenotype_task_{i}"
                    log_data[f"Validation Phenotype {label} AUPRC"] = auprc
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")

        return epoch_loss, taskwise_metrics, sofa_metrics, phenotype_metrics


  
# def test(
#     device: torch.device,
#     model: torch.nn.Module,
#     data_loader: torch.utils.data.DataLoader,
#     scaler: torch.amp.GradScaler,
#     accelerator: Accelerator,
#     binary_criterion: torch.nn,
#     threshold: float,
#     args: dict
# ):
    
#     logging.info("Start test...")
#     # val_loss = []
#     # val_precision = []  
#     # val_recall = []
#     # val_f1_score = []
#     # val_auroc = []
#     # val_auprc = []
#     test_loss = []
#     total_loss = 0  
#     total_samples = 0  

#     all_preds = []
#     all_labels = []
    
#     model.eval()
    
#     with torch.no_grad():
#         for step, batch in tqdm(enumerate(data_loader), desc="Test", total=len(data_loader)):
#             batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
#             input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
         
            
#             outputs = model(
#                 input_ids = input_ids,
#                 value_ids = value_ids,
#                 unit_ids = unit_ids,
#                 time_ids = time_ids,                
#                 continuous_ids = continuous_ids,
#                 position_ids = position_ids,
#                 token_type_ids = token_type_ids,
#                 age_ids = age_ids,
#                 gender_ids = gender_ids,
#                 task_token = task_token,
#                 attention_mask=attention_mask,
#                 global_attention_mask=None,
#                 labels=labels,
#                 return_dict=True,
#                 criterion=criterion,
#                 )
            
#             # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
#             # metrics = calculate_metrics(outputs, labels)
            
#             # val_loss.append(loss.item())
#             # val_precision.append(metrics['precision'])
#             # val_recall.append(metrics['recall'])
#             # val_f1_score.append(metrics['f1_score'])
#             # val_auroc.append(metrics['auroc'])
#             # val_auprc.append(metrics['auprc'])
            
#             # val_loss.append(loss.item())
#             # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
#             batch_size = labels.size(0)
#             total_loss += accelerator.gather(outputs['loss']).mean().item() * batch_size
#             total_samples += batch_size
#             # valid_loss.append(loss.item())
#             # metrics = calculate_metrics(outputs, labels)
            
#             all_preds.append(accelerator.gather(outputs).detach().cpu().numpy())
#             all_labels.append(accelerator.gather(labels).detach().cpu().numpy())
                        
            
#             if step % 30 == 0:   
#                 if accelerator.is_local_main_process:
#                     metrics = calculate_metrics_test(outputs, labels, thresholds=threshold)
#                     print(f"Test | Step {step+1} / {len(data_loader)} | Loss: {outputs['loss'].item()} | Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1_Score: {metrics['f1_score']} | AUROC: {metrics['auroc']} | AUPRC: {metrics['auprc']}")     
         
            
#     #     accelerator.wait_for_everyone()
        
#     #     if accelerator.is_local_main_process:
#     #         wandb.log({
#     #             "Val Loss": np.mean(val_loss),
#     #             "Val Precision": np.mean(val_precision),
#     #             "Val Recall": np.mean(val_recall),
#     #             "Val F1_score": np.mean(val_f1_score),
#     #             "Val Auroc": np.mean(val_auroc),
#     #             "Val Auprc": np.mean(val_auprc)
#     #         })
            
#     #         metrics = {
#     #             'precision': np.mean(val_precision),
#     #             'recall': np.mean(val_recall),
#     #             'f1_score': np.mean(val_f1_score),
#     #             'auroc': np.mean(val_auroc),
#     #             'auprc': np.mean(val_auprc)
#     #         }
        
#     # return np.mean(val_loss), metrics
#         epoch_loss = total_loss / total_samples
#         all_preds = torch.cat([torch.tensor(pred).to(device) for pred in all_preds])
#         all_labels = torch.cat([torch.tensor(label).to(device) for label in all_labels])
        
#         # all_preds = np.concatenate([pred for pred in all_preds])
#         # all_labels = np.concatenate([label for label in all_labels])

#         all_metrics = calculate_metrics_test(all_preds, all_labels, thresholds=threshold)
        

#         accelerator.wait_for_everyone()
#         if accelerator.is_local_main_process:
#             log_data = {
#                 "Test Loss": epoch_loss,
#                 **{f"Test {key}": value for key, value in all_metrics.items()},
#                             }
#             for k, v in log_data.items():
            # print(f"{k}: {v}")
#             wandb.log(log_data)
        
#     return epoch_loss, all_metrics