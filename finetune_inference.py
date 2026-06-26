
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



def calculate_binary_metrics(preds, labels, num_tasks=7):
    task_metrics = []

    for i in range(num_tasks):
        task_preds = preds[:, i]  
        task_labels = labels[:, i] 


        if len(np.unique(task_labels)) < 2:  
            auroc = np.nan  
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

        # ê¸°ë³¸ classification metrics
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

        # AUROC ?˜ˆ?™¸ ì²˜ë¦¬ (?•œ ê°œì˜ ?´?ž˜?Š¤ë§? ì¡´ìž¬?•˜?Š” ê²½ìš°)
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
    
    

def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    loss_weighter: torch.nn.Module,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    if args.selected_data == "hirid":
        needed_binary_idx = [2, 4, 5, 7, 8, 9, 10]
    else:
        needed_binary_idx = [idx for idx in range(args.num_binary_tasks)]

    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]
        
    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
    else:
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks

    if args.inference_mode:
        logging.info("Inference mode")
        test_loss, test_multitask_metrics, test_multiclass_metrics, test_phenotype_metrics = test(
            device, model, test_loader, accelerator,
            multitask_labels, multiclass_labels, multilabel_labels,
            binary_criterion, multiclass_criterion, args, needed_binary_idx
        )
        logging.info("[Inference Mode] Test finished.")
        return

    logging.info("Start training...")
    check_patience = 0
    best_auroc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(start_epoch, epochs), desc='Epochs', total=epochs-start_epoch, smoothing=0.1):
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        binary_task_losses = [[] for _ in range(num_binary_tasks)]
        multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
        phenotype_losses = []
        
        all_preds, all_labels = [], []
        all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
        all_sofa_labels = [[] for _ in range(num_sofa_tasks)]
        all_phenotype_preds, all_phenotype_labels = [], []

        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):
            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                if args.selected_data != "hirid":
                    (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                    task_ids, labels, multi_labels) = batch
                else:
                    (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                        position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                        task_ids, labels) = batch
                    
                with accelerator.autocast():
                    loss = 0.0
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

                    logits = outputs['binary_logits'].squeeze(1)
                    binary_labels = labels[:, :num_binary_tasks].float()
                    selected_logits = logits[:, needed_binary_idx]
                    # selected_labels = binary_labels[:, needed_binary_idx]
                    binary_losses = []
                    
                    for i in range(needed_binary_idx):
                        loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                        binary_losses.append(loss_i)
                        binary_task_losses[i].append(loss_i.item())

                    all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
                    all_labels.append(binary_labels.detach().cpu())
                    
                    binary_loss = sum(binary_losses)
                    # loss += binary_loss

                    sofa_logits = outputs['sofa_logits']
                    sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()
                    multiclass_losses = []
                    for i in range(num_sofa_tasks):
                        pred_i = sofa_logits[:, i, :]
                        label_i = sofa_labels[:, i]
                        loss_i = multiclass_criterion(pred_i, label_i)
                        multiclass_losses.append(loss_i)
                        multiclass_task_losses[i].append(loss_i.item())
                        all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                        all_sofa_labels[i].append(label_i.detach().cpu())
                    multiclass_loss = sum(multiclass_losses)
                    # loss += multiclass_loss
                    
                    if args.selected_data != "hirid":
                        phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                        phenotype_labels = multi_labels.float()
                        phenotype_loss = binary_criterion(phenotype_logits, phenotype_labels)
                        phenotype_losses.append(phenotype_loss.item())
                        # loss += phenotype_loss
                        
                        all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                        all_phenotype_labels.append(phenotype_labels.detach().cpu())
                        
                        loss = binary_loss + multiclass_loss + phenotype_loss
                        
                    else:
                        loss = binary_loss + multiclass_loss

                        

                

                accelerator.backward(loss)

                if (step + 1) % args.acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad()
                    
                total_loss += loss.item()

                
                if step != 0 and step % 100 == 0 and accelerator.is_local_main_process:
                    log_data = {"Step": step+1, "Step Loss": loss.item()}
                    if outputs['binary_logits'] is not None:
                        binary_preds = torch.cat(all_preds).numpy()
                        binary_labels = torch.cat(all_labels).numpy()
                        
                        if args.task_mode == 'binary' and args.task_idx is not None:
                            taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=1)
                            metrics = taskwise_metrics[0]
                            log_data.update({
                                f"Step Train {multitask_labels[args.task_idx]} AUROC": metrics['auroc'],
                                f"Step Train {multitask_labels[args.task_idx]} AUPRC": metrics['auprc'],
                            })
                        else:
                            taskwise_metrics = calculate_binary_metrics(
                                torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), num_tasks=num_binary_tasks)
                            for i, metrics in enumerate(taskwise_metrics):
                                log_data.update({
                                    f"Step Train {binary_tasks[i]} AUROC": metrics['auroc'],
                                    f"Step Train {binary_tasks[i]} AUPRC": metrics['auprc'],
                                })
                    else:
                        taskwise_metrics = None
                    
                    if outputs['sofa_logits'] is not None:
                        sofa_metrics = calculate_multiclass_metrics(
                            [torch.cat(p) for p in all_sofa_preds], [torch.cat(l) for l in all_sofa_labels], num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels)
                        for i, metrics in enumerate(sofa_metrics):
                            log_data.update({
                                f"Step Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                                f"Step Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                                f"Step Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                                f"Step Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                            })
                    else:
                        sofa_metrics = None
                    
                    if args.selected_data != "hirid":
                        if outputs['phenotype_logits'] is not None:
                            phenotype_metrics = calculate_phenotype_metrics(
                                torch.cat(all_phenotype_preds).numpy(), torch.cat(all_phenotype_labels).numpy())
                            log_data.update({
                                "Step Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                                "Step Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                                "Step Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                                "Step Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                            })
                        else:
                            phenotype_metrics = None

                    
                    
                    for k, v in log_data.items():
                        print(f"{k}: {v}")
                    wandb.log(log_data)


        epoch_loss = total_loss / len(data_loader)
        binary_preds = torch.cat(all_preds).numpy()
        binary_labels = torch.cat(all_labels).numpy()
        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))
        sofa_preds = [torch.cat(p).numpy() for p in all_sofa_preds]
        sofa_labels = [torch.cat(l).numpy() for l in all_sofa_labels]
        sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
                                                    num_tasks=num_sofa_tasks,
                                                    num_classes=args.num_multiclass_labels)
        
        if args.selected_data != "hirid":
            phenotype_preds = torch.cat(all_phenotype_preds).numpy()
            phenotype_labels = torch.cat(all_phenotype_labels).numpy()
            phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels, num_tasks=25)
        

        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                f"Train {binary_tasks[i]} AUROC": metrics['auroc'],
                f"Train {binary_tasks[i]} AUPRC": metrics['auprc'],
                f"Train {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
            })
        for i, metrics in enumerate(sofa_metrics):
            log_data.update({
                f"Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                f"Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                f"Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                f"Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],    
                f"Train {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
            })
        if args.selected_data != "hirid":
            log_data.update({
                "Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Train Phenotype Loss": np.mean(phenotype_losses),
            })
            
            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data[f"Train Phenotype {multilabel_labels[i]} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data[f"Train Phenotype {multilabel_labels[i]} AUPRC"] = auprc
        
        
        for k, v in log_data.items():
            print(f"{k}: {v}")
        wandb.log(log_data)

        # === Validation ===
        valid_loss, valid_multitask_metrics, valid_multiclass_metrics, valid_phenotype_metrics = validation(
            device, model, val_loader, scheduler, scaler, accelerator,
            multitask_labels, multiclass_labels, multilabel_labels, binary_criterion, multiclass_criterion, epoch + 1, args, needed_binary_idx
        )

        if args.selected_data != "hirid":
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [m['macro_auroc'] for m in valid_multiclass_metrics] + [valid_phenotype_metrics['macro_auroc']])
            mean_auprc = np.nanmean([m['auprc'] for m in valid_multitask_metrics] + [m['macro_auprc'] for m in valid_multiclass_metrics] + [valid_phenotype_metrics['macro_auprc']])
        else:
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [m['macro_auroc'] for m in valid_multiclass_metrics])
            mean_auprc = np.nanmean([m['auprc'] for m in valid_multitask_metrics] + [m['macro_auprc'] for m in valid_multiclass_metrics])

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(mean_auprc)

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_epoch = epoch + 1
            check_patience = 0

            if accelerator.is_local_main_process:
                wandb.log({
                    "Best Valid Epoch": best_epoch,
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
        
    if accelerator.is_local_main_process:
        print(f"Loading best model from epoch {best_epoch}")

    best_model_path = Path(save_path) / f"best_{args.exp_name}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    test_loss, test_multitask_metrics, test_multiclass_metrics, _ = test(
        device, model, test_loader, accelerator,
        multitask_labels, multiclass_labels, multilabel_labels,
        binary_criterion, multiclass_criterion, args, needed_binary_idx
    )


def validation(device, model, data_loader, scheduler, scaler, accelerator, multitask_labels, multiclass_labels, multilabel_labels, binary_criterion, multiclass_criterion,
               epoch, args, needed_binary_idx):
    if args.selected_data == "hirid":
        needed_binary_idx = [2, 4, 5, 7, 8, 9, 10]
    else:
        needed_binary_idx = [idx for idx in range(args.num_binary_tasks)]

    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]
        
    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
    else:
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks
        
    model.eval()
    total_loss = 0
    binary_task_losses = [[] for _ in range(num_binary_tasks)]
    multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
    phenotype_losses = []
    
    all_preds, all_labels = [], []
    all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
    all_sofa_labels = [[] for _ in range(num_sofa_tasks)]
    all_phenotype_preds, all_phenotype_labels = [], []
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            if args.selected_data != "hirid":
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                task_ids, labels, multi_labels) = batch
            else:
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                    task_ids, labels) = batch

            loss = 0.0
            outputs = model(
                input_ids=input_ids, value_ids=value_ids, unit_ids=unit_ids, time_ids=offset_ids,
                position_ids=position_ids, token_type_ids=token_type_ids,
                ordername_ids=ordercategoryname_ids, orderdescription_ids=ordercategorydescription_ids,
                age_ids=age_ids, gender_ids=gender_ids, task_token=task_ids,
                attention_mask=attention_mask, return_dict=True,
            )

            logits = outputs['binary_logits'].squeeze(1)
            binary_labels = labels[:, :num_binary_tasks].float()
            selected_logits = logits[:, needed_binary_idx]
            # selected_labels = labels_bin[:, needed_binary_idx]
            binary_losses = []
            # binary_loss = binary_criterion(selected_logits, selected_labels)

            for i in range(needed_binary_idx):
                loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                binary_losses.append(loss_i)
                binary_task_losses[i].append(loss_i.item())
      

            all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
            all_labels.append(binary_labels.detach().cpu())
            
            binary_loss = sum(binary_losses)

            sofa_logits = outputs['sofa_logits']
            sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()
            multiclass_losses = []
            for i in range(num_sofa_tasks):
                pred_i = sofa_logits[:, i, :]
                label_i = sofa_labels[:, i]
                loss_i = multiclass_criterion(pred_i, label_i)
                multiclass_losses.append(loss_i)
                multiclass_task_losses[i].append(loss_i.item())
                all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                all_sofa_labels[i].append(label_i.detach().cpu())
            multiclass_loss = sum(multiclass_losses)

            # total_loss += (binary_loss + multiclass_loss).item()
            
            if args.selected_data != "hirid":
                phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                phenotype_labels = multi_labels.float()
                phenotype_loss = binary_criterion(phenotype_logits, phenotype_labels)
                phenotype_losses.append(phenotype_loss.item())
                # loss += phenotype_loss
                
                all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                all_phenotype_labels.append(phenotype_labels.detach().cpu())
                
                loss = binary_loss + multiclass_loss + phenotype_loss
                
            else:
                loss = binary_loss + multiclass_loss

            total_loss += loss.item()

    epoch_loss = total_loss / len(data_loader)
    
    
    # epoch_loss = total_loss / (step + 1)
    
    all_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
    all_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
    taskwise_metrics = calculate_binary_metrics(all_preds, all_labels, num_tasks=len(needed_binary_idx))
    
    all_sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
    all_sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
    sofa_metrics = calculate_multiclass_metrics(all_sofa_preds, all_sofa_labels, num_tasks=args.num_sofa_tasks, num_classes=args.num_multiclass_labels)
    
    
    if args.selected_data != "hirid":
        all_phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        all_phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
        phenotype_metrics = calculate_phenotype_metrics(all_phenotype_preds, all_phenotype_labels)
    
    if accelerator.is_local_main_process:
        log_data = {"Epoch": epoch, "Validation Loss": epoch_loss}
        
        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                f"Validation {binary_tasks[i]} AUROC": metrics['auroc'],
                f"Validation {binary_tasks[i]} AUPRC": metrics['auprc'],
                f"Validation {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
            })
        for i, metrics in enumerate(sofa_metrics):
            log_data.update({
                f"Validation {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                f"Validation {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                f"Validation {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                f"Validation {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                f"Validation {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
            })
            
        if args.selected_data != "hirid":
            log_data.update({
                "Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Validation Phenotype Loss": np.mean(phenotype_losses),
            })

            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data[f"Validation Phenotype {multilabel_labels[i]} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data[f"Validation Phenotype {multilabel_labels[i]} AUPRC"] = auprc

    
    for k, v in log_data.items():
        print(f"{k}: {v}")
    wandb.log(log_data)
    if args.selected_data != "hirid":
        return epoch_loss, taskwise_metrics, sofa_metrics, phenotype_metrics
    else:
        return epoch_loss, taskwise_metrics, sofa_metrics, None

def test(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,   # »ç¿ëÇÏÁö ¾ÊÀ½(phenotype Á¦°Å)
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    args: dict,
    needed_binary_idx: list
):
          
    if args.selected_data == "hirid":
        needed_binary_idx = [2, 4, 5, 7, 8, 9, 10]
        multilabel_labels = []  
        
    else:
        needed_binary_idx = [idx for idx in range(args.num_binary_tasks)]

    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]
    
    
        
    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
    else:
        # if args.window == 48:
        #     num_binary_tasks = args.num_binary_tasks - 1
        # else:
        #     num_binary_tasks = args.num_binary_tasks
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks
        
        
    logging.info("Start test...")
    model.eval()
    total_loss = 0
    binary_task_losses = [[] for _ in range(num_binary_tasks)]
    multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
    phenotype_losses = []

    all_preds, all_labels = [], []
    all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
    all_sofa_labels = [[] for _ in range(num_sofa_tasks)]
    all_phenotype_preds, all_phenotype_labels = [], []
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Test", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            if args.selected_data != "hirid":
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                task_ids, labels, multi_labels) = batch
            else:
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids,
                    task_ids, labels) = batch

            loss = 0.0
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
            
            # === Binary Task Loss ===
            if outputs['binary_logits'] is not None:
                logits = outputs['binary_logits'].squeeze(1)
                binary_labels = labels[:, :num_binary_tasks].float()
                selected_logits = logits[:, needed_binary_idx]
                
                binary_losses = []
                
                # ÅÂ½ºÅ©º° °³º° loss ±â·Ï
                for i in range(len(needed_binary_idx)):
                    loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                    binary_losses.append(loss_i)
                    binary_task_losses[i].append(loss_i.item())
                    
                all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())
                
                binary_loss = sum(binary_losses)
            
            # === SOFA Task Loss ===
            if outputs['sofa_logits'] is not None:
                sofa_logits = outputs['sofa_logits']
                sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()

                multiclass_losses = []
                for i in range(num_sofa_tasks):
                    pred_i = sofa_logits[:, i, :]
                    label_i = sofa_labels[:, i]
                    loss_i = multiclass_criterion(pred_i, label_i)
                    multiclass_losses.append(loss_i)
                    multiclass_task_losses[i].append(loss_i.item())
                    all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                    all_sofa_labels[i].append(label_i.detach().cpu())
                multiclass_loss = sum(multiclass_losses)
                
            if args.selected_data != "hirid":
                # === Phenotype Task Loss ===
                if outputs['phenotype_logits'] is not None:
                    phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                    phenotype_labels = multi_labels.float()
                    phenotype_loss = binary_criterion(phenotype_logits, phenotype_labels)

                    phenotype_losses.append(phenotype_loss.item())
                    all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                    all_phenotype_labels.append(phenotype_labels.detach().cpu())
                    
                loss = binary_loss + multiclass_loss + phenotype_loss
            else:
                loss = binary_loss + multiclass_loss

            total_loss += loss.item()
    
    epoch_loss = total_loss / len(data_loader)
    # === Metric °è»ê ===
    binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
    binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
    taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))
    
    sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
    sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
    sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
                                                num_tasks=num_sofa_tasks,
                                                num_classes=args.num_multiclass_labels)
    
    if args.selected_data != "hirid":
        phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
        phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)
        
    

    # === Logging ===
    if accelerator.is_local_main_process:
        log_data = {"Test Loss": epoch_loss}
        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                f"Test {binary_tasks[i]} AUROC": metrics['auroc'],
                f"Test {binary_tasks[i]} AUPRC": metrics['auprc'],
                f"Test {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
            })
        for i, metrics in enumerate(sofa_metrics):
            log_data.update({
                f"Test {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                f"Test {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                f"Test {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                f"Test {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                f"Test {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
            })
            
        if args.selected_data != "hirid":
            log_data.update({
                "Test Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Test Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Test Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Test Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Test Phenotype Loss": np.mean(phenotype_losses),
            })

            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data[f"Test Phenotype {multilabel_labels[i]} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data[f"Test Phenotype {multilabel_labels[i]} AUPRC"] = auprc

    
        for k, v in log_data.items():
            print(f"{k}: {v}")
        wandb.log(log_data)
    
    if args.selected_data != "hirid":
        return epoch_loss, taskwise_metrics, sofa_metrics, phenotype_metrics
    else:
        return epoch_loss, taskwise_metrics, sofa_metrics, None