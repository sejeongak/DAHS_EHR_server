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


def calculate_taskwise_metrics(preds, labels, num_tasks=7):
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
    criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    # scheduler: ,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multilabel_labels: list,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    logging.info("Start training...")
    check_patience = 0
    
    best_auroc = 0.0
    best_epoch = 0
    best_f1_score = 0.0
    
    
    for epoch in tqdm(range(start_epoch, epochs), desc='Epochs', total=epochs-start_epoch, smoothing=0.1):
        
        if hasattr(data_loader, 'sampler') and isinstance(data_loader.sampler, RandomOversamplingDistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        model.train()
        
        total_loss = 0  
        all_preds = []
        all_labels = []
        task_losses = [[] for _ in range(args.num_tasks)]
        
        all_phenotype_preds = []
        all_phenotype_labels = []
        phenotype_losses = []
        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):
            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids, position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids, task_ids, labels, multi_labels = batch 
     
                with accelerator.autocast():
                    outputs = model(
                        input_ids = input_ids,
                        value_ids = value_ids,
                        unit_ids = unit_ids,
                        time_ids = offset_ids,                
                        position_ids = position_ids,
                        token_type_ids = token_type_ids,
                        ordername_ids = ordercategoryname_ids,
                        orderdescription_ids = ordercategorydescription_ids,
                        age_ids = age_ids,
                        gender_ids = gender_ids,
                        task_token = task_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=None,
                        # labels=labels,
                        return_dict=True,
                        # criterion=criterion,
                    )
                    
          
                    
                    logits = outputs['logits']
                    logits = logits.squeeze(1)
                    labels = labels.float()
                    
                    phenotype_logits = outputs['multilabel_logits']
                    phenotype_labels = multi_labels.float()
                    
                    # uncertainties = get_attr(model, "task_uncertainties")
                    
                    losses = []
                    
                    # print(uncertainties)
                    for i in range(args.num_tasks):
                        task_loss = criterion(logits[:, i], labels[:, i])
                        # Uncertainties
                        # weighted_loss = task_loss / (2 * model.module.task_uncertainties[i] ** 2) + torch.log(model.module.task_uncertainties[i])
                        # losses.append(weighted_loss)
                        losses.append(task_loss)
                        task_losses[i].append(task_loss.item())
                    multitask_loss = sum(losses) 
        
                    # loss = criterion(outputs, labels)
                    # print("Loss: ", loss.item())
                    
                    phenotype_loss = criterion(phenotype_logits, phenotype_labels)
                    # weighted_phenotype_loss = (phenotype_loss / (2 * model.module.phenotype_uncertainties ** 2)) + torch.log(model.module.phenotype_uncertainties)
                    phenotype_losses.append(phenotype_loss.item())
                     
                    loss = multitask_loss + phenotype_loss
                    # loss = multitask_loss + weighted_phenotype_loss
                    
                accelerator.backward(loss)
                
                # for name, param in model.named_parameters():
                #     if "task_uncertainties" in name:
                #         print(name, param.requires_grad, param.grad, param.data)
                    
                
                # if (step+1) % args.clip_interval == 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                if (step + 1) % args.acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # print(f"Output Min: {outputs.min()}, Max: {outputs.max()}")
                    # print(f"Step {step+1}: Updating weights")
                      # Optimizer Step 전에 Gradient 초기화
                    # accelerator.backward(loss)
                    # if not accelerator.use_fp16:
                    #     accelerator.clip_grad_norm_(model.parameters(), max_norm=0.1)  
                    #     accelerator.clip_grad_value_(model.parameters(), clip_value=1) 
                        
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient Clipping
                    optimizer.step()
                    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad()
             
                # print(f"Loss after step (should decrease in next batch): {loss.item()}")
                    
                
                # loss = criterion(outputs.view(10, args.num_labels), labels.view(-1))
                
                # batch_size = labels.size(0)
                # loss = outputs['loss']
                # total_loss += accelerator.gather(outputs['loss']).mean().item() * batch_size
                total_loss += loss.item() 
                # total_samples += batch_size
                probs = torch.sigmoid(logits)
                # all_preds.append(accelerator.gather(outputs).detach().cpu().numpy())
                # all_labels.append(accelerator.gather(labels).detach().cpu().numpy())
                all_preds.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                phenotype_probs = torch.sigmoid(phenotype_logits)
                all_phenotype_preds.append(phenotype_probs.detach().cpu())
                all_phenotype_labels.append(phenotype_labels.detach().cpu())
                

                # train_loss.append(loss.item())
                
                # train_precision_scores.append(metrics['precision'])
                # train_recall_scores.append(metrics['recall'])
                # train_f1_scores.append(metrics['f1_score'])
                # train_auroc.append(metrics['auroc'])
                # train_auprc.append(metrics['auprc'])
                                  

                if step != 0 and step % 100 == 0 and accelerator.is_local_main_process:
                    taskwise_metrics = calculate_taskwise_metrics(probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), num_tasks=args.num_tasks)
                    phenotype_metrics = calculate_phenotype_metrics(phenotype_probs.detach().cpu().numpy(), phenotype_labels.detach().cpu().numpy(), num_tasks=25)
                    log_data = {
                        "Step": step+1,
                        "Step Loss": loss.item(),
                    }

                    for i, metrics in enumerate(taskwise_metrics):
                        log_data.update({
                            f"Step Train {multitask_labels[i]} AUROC": metrics['auroc'],
                            f"Step Train {multitask_labels[i]} AUPRC": metrics['auprc'],
                        })
                    log_data.update({
                        "Step Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                        "Step Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                        "Step Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                        "Step Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                    })
                    for k, v in log_data.items():
                        print(f"{k}: {v}")

                    wandb.log(log_data)
                
        epoch_loss = total_loss / (step+1)
        
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        all_phenotype_preds = torch.cat(all_phenotype_preds).cpu().numpy()
        all_phenotype_labels = torch.cat(all_phenotype_labels).cpu().numpy()
 
        # print(all_preds.shape)
        # print(all_labels.shape)
        taskwise_metrics = calculate_taskwise_metrics(all_preds, all_labels, num_tasks=args.num_tasks)
        phenotype_metrics = calculate_phenotype_metrics(all_phenotype_preds, all_phenotype_labels, num_tasks=25)
        if accelerator.is_local_main_process:
            log_data = {
                "Epoch": epoch + 1,
                "Train Loss": epoch_loss,
            }

            for i, metrics in enumerate(taskwise_metrics):
                log_data.update({
                    # f"Train {multitask_labels[i]} Accuracy": metrics['accuracy'],
                    f"Train {multitask_labels[i]} AUROC": metrics['auroc'],
                    # f"Train {multitask_labels[i]} Precision": metrics['precision'],
                    # f"Train {multitask_labels[i]} Recall": metrics['recall'],
                    # f"Train {multitask_labels[i]} F1_score": metrics['f1_score'],
                    f"Train {multitask_labels[i]} AUPRC": metrics['auprc'],
                    f"Train {multitask_labels[i]} Loss": np.mean(task_losses[i]),
                })
            log_data.update({
                "Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Train Phenotype Loss": np.mean(phenotype_losses),
            })
            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data.update({
                    f"Train Phenotype {multilabel_labels[i]} AUROC": auroc,
                })
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data.update({
                    f"Train Phenotype {multilabel_labels[i]} AUPRC": auprc,
                })

            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")
            
        
        # model.eval()
        valid_loss, valid_multitask_metrics, valid_phenotype_metrics = validation(device, model, val_loader, scheduler, scaler, accelerator, multitask_labels, multilabel_labels, criterion, epoch+1, args)
        
        
        
        
        
        # if valid_metrics['auroc'] is not None:
        #     # scheduler.step(valid_metrics['auroc'])
        #     scheduler_plateau.step(valid_metrics['auroc'])
        # scheduler.step(valid_metrics['f1_score'])
            
                    
        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     best_epoch = epoch + 1
            
        #     if accelerator.is_local_main_process:
        #         wandb.log({
        #             "Best Epoch": best_epoch,
        #             "Best Loss": best_loss,
        #             "Best Precision": valid_metrics['precision'],
        #             "Best Recall": valid_metrics['recall'],
        #             "Best F1_score": valid_metrics['f1_score'],
        #             "Best Auroc": valid_metrics['auroc'],
        #             "Best Auprc": valid_metrics['auprc']
        #     }) 
            
        #         output_path = Path(save_path) / f"best_{args.mode}_model.pth"
        #         accelerator.save({
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }, output_path)
                
        #     check_patience = 0
        # else:
        #     check_patience += 1
            
        # if epoch+1 > epochs // 2 and check_patience >= patience:
        #     break
        
        # accelerator.wait_for_everyone()
        mean_auroc = np.mean([m['auroc'] for m in valid_multitask_metrics] + [valid_phenotype_metrics['macro_auroc']])
        mean_auprc = np.mean([m['auprc'] for m in valid_multitask_metrics] + [valid_phenotype_metrics['macro_auprc']])
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(mean_auroc)
        
        
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

        # ? Early Stopping
        if check_patience >= patience:
            print(f"Early Stopping triggered after {epoch + 1} epochs.")
            break
   
            
            
      
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multilabel_labels: list,
    criterion: torch.nn,
    epoch: int,
    args: dict
):
    
    logging.info("Start validation...")
    # val_loss = []
    # val_precision = []  
    # val_recall = []
    # val_f1_score = []
    # val_auroc = []
    # val_auprc = []

    total_loss = 0  
    all_preds = []
    all_labels = []
    task_losses = [[] for _ in range(args.num_tasks)]
    
    all_phenotype_preds = []
    all_phenotype_labels = []
    phenotype_losses = []
    model.eval()
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids, position_ids, token_type_ids, ordercategoryname_ids, ordercategorydescription_ids, task_ids, labels, multi_labels = batch
         
            
            outputs = model(
                input_ids = input_ids,
                value_ids = value_ids,
                unit_ids = unit_ids,
                time_ids = offset_ids,                
                position_ids = position_ids,
                token_type_ids = token_type_ids,
                ordername_ids = ordercategoryname_ids,
                orderdescription_ids = ordercategorydescription_ids,
                age_ids = age_ids,
                gender_ids = gender_ids,
                task_token = task_ids,
                attention_mask=attention_mask,
                global_attention_mask=None,
                # labels=labels,
                return_dict=True,
                # criterion=criterion,
                )
            
            # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
            # metrics = calculate_metrics(outputs, labels)
            
            # val_loss.append(loss.item())
            # val_precision.append(metrics['precision'])
            # val_recall.append(metrics['recall'])
            # val_f1_score.append(metrics['f1_score'])
            # val_auroc.append(metrics['auroc'])
            # val_auprc.append(metrics['auprc'])
            
            # val_loss.append(loss.item())
            # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
            logits = outputs['logits']
            logits = logits.squeeze(1)
            labels = labels.float()
            
            phenotype_logits = outputs['multilabel_logits']
            phenotype_labels = multi_labels.float()
            
            losses = []
            for i in range(args.num_tasks):
                task_loss = criterion(logits[:, i], labels[:, i])
                # Uncertainties
                # weighted_loss = task_loss / (2 * model.module.task_uncertainties[i] ** 2) + torch.log(model.module.task_uncertainties[i])
                # losses.append(weighted_loss)
                losses.append(task_loss)
                task_losses[i].append(task_loss.item())
            multitask_loss = sum(losses) 
            
            phenotype_loss = criterion(phenotype_logits, phenotype_labels)
            # weighted_phenotype_loss = (phenotype_loss / (2 * model.module.phenotype_uncertainties ** 2)) + torch.log(model.module.phenotype_uncertainties)
            phenotype_losses.append(phenotype_loss.item())
            # weighted_phenotype_losses.append(weighted_phenotype_loss.item())
            
            loss = multitask_loss + phenotype_loss
            # loss = multitask_loss + weighted_phenotype_loss
            
            # batch_size = labels.size(0)
            # loss = outputs['loss']
            total_loss += loss.item() 
            # total_samples += batch_size
            # valid_loss.append(loss.item())
            # metrics = calculate_metrics(outputs, labels)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs)
            all_labels.append(labels)
            
            phenotype_preds = torch.sigmoid(phenotype_logits)
            all_phenotype_preds.append(phenotype_preds)
            all_phenotype_labels.append(phenotype_labels)        
            
            if step % 100 == 0 and accelerator.is_local_main_process:
                taskwise_metrics = calculate_taskwise_metrics(
                    probs.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                    num_tasks=args.num_tasks
                )
                phenotype_metrics = calculate_phenotype_metrics(
                    phenotype_preds.detach().cpu().numpy(),
                    phenotype_labels.detach().cpu().numpy(),
                    num_tasks=25
                )
                
                log_data = {"Validation Step": step + 1, "Step Validation Loss": loss.item()}
                for i, metrics in enumerate(taskwise_metrics):
                    log_data.update({
                        f"Step Validation {multitask_labels[i]} AUROC": metrics['auroc'],
                        f"Step Validation {multitask_labels[i]} AUPRC": metrics['auprc'],
                        f"Step Validation {multitask_labels[i]} Loss": np.mean(task_losses[i]),
                    })
                log_data.update({
                    "Step Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                    "Step Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                    "Step Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                    "Step Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                    "Step Validation Phenotype Loss": phenotype_loss.item(),
                })
                
                for k, v in log_data.items():
                    print(f"{k}: {v}")
                wandb.log(log_data)
    #     accelerator.wait_for_everyone()
        
    #     if accelerator.is_local_main_process:
    #         wandb.log({
    #             "Val Loss": np.mean(val_loss),
    #             "Val Precision": np.mean(val_precision),
    #             "Val Recall": np.mean(val_recall),
    #             "Val F1_score": np.mean(val_f1_score),
    #             "Val Auroc": np.mean(val_auroc),
    #             "Val Auprc": np.mean(val_auprc)
    #         })
            
    #         metrics = {
    #             'precision': np.mean(val_precision),
    #             'recall': np.mean(val_recall),
    #             'f1_score': np.mean(val_f1_score),
    #             'auroc': np.mean(val_auroc),
    #             'auprc': np.mean(val_auprc)
    #         }
        
    # return np.mean(val_loss), metrics
        
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    all_preds = accelerator.gather(all_preds).cpu().numpy()
    all_labels = accelerator.gather(all_labels).cpu().numpy()
    
    all_phenotype_preds = torch.cat(all_phenotype_preds)
    all_phenotype_labels = torch.cat(all_phenotype_labels)
    
    all_phenotype_preds = accelerator.gather(all_phenotype_preds).cpu().numpy()
    all_phenotype_labels = accelerator.gather(all_phenotype_labels).cpu().numpy()

    epoch_loss = total_loss / len(data_loader)
    taskwise_metrics = calculate_taskwise_metrics(all_preds, all_labels, num_tasks=args.num_tasks)
    phenotype_metrics = calculate_phenotype_metrics(all_phenotype_preds, all_phenotype_labels, num_tasks=25)

    if accelerator.is_local_main_process:
        log_data = {
            "Epoch": epoch,
            "Validation Loss": epoch_loss
            }
        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                # f"Validation {multitask_labels[i]} Accuracy": metrics['accuracy'],
                f"Validation {multitask_labels[i]} AUROC": metrics['auroc'],
                # f"Validation {multitask_labels[i]} Precision": metrics['precision'],
                # f"Validation {multitask_labels[i]} Recall": metrics['recall'],
                # f"Validation {multitask_labels[i]} F1_score": metrics['f1_score'],
                f"Validation {multitask_labels[i]} AUPRC": metrics['auprc'],
                f"Validation {multitask_labels[i]} Loss": np.mean(task_losses[i]),
            })
        log_data.update({
            "Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
            "Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
            "Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
            "Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
            "Validation Phenotype Loss": np.mean(phenotype_losses),
        })
        for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
            log_data.update({
                f"Validation Phenotype [{multilabel_labels[i]}] AUROC": auroc,
            })
        for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
            log_data.update({
                f"Validation Phenotype [{multilabel_labels[i]}] AUPRC": auprc,
            })
            
        

        for k, v in log_data.items():
            print(f"{k}: {v}")
        wandb.log(log_data)
    
    return epoch_loss, taskwise_metrics, phenotype_metrics



  
# def test(
#     device: torch.device,
#     model: torch.nn.Module,
#     data_loader: torch.utils.data.DataLoader,
#     scaler: torch.amp.GradScaler,
#     accelerator: Accelerator,
#     criterion: torch.nn,
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