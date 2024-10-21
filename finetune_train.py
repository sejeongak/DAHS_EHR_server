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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score



def calculate_metrics(predictions, labels): 
    probs = F.softmax(predictions, dim=1)
    preds = torch.argmax(probs, dim=1)
    
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
        if auprc > 1:
            print("auprc > 1!!!!!!!!!!!!!!!!!")
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

from sklearn.metrics import precision_recall_curve

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

    

def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    logging.info("Start training...")
    check_patience = 0
    
    best_auprc = 0.0
    best_epoch = 0
    
    
    
    for epoch in tqdm(range(start_epoch, epochs), desc='Epochs', total=epochs-start_epoch, smoothing=0.1):

        
        model.train()
        
        total_loss = 0  
        total_samples = 0  
        # train_loss = []
        all_preds = []
        all_labels = []
        
        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):
            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
                
                outputs = model(
                    input_ids = input_ids,
                    value_ids = value_ids,
                    unit_ids = unit_ids,
                    time_ids = time_ids,                
                    continuous_ids = continuous_ids,
                    position_ids = position_ids,
                    token_type_ids = token_type_ids,
                    age_ids = age_ids,
                    gender_ids = gender_ids,
                    task_token = task_token,
                    attention_mask=attention_mask,
                    global_attention_mask=None,
                    labels=labels,
                    return_dict=True,
                )
                
             
                
                
                loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                metrics = calculate_metrics(outputs, labels)
                
                # train_loss.append(loss.item())
                
                # train_precision_scores.append(metrics['precision'])
                # train_recall_scores.append(metrics['recall'])
                # train_f1_scores.append(metrics['f1_score'])
                # train_auroc.append(metrics['auroc'])
                # train_auprc.append(metrics['auprc'])
                                
                all_preds.append(outputs.view(-1, args.num_labels).detach().cpu().numpy())  
                all_labels.append(labels.view(-1).detach().cpu().numpy())
                            
                
                accelerator.backward(loss)
                
                if (step + 1) % args.acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
                
                if step != 0 and step % 50 == 0:   
                    if accelerator.is_local_main_process:
                        print(f"Step {step+1} / {len(data_loader)} | Train Loss: {loss} | Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1_Score: {metrics['f1_score']} | AUROC: {metrics['auroc']} | AUPRC: {metrics['auprc']}")     
                        wandb.log({
                            "Step": step+1,
                            "Step Loss": loss.item(),
                            "Step Accuracy": metrics['accuracy'],
                            "Step Precision": metrics['precision'],
                            "Step Recall": metrics['recall'],
                            "Step F1_score": metrics['f1_score'],
                            "Step Auroc": metrics["auroc"],
                            "Step Auprc": metrics["auprc"]
                        }) 
                        
                    
        epoch_loss = total_loss / total_samples
        
        all_preds = torch.cat([torch.tensor(pred) for pred in all_preds])
        all_labels = torch.cat([torch.tensor(label) for label in all_labels])
        all_metrics = calculate_metrics(all_preds, all_labels)
       
                        
        accelerator.wait_for_everyone()
        
        # if accelerator.is_local_main_process:
        #     print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.4f} | Precision: {np.mean(train_precision_scores):.4f} | Recall: {np.mean(train_recall_scores):.4f} | F1_score: {np.mean(train_f1_scores):.4f} | AUROC: {np.mean(train_auroc):.4f} | AUPRC: {np.mean(train_auprc):.4f}")
                 
        #     wandb.log({
        #             "Epoch": epoch+1,
        #             "Train Loss": np.mean(train_loss),
        #             "Train Precision": np.mean(train_precision_scores),
        #             "Train Recall": np.mean(train_recall_scores),
        #             "Train F1_score": np.mean(train_f1_scores),
        #             "Train Auroc": np.mean(train_auroc),
        #             "Train Auprc": np.mean(train_auprc)
        #     }) 
        # learning_rates = {f"Learning Rate ({param_group.get('name', f'Group {i+1}')})": param_group['lr'] for i, param_group in enumerate(optimizer.param_groups)}

        if accelerator.is_local_main_process:
            log_data = {
                "Epoch": epoch + 1,
                "Train Loss": epoch_loss,
                **{f"Train {key}": value for key, value in all_metrics.items()},
                # **learning_rates
            }
            print(log_data)
            wandb.log(log_data)
                    
        
        
        
        # losses.append(np.mean(train_loss))
        # precision_scores.append(np.mean(train_precision_scores))
        # recall_scores.append(np.mean(train_recall_scores))
        # f1_scores.append(np.mean(train_f1_scores))
        # auroc.append(np.mean(train_auroc))
        # auprc.append(np.mean(train_auprc))
        
        # model.eval()
        valid_loss, valid_metrics = validation(device, model, val_loader, scaler, accelerator, criterion, args)
        
        scheduler.step(valid_metrics['auprc'])
            
                    
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
        if valid_metrics['auprc'] > best_auprc:
            best_auprc = valid_metrics['auprc']
            best_epoch = epoch + 1
            if accelerator.is_local_main_process:
                wandb.log({
                    "Best Epoch": best_epoch,
                    "Best Loss": valid_loss,
                    **{f"Best {key}": valid_metrics[key] for key in valid_metrics}
                })
                if args.pretrain:
                    output_path = os.path.join(save_path, f"best_{args.mode}_pretrain_model.pth")
                else:
                    output_path = os.path.join(save_path, f"best_{args.mode}_not_pretrain_model.pth")
                accelerator.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, output_path)
            check_patience = 0
        else:
            check_patience += 1
            
        if check_patience >= patience:
            print(f"Early Stopping triggered after {epoch + 1} epochs.")
            break

   
            
            
      
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    criterion: torch.nn,
    args: dict
):
    
    logging.info("Start validation...")
    # val_loss = []
    # val_precision = []  
    # val_recall = []
    # val_f1_score = []
    # val_auroc = []
    # val_auprc = []
    valid_loss = []
    total_loss = 0  
    total_samples = 0  

    all_preds = []
    all_labels = []
    
    model.eval()
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
         
            
            outputs = model(
                input_ids = input_ids,
                value_ids = value_ids,
                unit_ids = unit_ids,
                time_ids = time_ids,                
                continuous_ids = continuous_ids,
                position_ids = position_ids,
                token_type_ids = token_type_ids,
                age_ids = age_ids,
                gender_ids = gender_ids,
                task_token = task_token,
                attention_mask=attention_mask,
                global_attention_mask=None,
                labels=labels,
                return_dict=True,)
            
            # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
            # metrics = calculate_metrics(outputs, labels)
            
            # val_loss.append(loss.item())
            # val_precision.append(metrics['precision'])
            # val_recall.append(metrics['recall'])
            # val_f1_score.append(metrics['f1_score'])
            # val_auroc.append(metrics['auroc'])
            # val_auprc.append(metrics['auprc'])
            
            # val_loss.append(loss.item())
            loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            # valid_loss.append(loss.item())
            # metrics = calculate_metrics(outputs, labels)
            
            if step != 0 and step % 30 == 0:   
                if accelerator.is_local_main_process:
                    metrics = calculate_metrics(outputs, labels)
                    print(f"Validation | Step {step+1} / {len(data_loader)} | Loss: {loss.item()} | Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1_Score: {metrics['f1_score']} | AUROC: {metrics['auroc']} | AUPRC: {metrics['auprc']}")     
                        
                    
            all_preds.append(outputs.view(-1, args.num_labels).detach().cpu().numpy())  
            all_labels.append(labels.view(-1).detach().cpu().numpy())
                        
            
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
        epoch_loss = total_loss / total_samples
        all_preds = torch.cat([torch.tensor(pred).to(device) for pred in all_preds])
        all_labels = torch.cat([torch.tensor(label).to(device) for label in all_labels])
        
        # all_preds = np.concatenate([pred for pred in all_preds])
        # all_labels = np.concatenate([label for label in all_labels])

        all_metrics = calculate_metrics(all_preds, all_labels)
        

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            log_data = {
                "Val Loss": epoch_loss,
                **{f"Val {key}": value for key, value in all_metrics.items()},
                            }
            print(log_data)
            wandb.log(log_data)
        
    return epoch_loss, all_metrics