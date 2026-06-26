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

def compute_mlm_loss(predictions, labels):
    loss_fct = nn.CrossEntropyLoss()  
    return loss_fct(predictions.view(-1, predictions.size(-1)), labels.view(-1))

# def calculate_mlm_precision(predictions, labels):
#     predicted_labels = predictions.argmax(dim=-1)
#     valid_mask = labels != -100
    
#     correct_predictions = (predicted_labels == labels) & valid_mask
    
#     num_correct = correct_predictions.sum().item()
#     num_valid = valid_mask.sum().item()
    
#     precision = num_correct / num_valid if num_valid > 0 else 0.0
    
#     return precision

def calculate_mlm_precision(predictions, labels):
    correct_predictions = (predictions == labels)
    num_correct = correct_predictions.sum().item()
    num_valid = labels.numel()
    
    precision = num_correct / num_valid if num_valid > 0 else 0.0
    
    return precision




def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict
):
    logging.info("Start training...")
    check_patience = 0
    
    best_loss = 1e10
    best_epoch = 0
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", total=epochs-start_epoch):
        train_cls_loss = 0
        train_reg_loss = 0
        train_loss = 0
        all_predictions = []
        all_labels = []
        all_regressions = []
        all_values = []
    
        # start = time.time()
        
        additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(args.batch_size, 1).to(device)
        model.train()
        
        
        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader)):
            with accelerator.accumulate(model):
            
            
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)   
                input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels, values = batch
   

                batch_size = labels.size(0)
                # additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
                labels = torch.cat([additional_tokens, labels], dim=1)
                values = torch.cat([additional_tokens, values], dim=1)
                # with torch.autocast(device_type=device.type, dtype=torch.float16):   
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
                    mask_values=values,
                    return_dict=True,)
                
                prediction_scores = outputs["mlm_logits"][:, 3:, :]
                labels = labels[:, 3:]
                
                
                valid_mask = labels!=-100
                valid_predictions = prediction_scores.argmax(dim=-1)[valid_mask]
                valid_labels = labels[valid_mask]
                
                step_precision = calculate_mlm_precision(valid_predictions, valid_labels)

                all_predictions.append(valid_predictions)
                all_labels.append(valid_labels)
                
                valid_scores = prediction_scores[valid_mask]
                mlm_cls_loss = compute_mlm_loss(valid_scores, valid_labels)
                train_cls_loss += mlm_cls_loss.item()
                
                # loss = compute_mlm_loss(prediction_scores, labels)
                # precision = calculate_mlm_precision(prediction_scores, labels)
                
                # loss = torch.clamp(loss, min=1e-6)
                # train_precision.append(precision)
                # train_loss.append(loss.item())
                
                if args.regression_mode:
                    regression_logits = outputs["regression_logits"][:, 3:]
                    values = values[:, 3:]
                    regression_mask = values != -100
                    
                    valid_regression_logits = regression_logits[regression_mask]
                    valid_regression_values = values[regression_mask]
                    
                    mlm_reg_loss = nn.MSELoss()(valid_regression_logits, valid_regression_values)
                    
                    all_regressions.append(valid_regression_logits)
                    all_values.append(valid_regression_values)
                    
                    train_reg_loss += mlm_reg_loss.item()
                    
                    batch_loss = mlm_cls_loss * (1 - args.loss_alpha) + mlm_reg_loss * args.loss_alpha
                    train_loss += batch_loss.item()
                    
                    
                else:
                    train_loss = train_cls_loss
                    
                    accelerator.backward(mlm_cls_loss)
                
                # if args.acc > 1:
                #     loss = loss / args.acc
                    
                # scaler.scale(loss).backward()
                
                accelerator.backward(batch_loss)
        
                # if (step + 1) % args.acc == 0:
                #     scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
                #     # scale_before_step = scaler.get_scale()
                #     scaler.step(optimizer)
                #     scaler.update()
                
                #     # skip_update = scale_before_step == scaler.get_scale()
                #     # if not skip_update:
                #     #     scaler.update()
                        
                #     optimizer.zero_grad()
                #     scheduler.step()
                
                if (step+1) % args.clip_interval == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                if args.acc == 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                elif (step + 1) % args.acc == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                # if torch.isnan(batch_loss):
                #     print(f"Nan detected at epoch {epoch}, step {step}")
                #     print(f"Learning rate: {scheduler.get_lr()}")
                                    
                    
                if step != 0 and step % 300 == 0:   
                    if accelerator.is_local_main_process:
                        print(f"Epoch {epoch+1}, Step {step}, Learning Rate: {scheduler.get_lr()}")

                        print(f"Epoch {epoch+1} | Step {step} | Train Loss: {batch_loss.item():.4f} | MLM Precision: {step_precision:.4f} | Value Regression loss: {mlm_reg_loss.item():.4f}")     
                        wandb.log({
                            "Step": step,
                            "Step Loss": round(batch_loss.item(), 4),
                            "Step Precision": round(step_precision, 4),
                            "Step Value Regression loss": round(mlm_reg_loss.item(), 4),
                        }) 
                        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)   
        all_regressions = torch.cat(all_regressions, dim=0)
        all_values = torch.cat(all_values, dim=0)
        
        precision = calculate_mlm_precision(all_predictions, all_labels)
        reg_loss = nn.MSELoss()(all_regressions, all_values).item()
        avg_loss = train_loss / len(data_loader)
        
    
        accelerator.wait_for_everyone()

                
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Value Regression loss: {reg_loss:.4f}")
            wandb.log({
                "Epoch": epoch+1,
                "Train Loss": round(avg_loss, 4),
                "Train Precision": round(precision, 4),
                "Train Value Regression loss": round(reg_loss, 4)
            }) 
        
 
        valid_loss, valid_precision, valid_reg_loss = validation(device, model, val_loader, scaler, accelerator, args)
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} | Valid Loss: {valid_loss:.4f} | Valid Precision: {valid_precision:.4f} | Valid Value Regression loss: {valid_reg_loss:.4f}")
            wandb.log({
                "Validation Loss": round(valid_loss, 4),
                "Validation Precision": round(valid_precision, 4),
                "Validation Value Regression loss": round(valid_reg_loss, 4)
            })
        
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            
            if accelerator.is_local_main_process:
                wandb.log({
                    'Best Epoch': best_epoch,
                    'Best Loss': best_loss,
                    'Best Precision': valid_precision,
                    'Best Value Regression loss': valid_reg_loss
                })
                if args.resume:
                    output_path = Path(save_path) / f"best_pretrain_model_resume_{epoch+1}.pth"
                else:
                    output_path = Path(save_path) / f"best_pretrain_model_{args.exp_name}.pth"
                accelerator.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, output_path)
                
            check_patience = 0
        else:
            check_patience += 1
            
        if epoch > epochs // 2 and check_patience >= patience:
            break
        
        accelerator.wait_for_everyone()

        
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    args: dict
):
    
    logging.info("Start validation...")
    val_loss = 0
    val_cls_loss = 0
    val_reg_loss = 0
    all_predictions = []
    all_labels = []
    all_regressions = []
    all_values = []
    
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels, values = batch
            
            batch_size = labels.size(0)
            additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
            labels = torch.cat([additional_tokens, labels], dim=1)
            values = torch.cat([additional_tokens, values], dim=1)
            # with torch.autocast(device_type=device.type, dtype=torch.float16):
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
                mask_values=values,
                return_dict=True,)
                
            prediction_scores = outputs["mlm_logits"][:, 3:, :]
            labels = labels[:, 3:]
            
        
            valid_mask = labels!=-100
            valid_predictions = prediction_scores.argmax(dim=-1)[valid_mask]
            valid_labels = labels[valid_mask]
            

            all_predictions.append(valid_predictions)
            all_labels.append(valid_labels)
            
            valid_scores = prediction_scores[valid_mask]
            mlm_cls_loss = compute_mlm_loss(valid_scores, valid_labels)
            val_cls_loss += mlm_cls_loss.item()
            
            regression_logits = outputs["regression_logits"][:, 3:]
            values = values[:, 3:]
            regression_mask = values != -100
            
            valid_regression_logits = regression_logits[regression_mask]
            valid_regression_values = values[regression_mask]
            
            mlm_reg_loss = nn.MSELoss()(valid_regression_logits, valid_regression_values)
            
            all_regressions.append(valid_regression_logits)
            all_values.append(valid_regression_values)
            
            val_reg_loss += mlm_reg_loss.item()
            
            batch_loss = mlm_cls_loss * (1 - args.loss_alpha) + mlm_reg_loss * args.loss_alpha
            val_loss += batch_loss.item()
            
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)  
        all_regressions = torch.cat(all_regressions, dim=0)
        all_values = torch.cat(all_values, dim=0)
        
        precision = calculate_mlm_precision(all_predictions, all_labels)
        reg_loss = nn.MSELoss()(all_regressions, all_values).item()
        avg_loss = val_loss / len(data_loader)

        accelerator.wait_for_everyone()
     
            
    
    return avg_loss, precision, reg_loss