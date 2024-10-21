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
    mask = torch.ones_like(labels)
    mask[:, :3] = 0  
    
    predictions = predictions.reshape(-1, predictions.size(-1))
    labels = labels.reshape(-1)
    mask = mask.reshape(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(predictions[mask.bool()], labels[mask.bool()])
    
    return loss

def calculate_mlm_precision(predictions, labels):
    mask = torch.ones_like(labels)
    mask[:, :3] = 0  
    
    predicted_labels = predictions.argmax(dim=-1)
    valid_mask = labels != -100
    
    correct_predictions = (predicted_labels == labels) & valid_mask & mask.bool()
    
    num_correct = correct_predictions.sum().item()
    num_valid = (valid_mask & mask.bool()).sum().item()
    
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
        train_precision = []
        train_loss = []
    
        # start = time.time()
        model.train()
        
        
        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader)):
            with accelerator.accumulate(model):
            
            
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)   
                input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
   

                batch_size = labels.size(0)
                additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
                labels = torch.cat([additional_tokens, labels], dim=1)
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
                    return_dict=True,)
                
                prediction_scores = outputs.logits
                
                loss = compute_mlm_loss(prediction_scores, labels)
                precision = calculate_mlm_precision(prediction_scores, labels)
                
                loss = torch.clamp(loss, min=1e-6)
                
                
                train_precision.append(precision)
                train_loss.append(loss.item())
                
                
                # if args.acc > 1:
                #     loss = loss / args.acc
                    
                # scaler.scale(loss).backward()
                accelerator.backward(loss)
        
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
                    
                if torch.isnan(loss):
                    print(f"Nan detected at epoch {epoch}, step {step}")
                    print(f"Learning rate: {scheduler.get_lr()}")
                                    
                    
                if step != 0 and step % 300 == 0:   
                    if accelerator.is_local_main_process:
                        print(f"Epoch {epoch+1}, Step {step}, Learning Rate: {scheduler.get_lr()}")

                        print(f"Epoch {epoch+1} | Step {step} | Train Loss: {np.mean(train_loss):.4f} | Precision: {np.mean(train_precision):.4f}")     
                        wandb.log({
                            "Step": step,
                            "Step Loss": np.mean(train_loss),
                            "Step Precision": np.mean(train_precision)
                        }) 
                        
                        
    
        accelerator.wait_for_everyone()

                
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.4f} | Precision: {np.mean(train_precision):.4f}")     
            wandb.log({
                "Epoch": epoch+1,
                "Train Loss": np.mean(train_loss),
                "Train Precision": np.mean(train_precision)
            }) 
        
 
        valid_loss, valid_precision = validation(device, model, val_loader, scaler, accelerator, args)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            
            if accelerator.is_local_main_process:
                wandb.log({
                    'Best Epoch': best_epoch,
                    'Best Loss': best_loss,
                    'Best Precision': valid_precision
                })
                if args.resume:
                    output_path = Path(save_path) / f"best_pretrain_model_resume_{epoch+1}.pth"
                else:
                    output_path = Path(save_path) / "best_pretrain_model.pth"
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
    val_loss = []
    val_precision = []
    
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
            
            batch_size = labels.size(0)
            additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
            labels = torch.cat([additional_tokens, labels], dim=1)
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
                return_dict=True,)
                
            prediction_scores = outputs.logits[:, 3:, :]
            labels = labels[:, 3:]
            
        
            loss = compute_mlm_loss(prediction_scores, labels)
            loss = torch.clamp(loss, min=1e-6)
            precision = calculate_mlm_precision(prediction_scores, labels)
            val_precision.append(precision)
            val_loss.append(loss.item())

        accelerator.wait_for_everyone()
     
            
        if accelerator.is_local_main_process:
            wandb.log({
                "Validation Loss": np.mean(val_loss),
                "Validation Precision": np.mean(val_precision)
            })
    return np.mean(val_loss), np.mean(val_precision)