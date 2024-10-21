from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch import nn, optim
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts
from transformers import LongformerConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.longformer.modeling_longformer import (
    LongformerForMaskedLM,
    LongformerModel,
    LongformerPreTrainedModel,
    LongformerSelfAttention,
    LongformerForSequenceClassification,
    LongformerLMHead,
    LongformerMaskedLMOutput,
    LongformerLayer,
    LongformerBaseModelOutput,
    LongformerBaseModelOutputWithPooling,
    LongformerSequenceClassifierOutput,
)

from models.embedding import EHREmbedding



class LongformerPretrain(pl.LightningModule):
    """ Longformer model for pretraining"""
    
    def __init__(
        self,
        vocab_size,
        itemid_size,
        max_position_embeddings,
        unit_size,
        continuous_size,
        task_size,
        max_age,
        gender_size,
        embedding_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        learning_rate,
        dropout_prob, 
        gpu_mixed_precision=True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.itemid_size = itemid_size
        self.max_position_embeddings = max_position_embeddings
        self.unit_size = unit_size
        self.continuous_size = continuous_size
        self.task_size = task_size
        self.max_age = max_age
        self.gender_size = gender_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.gpu_mixed_precision = gpu_mixed_precision
        self.train_mlm_precisions = [] 
        self.val_mlm_precisions = []
        
        self.config = LongformerConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.embedding_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            attention_window = [512] * self.num_hidden_layers,
        )
        
        self.config.vocab_size = self.itemid_size
        
        # LongformerForMaskedLM
        self.embeddings = EHREmbedding(
            config=self.config,
            itemid_size=itemid_size,
            unit_size=unit_size,
            max_age=max_age,
            max_len=max_position_embeddings,
            continuous_size=continuous_size,
            gender_size=gender_size,
            task_size=task_size,
            use_itemid=True,
            inputs_embeds=None,
        )
        
        self.model = LongformerForMaskedLM(config=self.config)
        # self.model.longformer.embeddings = self.embeddings
        
        self.post_init()
        
    def _init_weights(self, module: torch.nn.Module) -> None:
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def post_init(self):
        self.model.apply(self._init_weights)
        
    
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        continuous_ids,
        position_ids,
        token_type_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # self.embeddings.cache_input(value_ids=value_ids, 
        #                     unit_ids=unit_ids, 
        #                     time_ids=time_ids, 
        #                     continuous_ids=continuous_ids, 
        #                     age_ids=age_ids, 
        #                     gender_ids=gender_ids, 
        #                     task_token=task_token)
        combined_embed = self.embeddings(
            input_ids = input_ids,
            value_ids = value_ids,
            unit_ids = unit_ids,
            time_ids = time_ids,
            continuous_ids = continuous_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            age_ids = age_ids,
            gender_ids = gender_ids,
            task_ids = task_token
        )
        
        if global_attention_mask is None:
            global_attention = torch.zeros_like(attention_mask)
            global_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_prefix = torch.ones((attention_mask.shape[0], 3)).to(self.device)
            attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            
        # print(combined_embed.shape)
        # print(attention_mask.shape)
        # print(global_attention_mask.shape)
        return self.model(
            inputs_embeds=combined_embed,
            # position_ids=position_ids,
            # token_type_ids=type_ids,
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
    def calculate_mlm_precision(self, prediction_scores, labels):
        mask = labels != -100
        _, predictions = torch.max(prediction_scores, dim=-1)
        
        correct_predictions = (predictions == labels) & mask
        precision = correct_predictions.sum() / mask.sum()
    
        return precision.item()
    
    def on_train_epoch_end(self):
        
        avg_train_precision = sum(self.train_mlm_precisions) / len(self.train_mlm_precisions)
        self.log('avg_train_mlm_precision', avg_train_precision, prog_bar=True, sync_dist=True)
        self.train_mlm_precisions.clear()
    
    def on_validation_epoch_end(self):
        
        avg_val_precision = sum(self.val_mlm_precisions) / len(self.val_mlm_precisions)
        self.log('avg_val_mlm_precision', avg_val_precision, prog_bar=True, sync_dist=True)
        self.val_mlm_precisions.clear() 
    
    
    def training_step(self, batch, batch_idx):
        
        batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
        
        batch_size = labels.size(0)
        additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        labels = torch.cat([additional_tokens, labels], dim=1)
        with torch.autocast(enabled=self.gpu_mixed_precision, device_type='cuda'):
            
            outputs = self.forward(
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
            prediction_scores = outputs.logits[:, 3:, :]
            labels = labels[:, 3:]
        
            
            assert prediction_scores.shape[:-1] == labels.shape, "Prediction and Labels shape mismatch after slicing."
        
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.reshape(-1, self.itemid_size), labels.reshape(-1))
            
            mlm_precision = self.calculate_mlm_precision(prediction_scores, labels)
            self.train_mlm_precisions.append(mlm_precision)
                  
        (current_lr,) = self.lr_schedulers().get_last_lr()
        
        if batch_idx % 100 == 0:
            self.log_dict(
                dictionary={"train loss": loss, "train_learning_rate": current_lr},
                on_step=True,
                prog_bar=True,
                sync_dist=True,
            )
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
        batch_size = labels.size(0)
        additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        labels = torch.cat([additional_tokens, labels], dim=1)
        
        with torch.autocast(enabled=self.gpu_mixed_precision, device_type='cuda'):
            outputs = self.forward(
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
            
            prediction_scores = outputs.logits[:, 3:, :]
            labels = labels[:, 3:]

            assert prediction_scores.shape[:-1] == labels.shape, "Prediction and Labels shape mismatch after slicing."
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.reshape(-1, self.itemid_size), labels.reshape(-1))
            
            mlm_precision = self.calculate_mlm_precision(prediction_scores, labels)
            self.val_mlm_precisions.append(mlm_precision)
            
        (current_lr,) = self.lr_schedulers().get_last_lr()
        
        if batch_idx % 100 == 0:
            self.log_dict(
                dictionary={"val_loss": loss, "valid_learning_rate": current_lr},
                on_step=True,
                prog_bar=True,
                sync_dist=True,
            )   
        return {'valid_loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # Change optimizer if DeepSpeed strategy is used
        # optimizer = DeepSpeedCPUAdam(  # noqa: ERA001
        #     self.parameters(), lr=self.learning_rate, adamw_mode=True
        # )  # noqa: ERA001
        
        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(n_steps * 0.1)
        n_decay_steps = int(n_steps * 0.9)
        
        warmup = LinearLR(optimizer, 
                          start_factor=0.01,
                          end_factor=1.0,
                          total_iters=n_warmup_steps)
        
        decay = LinearLR(optimizer,
                         start_factor=1.0,
                         end_factor=0.01,
                         total_iters=n_decay_steps)
        
        scheduler = SequentialLR(optimizer, 
                                 schedulers=[warmup, decay],
                                 milestones=[n_warmup_steps])

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
       

class LongformerFinetune(pl.LightningModule):
    """ Longformer model for finetuning"""
    
    def __init__(
        self,
        pretrained_model: LongformerPretrain,
        problem_type: str = "single_label_classification",
        num_labels: int=2,
        learning_rate: float=5e-5,
        classifier_dropout: float=0.1,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.test_outputs = []
        
        self.config = pretrained_model.config
        self.config.num_labels = self.num_labels
        self.config.classifier_dropout = self.classifier_dropout   
        self.config.problem_type = problem_type
        
        
        self.model = LongformerForSequenceClassification(config=self.config)
        self.model.longformer = pretrained_model.model.longformer
        
        self.post_init()
        
    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def post_init(self):
        self.model.apply(self._init_weights)
        
    def forward(
        self,
        input_ids,
        value_ids,
        unit_ids,
        time_ids,
        continuous_ids,
        position_ids,
        token_type_ids,
        age_ids,
        gender_ids,
        task_token,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        self.model.longformer.embeddings.cache_input(value_ids, unit_ids, time_ids, continuous_ids, age_ids, gender_ids, task_token)
        
        global_attention = torch.zeros_like(attention_mask)
        global_prefix = torch.ones((attention_mask.shape[0], 3))
        global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        attention_prefix = torch.ones((attention_mask.shape[0], 3))
        attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
            
            
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
        
        # global_attention_mask = torch.zeros_like(input_ids)
        # global_attention_mask[:, :3] = 1
        
        with autocast():
            loss = self(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=time_ids,
                continuous_ids=continuous_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_token,
                attention_mask=attention_mask,
                global_attention_mask=None,
                labels=labels,
                return_dict=True,
            ).loss
            
        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train loss": loss, "learning_rate": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
        
        # global_attention = torch.zeros_like(attention_mask)
        # global_prefix = torch.ones((attention_mask.shape[0], 3))
        # global_attention_mask = torch.cat([global_prefix, global_attention], dim=1)
        
        # attention_prefix = torch.ones((attention_mask.shape[0], 3))
        # attention_mask = torch.cat([attention_prefix, attention_mask], dim=1)
        
        with autocast():
            loss = self(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=time_ids,
                continuous_ids=continuous_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_token,
                attention_mask=attention_mask,
                global_attention_mask=None,
                labels=labels,
                return_dict=True,
            ).loss
            
        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val loss": loss, "learning_rate": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss
    
    def task_step(self, batch, batch_idx):
        input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
        
        
        with autocast():
            outputs = self(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=time_ids,
                continuous_ids=continuous_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_token,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=-1)
        log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}
        
        self.test_outputs.append(log)
        
        return log
    
    
    def on_test_epoch_end(self):
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()
        
        self.test_outputs = {
            "labels": labels,
            "logits": logits,
        }
        
        if self.config.problem_type == "multi_label_classification":
            preds_one_hot = np.eye(labels.shape[1])[preds]
            accuracy = accuracy_score(labels, preds_one_hot)
            f1 = f1_score(labels, preds_one_hot, average="micro")
            precision = precision_score(labels, preds_one_hot, average="micro")
            recall = recall_score(labels, preds_one_hot, average="micro")
            roc_auc = roc_auc_score(labels, preds_one_hot, average="micro")
        else: # single_label_classification
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="micro")
            precision = precision_score(labels, preds, average="micro")
            recall = recall_score(labels, preds, average="micro")
            roc_auc = roc_auc_score(labels, preds, average="micro")
        
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_f1", f1)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_auc", roc_auc)
        
        return loss
    
    def configure_optimizers(self,):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(n_steps * 0.1)
        n_decay_steps = int(n_steps * 0.9)
        
        warmup = LinearLR(optimizer, 
                          start_factor=0.01,
                          end_factor=1.0,
                          total_iters=n_warmup_steps)
        
        decay = LinearLR(optimizer,
                        start_factor=1.0,
                        end_factor=0.01,
                        total_iters=n_decay_steps)
        
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup, decay],
                                 milestones=[n_warmup_steps])
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        
        
        
        
        
        