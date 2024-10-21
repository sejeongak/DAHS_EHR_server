# Python
import os
from pathlib import Path
from typing import Union, Tuple, List
import zipfile
import sklearn.metrics as skm
import torch.nn.functional as F

# Data
import numpy as np
import pandas as pd

# Local

# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# ETC
from tqdm import tqdm

# def cal_acc(label, pred):
#     logs = nn.LogSoftmax()
#     label = label.cpu().numpy()
#     ind = np.where(label!=-1)[0]
#     truepred = pred.detach().cpu().numpy()
#     truepred = truepred[ind]
#     truelabel = label[ind]
#     truepred = logs(torch.tensor(truepred))
#     outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
#     precision = skm.precision_score(truelabel, outs, average='micro')
#     return precision

def cal_acc(label, pred):
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = F.log_softmax(torch.tensor(truepred), dim=-1)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision