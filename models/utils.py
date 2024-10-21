import pandas as pd
import numpy as np
import os
import yaml
import torch
from os.path import join
import pickle

def age_vocab(min_age, max_age, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])
    
    for i in range(min_age,max_age+1):
        age2idx[str(i)] = len(symbol) + i-min_age
        idx2age[len(symbol) + i-min_age] = str(i)
   
    return age2idx, idx2age

  
def offset_vocab(min_offset, max_offset, symbol=None):
    offset2idx = {}
    idx2offset = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']
    
    for i in range(len(symbol)):
        offset2idx[str(symbol[i])] = i
        idx2offset[i] = str(symbol[i])
        
    for i in range(min_offset, max_offset+1):
        offset2idx[str(i)] = len(symbol) + i-min_offset
        idx2offset[len(symbol) + i-min_offset] = str(i)
        
    return offset2idx, idx2offset



def load_config(config_dir, model_type):
    """ Load the model configuration from a yaml file.
    
    Parameters
    ----------
    config_dir: str
        Directory containing the model configuration files

    model_type: str
        Model type to load configuration for

    Returns
    -------
    Any
        Model configuration

    """
    config_file = join(config_dir, f"{model_type}.yaml")
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    

def load_pretrain_data(
    data_dir,
    sequence_file,
    id_file
):
    """ Load the pretraining data
    Parameters
    ----------
    data_dir: str
        Directory containing the data files
    sequence_file: str
        Sequence file name
    id_file: str
        ID file name

    Returns
    -------
    pd.DataFrame
        Pretraining data

    """
    sequence_path = join(data_dir, sequence_file)
    id_path = join(data_dir, id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pd.read_parquet(sequence_path)
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    return data.loc[data["patient_id"].isin(patient_ids["pretrain"])]

def load_finetune_data(
    data_dir: str,
    sequence_file: str,
    id_file: str,
    valid_scheme: str,
    num_finetune_patients: str,
) -> pd.DataFrame:
    """Load the finetuning data.

    Parameters
    ----------
    data_dir: str
        Directory containing the data files
    sequence_file: str
        Sequence file name
    id_file: str
        ID file name
    valid_scheme: str
        Validation scheme
    num_finetune_patients: str
        Number of finetune patients

    Returns
    -------
    pd.DataFrame
        Finetuning data

    """
    sequence_path = join(data_dir, "patient_sequences", sequence_file)
    id_path = join(data_dir, "patient_id_dict", id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pd.read_parquet(sequence_path)
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    fine_tune = data.loc[
        data["patient_id"].isin(
            patient_ids["finetune"][valid_scheme][num_finetune_patients],
        )
    ]
    fine_test = data.loc[data["patient_id"].isin(patient_ids["test"])]
    return fine_tune, fine_test

# def get_run_id(
#     checkpoint_dir: str,
#     retrieve: bool = False,
#     run_id_file: str = "wandb_run_id.txt",
#     length: int = 8,
# ) -> str:
#     """Fetch the run ID for the current run.

#     If the run ID file exists, retrieve the run ID from the file.
#     Otherwise, generate a new run ID and save it to the file.

#     Parameters
#     ----------
#     checkpoint_dir: str
#         Directory to store the run ID file
#     retrieve: bool, optional
#         Retrieve the run ID from the file, by default False
#     run_id_file: str, optional
#         Run ID file name, by default "wandb_run_id.txt"
#     length: int, optional
#         String length of the run ID, by default 8

#     Returns
#     -------
#     str
#         Run ID for the current run

#     """
#     run_id_path = os.path.join(checkpoint_dir, run_id_file)
#     if retrieve and os.path.exists(run_id_path):
#         with open(run_id_path, "r") as file:
#             run_id = file.read().strip()
#     else:
#         run_id = str(uuid.uuid4())[:length]
#         with open(run_id_path, "w") as file:
#             file.write(run_id)
#     return run_id


def load_finetuned_model(
    model_type,
    model_path,
    tokenizer,
    pre_model_config,
    fine_model_config,
    device
):
    """
    Return a loaded finetuned model from model_path, using tokenizer information.

    If config arguments are not provided, the default configs built into the
    PyTorch classes are used.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model to load.
    tokenizer : tokenizer object
        Loaded tokenizer object that is used with the model.
    pre_model_config : dict, optional
        Config to override default values of a pretrained model.
    fine_model_config : dict, optional
        Config to override default values of a finetuned model.
    device : str, optional
        CUDA device to use for loading the model. Defaults to GPU if not specified.

    Returns
    -------
    torch.nn.Module
    The loaded PyTorch model.
    """
    
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Create the skeleton of a pretrained and finetuned model
    if model_type == 'longformer':
        pretrained_model = 