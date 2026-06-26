from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from collections import Counter
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
import torch.distributed as dist
import math
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
    
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = [self.dataset[idx][-1].item() for idx in range(len(self.dataset))]  
        self.class_counts = Counter(self.labels)
        

        total_samples = sum(self.class_counts.values())
        self.weights = [1.0 / self.class_counts[label] for label in self.labels]

        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=len(self.dataset), replacement=True)
        
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
# class BalancedWeightSampler(Sampler):
#     def __init__(self, dataset):

#         self.dataset = dataset
#         self.labels = [self.dataset[idx][-1].item() for idx in range(len(self.dataset))]  
#         self.class_counts = Counter(self.labels)  
#         self.num_samples = len(self.dataset)  

 
#         self.weights = [1.0 / self.class_counts[label] for label in self.labels]

#     def __iter__(self):
#         sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.num_samples, replacement=True)
#         return iter(sampler)

#     def __len__(self):
#         return self.num_samples


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):

        self.dataset = dataset
        self.batch_size = batch_size
        
   
        self.labels = [self.dataset[i][-1].item() for i in range(len(dataset))]
        
        self.class_indices = {label: np.where(np.array(self.labels) == label)[0]
                               for label in set(self.labels)}
        
        self.num_classes = len(self.class_indices)
        self.batch_class_size = self.batch_size // self.num_classes
        
    def __iter__(self):

        indices = []
        for label, indices_for_class in self.class_indices.items():

            indices.extend(np.random.choice(indices_for_class, self.batch_class_size, replace=False))
        

        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Get labels for all samples
        labels = [dataset[i][-1].item() for i in range(len(dataset))]

        # Group indices by class
        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        # Shuffle class indices
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])

        # Calculate class ratios from dataset
        total_samples = len(labels)
        self.class_ratios = {
            label: len(indices) / total_samples for label, indices in self.class_indices.items()
        }

        # Calculate class samples per batch
        self.class_samples_per_batch = {
            label: int(round(self.class_ratios[label] * batch_size))
            for label in self.class_ratios
        }

        # Ensure total batch size matches
        total_class_samples = sum(self.class_samples_per_batch.values())
        if total_class_samples != batch_size:
            difference = batch_size - total_class_samples
            for label in sorted(self.class_samples_per_batch.keys()):
                self.class_samples_per_batch[label] += difference
                break

        # Precompute indices in flat list
        self.indices = self._create_indices()
        
    def _create_indices(self):
        indices = []

        # Generate full batches
        for _ in range(len(self.dataset) // self.batch_size):
            for label, count in self.class_samples_per_batch.items():
                indices.extend(self.class_indices[label][:count])
                self.class_indices[label] = self.class_indices[label][count:]

        # Handle remaining samples for the last batch
        remaining_indices = []
        for label in self.class_indices:
            remaining_indices.extend(self.class_indices[label])

        # Add remaining indices to create the last batch
        if remaining_indices:
            indices.extend(remaining_indices)

        return indices


    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)


class RandomOversamplingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size

  
        labels = [dataset[i][-1].item() for i in range(len(dataset))]

  
        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

  
        max_samples = max(len(indices) for indices in self.class_indices.values())


        self.oversampled_class_indices = {
            label: np.random.choice(indices, max_samples, replace=True).tolist()
            for label, indices in self.class_indices.items()
        }

        num_classes = len(self.oversampled_class_indices)
        self.samples_per_class = self.batch_size // num_classes

        remaining_samples = self.batch_size % num_classes
        self.class_samples_per_batch = {
            label: self.samples_per_class + (1 if i < remaining_samples else 0)
            for i, label in enumerate(sorted(self.oversampled_class_indices.keys()))
        }

        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        num_batches = len(self.dataset) // self.batch_size

        for label in self.oversampled_class_indices:
            np.random.shuffle(self.oversampled_class_indices[label])
            
        for _ in range(num_batches):
            batch = []
            for label, count in self.class_samples_per_batch.items():
                batch.extend(self.oversampled_class_indices[label][:count])
                self.oversampled_class_indices[label] = self.oversampled_class_indices[label][count:]

            np.random.shuffle(batch)
            indices.extend(batch)

        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
class RandomOversamplingSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

        labels = [dataset[i][-1].item() for i in range(len(dataset))]

        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        max_samples = max(len(indices) for indices in self.class_indices.values())

        self.oversampled_indices = []
        for label, indices in self.class_indices.items():
            oversampled = np.random.choice(indices, max_samples, replace=True).tolist()
            self.oversampled_indices.extend(oversampled)

        np.random.shuffle(self.oversampled_indices)

    def __iter__(self):
        return iter(self.oversampled_indices)

    def __len__(self):
        return len(self.oversampled_indices)
    


class RandomOversamplingDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, ratio=1.0, seed=42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.ratio = ratio
        self.seed = seed

        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        
        self.num_replicas = num_replicas
        self.rank = rank

        labels = [dataset[i][-1].item() for i in range(len(dataset))]

        self.class_indices = {label: [] for label in set(labels)}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
            
        max_samples = max(len(indices) for indices in self.class_indices.values())

        self.oversampled_indices = []
        
        np.random.seed(self.seed)

        for label, indices in self.class_indices.items():
            if len(indices) < max_samples:
                target_samples = int(max_samples * self.ratio) if label != max_samples else max_samples
                oversampled = np.random.choice(indices, target_samples, replace=True).tolist()
            else:
                oversampled = indices
            self.oversampled_indices.extend(oversampled)

        if self.shuffle:
            np.random.shuffle(self.oversampled_indices)

        self.total_size = len(self.oversampled_indices)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)  
        
        split_size = self.num_samples
        start_idx = self.rank * split_size
        end_idx = min((self.rank + 1) * split_size, self.total_size)

        self.oversampled_indices = self.oversampled_indices[start_idx:end_idx]

    def __iter__(self):
        return iter(self.oversampled_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        np.random.seed(self.seed + epoch) 
        if self.shuffle:
            np.random.shuffle(self.oversampled_indices)