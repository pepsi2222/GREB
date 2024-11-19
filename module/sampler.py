import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, ConcatDataset
import random
import numpy as np
from collections import defaultdict
import bisect

class DomainSampler(Sampler):
    def __init__(self, dataset: ConcatDataset, batch_size: int, shuffle: bool, iter_dict: dict, scale_dict: dict):
        self.dataset = dataset
        self.domain_indices = self._create_domain_indices()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter_dict = iter_dict
        self.iter_idx = np.concatenate(
                            [np.repeat(domain, cnt) for domain, cnt in iter_dict.items()])
        self.scale = scale_dict

    def _create_domain_indices(self):
        domain_indices = defaultdict(list)
        for idx in range(self.dataset.cumulative_sizes[-1]):
            domain_idx = bisect.bisect_right(self.dataset.cumulative_sizes, idx)
            domain = self.dataset.datasets[domain_idx].name
            domain_indices[domain].append(idx)
        return domain_indices

    def __iter__(self):
        iter_idx = list(self.iter_idx)
        iters = dict()
        for domain in self.iter_dict:
            if self.shuffle:
                random.shuffle(self.domain_indices[domain])
                self._swap_pairs(domain)
            iters[domain] = iter(self.domain_indices[domain])
        
        step = 0
        indices = []
        scale = self.scale
        while len(iter_idx) > 0:
            domain = iter_idx[step % len(iter_idx)]
            cur_iter = iters[domain]
            try:
                batch = [next(cur_iter)for _ in range(self.batch_size)]
                indices.extend(batch)
                step += 1
            except StopIteration:
                if scale[domain] > 1:
                    if self.shuffle:
                        random.shuffle(self.domain_indices[domain])
                        self._swap_pairs(domain)
                    iters[domain] = iter(self.domain_indices[domain])
                    scale[domain] -= 1
                else:
                    del iter_idx[step % len(iter_idx)]

        return iter(indices)

    def __len__(self):
        return len(self.dataset)
    

    def _swap_pairs(self, domain):
        idx = self.dataset.name2idx[domain]
        self.dataset.datasets[idx].pairs = [(y, x) for x, y in self.dataset.datasets[idx].pairs]