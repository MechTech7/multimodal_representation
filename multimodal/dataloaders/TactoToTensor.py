import torch
import numpy as np

class ToTensor(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        for k in sample.keys():
            if k.startswith('flow'):
                sample[k] = sample[k].transpose((2, 0, 1))
        
        new_dict = dict()
        for k, v in sample.items():
            if self.device is None:
                new_dict[k] = torch.FloatTensor(v)
            else:
                new_dict[k] = torch.from_numpy(v).float()
        
        return new_dict