import torch
import pandas as pd
from torch.utils.data import Dataset
    
class PatchDataset(Dataset):
    def __init__(self, features_pt:str, label:int):
        super().__init__()
        self.features = torch.load(features_pt)
        self.label = label
        self.length = len(self.features)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.features[index]
    