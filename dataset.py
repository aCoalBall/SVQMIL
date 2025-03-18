import os
import numpy as np
import torch
import h5py
import pandas as pd
from PIL import Image
from constants import TRANSFORM
from torch.utils.data import DataLoader, Dataset


class Patch_Dataset(Dataset):
    def __init__(self, dir_path:str, transform=TRANSFORM):
        super().__init__()
        self.image_paths = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]
        self.length = len(self.image_paths)
        self.transform = TRANSFORM
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        patch = Image.open(path)
        if self.transform:
            patch = self.transform(patch)
        return patch

class WSI_Feature_Dataset(Dataset):
    def __init__(self, slidenames_list:list, feature_dir:str):
        super().__init__()
        self.feature_dir = feature_dir
        self.slidenames_list = slidenames_list
        self.length = len(self.slidenames_list)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index:int):
        slidename = self.slidenames_list[index]
        feature_path = os.path.join(self.feature_dir, slidename + 'h5')
        h5 = h5py.File(feature_path, 'r')
        features = h5['features']
        coords = list(h5['coords'])

        return features, coords
    
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
    


def get_split_slidenames(split_csv:str, label_csv:str):
    split = pd.read_csv(split_csv)

    labels = pd.read_csv(label_csv).drop(columns='case_id')
    labels['label'] = labels['label'].map({'normal': 0, 'tumor': 1})
    labels = labels.set_index('slide_id')['label'].to_dict()

    train_slides = [x for x in split['train'].values if str(x) != 'nan'] 
    val_slides = [x for x in split['val'].values if str(x) != 'nan'] 
    test_slides = [x for x in split['test'].values if str(x) != 'nan'] 

    train_set = []
    for slidename in train_slides:
        try:
            label = labels[slidename]
            train_set.append((slidename, label))
        except:
            continue

    val_set = []
    for slidename in val_slides:
        try:
            label = labels[slidename]
            val_set.append((slidename, label))
        except:
            continue

    test_set = []
    for slidename in test_slides:
        try:
            label = labels[slidename]
            test_set.append((slidename, label))  
        except:
            continue 
    
    return train_set, val_set, test_set


class FeatureDataset(Dataset):
    def __init__(self, feature_h5:str='../res50_imgn/Camelyon16_patch224_ostu_train/h5_files/tumor_026.h5'):
        super().__init__()
        h5 = h5py.File(feature_h5, 'r')
        self.features = list(h5['features'])
        self.coords = list(h5['coords'])
        self.length = len(self.features)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.features[index], self.coords[index]