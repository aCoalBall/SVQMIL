import h5py
from lxml import etree
import pandas as pd
from shapely import Polygon
import os
import openslide
import torch
    

def partition_tissues(xml_path, h5_path, patch_size=224):
    # Load Annotations
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for annotation_elem in root.findall('.//Annotation'):
        coordinates = []
        for coord_elem in annotation_elem.findall('.//Coordinate'):
            x = float(coord_elem.get('X')) 
            y = float(coord_elem.get('Y'))
            coordinates.append((x, y))
        annotations.append(coordinates)
    #Get polygons (range of tumor tissues)
    polygons = []
    for a in annotations:
        polygon = Polygon(a)
        polygons.append(polygon)
    #Get patches
    h5 = h5py.File(h5_path, 'r')
    coords = h5['coords'][:]
    h5.close()
    #Check which patches are inside the tumor tissue, which are on the boudary, which are not
    tumor_coords = []
    boundary_coords = []
    normal_coords = []
    for c in coords:
        c_poly = Polygon([c, [c[0], c[1] + patch_size], [c[0] + patch_size, c[1]], [c[0] + patch_size, c[1] + patch_size]])
        normal = True
        for p in polygons:
            if p.contains(c_poly):
                tumor_coords.append(c)
                normal = False
                break
            elif p.intersects(c_poly):
                boundary_coords.append(c)
                normal = False
                break
        if normal:
            normal_coords.append(c)
    return tumor_coords, boundary_coords, normal_coords


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


def collect_patches(feat_dir, split_csv, split,
                    label_csv, 
                    sample_path='sample_feats'):
    
    train_set, val_set, test_set = get_split_slidenames(split_csv, label_csv)
    t_bags = []
    n_bags = []
    for slide, label in train_set:
        print('processing ', slide, flush=True)
        feat_path = os.path.join(feat_dir, slide + '.pt')
        feats = torch.load(feat_path)

        if label==0:
            n_bags.append(feats)
        else:
            t_bags.append(feats)

    t_bags = torch.vstack(t_bags)
    n_bags = torch.vstack(n_bags)

    os.makedirs(os.path.join(sample_path, 'split%d'%split), exist_ok=True)
    pos_path = os.path.join(sample_path, 'split%d/tumor.pth'%split)
    neg_path = os.path.join(sample_path, 'split%d/normal.pth'%split)
    torch.save(t_bags, pos_path)
    torch.save(n_bags, neg_path)
