import os
from collections import Counter
import pickle

import numpy as np
import torch
import torch.nn as nn
import h5py
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from constants import *
from models.aggregator import FeatureEncoder, Decoder, ABMIL_Head
from models.svq import split_VQMIL
from utils.partition import partition_tissues, get_split_slidenames


def get_codeword_frequency(save_file='tf_idf_train.pkl', h5_coord_dir = '/home/coalball/projects/WSI/vqmil/patches/Camelyon16_patch224_ostu/patches',
                           xml_dir = '/home/coalball/projects/WSI/vqmil/annotations',
                           feat_dir = '/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                           checkpoint = 'checkpoints/svqmil_s0.pth', split=32, dataset='train'):
    dim=512
    num_embeddings = 64
    split_vq = split
    encoder = FeatureEncoder(out_dim=dim)
    decoder = Decoder(input_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = split_VQMIL(train_mode=CLS, encoder=encoder, decoder=decoder, cls_head=head, dim=dim, num_embeddings=num_embeddings, split=split_vq, commitment_cost=0.25, ema=False).to(DEVICE)
    
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv='splits/task_camelyon16/splits_0.csv', label_csv='labels/labels_all.csv')

    tumor_tf_sum = [0,] * 64
    normal_tf_sum = [0,] * 64
    contain_in_docs = [0,] * 64
    num_docs = 0
    
    if dataset == 'train':
        target_slides = train_slides
    else:
        target_slides = test_slides
    for slide, label in target_slides:
        if label == 1:
            try:
                h5_path = os.path.join(h5_coord_dir, slide + '.h5')
                xml_path = os.path.join(xml_dir, slide + '.xml')
                tumor, _, normal = partition_tissues(xml_path=xml_path, h5_path=h5_path)
                tumor = set([tuple(c) for c in tumor])
                normal = set([tuple(c) for c in normal])

                h5 = h5py.File(h5_path, 'r')
                coords = h5['coords'][:]
                coords = [tuple(c) for c in coords]

                feat_path = os.path.join(feat_dir, slide + '.pt')
                feat = torch.load(feat_path).to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(feat)

                if split == 1:
                    encodings = [[e.cpu().item(),] for e in encodings.squeeze()]
                else:
                    encodings = [list(e.cpu().numpy()) for e in encodings.squeeze()]
                
                for i in range(len(encodings)):
                    coord = coords[i]
                    #each encoding as a document
                    encoding = encodings[i]
                    
                    #Count TF
                    if coord in tumor:
                        tumor_tf = [0,] * 64
                        for e in encoding:
                            tumor_tf[e] += 1
                        for e in range(64):
                            tumor_tf[e] = tumor_tf[e] / 32
                        for e in range(64):
                            tumor_tf_sum[e] = tumor_tf_sum[e] + tumor_tf[e]
                    elif coord in normal:
                        normal_tf = [0,] * 64
                        for e in encoding:
                            normal_tf[e] += 1
                        for e in range(64):
                            normal_tf[e] = normal_tf[e] / 32
                        for e in range(64):
                            normal_tf_sum[e] = normal_tf_sum[e] + normal_tf[e]

                    for code in encoding:
                        contain_in_docs[code] += 1
                    #Update total number of instances (docs)
                    num_docs += 1
            except:
                continue

    with open(save_file, 'wb') as file:
        pickle.dump((tumor_tf_sum, normal_tf_sum, num_docs, contain_in_docs), file)



def get_codeword_frequency_slides(save_file='tf_idf_slides.pkl', h5_coord_dir = '/home/coalball/projects/WSI/vqmil/patches/Camelyon16_patch224_ostu/patches',
                           xml_dir = '/home/coalball/projects/WSI/vqmil/annotations',
                           feat_dir = '/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                           checkpoint = 'checkpoints/svqmil_s0.pth', split=32):
    dim=512
    num_embeddings = 64
    split_vq = split
    encoder = FeatureEncoder(out_dim=dim)
    decoder = Decoder(input_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = split_VQMIL(train_mode=CLS, encoder=encoder, decoder=decoder, cls_head=head, dim=dim, num_embeddings=num_embeddings, split=split_vq, commitment_cost=0.25, ema=False).to(DEVICE)
    
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv='splits/task_camelyon16/splits_0.csv', label_csv='labels/labels_all.csv')

    tumor_tf_sum = [0,] * 64
    normal_tf_sum = [0,] * 64
    contain_in_docs_sum = [0,] * 64
    num_docs = 0
    
    for slide, label in train_slides:

        h5_path = os.path.join(h5_coord_dir, slide + '.h5')
        h5 = h5py.File(h5_path, 'r')
        coords = h5['coords'][:]
        coords = [tuple(c) for c in coords]

        feat_path = os.path.join(feat_dir, slide + '.pt')
        feat = torch.load(feat_path).to(DEVICE)
        vq_loss, pred, encodings, A, Z = model(feat)

        if split == 1:
            encodings = [[e.cpu().item(),] for e in encodings.squeeze()]
        else:
            encodings = [list(e.cpu().numpy()) for e in encodings.squeeze()]

        if label == 1:
            
            tumor_tf = [0,] * 64
            contain_in_docs = [0,] * 64

            
            for i in range(len(encodings)):
                #each encoding as a document
                encoding = encodings[i]                
                #Count TF
                for code in encoding:
                    tumor_tf[code] += 1
                    contain_in_docs[code] = 1
            #Update total number of slides (docs)
            tumor_tf = [x / len(encodings * 32) for x in tumor_tf]
            for i in range(64):
                tumor_tf_sum[i] = tumor_tf_sum[i] + tumor_tf[i]
                contain_in_docs_sum[i] += contain_in_docs[code]

            num_docs += 1
        
        else:
            normal_tf = [0,] * 64
            contain_in_docs = [0,] * 64

            
            for i in range(len(encodings)):
                #each encoding as a document
                encoding = encodings[i]                
                #Count TF
                for code in encoding:
                    normal_tf[code] += 1
                    contain_in_docs[code] = 1
            #Update total number of slides (docs)
            normal_tf = [x / len(encodings * 32) for x in normal_tf]
            for i in range(len(normal_tf_sum)):
                normal_tf_sum[i] = normal_tf_sum[i] + normal_tf[i]
                contain_in_docs_sum[i] += contain_in_docs[code]
            num_docs += 1          

    with open(save_file, 'wb') as file:
        pickle.dump((tumor_tf_sum, normal_tf, num_docs, contain_in_docs_sum), file)


def draw_umap_for_clustering(ckpt_svq='checkpoints/svqmil_s1.pth', feat_dir='/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/pt_files',
                   xml_dir='/home/coalball/projects/WSI/vqmil/annotations', h5_coord_dir='/home/coalball/projects/WSI/vqmil/patches/Camelyon16_patch224_ostu/patches'):
    dim = 512
    encoder = FeatureEncoder(out_dim=dim)
    decoder = Decoder(input_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = split_VQMIL(train_mode=CLS, encoder=encoder, decoder=decoder, cls_head=head, dim=dim, num_embeddings=64, split=32, commitment_cost=0.25, ema=False).to(DEVICE)
    state_dict = torch.load(ckpt_svq)
    model.load_state_dict(state_dict, strict=True)

    encoder = model.encoder
    vqer = model.vqer
    model = nn.Sequential(encoder, vqer).to(DEVICE)


    #get positive slides in the test set
    _, _, test_slides = get_split_slidenames(split_csv='splits/task_camelyon16/splits_1.csv', label_csv='labels/labels_all.csv')
    slides = []
    for slide, label in test_slides:
        if label == 1:
            slides.append(slide)


    #Draw umap for each positive slide
    for slide in slides:

        #get coords and coords of tumor
        xml_path = os.path.join(xml_dir, slide + '.xml')
        h5_coord_path = os.path.join(h5_coord_dir, slide + '.h5')
        tumor, boundary, normal = partition_tissues(xml_path=xml_path, 
                                        h5_path=h5_coord_path)
        positives = set([tuple(c) for c in tumor + boundary])

        h5 = h5py.File(h5_coord_path, 'r')
        coords = h5['coords'][:]
        coords = [tuple(c) for c in coords]


        #get original embedddings
        feat_path = os.path.join(feat_dir, slide + '.pt')
        feat = torch.load(feat_path).to(DEVICE)

        #get quantized embeddings
        _, quantized, encodings = model(feat)

        #do umap for original 
        feat = feat.detach().to('cpu').numpy()
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        Y_umap = umap_model.fit_transform(feat)

        #do umap for quantized
        quantized = quantized.detach().to('cpu').numpy()
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(quantized)

        #get instance labels
        labels = []
        for c in coords:
            if c in positives:
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)
        custom_colors = ['blue', 'blue']
        cmap = ListedColormap(custom_colors)


        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].scatter(Y_umap[:, 0], Y_umap[:, 1], c=labels, cmap=cmap, s=5, alpha=0.7)
        #axs[0][0].set_title("perlinmianry features from the vision encoder")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap=cmap, s=5, alpha=0.7)
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        save_path = os.path.join('figures', slide + '_umap.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()
        plt.close() 
        print('Done')
        #pass



