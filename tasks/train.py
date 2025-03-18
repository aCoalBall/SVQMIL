import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from constants import DEVICE, RECON, BOTH, CLS
from dataset import PatchDataset
from models.vq import VectorQuantizer, VQMIL
from models.svq import split_VectorQuantizer, split_VQMIL
from models.aggregator import FeatureEncoder, Decoder, ABMIL_Head
from utils.partition import get_split_slidenames


def train_mil(split:int, feature_dir, split_csv, label_csv, dim=512,
                           lr=1e-4, epoch=400, record_time=False):
    encoder = FeatureEncoder(out_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = nn.Sequential(encoder, head).to(DEVICE)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv, label_csv)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_features(slidename, feature_dir):
        feature_path = os.path.join(feature_dir, slidename + '.pt')
        feature = torch.load(feature_path)
        return feature

    total_time = 0
    for e in range(epoch):
        #Train
        s_time = time.time()
        model.train()
        random.shuffle(train_slides)
        for slidename, label in train_slides:
            optimizer.zero_grad()
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            pred, A, Z = model(feature)
            cls_loss = loss_fn(pred, label) 
            loss = cls_loss
            loss.backward()
            optimizer.step()
        e_time = time.time()
        total_time += (e_time - s_time)

        #test
        if (e + 1) % 100 == 0:
            ts_time = time.time()
            model.eval()
            preds = []
            pred_labels = []
            labels = []
            for slidename, label in test_slides:
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                pred, A, Z = model(feature)
                cls_loss = loss_fn(pred, label)
                preds.append(pred[1].item())
                pred_labels.append(torch.argmax(pred).item())
                labels.append(label.item())
            test_acc = accuracy_score(labels, pred_labels)
            test_f1 = f1_score(labels, pred_labels)
            test_auc = roc_auc_score(labels, preds)
            te_time = time.time()
            print('inference time: ', te_time - ts_time)
            print('epoch : ', e + 1)
            print('test acc : ', test_acc)
            print('test f1 : ', test_f1)
            print('test auc : ', test_auc)
            print('\n')

            #torch.save(model.state_dict(), f=model_save_dir)    
            print('\n', flush=True)
    print(total_time, flush=True)


def train_vqmil(split:int, feature_dir, split_csv, label_csv, 
                           dim=512, num_embeddings=32, lr=1e-4, epoch=400,
                           commitment_cost=0.25):
    encoder = FeatureEncoder(out_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = VQMIL(encoder=encoder, head=head, dim=dim, num_embeddings=num_embeddings, commitment_cost=commitment_cost).to(DEVICE)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv, label_csv)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    #early_stopping = EarlyStopping(patience=20, stop_epoch=150, verbose=False)

    def get_features(slidename, feature_dir):
        feature_path = os.path.join(feature_dir, slidename + '.pt')
        feature = torch.load(feature_path)
        return feature

    for e in range(epoch):
        #Train
        model.train()
        random.shuffle(train_slides)
        for slidename, label in train_slides:
            optimizer.zero_grad()
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            vq_loss, pred, encodings, A, Z = model(feature)
            cls_loss = loss_fn(pred, label) 
            loss = vq_loss + cls_loss
            loss.backward()
            optimizer.step()

        #test
        if (e + 1) % 100 == 0:
            model.eval()
            preds = []
            pred_labels = []
            labels = []
            for slidename, label in test_slides:
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(feature)
                cls_loss = loss_fn(pred, label)
                preds.append(pred[1].item())
                pred_labels.append(torch.argmax(pred).item())
                labels.append(label.item())
            test_acc = accuracy_score(labels, pred_labels)
            test_f1 = f1_score(labels, pred_labels)
            test_auc = roc_auc_score(labels, preds) 
            print('epoch : ', e + 1)
            print('test acc : ', test_acc)
            print('test f1 : ', test_f1)
            print('test auc : ', test_auc)
            print('\n')

            #torch.save(model.state_dict(), f=model_save_dir)    
            print('\n', flush=True)



def train_s_vqmil(feature_dir, split, split_csv, model_save_dir, label_csv, split_vq=32,
                           dim=512, num_embeddings=64, lr=1e-4, recon=True, recon_epoch=400, 
                           epoch=100, commitment_cost=0.25, ema=False, pseudo=False, pseudo_bag_size=256, save=False, record_time=False):
    encoder = FeatureEncoder(out_dim=dim)
    decoder = Decoder(input_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = split_VQMIL(train_mode=CLS, encoder=encoder, decoder=decoder, cls_head=head, dim=dim, num_embeddings=num_embeddings, split=split_vq, commitment_cost=commitment_cost, ema=ema).to(DEVICE)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv, label_csv)

    cls_loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    normal_dataset = PatchDataset(features_pt='/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/split%d/normal.pt'%split, label=0)
    tumor_dataset = PatchDataset(features_pt='/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/split%d/tumor.pt'%split, label=1)
    normal_loader = DataLoader(dataset=normal_dataset, batch_size=pseudo_bag_size, shuffle=True)
    tumor_loader = DataLoader(dataset=tumor_dataset, batch_size=pseudo_bag_size, shuffle=True)

    #early_stopping = EarlyStopping(patience=20, stop_epoch=150, verbose=False)

    def get_features(slidename, feature_dir):
        feature_path = os.path.join(feature_dir, slidename + '.pt')
        feature = torch.load(feature_path)
        return feature

    if recon:
        for e in range(recon_epoch):
            #Reconstruction Train
            model.set_train_mode(RECON)
            model.train()
            random.shuffle(train_slides)
            for slidename, label in train_slides:
                optimizer.zero_grad()
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                vq_loss, recon_feature = model(feature)
                recon_loss = recon_loss_fn(recon_feature, feature) 
                loss = vq_loss + recon_loss
                recon_loss.backward()
                optimizer.step()
    
    total_time = 0
    for e in range(epoch):

        #Classification Train
        model.set_train_mode(CLS)
        model.train()

        if pseudo:
            tumor_label = torch.tensor(tumor_loader.dataset.label).to(DEVICE)
            normal_label = torch.tensor(normal_loader.dataset.label).to(DEVICE)
            for normal_batch, tumor_batch in zip(normal_loader, tumor_loader):
                tumor_batch = tumor_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(tumor_batch)
                cls_loss = cls_loss_fn(pred, tumor_label)
                loss = vq_loss + cls_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                normal_batch = normal_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(normal_batch)
                cls_loss = cls_loss_fn(pred, normal_label)
                loss = vq_loss + cls_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        s_time = time.time()

        random.shuffle(train_slides)
        for slidename, label in train_slides:
            optimizer.zero_grad()
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            vq_loss, pred, encodings, A, Z = model(feature)
            cls_loss = cls_loss_fn(pred, label) 
            loss = vq_loss + cls_loss
            loss.backward()
            optimizer.step()
        
        e_time = time.time()
        total_time += (e_time - s_time)

        #test
        if (e + 1) % 50 == 0:
            ts_time = time.time()
            model.eval()
            preds = []
            pred_labels = []
            labels = []
            for slidename, label in test_slides:
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(feature)
                cls_loss = cls_loss_fn(pred, label)
                preds.append(pred[1].item())
                pred_labels.append(torch.argmax(pred).item())
                labels.append(label.item())
            test_acc = accuracy_score(labels, pred_labels)
            test_f1 = f1_score(labels, pred_labels)
            test_auc = roc_auc_score(labels, preds) 
            te_time = time.time()
            print('inference time: ', te_time - ts_time)
            print('epoch : ', e + 1)
            print('test acc : ', test_acc)
            print('test f1 : ', test_f1)
            print('test auc : ', test_auc)
            print('\n')

            if save:
                torch.save(model.state_dict(), f=model_save_dir)    
            print('\n', flush=True)

    if record_time:
        print(total_time, flush=True)



def train_s_vqmil_clip(feature_dir, split, split_csv, model_save_dir, label_csv, split_vq=32,
                           dim=512, num_embeddings=64, lr=1e-4, recon=True, recon_epoch=400, grad_clip=5.0, 
                           epoch=100, commitment_cost=0.25, ema=False, pseudo=False, pseudo_bag_size=256, save=False):
    encoder = FeatureEncoder(out_dim=dim)
    decoder = Decoder(input_dim=dim)
    head = ABMIL_Head(M=dim, L=dim)
    model = split_VQMIL(train_mode=CLS, encoder=encoder, decoder=decoder, cls_head=head, dim=dim, num_embeddings=num_embeddings, split=split_vq, commitment_cost=commitment_cost, ema=ema).to(DEVICE)

    train_slides, val_slides, test_slides = get_split_slidenames(split_csv, label_csv)

    cls_loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    normal_dataset = PatchDataset(features_pt='/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/split%d/normal.pt'%split, label=0)
    tumor_dataset = PatchDataset(features_pt='/home/coalball/projects/WSI/vqmil/data_feat/res50_imgn/Camelyon16_patch224_ostu_train/split%d/tumor.pt'%split, label=1)
    normal_loader = DataLoader(dataset=normal_dataset, batch_size=pseudo_bag_size, shuffle=True)
    tumor_loader = DataLoader(dataset=tumor_dataset, batch_size=pseudo_bag_size, shuffle=True)

    #early_stopping = EarlyStopping(patience=20, stop_epoch=150, verbose=False)

    def get_features(slidename, feature_dir):
        feature_path = os.path.join(feature_dir, slidename + '.pt')
        feature = torch.load(feature_path)
        return feature

    if recon:
        for e in range(recon_epoch):
            #Reconstruction Train
            model.set_train_mode(RECON)
            model.train()
            random.shuffle(train_slides)
            for slidename, label in train_slides:
                optimizer.zero_grad()
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                vq_loss, recon_feature = model(feature)
                recon_loss = recon_loss_fn(recon_feature, feature) 
                loss = vq_loss + recon_loss
                recon_loss.backward()
                optimizer.step()
    

    for e in range(epoch):

        #Classification Train
        model.set_train_mode(CLS)
        model.train()

        if pseudo:
            tumor_label = torch.tensor(tumor_loader.dataset.label).to(DEVICE)
            normal_label = torch.tensor(normal_loader.dataset.label).to(DEVICE)
            for normal_batch, tumor_batch in zip(normal_loader, tumor_loader):
                tumor_batch = tumor_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(tumor_batch)
                cls_loss = cls_loss_fn(pred, tumor_label)
                loss = vq_loss + cls_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                normal_batch = normal_batch.to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(normal_batch)
                cls_loss = cls_loss_fn(pred, normal_label)
                loss = vq_loss + cls_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        random.shuffle(train_slides)
        for slidename, label in train_slides:
            optimizer.zero_grad()
            feature = get_features(slidename, feature_dir).to(DEVICE)
            label = torch.tensor(label).to(DEVICE)
            vq_loss, pred, encodings, A, Z = model(feature)
            cls_loss = cls_loss_fn(pred, label) 
            loss = vq_loss + cls_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        #test
        if (e + 1) % 100 == 0:
            model.eval()
            preds = []
            pred_labels = []
            labels = []
            for slidename, label in test_slides:
                feature = get_features(slidename, feature_dir).to(DEVICE)
                label = torch.tensor(label).to(DEVICE)
                vq_loss, pred, encodings, A, Z = model(feature)
                cls_loss = cls_loss_fn(pred, label)
                preds.append(pred[1].item())
                pred_labels.append(torch.argmax(pred).item())
                labels.append(label.item())
            test_acc = accuracy_score(labels, pred_labels)
            test_f1 = f1_score(labels, pred_labels)
            test_auc = roc_auc_score(labels, preds) 
            print('epoch : ', e + 1)
            print('test acc : ', test_acc)
            print('test f1 : ', test_f1)
            print('test auc : ', test_auc)
            print('\n')

            if save:
                torch.save(model.state_dict(), f=model_save_dir)    
            print('\n', flush=True)

