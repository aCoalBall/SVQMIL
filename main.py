import os
import argparse
import random
import torch

from constants import *
from tasks.train import train_vqmil, train_s_vqmil, train_mil

def get_args_parser():
    parser = argparse.ArgumentParser('Experiments', add_help=False)
    parser.add_argument('--task', default='vqmil', type=str)
    parser.add_argument('--dataset', default='camelyon16', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--split_dir', default='splits/task_camelyon16', type=str)
    parser.add_argument('--label_csv', default='labels/labels_all.csv', type=str)
    parser.add_argument('--split', default=32, type=int)
    parser.add_argument('--num_emb', default=64, type=int)
    parser.add_argument('--ema', action='store_true')
    return parser

def main(args):
    
    if args.task == 'svqmil':
        if args.dataset == 'camelyon16':
            for i in range(5):
                train_s_vqmil(feature_dir=args.feature_dir, model_save_dir='checkpoints/split%d/svqmil_s%d.pth'%(args.split, i), split=i,
                    split_csv=os.path.join(args.split_dir, 'splits_%d.csv'%i), label_csv=args.label_csv, split_vq=args.split,
                    dim=512, num_embeddings=args.num_emb,
                    lr=1e-4, recon=False, epoch=300, ema=args.ema, pseudo=False, save=False)
        elif args.dataset == 'tcga':
            for i in range(3):
                train_s_vqmil(feature_dir=args.feature_dir, model_save_dir='checkpoints/tcga/svqmil_s%d.pth'%i, split=i,
                    split_csv='splits/task_tcga/splits_%d.csv'%i, 
                    label_csv='labels/labels_tcga.csv', split_vq=args.split,
                    dim=512, num_embeddings=args.num_emb,
                    lr=1e-4, recon=False, epoch=100, ema=args.ema, pseudo=False, save=False)
    
if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    args = get_args_parser()
    args = args.parse_args()
    main(args)