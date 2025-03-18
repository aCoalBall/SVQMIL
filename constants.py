import torch
import torchvision.transforms as transforms

NUM_THREADS=6

CLS = 0
RECON = 1
BOTH = 2

TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


DEVICE = torch.device('cuda')


wsi_dict = {
    'camelyon16': '/home/coalball/projects/WSI/camelyon16/wsis',
}

label_dict = {
    'camelyon16': 'labels/labels_all.csv',
}

h5_dict = {
    'camelyon16': 'patches/Camelyon16_patch224_ostu/patches',
}

annotation_dict = {
    'camelyon16': 'annotations',
}

feature_dict = {
    'camelyon16': 'data_feat/res50_imgn/Camelyon16_patch224_ostu_train/h5_files',
    'imgnet' : '../data/imgnet/resnet_features.pth',
}
