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