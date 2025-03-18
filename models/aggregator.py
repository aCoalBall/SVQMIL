import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    def __init__(self, out_dim=128) -> None:
        super(FeatureEncoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, out_dim))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

class ABMIL(nn.Module):
    def __init__(self, M=512, L=512):
        super(ABMIL, self).__init__()
        self.M = M
        self.L = L

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, 1) 

    def forward(self, H):
        A_V = self.attention_V(H) 
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)   
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1) 
        Z = torch.mm(A, H) 
        return Z
    

class ABMIL_Head(nn.Module):
    def __init__(self, M=512, L=512):
        super(ABMIL_Head, self).__init__()
        self.M = M
        self.L = L

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, 1) 
        self.classifier = nn.Linear(self.M, 2)

    def forward(self, H):
        A_V = self.attention_V(H) 
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)   
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1) 
        Z = torch.mm(A, H) 
        Y_prob = self.classifier(Z)
        Y_prob = Y_prob.squeeze()
        return Y_prob, A, Z
    
class Decoder(nn.Module):
    def __init__(self, input_dim=512, out_dim=1024) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU())
        self.fc2 = nn.Linear(512, out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x