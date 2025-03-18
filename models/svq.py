import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import CLS, RECON, BOTH

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=512, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) #the codebook
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        # Reshape input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=512, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) #the codebook
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        input_shape = inputs.shape
        # Reshape input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices


class split_VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=512, split=32, commitment_cost=0.25, ema=False):
        super().__init__()
        self.part = embedding_dim // split
        self.split = split
        if ema:
            self.vq = VectorQuantizerEMA(num_embeddings=num_embeddings, embedding_dim=self.part, commitment_cost=commitment_cost)
        else:
            self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=self.part, commitment_cost=commitment_cost)

    def forward(self, z_e):
        loss, quantized, encoding_indices = self.vq(z_e.reshape(-1, self.part))
        return loss, quantized.reshape(z_e.shape[0], -1), encoding_indices.reshape(z_e.shape[0], -1)
    
    
class split_VQMIL(nn.Module):
    def __init__(self, train_mode, encoder, decoder, cls_head, dim, num_embeddings, split, commitment_cost=0.25, ema=False):
        super().__init__()
        self.train_mode = train_mode
        self.encoder = encoder
        self.decoder = decoder
        self.vqer = split_VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=dim, split=split, commitment_cost=commitment_cost, ema=ema)
        self.cls_head = cls_head
    
    def set_train_mode(self, train_mode):
        self.train_mode = train_mode
    
    def forward(self, x):
        x = self.encoder(x)
        vq_loss, x, encodings = self.vqer(x)
        if self.train_mode == CLS:
            x, A, Z = self.cls_head(x)
            return vq_loss, x, encodings, A, Z
        
        elif self.train_mode == RECON:
            return vq_loss, self.decoder(x)
        
        elif self.train_mode == BOTH:
            x, A, Z = self.cls_head(x)
            recon = self.decoder(x)
            return vq_loss, x, encodings, A, Z, recon

class split_VQMIL_no_head(nn.Module):
    def __init__(self, encoder, dim, num_embeddings, split, commitment_cost=0.25) -> None:
        super().__init__()
        self.encoder = encoder
        self.vqer = split_VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=dim, split=split, commitment_cost=commitment_cost)
    
    def forward(self, x):
        x = self.encoder(x)
        vq_loss, x, encodings = self.vqer(x)
        return vq_loss, x, encodings