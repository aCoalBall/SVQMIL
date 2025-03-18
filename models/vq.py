from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=128, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
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
    
class VQMIL(nn.Module):
    def __init__(self, encoder, head, dim, num_embeddings, commitment_cost=0.25) -> None:
        super().__init__()
        self.encoder = encoder
        self.vqer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=dim, commitment_cost=commitment_cost)
        self.head = head
    
    def forward(self, x):
        x = self.encoder(x)
        vq_loss, x, encodings = self.vqer(x)
        x, A, Z = self.head(x)
        return vq_loss, x, encodings, A, Z
    

