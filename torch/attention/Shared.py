
import torch
from torch import nn
import math

class InputEmbeddings(nn.Module):
    """Converts token indices to embeddings and scales them by sqrt(d_model)"""
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # This is actually a module containing vocab_size tensors d_model long each
        self.Embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, Tokens):
        # If I give it a tensor with size (batch_size,seq_length) containing the index of each token
        # It'll return a tensor (batch_size,seq_length,d_model) => it'll basically add a dimension 
        # containing the embedding of each token
        return self.Embedding(Tokens) * math.sqrt(self.d_model)


class PosEncoding(nn.Module):
    """Takes in a sequence of embeddings and adds to them their corresponding positional encoding"""
    
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        
        PE = torch.zeros((self.seq_length, self.d_model))
        
        # This is like the pos axis for us, we can apply operation on it,
        # just like the vector*scalar we'll do next
        pos = torch.arange(0, self.seq_length).unsqueeze(dim=1)
        
        # [1e5^(2*i/d_model)] with i going from 1 to d_model/2
        in_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(1e5)/self.d_model))
        # what i understood from this line is that the d_model needs to be even 
        # or else we need to create another list in_term_odd
        PE[:, 0::2] = torch.sin(pos * in_term)
        PE[:, 1::2] = torch.cos(pos * in_term)
        # We do it like this since we want the positional encoding to be part of the class attributes
        # but since it is involved with the calculation of the embedding, it will receive a gradient.
        PE = PE.unsqueeze(dim=0)

        self.register_buffer('PE', PE, persistent=False)  # (batch_size,seq_length,d_model)
    
    def forward(self, E):
        return self.PE[:, :E.shape[1], :] + E

class MLPBlock(nn.Module):
    """Multi-Layer Perceptron block with two linear layers and ReLU activation"""
    
    def __init__(self, d_mlp, d_model):
        super().__init__()
        self.L1 = nn.Linear(d_model, d_mlp)
        self.L2 = nn.Linear(d_mlp, d_model)
    
    def forward(self, E):
        return self.L2(torch.relu(self.L1(E)))

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, E):
        mean = E.mean(dim=-1, keepdim=True) 
        var = E.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (E - mean) / torch.sqrt(var + self.eps)
        # same dimensions as E , just that every element in E has been replaced by 
        # the mean / var  of the exact embedding it belongs to .
        # the use of eps is standard practice to avoid devision by zero( var > 0)
        return self.gamma * normalized + self.beta

class ResConnection(nn.Module): 
    def __init__(self, d_model):
        super().__init__()
        self.Norm = LayerNormalization(d_model)
        
    def forward(self, E, curr_layer):
        return E + self.Norm(curr_layer(E))


class Projection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.proj(x)
    