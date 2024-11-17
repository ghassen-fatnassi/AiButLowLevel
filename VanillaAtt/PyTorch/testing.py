import torch
from torch import nn
import math

# The class decomposition is inspired from Umar Jamil tutorial
# even though he made a mistake in the multiheadattention part
# where he did not Project the QKV of each embedding using a separate linear projection 
# but just sliced and put them in order into the heads 
# which is different from what I've found in the paper

# This code follows closely the paper "Attention is All You Need" based on my interpretation 
# but tailored for decoder-only architecture (autoregressive).
# If there's any mistake, I'll gladly fix it after a pull request or an issue.

class InputEmbeddings(nn.Module):
    """Converts token indices to embeddings and scales them by sqrt(d_model)"""
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # This is actually a module containing vocab_size tensors d_model long each
        self.Embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, Tokens):
        # If I give it a tensor with size (batch_size, seq_length) containing the index of each token
        # It'll return a tensor (batch_size, seq_length, d_model)
        # => It'll basically add a dimension containing the embedding of each token
        embedding = self.Embedding(Tokens) * math.sqrt(self.d_model)
        print(f"InputEmbeddings output shape: {embedding.shape}")
        return embedding

class PosEncoding(nn.Module):
    """Takes in a sequence of embeddings and adds to them their corresponding positional encoding"""
    
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        
        PE = torch.zeros((self.seq_length, self.d_model))
        
        # This is like the pos axis for us, we can apply operations on it,
        # just like the vector*scalar we'll do next
        pos = torch.arange(0, self.seq_length).unsqueeze(dim=1)
        
        # [1e5^(2*i/d_model)] with i going from 1 to d_model/2
        in_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(1e5)/self.d_model))
        
        PE[:, 0::2] = torch.sin(pos * in_term)
        PE[:, 1::2] = torch.cos(pos * in_term)
        
        PE.unsqueeze(dim=0)  # Add batch dimension

        self.register_buffer('PE', PE, persistent=False)  # (seq_length, d_model)
        
        
    
    def forward(self, E):
        E = self.PE[:, :E.shape[1], :] + E
        print(f"PosEncoding output shape: {E.shape}")
        return E

class MLPBlock(nn.Module):
    """Multi-Layer Perceptron block with two linear layers and ReLU activation"""
    
    def __init__(self, d_mlp, d_model):
        super().__init__()
        self.L1 = nn.Linear(d_model, d_mlp)
        self.L2 = nn.Linear(d_mlp, d_model)
    
    def forward(self, E):
        # Apply ReLU once to avoid positive-only embeddings for attention
        output = self.L2(torch.relu(self.L1(E)))
        print(f"MLPBlock output shape: {output.shape}")
        return output

class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, E):
        output = self.layer_norm(E)
        print(f"LayerNormalization output shape: {output.shape}")
        return output

class ResConnection(nn.Module): 
    """Residual Connection with LayerNorm and next layer"""
    def __init__(self, d_model):
        super().__init__()
        self.Norm = LayerNormalization(d_model)
    
    def forward(self, E, next_layer):
        output = E + self.Norm(next_layer(E))
        print(f"ResConnection output shape: {output.shape}")
        return output

class MultiHeadAtt(nn.Module):
    """Multi-Head Attention module"""
    
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_qkv = d_model // num_heads
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.H_Q = nn.ModuleList([nn.Linear(d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
        self.H_K = nn.ModuleList([nn.Linear(d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
        self.H_V = nn.ModuleList([nn.Linear(d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
    
    @staticmethod
    def attention(q, k, v, mask):
        d_k = q.shape[-1]  # (batch, seq_length, d_k)
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float('inf'))
        
        att_weights = torch.softmax(att_scores, dim=-1)
        values = torch.matmul(att_weights, v)
        print(f"MultiHeadAtt attention output shape: {values.shape}")
        return values
    
    def forward(self, E, mask):
        Q = self.W_Q(E)
        K = self.W_K(E)
        V = self.W_V(E)
        
        Heads_Q = torch.cat([hq(Q) for hq in self.H_Q], dim=2)
        Heads_K = torch.cat([hk(K) for hk in self.H_K], dim=2)
        Heads_V = torch.cat([hv(V) for hv in self.H_V], dim=2)
        
        heads_out = []
        for i in range(self.num_heads):
            heads_out.append(MultiHeadAtt.attention(Heads_Q[:, :, i:i+1], Heads_K[:, :, i:i+1], Heads_V[:, :, i:i+1], mask))
        
        heads_together = torch.cat(heads_out, dim=2)
        aggregated_values = self.W_O(heads_together)
        
        print(f"MultiHeadAtt forward output shape: {aggregated_values.shape}")
        return aggregated_values

class DecoderBlock(nn.Module):
    """Decoder block consisting of multi-head attention and MLP"""
    
    def __init__(self, num_heads, d_model, d_mlp):
        super().__init__()
        self.attention = MultiHeadAtt(num_heads, d_model)
        self.res_conn1 = ResConnection(d_model)
        self.mlp = MLPBlock(d_mlp, d_model)
        self.res_conn2 = ResConnection(d_model)
    
    def forward(self, E, mask):
        attention_output = self.attention(E, mask)
        output1 = self.res_conn1(E, lambda x: attention_output)
        mlp_output = self.mlp(output1)
        output2 = self.res_conn2(output1, lambda x: mlp_output)
        print(f"DecoderBlock output shape: {output2.shape}")
        return output2

