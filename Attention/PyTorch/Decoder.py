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
        
        PE[:, 0::2] = torch.sin(pos * in_term)
        PE[:, 1::2] = torch.cos(pos * in_term)
        
        # Unsqueeze() so we leave a place for the batch dimension so we can do a modular forward(x) function
        PE = PE.unsqueeze(dim=0)

        # We do it like this since we want the positional encoding to be part of the class attributes
        # but since it is involved with the calculation of the embedding, it will receive a gradient.
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
        return self.gamma * normalized + self.beta

class ResConnection(nn.Module): 
    def __init__(self, d_model):
        super().__init__()
        self.Norm = LayerNormalization(d_model)
        
    def forward(self, E, next_layer):
        return E + self.Norm(next_layer(E))

class MultiHeadAtt(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_qkv = d_model // self.num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
            
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.H_Q = nn.ModuleList([nn.Linear(self.d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
        self.H_K = nn.ModuleList([nn.Linear(self.d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
        self.H_V = nn.ModuleList([nn.Linear(self.d_model, self.d_qkv, bias=False) for _ in range(self.num_heads)])
        
    @staticmethod
    def attention(q, k, v, mask):
        d_k = q.shape[-1]
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
        att_weights = torch.softmax(att_scores, dim=-1)
        values = torch.matmul(att_weights, v)
        return values

    def forward(self, E, mask=None):
        Q = self.W_Q(E)
        K = self.W_K(E)
        V = self.W_V(E)

        Heads_Q = torch.stack([hq(Q) for hq in self.H_Q])
        Heads_K = torch.stack([hk(K) for hk in self.H_K])
        Heads_V = torch.stack([hv(V) for hv in self.H_V])

        heads_out = []
        for i in range(self.num_heads):
            heads_out.append(self.attention(Heads_Q[i], Heads_K[i], Heads_V[i], mask))

        heads_together_strong = torch.cat(heads_out, dim=2)
        aggregated_values = self.W_O(heads_together_strong)
        return aggregated_values

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_mlp, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAtt(num_heads, d_model)
        self.res_conn1 = ResConnection(d_model)
        self.mlp = MLPBlock(d_mlp, d_model)
        self.res_conn2 = ResConnection(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, E, mask=None):
        # Self attention with residual connection
        att_output = self.res_conn1(E, lambda x: self.dropout(self.self_attention(x, mask)))
        # MLP with residual connection
        mlp_output = self.res_conn2(att_output, lambda x: self.dropout(self.mlp(x)))
        return mlp_output

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_mlp, num_heads, num_layers, seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PosEncoding(seq_length, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_mlp, num_heads) for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, mask=None):
        # Convert tokens to embeddings and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)

class Projection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Project the decoder output back to vocabulary space
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_mlp, num_heads, N, seq_length):
        super().__init__()
        self.decoder = Decoder(vocab_size, d_model, d_mlp, num_heads, N, seq_length)
        self.projection = Projection(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Get decoder output
        decoder_output = self.decoder(x, mask)
        # Project to vocabulary space
        logits = self.projection(decoder_output)
        return logits

    def generate(self, input_ids, max_length, temperature=1.0):
        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            # Create attention mask for autoregressive generation
            seq_len = input_ids.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.unsqueeze(0)  # Add batch dimension
            
            # Get model predictions
            logits = self.forward(input_ids, mask)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Concatenate with input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids