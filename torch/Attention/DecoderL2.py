import torch
from torch import nn
import math
from Shared import InputEmbeddings,Projection,MLPBlock,ResConnection,LayerNormalization,PosEncoding

# this is the same impelementation of DecoderL1.py 
# but i'm aiming to make this more parallel even at the cost of the code being less readable
class MultiHeadAtt(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_qkv = d_model // self.num_heads
        self.dropout = nn.Dropout(dropout)  
        self.temperature = temperature 
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
            
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        # i will remove the ModuleList of linear (d_model => d_kqv) 
        # and replace it by one linear from (d_model => d_model)  and then split them among the heads
        # the computation is equivalent
        self.H_Q = nn.Linear(self.d_model,self.d_model,bias=False)
        self.H_K = nn.Linear(self.d_model,self.d_model,bias=False)
        self.H_V = nn.Linear(self.d_model,self.d_model,bias=False)
        #AGGREGATION LINEAR LAYER   
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)
        
    @staticmethod
    def attention(q, k, v, mask, temperature):
        d_k = q.shape[-1]
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0,-1e9)
        att_scores=att_scores / temperature
        att_weights = att_scores.softmax(dim=-1)
        att_weights = torch.nn.functional.dropout(att_weights, p=0.1)
        weighted_values = torch.matmul(att_weights, v)
        return weighted_values

    def forward(self, E, mask=None):
        Q = self.W_Q(E)
        K = self.W_K(E)
        V = self.W_V(E)

        Heads_Q = self.H_Q(Q)
        Heads_K = self.H_Q(K)
        Heads_V = self.H_Q(V)
        
        Heads_Q=Heads_Q.view(Heads_Q.shape[0],Heads_Q.shape[1],self.num_heads,self.d_qkv).transpose(1,2)
        Heads_K=Heads_K.view(Heads_K.shape[0],Heads_K.shape[1],self.num_heads,self.d_qkv).transpose(1,2)
        Heads_V=Heads_V.view(Heads_V.shape[0],Heads_V.shape[1],self.num_heads,self.d_qkv).transpose(1,2)
        
        heads_out=MultiHeadAtt.attention(Heads_Q,Heads_K,Heads_V,mask,self.temperature)
        #(b_size,h,seq_len,d_model) transpose(1,2) => (b_size,seq_len,h,d_kqv) view() => (b_size,seq_len,d_model)
        heads_together_strong=heads_out.transpose(1,2).contiguous().view(heads_out.shape[0],heads_out.shape[2],self.d_model)
        aggregated_values = self.W_O(heads_together_strong)
        return aggregated_values

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_mlp, num_heads, dropout=0.1, temperature=1.0):
        super().__init__()
        self.self_attention = MultiHeadAtt(num_heads, d_model, dropout, temperature)
        self.res_conn1 = ResConnection(d_model)
        self.mlp = MLPBlock(d_mlp, d_model)
        self.res_conn2 = ResConnection(d_model)
        
    def forward(self, E, mask=None):
        att_output = self.res_conn1(E, lambda x: self.self_attention(x, mask))
        mlp_output = self.res_conn2(att_output, lambda x: self.mlp(x))
        return mlp_output

class Decoder(nn.Module):
    def __init__(self,d_model, d_mlp, num_heads, num_blocks, dropout=0.1, temperature=1.0):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_mlp, num_heads, dropout, temperature) for _ in range(num_blocks)])
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, d_mlp, num_heads, N, max_length, dropout=0.1, temperature=1.0):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PosEncoding(max_length, d_model)
        self.decoder = Decoder(d_model, d_mlp, num_heads, N, dropout, temperature)
        self.proj = Projection(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        decoder_output = self.decoder(x, mask)
        return self.proj(decoder_output)
    
    def generate(self, input_ids, max_length, temperature=1.0):
        for _ in range(max_length - input_ids.shape[1]):
            seq_len = input_ids.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()
            mask = mask.unsqueeze(0)
            logits = self.forward(input_ids, mask)
            next_token_logits = logits[:, -1, :] / temperature
            pre_next_token=torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(pre_next_token, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            print(input_ids)
        return input_ids

d_model=10
d_mlp=24
vocab_size=10
heads=5
blocks=10
seq_len=3
max_length=50
batch_size=5
dropout=0.1
Temperature=2

tokens=torch.randint(high=vocab_size-1,size=[batch_size,seq_len]) #vocab_size-1 since it's 0 indexed
T=Transformer(vocab_size,d_model,d_mlp,heads,blocks,max_length,dropout,Temperature)
out=T.generate(tokens,max_length)
