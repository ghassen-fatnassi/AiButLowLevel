import torch
from torch import nn
#don't forget to implement dropout in its correspoding blocks
import math

# The class decomposition is inspired from Umar Jamil tutorial
# even tho he made a mistake in the multiheadattention part 
# where he did not Project the QKV of each embedding using a seperate linear projection 
# but just sliced and put them in order into the heads 
# which is different to what i've found in the paper

# This code follows closely the paper "Attention is All You Need" based on my interpretation 
# but tailored for decoder only arch (autoregressive)
# If there's any mistake I'll gladly fix it after a pull request or an issue


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
        # If the d_model is not even, this will cause an error,
        # since the number of even and odd indices will not be equal
        
        # We do it like this since we want the positional encoding to be part of the class attributes
        # but since it is involved with the calculation of the embedding, it will receive a gradient.
        # We can't do requires_grad=False since that will only make the attribute not receive the gradient,
        # while the calculation will be calculated nevertheless, and as the size of the input sequence gets bigger
        # the gradient will become larger too, inference time-- and vram++ (very bad for us)
        self.register_buffer('PE', PE, persistent=False)  # (batch_size,seq_length,d_model)
        
        # Unsqueeze() so we leave a place for the batch dimension so we can do a modular forward(x) function
        self.PE.unsqueeze(dim=0)
    
    def forward(self, E):
        E = self.PE[:, :E.shape[1], :] + E
        return E

class MLPBlock(nn.Module):
    """Multi-Layer Perceptron block with two linear layers and ReLU activation"""
    
    def __init__(self, d_mlp, d_model):
        super().__init__()
        self.L1 = nn.Linear(d_model, d_mlp)
        self.L2 = nn.Linear(d_mlp, d_model)
    
    def forward(self, E):
        # I'm trying to use operation fusion whenever i can to make more optimized code
        # We only apply relu once and not in the output since we don't want the final output 
        # Embeddings of the MLP to be positive only because it'll go inside an Attention block 
        # and that will cause all scalar products to be positive.
        # Which will limit the superposition phenomena and will cause the model to underperform 
        # (you can learn more about this in anthropic papers in their website, highly recommended 
        # for people looking for intuition about the transformer model)
        return self.L2(torch.relu(self.L1(E)))

class LayerNormalization(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        pass
    def forward(self,E):
        pass

class ResConnection(nn.Module): 
    # LayerNorm & the sublayer are implemented here 
    # sublayer can be an attention block or an mlp block 
    # => this is basically a wrapper that also implements LayerNorm
    def __init__(self,d_model):
        super().__init__()
        self.Norm=LayerNormalization(d_model)
    def forward(self,E,next_layer):
        
        return E + self.Norm(next_layer(E))

class MultiHeadAtt(nn.Module):

    # imma construct this class to be modular and work in an encoder or a decoder (masked & unmasked)
    # it's okay if u see some things don't align with your intuition , the math is correct nevertheless

    def __init__(self,num_heads,d_model):
        super().__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        self.d_qkv=d_model/self.num_heads
        if( d_model % num_heads == 0 ):
            raise "d_model is not divisible by h"
        # I'm setting bias to false since there's no relu after this operation 
        # and also since there's a layernorm directly after the attention mechanism
        self.W_Q=nn.Linear(self.d_model,self.d_model,bias=False)
        self.W_K=nn.Linear(self.d_model,self.d_model,bias=False)
        self.W_V=nn.Linear(self.d_model,self.d_model,bias=False)
        self.W_O=nn.Linear(self.d_model,self.d_model,bias=False) # concat(head1,head2,..headi,..headn)*agg
        
        self.H_Q=nn.ModuleList([nn.Linear(self.d_model,self.d_qkv,bias=False)])
        self.H_K=nn.ModuleList([nn.Linear(self.d_model,self.d_qkv,bias=False)])
        self.H_V=nn.ModuleList([nn.Linear(self.d_model,self.d_qkv,bias=False)])
        
    @staticmethod 
    #equivalent to MultiHeadAtt.attention=staticmethod(MultiHeadAtt)
    def attention(q,k,v,mask):
        pass    

    def forward(self,E,mask):
        # this will work for cross attention and self attention , 
        # we just gotta make E_enc=E_dec
        Q=self.W_Q(E)
        K=self.W_K(E)
        V=self.W_V(E)
        Heads_Q=torch.tensor([hq(Q) for hq in self.H_Q])
        Heads_K=torch.tensor([hk(K) for hk in self.H_K])
        Heads_V=torch.tensor([hv(V) for hv in self.H_V])
        heads_out=[]
        for i in range(self.num_heads):
            heads_out.append(MultiHeadAtt.attention(Heads_Q[i],Heads_K[i],Heads_V[i],mask))
        heads_together_strong=torch.cat(heads_out,dim=2)
        # since each of them is (batch_size,seq_length,d_qkv)
        return self.W_O(heads_together_strong)

