import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttHead(nn.Module):
    def __init__(self,emb_dim,d_k):
        self.WQ=nn.Linear(d_k,emb_dim)
        self.WK=nn.Linear(d_k,emb_dim)
        self.WV=nn.Linear(emb_dim,emb_dim)
        self.Q=torch.zeros([])
        self.K=torch.zeros([])
        self.V=torch.zeros([])
    def forward(self,E):#X is a matrix where the rows are the tokens' embeddings
        #E is (emb_dim,num_tokens)
        #self.WQ is (d_k,emb_dim)
        #self.WQ is (d_k,emb_dim)
        #self.WQ is (emb_dim,emb_dim) 
        self.Q=self.WQ @ E #returns (d_k,num_tokens)
        self.K=self.WK @ E #returns (d_k,num_tokens)
        self.V=self.WV @ E #returns (emb_dim,num_tokens)
        E=E+
