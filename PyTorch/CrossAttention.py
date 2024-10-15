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
    def forward(self,X):#X is a matrix where the rows are the tokens' embeddings
        self.Q=self.WQ @ X
        self.K=self.WK @ X
        self.V=self.WV @ X
        