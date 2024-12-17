import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,Sample_dim,Latent_dim):
        super().__init__()
        self.latent_dim=Latent_dim
        self.sample_dim=Sample_dim
        self.enc=Encoder(self.sample_dim,self.latent_dim)
        self.dec=Decoder(self.latent_dim,self.sample_dim)
        self.eps=torch.zeros(self.latent_dim) #need to change to be compatible with batch dimension
        pass

    @staticmethod #I did this since I don't need any class specific data to create a normal dist N(0,1)
    def sample(dim):
        return torch.randn(dim)

    def forward(self,x):
        mean,var=self.enc(x)
        self.eps=VAE.sample(self.latent_dim)
        z=mean+torch.exp(0.5*torch.log(var))*self.eps #turns out this is more numerically stable
        return self.dec(z)

class Encoder(nn.Module):
    def __init__(self,Sample_dim,Latent_dim):
        super().__init__()
        self.latent_dim=Latent_dim
        self.sample_dim=Sample_dim
        self.get_mean=nn.Linear(self.sample_dim,self.latent_dim)
        self.get_var=nn.Linear(self.sample_dim,self.latent_dim)

    def forward(self,x):
        return self.get_mean(x),self.get_var(x)

class Decoder(nn.Module):
    def __init__(self,Sample_dim,Latent_dim):
        super().__init__()
        self.latent_dim=Latent_dim
        self.sample_dim=Sample_dim
        self.create=nn.Linear(self.latent_dim,self.sample_dim)
        pass
    def forward(self,z):
        return self.create(z)        

print('hello')

v=VAE(10,2)
print(v(torch.rand(10)))