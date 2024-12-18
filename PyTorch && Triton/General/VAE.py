import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_hidden_dims:list,dec_hidden_dims:list):
        super(VAE, self).__init__()
        self.x_dim=input_dim
        self.z_dim=latent_dim

        """constructing the encoder"""
        core=[nn.Sequential(nn.Linear(self.x_dim,enc_hidden_dims[0]),nn.ReLU())]
        core.extend([
            nn.Sequential(
                nn.Linear(enc_hidden_dims[i],enc_hidden_dims[i+1]),
                nn.ReLU())
            for i in range(len(enc_hidden_dims)-1)])
        core.append(nn.Linear(enc_hidden_dims[-1],2*self.z_dim))
        #2*self.z_dim since half for mu and half for var (to make sure computation is as parallel as possible)
        self.encoder=nn.ModuleList(core)

        """constructing the decoder"""
        core=[nn.Sequential(nn.Linear(self.z_dim,dec_hidden_dims[0]),nn.ReLU())]
        core.extend([
            nn.Sequential(
                nn.Linear(dec_hidden_dims[i],dec_hidden_dims[i+1]),
                nn.ReLU())
            for i in range(len(dec_hidden_dims)-1)])
        core.append(nn.Sequential(nn.Linear(dec_hidden_dims[-1],self.x_dim))) 
        self.decoder=nn.ModuleList(core)
        # used the identity function as the last activation function here 
        # as i'm going for the gaussian assumption of the output output in (R) (pixel values are continuous)
        # i'm using MSE as loss function

    @staticmethod
    def generate(model,device,num_samples=10):
        model.eval()
        with torch.inference_mode():
            z=torch.randn(num_samples,model.z_dim).to(device)
            samples=model.decode(z)
        return samples

    def encode(self, x):
        for layer in self.encoder:
            x=layer(x)
        mu,log_var=x.chunk(2,dim=1)
        return mu,log_var

    @staticmethod
    def reparameterize(mu, log_var):
        eps=torch.randn_like(log_var)
        return mu+eps*torch.exp(0.5*log_var) 

    def decode(self, z):
        for layer in self.decoder:
            z=layer(z)
        return z

    def forward(self, x):
        mu,log_var=self.encode(x)
        z=self.reparameterize(mu,log_var)
        return mu,log_var,self.decode(z)

