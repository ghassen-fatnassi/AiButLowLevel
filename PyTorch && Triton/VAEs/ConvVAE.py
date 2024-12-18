import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self,in_size,in_channels,latent_dim,hidden_channels:list):
        super(VAE, self).__init__()
        self.in_size=in_size # the input should be a (in_channels,in_size,in_size) image
        self.in_channels=in_channels
        self.z_dim=latent_dim

        """constructing the encoder"""
        core=[
            nn.Sequential(
                nn.Conv2d(self.in_channels,hidden_channels[0],kernel_size=4,stride=2),
                nn.ReLU())
            ]
        core.extend([
            nn.Sequential(
                nn.Conv2d(hidden_channels[i],hidden_channels[i+1],kernel_size=4,stride=2),
                nn.ReLU())
            for i in range(len(hidden_channels)-1)])
        ir=hidden_channels[-1]*(in_size/len(hidden_channels))*(in_size/hidden_channels)
        core.append(nn.Sequential(nn.Flatten(),nn.Linear(ir,2*self.z_dim)))
        #2*self.z_dim since half for mu and half for var (to make sure computation is as parallel as possible)
        self.encoder=nn.ModuleList(core)

        """constructing the decoder"""
        core=[nn.Sequential(nn.Linear(self.z_dim,hidden_channels[0]),nn.ReLU())]
        core.extend([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels[i],hidden_channels[i-1]),
                nn.ReLU())
            for i in range(1,len(hidden_channels))])
        core.append(nn.Sequential(nn.ConvTranspose2d(hidden_channels[-1],self.in_channels))) 
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
