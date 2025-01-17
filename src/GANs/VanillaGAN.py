import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

latent_dim = 100
batch_size = 64
epochs = 50
lr = 0.0002

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
discriminator = Discriminator().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

adversarial_loss = nn.BCELoss().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)
adversarial_loss = adversarial_loss.to(device)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)
        valid = torch.ones((batch_size, 1), device=device)
        fake = torch.zeros((batch_size, 1), device=device)

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(data_loader)} "
                  f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    with torch.no_grad():
        z = torch.randn(25, latent_dim, device=device)
        gen_imgs = generator(z).cpu()
        gen_imgs = (gen_imgs + 1) / 2

        plt.figure(figsize=(5, 5))
        for k in range(25):
            plt.subplot(5, 5, k + 1)
            plt.imshow(gen_imgs[k].squeeze(0), cmap="gray")
            plt.axis("off")
        plt.show()

print("Training completed!")
