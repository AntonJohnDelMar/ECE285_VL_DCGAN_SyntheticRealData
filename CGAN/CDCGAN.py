import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt

# Create image folder
os.makedirs("images", exist_ok=True)

# Hyperparameters
latent_dim = 100
img_size = 32
channels = 1
n_classes = 10
embedding_dim = 50
batch_size = 64
n_epochs = 200
sample_interval = 400
lr = 0.0002
b1 = 0.5
b2 = 0.999

img_shape = (channels, img_size, img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, embedding_dim)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embedding_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), dim=1)
        out = self.l1(x).view(x.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, img_size * img_size)

        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * (img_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).view(labels.size(0), 1, img_size, img_size)
        d_in = torch.cat((img, label_input), dim=1)
        return self.model(d_in)

# Loss
adversarial_loss = nn.BCELoss()

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# transform photos
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Assuming img_size is an int
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Data loader
dataloader = DataLoader(
    datasets.ImageFolder(
        root="Dataset/Train",
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Sample generator output
def sample_image(n_row, batches_done):
    z = torch.randn(n_row ** 2, latent_dim).to(device)
    labels = torch.tensor([i for i in range(n_row) for _ in range(n_row)], dtype=torch.long).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"images/{batches_done}.png", nrow=n_row, normalize=True)


# Training
g_losses = []
d_losses = []
for epoch in range(n_epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
    epoch_g_loss = 0
    epoch_d_loss = 0

    for i, (imgs, labels) in enumerate(pbar):
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, n_classes, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        pbar.set_postfix(D_loss=f"{d_loss.item():.4f}", G_loss=f"{g_loss.item():.4f}")

        if (epoch * len(dataloader) + i) % sample_interval == 0:
            sample_image(n_row=10, batches_done=epoch * len(dataloader) + i)

    g_losses.append(epoch_g_loss / len(dataloader))
    d_losses.append(epoch_d_loss / len(dataloader))


torch.save(generator.state_dict(), "generator.pth"); 
torch.save(discriminator.state_dict(), "discriminator.pth"); 