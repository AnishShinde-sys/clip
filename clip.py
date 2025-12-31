import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

def preprocess(image, image_size=224):
    
    transform = transforms.Compose([ transforms.Resize(image_size), transforms.ToTensor() ])
    return transform(image)

class Encoder(nn.Module): 

     def __init__(self, image_size, patch_size, embed_dim): 
        super().__init__()
        # Split image into patches and project to embed_dim
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x): 
        patches = self.patch_embed(x)
        return patches

class MLP(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp = layers

    def forward(self, x):
        return self.mlp(x)