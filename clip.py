import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPTokenizer

def image_preprocess(image, image_size=224):

    transform = transforms.Compose([ transforms.Resize(image_size), transforms.ToTensor() ])
    return transform(image)

class ImageEncoder(nn.Module): 

     def __init__(self, image_size, patch_size, embed_dim): 
        super().__init__()
        # Split image into patches and project to embed_dim
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate number of patches
        num_patches = (image_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        

    def forward(self, x): 
        # Get batch size
        batch_size = x.shape[0]
        
        patches = self.patch_embed(x)
        
        # Reshape to sequence: [batch, embed_dim, 14, 14] -> [batch, embed_dim, 196]
        patches = patches.flatten(2).transpose(1, 2)
        
        # Prepend CLS token: expand [1, 1, embed_dim] -> [batch, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate: [batch, 1, embed_dim] + [batch, 196, embed_dim] -> [batch, 197, embed_dim]
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add positional embeddings: [batch, 197, embed_dim] + [1, 197, embed_dim]
        x = x + self.pos_embed
        
        return x


def TextPreprocess(text, vocab_size=49408):
    transform = transforms.Compose([ transforms.ToTensor() ])
    return transform(text)



class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512,):
        super().__init__()

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=12)

    def forward(self, x):
        return self.attention(x)

class MLP(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
        )
        self.mlp = layers


    def forward(self, x):
        return self.mlp(x)
    
class CLIP(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(image_size, patch_size, embed_dim)
        self.attention = Attention(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, image, text_tokens):
        image_tokens = self.image_encoder(preprocess(image))  # [batch, 197, embed_dim]
        text_tokens = self.encoder(text_tokens)  # [batch, 197, embed_dim]
        x = torch.cat([image_tokens, text_tokens], dim=1)  # [batch, 394, embed_dim]
        x = self.attention(x)  # [batch, 394, embed_dim]