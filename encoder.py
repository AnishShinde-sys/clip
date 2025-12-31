import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class Encoder(nn.Module): 

    def __init__(self,image_size,patch_size,embed_dim): 

        super().__init__()
        
        #  Resize image into 224 by 224 
        self.resize = transforms.Resize((224, 224))

        # Split image into patches and project to embed_dim
        self.patch_embed = nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)


    def forward(self,x): 
        
        resize = self.resize(x)
        
        patchs = self.patch_embed(resize)
        
        return patchs 

 