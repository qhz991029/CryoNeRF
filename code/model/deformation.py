import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint_sequential


class DeformationEncoder(nn.Module):
    def __init__(self, encoder_type: str, latent_dim: int) -> None:
        super().__init__()
        
        self.deformation_encoder = timm.create_model(encoder_type, num_classes=0, global_pool="", in_chans=1)
        # if encoder_type == "resnet18":
        #     self.output_layer = nn.Linear(32768, latent_dim)
        # elif encoder_type == "resnet34":
        #     self.output_layer = nn.Linear(32768, latent_dim)
        # else:
        #     self.output_layer = nn.LazyLinear(latent_dim)
        self.output_layer = nn.LazyLinear(latent_dim)
            
    def forward(self, images):
        x = F.relu(self.deformation_encoder(images).reshape(images.shape[0], -1))
        latent_variable = self.output_layer(x)
        
        return latent_variable
    

class DeformationDecoder(nn.Module):
    def __init__(self, enc_dim=10, latent_dim=16, hid_dim=160, hid_layer_num=2, checkpointing=False) -> None:
        super().__init__()
        
        self.checkpointing = checkpointing
        self.hid_layer_num = hid_layer_num
        self.module_list = \
            [nn.Linear(6 * enc_dim + latent_dim, hid_dim), nn.ReLU()] + \
            [nn.Linear(hid_dim, hid_dim), nn.ReLU()] * hid_layer_num + \
            [nn.Linear(hid_dim, 3)]
        
        self.mlp = nn.Sequential(*self.module_list)
        
    def forward(self, encoded_pos, latent_variable):
        encoded_pos_with_latent = torch.cat([encoded_pos, repeat(latent_variable, "B Dim -> B N L Dim",
                                                                 N=encoded_pos.shape[1], L=encoded_pos.shape[2])], dim=-1)
        if self.checkpointing:
            delta_coord = checkpoint_sequential(self.mlp, self.hid_layer_num + 2, encoded_pos_with_latent)
        else:
            delta_coord = self.mlp(encoded_pos_with_latent)

        return delta_coord