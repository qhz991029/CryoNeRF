import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class NeuralRadianceField(nn.Module):
    def __init__(self, enc_dim=10, hid_dim=160, hid_layer_num=2, checkpointing=False):
        super().__init__()
        
        self.checkpointing = checkpointing
        self.hid_layer_num = hid_layer_num
        self.module_list = \
            [nn.Linear(6 * enc_dim, hid_dim), nn.ReLU()] + \
            [nn.Linear(hid_dim, hid_dim), nn.ReLU()] * hid_layer_num + \
            [nn.Linear(hid_dim, 1)]
        
        self.mlp = nn.Sequential(*self.module_list)

    def forward(self, encoded_pos):
        if self.checkpointing:
            density = checkpoint_sequential(self.mlp, self.hid_layer_num + 2, encoded_pos)
        else:
            density = self.mlp(encoded_pos)

        return density