import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


class NeuralRadianceField(nn.Module):
    def __init__(self, enc_dim=10, hid_dim=160, hid_layer_num=2, checkpointing=False):
        super().__init__()
        
        self.checkpointing = checkpointing
        self.hid_layer_num = hid_layer_num
        self.module_list = [nn.Linear(6 * enc_dim, hid_dim), nn.ReLU()]
        for i in range(hid_layer_num):
            self.module_list += [nn.Linear(hid_dim, hid_dim), nn.ReLU()]
        self.module_list += [nn.Linear(hid_dim, 1)]
        
        self.mlp = nn.Sequential(*self.module_list)

    def forward(self, encoded_pos):
        if self.checkpointing:
            density = checkpoint_sequential(self.mlp, len(self.mlp), encoded_pos)
        else:
            density = self.mlp(encoded_pos)

        return density