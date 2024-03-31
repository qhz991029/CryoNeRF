import timm
import torch
import torch.nn as nn

model = timm.create_model("resnet18", num_classes=0, global_pool="", in_chans=1).cuda()
print(model(torch.randn((1, 1, 256, 256)).cuda()).shape)