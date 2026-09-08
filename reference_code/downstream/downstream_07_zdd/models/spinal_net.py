import torch
import torchvision.ops
from .SwinTransformer import SwinTransformerLayer
from .dec_net import DecNet
from . import resnet
import torch.nn as nn
import numpy as np
from .hrnet import HighResolutionNet


class SpineNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(SpineNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        # channels = [3, 64, 128, 256, 512, 1024]
        channels = [3, 64, 96, 256, 512, 1024]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = HighResolutionNet()
        self.dec_duel = DecNet(heads, final_kernel, head_conv, channels[self.l1])

    def forward(self, x):
        x = self.base_network(x)  # 返回4个部分：feat0, feat1, x0_seg_mask, x1_seg_mask
        dec_dict0, dec_dict1 = self.dec_duel(x[0], x[1], x[2], x[3])
        return dec_dict0, dec_dict1
