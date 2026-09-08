import torch.nn as nn
import torch.nn.functional as F
import torch
from .SE import se_block
from .CA_x import se_block as CA_x
from .CA import se_block as CA
from .CA2 import se_block as CA2
from .SA import SpatialAttention as S1
from .SA2 import SpatialAttention as S2
from .SA3 import SpatialAttention as S3
from .CBAM import cbam


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):  # x.size() 30,40,50,30
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)  # 30,2,50,30
        x = self.conv1(x)  # 30,1,50,30
        x = self.sigmoid(x)  # 30,1,50,30
        return x * inputs


class FWN(nn.Module):
    def __init__(self, input_dim):
        super(FWN, self).__init__()
        self.W = nn.Parameter(torch.rand(input_dim))
        self.W.requires_grad = True
        self.b = nn.Parameter(torch.rand(input_dim))
        self.b.requires_grad = True

    def forward(self, inputs):
        # out = torch.tanh(inputs * self.W + self.b)
        out = inputs * self.W + self.b
        return out


class MAM(nn.Module):  # 用MAM表示skip connection
    def __init__(self, in_channel):
        super(MAM, self).__init__()
        self.in_channel = in_channel
        self.ca = CA_x(self.in_channel)

    def forward(self, high_feature, low_feature):
        ca_high_feature, w = self.ca(high_feature)
        w_low = 1 - w
        ca_low_feature = low_feature * w_low
        return ca_low_feature + ca_high_feature


class attention_xmodule_3(nn.Module):
    def __init__(self, in_channel):
        super(attention_xmodule_3, self).__init__()
        self.attention = CA(in_channel)
        self.cv = nn.Conv2d(2 * in_channel, in_channel, 1)

    def forward(self, x, y):
        ca_x = self.attention(x)
        ca_y = self.attention(y)
        ca_out = self.cv(torch.cat((ca_x, ca_y), dim=1))
        return ca_out


class attention_cbam_cat(nn.Module):
    def __init__(self, in_channel):
        super(attention_cbam_cat, self).__init__()
        self.cbam = cbam(in_channel)
        self.cv=nn.Conv2d(2*in_channel,in_channel,1)

    def forward(self, x, y):
        x=self.cbam(x)
        fuse=torch.cat((x,y),dim=1)
        return self.cv(fuse)


class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(c_up),
                                          nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.ReLU(inplace=True))
        # self.spatial = SpatialAttention()
        # self.cv = nn.Conv2d(c_low, c_up, 1)
        # self.se = se_block(c_low)
        # # self.input_dim = input_dim
        # self.FWN = FWN(self.input_dim)
        # self.MAM = MAM(c_up)
        # self.attention3 = attention_xmodule_3(c_up)
        # self.cbam=attention_cbam_cat(c_up)

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))  # 上采样,尺寸和通道都上一层相同
        # return self.cat_conv(torch.cat((x_up, x_low), 1)) # 拼接 后卷积
        return x_up + x_low
        # out=self.MAM(x_up,x_low)
        # return out
        # out=self.cbam(x_low,x_up)
        # return out
        # out=self.attention3(x_low,x_up)
        # return out

