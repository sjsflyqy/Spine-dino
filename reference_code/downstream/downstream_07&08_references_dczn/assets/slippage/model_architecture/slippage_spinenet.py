import torch.nn as nn
import torch
import torch.nn.functional as F

from . import hrnet18_backbone


class SpineNet(nn.Module):
    def __init__(self):
        super(SpineNet, self).__init__()
        # 滑脱关键点模型采用 HRNet18 主干，保持高分辨率特征，
        # 再通过解码头输出 hm / reg / wh 三个分支。
        self.base_network = hrnet18_backbone.hrnet18(pretrained=True)
        heads = {'hm': 1, 'reg': 2, 'wh': 8 }
        self.dec_net = DecNet(heads, 1, 256, 64)

    def forward(self, x):
        x = self.base_network(x)
        dec_dict = self.dec_net(x)
        return dec_dict

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet, self).__init__()
        # HRNet 的四路特征会在这里逐级融合，最终在最高分辨率上输出关键点参数。
        self.dec_c2 = CombinationModule(36, 18, batch_norm=True)
        self.dec_c3 = CombinationModule(72, 36, batch_norm=True)
        self.dec_c4 = CombinationModule(144, 72, batch_norm=True)
        self.heads = heads
        channel =18
        head_conv = 256

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])

        return dec_dict


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        # 先把低分辨率特征上采样，再与高分辨率特征做通道和空间注意力融合。
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
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
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c_low, c_up, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(c_up, c_up, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sab = SAB()

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x = torch.cat((x_up, x_low), 1)
        x_avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        x_max = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        weight = self.sigmoid(x_max + x_avg)
        x_up = x_up*weight
        x_low = x_low*weight #newffca
        x_up = x_up*self.sab(x_up)   #sab
        x_low = x_low*self.sab(x_low)  #sab

        return self.cat_conv(torch.cat((x_up, x_low), 1))
