import torch.nn as nn
import torch
from .model_parts import CombinationModule
import torch.nn.functional as F
from .SE import se_block
from .CBAM import cbam


class Connection1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cv = nn.Conv2d(2 * self.out_channel, self.out_channel, 1)

    def forward(self, feature_A, feature_L, mask_A, mask_L):
        temp_feature_A = feature_A
        temp_feature_L = feature_L
        seg_feature_A = feature_A * mask_A
        seg_feature_L = feature_L * mask_L
        fuse = seg_feature_A + seg_feature_L
        out_feature_A = self.cv(torch.cat((temp_feature_A, fuse), dim=1))
        out_feature_L = self.cv(torch.cat((temp_feature_L, fuse), dim=1))
        return out_feature_A, out_feature_L

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet, self).__init__()
        self.dec_c2 = CombinationModule(256, 96, batch_norm=True)  # 128,96
        self.dec_c3 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c4 = CombinationModule(1024, 512, batch_norm=True)
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7 // 2, bias=True),
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
        # --------------------------------------基于分割的正侧面特征融合------------------
        self.f1 = Connection1(96, 96)
        self.f2 = Connection1(256, 256)
        self.f3 = Connection1(512, 512)
        self.f4 = Connection1(1024, 1024)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, y, x_seg, y_seg):  # x y代表正侧面原始的编码器特征

        # -----------------下面先对mask调整成4个不同的尺度，以适用于四个尺度的编码器特征----------------
        mask_D4_A = F.interpolate(x_seg, size=x[-1].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D3_A = F.interpolate(x_seg, size=x[-2].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D2_A = F.interpolate(x_seg, size=x[-3].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D1_A = F.interpolate(x_seg, size=x[-4].shape[2:], mode='bilinear',
                                  align_corners=False)
        # ------
        mask_D4_L = F.interpolate(y_seg, size=y[-1].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D3_L = F.interpolate(y_seg, size=y[-2].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D2_L = F.interpolate(y_seg, size=y[-3].shape[2:], mode='bilinear',
                                  align_corners=False)
        mask_D1_L = F.interpolate(y_seg, size=y[-4].shape[2:], mode='bilinear',
                                  align_corners=False)
        # ----------------- 基于分割结果进行正侧面特征融合，从而对编码器特征进行增强------------
        x[-1], y[-1] = self.f4(x[-1], y[-1], mask_D4_A, mask_D4_L)
        # x[-2], y[-2] = self.f3(x[-2], y[-2], mask_D3_A, mask_D3_L)
        # x[-3], y[-3] = self.f2(x[-3], y[-3], mask_D2_A, mask_D2_L)
        # x[-4], y[-4] = self.f1(x[-4], y[-4], mask_D1_A, mask_D1_L)
        # ---------------------从E4开始依次向上经过解码器-----------------------
        c4_combine_A = self.dec_c4(x[-1], x[-2])
        c3_combine_A = self.dec_c3(c4_combine_A, x[-3])
        c2_combine_A = self.dec_c2(c3_combine_A, x[-4])
        c4_combine_L = self.dec_c4(y[-1], y[-2])
        c3_combine_L = self.dec_c3(c4_combine_L, y[-3])
        c2_combine_L = self.dec_c2(c3_combine_L, y[-4])
        dec_dict_A = {}
        dec_dict_L = {}
        for head in self.heads:
            dec_dict_A[head] = self.__getattr__(head)(c2_combine_A)
            if 'hm' in head:
                dec_dict_A[head] = torch.sigmoid(dec_dict_A[head])
        for head in self.heads:
            dec_dict_L[head] = self.__getattr__(head)(c2_combine_L)
            if 'hm' in head:
                dec_dict_L[head] = torch.sigmoid(dec_dict_L[head])
        dec_dict_A['seg_img'] = x_seg
        dec_dict_L['seg_img'] = y_seg
        return dec_dict_A, dec_dict_L
