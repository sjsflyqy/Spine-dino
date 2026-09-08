import torch.nn
import torch.nn as nn

BN_MOMENTUM = 0.1

__all__ = ['HighResolutionNet']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


# 自己加的，用于stage4,原文是只输出分辨率最高的特征层，这里是输出分辨率最低的那层
class StageModuleLast(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        return torch.cat((self.down3(self.down2(self.down1(x[0]))),
                          self.down3(self.down2(x[1])),
                          self.down3(x[2]),
                          x[3]), dim=1)


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        self.upsam1 = torch.nn.Upsample(scale_factor=2)  # z

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )
        self.stage2_1 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )  # z

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        self.transition2_1 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])  # z
        self.upsam2 = torch.nn.Upsample(scale_factor=2)  #
        self.down2 = nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False)  #
        self.down2_1 = nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False)  #
        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        self.stage3_1 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )  # z

        self.upsam3 = torch.nn.Upsample(scale_factor=2)  # z
        self.down31 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)

        )  # z
        self.down32 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)

        )  # z
        self.down31_1 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)

        )  # z
        self.down32_1 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)

        )  # z
        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        self.transition3_1 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])  # z

        # Stage4

        self.stage4_seg = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        self.stage4_1_seg = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )  # z

        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModuleLast(input_branches=4, output_branches=1, c=base_channel)
        )

        self.stage4_1 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModuleLast(input_branches=4, output_branches=1, c=base_channel)
        )  # z

        # Final layer
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)
        self.cv = nn.Conv2d(96, 128, 1)  #
        self.seg_out =nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0, x1 = x[0], x[1]
        feat0, feat1 = [], []

        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)
        x0 = self.bn2(x0)
        x0 = self.relu(x0)  # (2,64 256 128)b,c,h,w

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)  # #64 256 128

        # ---------------经过transition1-----------------获得E1
        x0 = self.layer1(x0)
        x0 = [trans(x0) for trans in self.transition1]
        feat0.append(
            torch.cat((x0[0], self.upsam1(x0[1])), dim=1))

        x1 = self.layer1(x1)
        x1 = [trans(x1) for trans in self.transition1]

        feat1.append(torch.cat((x1[0], self.upsam1(x1[1])), dim=1))

        # ----------------经过stage2和transition2-----------------获得E2
        x0 = self.stage2(x0)
        x0 = [
            self.transition2[0](x0[0]),
            self.transition2[1](x0[1]),
            self.transition2[2](x0[-1])
        ]
        feat0.append(torch.cat((self.down2(x0[0]), x0[1], self.upsam2(x0[2])),
                               dim=1))

        x1 = self.stage2_1(x1)
        x1 = [
            self.transition2_1[0](x1[0]),
            self.transition2_1[1](x1[1]),
            self.transition2_1[2](x1[-1])
        ]
        feat1.append(torch.cat((self.down2_1(x1[0]), x1[1], self.upsam2(x1[2])), dim=1))

        # --------------------经过stage3和transition3-------------------获得E3
        x0 = self.stage3(x0)
        x0 = [
            self.transition3[0](x0[0]),
            self.transition3[1](x0[1]),
            self.transition3[2](x0[2]),
            self.transition3[3](x0[-1]),
        ]
        feat0.append(torch.cat((self.down32(self.down31(x0[0])), self.down32(x0[1]), x0[2], self.upsam3(x0[3])),
                               dim=1))

        x1 = self.stage3_1(x1)
        x1 = [
            self.transition3_1[0](x1[0]),
            self.transition3_1[1](x1[1]),
            self.transition3_1[2](x1[2]),
            self.transition3_1[3](x1[-1]),
        ]
        feat1.append(
            torch.cat((self.down32_1(self.down31_1(x1[0])), self.down32_1(x1[1]), x1[2], self.upsam3(x1[3])), dim=1))

        # ---------------------经过stage4--------------获得E4
        temp0 = x0
        x0 = self.stage4(x0)
        feat0.append(x0)
        x0_seg = self.stage4_seg(temp0)
        x0_seg=torch.squeeze(torch.stack(x0_seg),0)
        x0_seg_mask = self.seg_out(x0_seg)
        x0_seg_mask=torch.sigmoid(x0_seg_mask)
        temp1 = x1
        x1 = self.stage4(x1)
        feat1.append(x1)
        x1_seg = self.stage4_1_seg(temp1)
        x1_seg=torch.squeeze(torch.stack(x1_seg),0)
        x1_seg_mask = self.seg_out(x1_seg)
        x1_seg_mask = torch.sigmoid(x1_seg_mask)
        return feat0, feat1, x0_seg_mask, x1_seg_mask
