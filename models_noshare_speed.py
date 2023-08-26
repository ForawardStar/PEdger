import torch.nn as nn
import torch.nn.functional as F
import torch
import time

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel,track_running_stats=False)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class ConvINReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2 + dilation - 1
        super(ConvINReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channel,track_running_stats=False),
            nn.LeakyReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, dilation):
        super(InvertedResidual, self).__init__()
        expand_ratio = 2.
        hidden_channel = int(in_channel * expand_ratio)

        layers = []
        # 1x1 pointwise conv
        layers.append(ConvINReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvINReLU(hidden_channel, hidden_channel, groups=hidden_channel, dilation=dilation),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel,track_running_stats=False),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class Guider_noshare(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Guider_noshare, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.nfc = 32
        self.min_nfc = 32
        self.num_layer = 3

        self.head = ConvBlock(in_channels, 64, 3, 1,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()

        for i in range(self.num_layer - 2):
            block = InvertedResidual(64, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail_state1 = nn.Sequential(
            InvertedResidual(64, 1),
            #InvertedResidual(64, 1),
            #InvertedResidual(64, 1),
            InvertedResidual(64, 2),
            #InvertedResidual(64, 2),
            #InvertedResidual(64, 4),
            InvertedResidual(64, 4),
            #InvertedResidual(64, 8),
            InvertedResidual(64, 8),
            InvertedResidual(64, 1),
        )

        self.tail_mask_s2d1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_d2s1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.channel_expand1 = nn.Conv2d(1, 64, 1)

        self.tail_state2 = nn.Sequential(
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 2),
            InvertedResidual(64, 2),
            #InvertedResidual(64, 4),
            InvertedResidual(64, 4),
            #InvertedResidual(64, 8),
            InvertedResidual(64, 8),
            InvertedResidual(64, 1),
        )

        self.tail_mask_s2d2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_d2s2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.channel_expand2 = nn.Conv2d(1, 64, 1)

        self.tail_state3 = nn.Sequential(
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 2),
            InvertedResidual(64, 2),
            InvertedResidual(64, 4),
            InvertedResidual(64, 4),
            #InvertedResidual(64, 8),
            InvertedResidual(64, 8),
            InvertedResidual(64, 1),
        )

        self.tail_mask_s2d3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_d2s3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.channel_expand3 = nn.Conv2d(1, 64, 1)

        self.tail_state4 = nn.Sequential(
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 1),
            InvertedResidual(64, 2),
            InvertedResidual(64, 2),
            InvertedResidual(64, 4),
            InvertedResidual(64, 4),
            InvertedResidual(64, 8),
            InvertedResidual(64, 8),
            InvertedResidual(64, 1),
        )

        self.tail_mask_s2d4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.tail_mask_d2s4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16,track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )


        self.score_final = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        input_curr = x

        mask_features = []
        single_features = []
        state_curr = 0   
        start = time.perf_counter()
        torch.cuda.synchronize()
        x1 = self.head(x)
        tail_features = self.body(x1)
        prev_features = 0
        for step in range(4):
            if step == 0:
                state_curr1 = self.tail_state1(tail_features)
                s2d_features1 = self.tail_mask_s2d1(state_curr1)
                d2s_features1 = self.tail_mask_d2s1(state_curr1)
                
                mask_features.append(s2d_features1)
                single_features.append(
                    F.interpolate(d2s_features1, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
                tail_features1 = self.channel_expand1(s2d_features1)
            elif step == 1:
                state_curr2 = self.tail_state2(F.max_pool2d(tail_features1 + state_curr1, kernel_size=2, stride=2, padding=0))
                s2d_features2 = self.tail_mask_s2d2(state_curr2)
                d2s_features2 = self.tail_mask_d2s2(state_curr2)
                
                features2 = F.max_pool2d(s2d_features1, kernel_size=2, stride=2, padding=0).detach() + s2d_features2
                
                mask_features.append(features2)
                single_features.append(
                    F.interpolate(d2s_features2, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
                tail_features2 = self.channel_expand2(features2)
            elif step == 2:
                state_curr3 = self.tail_state3(F.max_pool2d(tail_features2 + state_curr2, kernel_size=2, stride=2, padding=0))
                s2d_features3 = self.tail_mask_s2d3(state_curr3)
                d2s_features3 = self.tail_mask_d2s3(state_curr3)
                
                features3 = F.max_pool2d(features2, kernel_size=2, stride=2, padding=0).detach() + s2d_features3

                mask_features.append(features3)
                single_features.append(
                    F.interpolate(d2s_features3, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))
                tail_features3 = self.channel_expand3(features3)
            elif step == 3:
                state_curr4 = self.tail_state4(F.max_pool2d(tail_features3 + state_curr3, kernel_size=2, stride=2, padding=0))
                s2d_features4 = self.tail_mask_s2d4(state_curr4)
                d2s_features4 = self.tail_mask_d2s4(state_curr4)
                
                features4 = F.max_pool2d(features3, kernel_size=2, stride=2, padding=0).detach() + s2d_features4

                mask_features.append(features4)
                single_features.append(
                    F.interpolate(d2s_features4, size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear'))

        s2d_1 = mask_features[0]
        s2d_2 = F.interpolate(mask_features[1], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        s2d_3 = F.interpolate(mask_features[2], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')
        s2d_4 = F.interpolate(mask_features[3], size=(input_curr.shape[2], input_curr.shape[3]), mode='bilinear')

        d2s_1 = single_features[0] + single_features[1].detach() + single_features[2].detach() + single_features[
            3].detach()
        d2s_2 = single_features[1] + single_features[2].detach() + single_features[3].detach()
        d2s_3 = single_features[2] + single_features[3].detach()
        d2s_4 = single_features[3]

        fuse = self.score_final(torch.cat([s2d_1, s2d_2, s2d_3, s2d_4, d2s_1, d2s_2, d2s_3, d2s_4], dim=1))
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - start
        print("inference time noshare: ", inference_time)

        return [s2d_1, s2d_2, s2d_3, s2d_4, d2s_1, d2s_2, d2s_3, d2s_4, fuse]
