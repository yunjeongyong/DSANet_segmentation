# import torch
# from torch import nn
# import torch.nn.functional as F
#
# # 깃허브 주소: https://github.com/bryandlee/animegan2-pytorch/blob/main/model.py
#
#
# class ConvNormLReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
#         pad_layer = {
#             "zero": nn.ZeroPad2d,
#             "same": nn.ReplicationPad2d,
#             "reflect": nn.ReflectionPad2d,
#         }
#         if pad_mode not in pad_layer:
#             raise NotImplementedError
#
#         super(ConvNormLReLU, self).__init__(
#             pad_layer[pad_mode](padding),
#             nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
#             nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#
# class InvertedResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, expansion_ratio=2):
#         super(InvertedResBlock, self).__init__()
#
#         self.use_res_connect = in_ch == out_ch
#         bottleneck = int(round(in_ch * expansion_ratio))
#         layers = []
#         if expansion_ratio != 1:
#             layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
#
#         # dw
#         layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
#         # pw
#         layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
#         layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))
#
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, input):
#         out = self.layers(input)
#         if self.use_res_connect:
#             out = input + out
#         return out
#
#
# class Generator(nn.Module):
#     def __init__(self,):
#         super().__init__()
#
#         self.block_a = nn.Sequential(
#             ConvNormLReLU(3, 32, kernel_size=7, padding=3),
#             ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
#             ConvNormLReLU(64, 64)
#         )
#
#         self.block_b = nn.Sequential(
#             ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
#             ConvNormLReLU(128, 128)
#         )
#
#         self.block_c = nn.Sequential(
#             ConvNormLReLU(128, 128),
#             InvertedResBlock(128, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             ConvNormLReLU(256, 128),
#         )
#
#         # self.block_d = nn.Sequential(
#         #     ConvNormLReLU(128, 128),
#         #     ConvNormLReLU(128, 128)
#         # )
#         #
#         # self.block_e = nn.Sequential(
#         #     ConvNormLReLU(128, 64),
#         #     ConvNormLReLU(64, 64),
#         #     ConvNormLReLU(64, 32, kernel_size=7, padding=3)
#         # )
#         #
#         # self.out_layer = nn.Sequential(
#         #     nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
#         #     nn.Tanh()
#         # )
#
#     def forward(self, input, align_corners=True):
#         out_features = []
#         out = self.block_a(input)
#         half_size = out.size()[-2:]
#         out = self.block_b(out)
#         for i in range(len(self.block_c)):
#             out = self.block_c[i](out)
#             print('out.shape', out)
#             out_features.append(out)
#
#
#
#         # out = self.block_c(out)
#
#         # if align_corners:
#         #     out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
#         # else:
#         #     out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         # out = self.block_d(out)
#         #
#         # if align_corners:
#         #     out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
#         # else:
#         #     out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         # out = self.block_e(out)
#         #
#         # out = self.out_layer(out)
#         return out_features
#
#
# if __name__ == "__main__":
#     img1 = torch.randn(1, 3, 512, 512)
#     output = Generator()
#     output = output(img1)
#     print('len(output)', len(output))
#     print('output',output)
#     for i in output:
#         print('output.shape', i.shape)
#
#     # print(output.shape)






###############################################
import torch
from torch import nn
import torch.nn.functional as F


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator_d(nn.Module):
    def __init__(self,):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            # ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, out, half_size, align_corners=True):
    # def forward(self, input, align_corners=True):
    #     out = self.block_a(input)
    #     print('first_out', out.shape)
    #     half_size = out.size()[-2:]
    #     print('half_size', half_size)
    #     out = self.block_b(out)
    #     out = self.block_c(out)
    #     y1, y2 = out.chunk(chunks=2, dim=1)
    #     print('y1',y1.shape)
    #     print('y2', y2.shape)
    #     # y3 = y1 + y2
    #     # y3 = torch.cat([y1, y2], dim=1)
    #     # print('y1+y2', y3.shape)
    #
    #     out = y1
    #     print('out.shape', out.shape)
    #     # y1 = y1 + y2

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        #print('eirwoiefowjesdf', out.shape)
        # conv = ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1))
        # out = conv(out)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


if __name__ == "__main__":
    img1 = torch.randn(1, 3, 512, 512)
    output = Generator_d()
    output = output(img1)
    # print('y1.shape', y1.shape)
    # print('y2.shape', y2.shape)
    print('output.shape', output.shape)
