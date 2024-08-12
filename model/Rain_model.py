from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
        self.convfinal = nn.Sequential(
            nn.Conv2d(32, 6, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        input = input.cuda()
        x = input.cuda()
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        #if self.use_GPU:
        h = h.cuda()
        c = c.cuda()


        #flag = self.iteration-1
        for i in range(5):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            x = x + input
        x = torch.cat((input, x), 1)
        x = self.conv0(x)
        x = torch.cat((x, h), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        x = h
        resx = x
        x = F.relu(self.res_conv1(x) + resx)
        resx = x
        x = F.relu(self.res_conv2(x) + resx)
        resx = x
        x = F.relu(self.res_conv3(x) + resx)
        resx = x
        x = F.relu(self.res_conv4(x) + resx)
        resx = x
        x = F.relu(self.res_conv5(x) + resx)
        x = self.convfinal(x)
        mu_b = x[:, :3, :, :] + input  # revising the prenet for obtaining parameters including mu_b and logvar_b
        logvar_b = x[:, 3:, :, :]
        return mu_b, logvar_b


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Derain(nn.Module):
    def __init__(self, args):
        super(Derain, self).__init__()
        self.derainet = PReNet(args)
    def forward(self, input):
        mu_b, logvar_b = self.derainet(input)
        return mu_b, logvar_b

class EDNet(nn.Module):  # RNet + G
    def __init__(self,nc,nz,nef):
        super(EDNet,self).__init__()
        self.nc = nc
        self.nz = nz
        self.nef= nef
        self.encoder = Encoder(self.nc,self.nef,self.nz)
        self.decoder = Decoder(self.nz,self.nef,self.nc)
    def sample (self, input):
        return self.decoder(input)

    def forward(self, input):
        distributions, _ = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return R, mu, logvar, z






# class Encoder(nn.Module):  # RNet
#     def __init__(self, nc, nef, nz):
#         super(Encoder, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nef*8, nef*16, 4, 1),
#             nn.ReLU(True),
#             View((-1, nef*16 * 1 * 1)),
#             nn.Linear(nef*16, nz* 2),
#         )
#     def forward(self, input):
#         distributions = self.main(input)
#         return distributions
#
# class Decoder(nn.Module):
#     def __init__(self, nz, nef, nc):
#         super(Decoder, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(nz, nef*16),
#             View((-1, nef*16, 1, 1)),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(nef*16, nef * 8, 4, 1, 0, bias=False),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
#             nn.ReLU(True)
#         )
#     def forward(self, input):
#         R = self.main(input)
#         return R
#
class Encoder(nn.Module):  # RNet
    def __init__(self, nc, nef, nz):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef*8, nef*16, 4, 1),
            nn.ReLU(True),
            View((-1, nef*16 * 1 * 1)),
            nn.Linear(nef*16, nz * 2),
        )
    def forward(self, input, nef=32, nz=128):
        linear = nn.Linear(nef * 16, nz)
        out_features = []
        distributions = self.main(input)
        a = 0
        for i in range(len(self.main)-1):
            input = self.main[i](input)
            print('encoder',input.shape)
            if i == 10:
                input = self.main[i](input)
                input = linear(input)
                print('input',input.shape)
                out_features.append(input)
                a += 1


        return distributions, out_features[0]

class Decoder(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef*16),
            View((-1, nef*16, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef*16, nef * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )
    def forward(self, input):
        #
        # linear = nn.Linear(nef * 16, nz)
        out_features = []
        R = self.main(input)
        # print('R',R.shape)
        # a = 0
        # for i in range(len(self.main)-1):
        #     input = self.main[i](input)
        #     print('decoder',input.shape)
        # # if i == 10:
        # #     input = self.main[i](input)
        #     # input = linear(input)
        #     # out_features.append(input)
        #     a += 1
        return R


if __name__ == "__main__":
    # a = torch.randn(2, 3, 256, 256)
    # encoder = Encoder(3, 32, 256)
    # decoder = Decoder(256,32, 3)
    # encoder_result, output = encoder(a)
    # print('encoder의 output', output.shape)
    # print(encoder_result.shape)
    # prenet = PReNet(recurrent_iter=6).cuda()
    # prenet_output1, prenet_output2 = prenet(a)
    # # print(encoder_result.shape)
    # # print(output)
    # # decoder_result = decoder(encoder_result)
    # # print(decoder_result.shape)
    # ednet = EDNet(3, 256, 32)
    # ednet_output, _, _, z = ednet(a)
    # print('z',z.shape)
    # # input_fake = prenet_output1 + ednet_output
    # print(ednet_output.shape)
    # print(prenet_output1.shape)
    # print(prenet_output2.shape)
    # input_fake = prenet_output1 + ednet_output
    # print(input_fake.shape)



    input = torch.randn(1, 3, 64, 64)
    encoder = Encoder(3, 32, 128)
    decoder = Decoder(128, 32, 3)

    distribution, encoder_output = encoder(input)
    print('encoder_output',encoder_output.shape)
    decoder_output = decoder(encoder_output)
    print('decoder_output',decoder_output.shape)


    # netDerain = PReNet(recurrent_iter=6).cuda()
    # netEDNet = EDNet(3, 128, 32)
    # mu_b, logvar_b = netDerain(input)
    # rain_make, mu_z, logvar_z, _ = netEDNet(input)
    # mu_b = mu_b.cuda()
    # rain_make = rain_make.cuda()
    #
    # input_fake = mu_b + rain_make
    # print(rain_make.shape)

