# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.networks.base_network import BaseNetwork
from models.networks.architecture import SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = int(opt.ngf // 1)
        f = 8 * 2
        print(f"nf={nf}")
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        ic = 4*3+3
        self.fc = nn.Conv2d(ic, f * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(f * nf, f * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(f * nf, f * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(f * nf, f * nf, opt)
        self.up_0 = SPADEResnetBlock(f * nf, f * nf, opt)
        self.up_1 = SPADEResnetBlock(f * nf, f // 2 * nf, opt)
        self.up_2 = SPADEResnetBlock(f //2 * nf, f // 4 * nf, opt)
        self.up_3 = SPADEResnetBlock(f //4 * nf, f // 8 * nf, opt)
        # add

        final_nc = nf*2
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, warp_out=None):
        print(f"input size = {input.shape}")
        img_size = 512
        seg = torch.cat((F.interpolate(warp_out[0], size=(img_size, img_size)), F.interpolate(warp_out[1], size=(img_size, img_size)), F.interpolate(warp_out[2], size=(img_size, img_size)), warp_out[3], input), dim=1)
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        print(x.shape)
        x = self.up(x)
        x = self.up_3(x, seg)
        print(x.shape)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        print(x.shape)
        x = torch.tanh(x)
        return x
