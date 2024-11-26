from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# Attention Feature Fusion Module
# --------------------------------------------
class ASM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=False):
        super(ASM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias)

    def forward(self, feat1, feat2):
        x1 = self.conv1(feat1)
        x2 = self.conv2(feat2)
        feat = x1 + x2
        x3 = torch.sigmoid(self.conv3(feat))
        out = x1 * x3
        out = out + feat1
        return out


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


class QGM(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(QGM, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res(x) + beta
        return x + res


class HQGN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(HQGN, self).__init__()

        self.nb = nb
        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                   nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )

        self.enc_gamma_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.enc_beta_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.enc_gamma_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.enc_beta_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.enc_gamma_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.enc_beta_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

        self.dec_gamma_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.dec_beta_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.dec_gamma_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.dec_beta_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.dec_gamma_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.dec_beta_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = nn.ModuleList([*[QGM(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                      downsample_block(nc[0], nc[1], bias=True, mode='2')])
        self.m_down2 = nn.ModuleList([*[QGM(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                      downsample_block(nc[1], nc[2], bias=True, mode='2')])
        self.m_down3 = nn.ModuleList([*[QGM(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.affm1 = ASM(n_feat=nc[0], kernel_size=3, bias=False)
        self.affm2 = ASM(n_feat=nc[1], kernel_size=3, bias=False)
        self.affm3 = ASM(n_feat=nc[2], kernel_size=3, bias=False)

        self.m_body_encoder = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([*[QGM(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                    *[QGM(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                    *[QGM(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)]])

        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x, qf, feat1, feat2, feat3):  # feature = [4, 512, 16, 16]

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        ###### qf调制参数对部分 ######
        qf_embedding = self.qf_embed(qf)  # qf:[4, 1]
        enc_gamma_3 = self.enc_gamma_3(qf_embedding)
        enc_beta_3 = self.enc_beta_3(qf_embedding)

        enc_gamma_2 = self.enc_gamma_2(qf_embedding)
        enc_beta_2 = self.enc_beta_2(qf_embedding)

        enc_gamma_1 = self.enc_gamma_1(qf_embedding)
        enc_beta_1 = self.enc_beta_1(qf_embedding)

        dec_gamma_3 = self.dec_gamma_3(qf_embedding)
        dec_beta_3 = self.dec_beta_3(qf_embedding)

        dec_gamma_2 = self.dec_gamma_2(qf_embedding)
        dec_beta_2 = self.dec_beta_2(qf_embedding)

        dec_gamma_1 = self.dec_gamma_1(qf_embedding)
        dec_beta_1 = self.dec_beta_1(qf_embedding)

        ###### 编码部分 ######
        x1 = self.m_head(x)  # [4, 64, 128, 128]

        feat1_cat = self.affm1(x1, feat1)
        for i in range(self.nb):
            feat1_cat = self.m_down1[i](feat1_cat, enc_gamma_1, enc_beta_1)
        f1 = feat1_cat
        x2 = self.m_down1[4](feat1_cat)  # [4, 128, 64, 64]

        feat2_cat = self.affm2(x2, feat2)
        for i in range(self.nb):
            feat2_cat = self.m_down2[i](feat2_cat, enc_gamma_2, enc_beta_2)
        f2 = feat2_cat
        x3 = self.m_down2[4](feat2_cat)  # [4, 256, 32, 32]

        feat3_cat = self.affm3(x3, feat3)
        for i in range(self.nb):
            feat3_cat = self.m_down3[i](feat3_cat, enc_gamma_3, enc_beta_3)
        f3 = feat3_cat
        x4 = feat3_cat

        x_enc = self.m_body_encoder(feat3_cat)

        ###### 解码部分 ######
        x_dec = self.m_body_decoder(x_enc)

        x_dec4 = x_dec + x4
        x_dec3 = x_dec4
        for i in range(self.nb):
            x_dec3 = self.m_up3[i](x_dec3, dec_gamma_3, dec_beta_3)
        f4 = x_dec3

        x_dec3 = x_dec3 + x3
        x_dec2 = self.m_up2[0](x_dec3)
        for i in range(self.nb):
            x_dec2 = self.m_up2[i+1](x_dec2, dec_gamma_2, dec_beta_2)
        f5 = x_dec2

        x_dec2 = x_dec2 + x2
        x_dec1 = self.m_up1[0](x_dec2)
        for i in range(self.nb):
            x_dec1 = self.m_up1[i+1](x_dec1, dec_gamma_1, dec_beta_1)
        f6 = x_dec1

        x_dec1 = x_dec1 + x1
        out = self.m_tail(x_dec1)
        out = out[..., :h, :w]
        # print(f1.size(), f2.size(), f3.size(), f4.size(), f5.size(), f6.size())
        features = (f1, f2, f3, f4, f5, f6)

        return out, features


if __name__ == "__main__":
    x = torch.randn(4, 1, 128, 128)
    qf = torch.randn(4, 1)
    out1 = torch.randn(4, 64, 128, 128)
    out2 = torch.randn(4, 128, 64, 64)
    out3 = torch.randn(4, 256, 32, 32)
    hqgn = HQGN()
    y, feats = hqgn(x, qf, out1, out2, out3)
    print(y.shape)