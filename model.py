import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes_in, padding=angRes_in, bias=False)
        self.DistgGroup = CascadedDistgGroup(n_group, n_block, angRes_in, channels)
        self.UpSample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        )

    def forward(self, x):
        x = SAI2MacPI(x, self.angRes_in)
        buffer = self.init_conv(x)
        buffer = self.DistgGroup(buffer)
        out = self.UpSample(buffer)
        out = MacPI2SAI(out, self.angRes_out)
        return out


class CascadedDistgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadedDistgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DistgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.fuse = nn.Conv2d(n_group * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        for i in range(self.n_group):
            x = self.Group[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DistgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))
        self.Blocks = nn.Sequential(*Blocks)
        self.fuse = nn.Conv2d((n_block + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        temp.append(x)
        for i in range(self.n_block):
            x = self.Blocks[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DistgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels, channels

        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes],
                      padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            PixelShuffle1D(angRes),
        )
        self.squeezeConv = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((x, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.squeezeConv(buffer)
        y = self.SpaConv(buffer) + buffer
        return y


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tentor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out



if __name__ == "__main__":
    net = Net(angRes_in=2, angRes_out=7).cuda()
    from thop import profile
    input = torch.randn(1, 1, 192, 192).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops * 2 / 1e9))