import torch
from torch import nn
from torch.nn import functional as F

class ConvUP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ConvDN(nn.Module):
    def __init__(self, in_channels, mid_channels,last = False):
        super().__init__()
        self.last = last
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,mid_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels)
        )
    def forward(self, x):
        if self.last:
            return torch.sigmoid(self.conv(x))
        else:
            return F.leaky_relu_(self.conv(x))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enlay1 = ConvUP(1,32)
        self.enlay2 = ConvUP(32,64)
        self.enlay3 = ConvUP(64,128)
        self.enlay4 = ConvUP(128,128)
        self.enlay5 = ConvUP(128,128)

        self.delay1 = ConvDN(128,128)
        self.delay2 = ConvDN(128,128)
        self.delay3 = ConvDN(128,64)
        self.delay4 = ConvDN(64,32)
        self.delay5 = ConvDN(32,2,last = True)

    def forward(self, x_img):
        x = x_img
        
        x = self.enlay1(x)
        x = self.enlay2(x)
        x = self.enlay3(x)
        x = self.enlay4(x)
        x = self.enlay5(x)
        
        x = self.delay1(x)
        x = self.delay2(x)
        x = self.delay3(x)
        x = self.delay4(x)
        x = self.delay5(x)
        return x
