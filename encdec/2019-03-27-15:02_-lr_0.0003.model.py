import torch
import torch.nn as nn
import torch.utils.data
# from ipdb import set_trace as stop

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()

        self.upsample = upsample

        if (upsample):
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = torch.nn.functional.interpolate(out, scale_factor=2)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out

class block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(block, self).__init__()
        self.A01 = ConvBlock(in_planes, 32, kernel_size=3)
        
        self.C01 = ConvBlock(32, 64, stride=2)
        self.C02 = ConvBlock(64, 64)
        self.C03 = ConvBlock(64, 64)
        self.C04 = ConvBlock(64, 64, kernel_size=1)

        self.C11 = ConvBlock(64, 64)
        self.C12 = ConvBlock(64, 64)
        self.C13 = ConvBlock(64, 64)
        self.C14 = ConvBlock(64, 64, kernel_size=1)
        
        self.C21 = ConvBlock(64, 128, stride=2)
        self.C22 = ConvBlock(128, 128)
        self.C23 = ConvBlock(128, 128)
        self.C24 = ConvBlock(128, 128, kernel_size=1)
        
        self.C31 = ConvBlock(128, 256, stride=2)
        self.C32 = ConvBlock(256, 256)
        self.C33 = ConvBlock(256, 256)
        self.C34 = ConvBlock(256, 256, kernel_size=1)
        
        self.C41 = ConvBlock(256, 128, upsample=True)
        self.C42 = ConvBlock(128, 128)
        self.C43 = ConvBlock(128, 128)
        self.C44 = ConvBlock(128, 128)
        
        self.C51 = ConvBlock(128, 64, upsample=True)
        self.C52 = ConvBlock(64, 64)
        self.C53 = ConvBlock(64, 64)
        self.C54 = ConvBlock(64, 64)
        
        self.C61 = ConvBlock(64, 64, upsample=True)
        self.C62 = ConvBlock(64, 64)
        self.C63 = ConvBlock(64, 64)

        self.C64 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.C64.weight)
        nn.init.constant_(self.C64.bias, 0.1)

        
    def forward(self, x):



        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = self.C04(C03)
        C04 += C01
        
        # N/2 -> N/2
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        # N/2 -> N/4
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        # N/4 -> N/8
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C41 += C24
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41
        
        C51 = self.C51(C44)
        C51 += C14
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = self.C61(C54)        
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        C64 = self.C64(C63)

        # T -> 0-7
        # vz -> 7-14
        # tau -> 14-21
        # logP -> 21-28
        # np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)) -> 28-35
        # np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)) -> 35-42
        # Bz -> 42-49

        out = C64
        out[:,0:7,:,:] += x[:,0:1,:,:]
        out[:,14:28,:,:] += x[:,0:1,:,:]
        
        # out = C64 + x[:,0:1,:,:]
        # out[:,7:14,:,:] -= x[:,0:1,:,:]
        # out[:,28:49,:,:] -= x[:,0:1,:,:]

        # out = C64
        # out[:,0:7,:,:] += x[:,0:1,:,:]
        # out[:,14:21,:,:] += x[:,0:1,:,:]
        
        return out