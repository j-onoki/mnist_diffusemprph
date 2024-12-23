import torch
import torch.nn as nn
import torch.nn.functional as f
import ddpm_function as df
from torch import einsum
import einops
from einops import rearrange



#畳み込みブロック
class ConvBlock(nn.Module):
    def __init__(self, dimin, dimout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dimin, dimout, 3, padding = 1),
            nn.LeakyReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dimout, dimout, 3, padding = 1), 
            nn.LeakyReLU()
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        
        return x

#ダウンサンプリング
class DownSampling(nn.Module):
    def __init__(self, layers, dimin0, dimout0):
        super().__init__()
        self.module1 = nn.ModuleList([])
        self.module2 = nn.ModuleList([])
        dimin = dimin0
        dimout = dimout0
        for i in range(layers):
            self.module1.append(ConvBlock(dimin, dimout))
            self.module2.append(nn.Conv2d(dimout, dimout, 3, 2, 1))
            dimin = dimout
            dimout *= 2   
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        cache = []
        for (m1, m2) in zip(self.module1, self.module2):
            x = m1(x)
            cache.append(x)
            x = m2(x)
            x = self.LeakyReLU(x)
        return x, cache

#中間層
class MidBlock(nn.Module):
    def __init__(self, dimin, dimout):
        super().__init__()
        self.conv1 = ConvBlock(dimin, dimout)

    def forward(self, x):
        x = self.conv1(x)
        return x

#アップサンプリング
class UpSampling(nn.Module):
    def __init__(self, layers, dimin0, dimout0):
        super().__init__()
        self.module1 = nn.ModuleList([])
        self.upconv = nn.ModuleList([])
        dimin = dimin0
        dimout = dimout0
        for j in range(layers):
            self.module1.append(ConvBlock(dimin, dimout))
            self.upconv.append(nn.ConvTranspose2d(dimin, dimout, 3, 2, 1, output_padding=1))
            dimin = dimout
            dimout //= 2
        
    def forward(self, x, cache):
        n = len(cache) - 1
        for (m1,u) in zip(self.module1, self.upconv):
            x = u(x)
            x = torch.cat((x, cache[n]), dim=1)
            x = m1(x)
            n -= 1
        return x

#U-Net
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downs = DownSampling(4, 2, 64)
        self.mid = MidBlock(512, 1024)
        self.ups = UpSampling(4, 1024, 512)
        self.conv= nn.Conv2d(64, 2, 1)

    def forward(self, epsilonhat, m):
        x = torch.cat((epsilonhat, m), dim=1)
        x, cache = self.downs(x)
        x = self.mid(x)
        x = self.ups(x, cache)
        x = self.conv(x)
        return x


            