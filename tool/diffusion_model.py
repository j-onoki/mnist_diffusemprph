import torch
import torch.nn as nn
import torch.nn.functional as f
import ddpm_function as df
from torch import einsum
import einops
from einops import rearrange

#残差結合
class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
 
  def forward(self, x, *args, **kwargs):

    return self.fn(x, *args, **kwargs) + x

#Attentionの前にGroupNormalization
class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
 
  def forward(self, x):
    x = self.norm(x)
    return self.fn(x)

#Attention
class Attention(nn.Module):
  def __init__(self, dim, heads=4, dim_head=32):
    super().__init__()
    self.scale = dim_head ** (- 0.5) # d^(-1/2) 
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
    self.to_out = nn.Conv2d(hidden_dim, dim, 1)
 
  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim=1) # Q, K, Vの3つにわける
    q, k, v = map(
        lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
    )
    q = q * self.scale
 
    sim = einsum("b h d i, b h d j -> b h i j", q, k)
    sim = sim - sim.amax(dim=-1, keepdim=True).detach()
    attn = sim.softmax(dim=-1)
 
    out = einsum("b h i j, b h d j -> b h i d", attn ,v)
    out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
    return self.to_out(out)

#linear attention 
class LinearAttention(nn.Module):
  def __init__(self, dim, heads=4, dim_head=32):
    super().__init__()
    self.scale = dim_head ** (- 0.5)
    self.heads = heads
    hidden_dim = dim_head * heads
    self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
    self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                nn.GroupNorm(1, dim))
     
  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.to_qkv(x).chunk(3, dim=1)
    q, k, v = map(
        lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
    )
 
    q = q.softmax(dim=-2)
    k = k.softmax(dim=-1)
 
    q = q * self.scale
    context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
 
    out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
    out =rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
    return self.to_out(out)

#時間の位置符号化
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.dim = dim
    
    def forward(self, t):
        t = df.embeddings(t, self.dim)
        time_emb = self.time_mlp(t)
        return self.time_mlp(t)

#畳み込みブロック
class ConvBlock(nn.Module):
    def __init__(self, dimin, dimout, groups):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dimin, dimout, 3, padding = 1),
            nn.GroupNorm(groups, dimout), 
            nn.ReLU()
        )
        self.time_mlp=TimeEmbedding(dimout)
        self.block2 = nn.Sequential(
            nn.Conv2d(dimout, dimout, 3, padding = 1),
            nn.GroupNorm(groups, dimout), 
            nn.ReLU()
        )
        self.res_conv = nn.Conv2d(dimin, dimout, 1)

    def forward(self, x, t):
              
        res = self.res_conv(x)
        x = self.block1(x)
        time_emb = self.time_mlp(t)
        x = time_emb.reshape(time_emb.size()[0], time_emb.size()[1], 1, 1) + x
        x = self.block2(x)
        
        return x + res

#ダウンサンプリング
class DownSampling(nn.Module):
    def __init__(self, layers, dimin0, dimout0, groups):
        super().__init__()
        self.module1 = nn.ModuleList([])
        self.module2 = nn.ModuleList([])
        self.module3 = nn.ModuleList([])
        dimin = dimin0
        dimout = dimout0
        for i in range(layers):
            self.module1.append(ConvBlock(dimin, dimout, groups))
            self.module2.append(ConvBlock(dimout, dimout, groups))                              
            self.module3.append(Residual(PreNorm(dimout, LinearAttention(dimout))))
            dimin = dimout
            dimout *= 2
        #self.attn = PreNorm(dimout,)    
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x, t):
        cache = []
        for (m1, m2, m3) in zip(self.module1, self.module2, self.module3):
            x = m1(x, t)
            x = m2(x, t)
            x = m3(x)
            cache.append(x)
            x = self.pool(x)
        return x, cache

#中間層
class MidBlock(nn.Module):
    def __init__(self, dimin, dimout, groups):
        super().__init__()
        self.conv1 = ConvBlock(dimin, dimout, groups)
        self.attn = Residual(PreNorm(dimout, Attention(dimout)))
        self.conv2 = ConvBlock(dimout, dimout, groups)

    def forward(self, x, t):
        x = self.conv1(x, t)
        x = self.attn(x)
        x = self.conv2(x, t)
        return x

#アップサンプリング
class UpSampling(nn.Module):
    def __init__(self, layers, dimin0, dimout0, groups):
        super().__init__()
        self.module1 = nn.ModuleList([])
        self.module2 = nn.ModuleList([])
        self.module3 = nn.ModuleList([])
        self.upconv = nn.ModuleList([])
        dimin = dimin0
        dimout = dimout0
        for j in range(layers):
            self.module1.append(ConvBlock(dimin, dimout, groups))
            self.module2.append(ConvBlock(dimout, dimout, groups))                              
            self.module3.append(Residual(PreNorm(dimout, LinearAttention(dimout))))
            self.upconv.append(nn.ConvTranspose2d(dimin, dimout, 4, 2, 1))
            dimin = dimout
            dimout //= 2
        
    def forward(self, x, t, cache):
        n = len(cache) - 1
        for (m1, m2, m3,u) in zip(self.module1,self.module2,self.module3,self.upconv):
            x = u(x)
            x = torch.cat((x, cache[n]), dim=1)
            x = m1(x, t)
            x = m2(x, t)
            x = m3(x)
            n -= 1
        return x

#U-Net
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downs = DownSampling(4, 3, 64, 8)
        self.mid = MidBlock(512, 1024, 8)
        self.ups = UpSampling(4, 1024, 512, 8)
        self.conv= nn.Conv2d(64, 1, 1)

    def forward(self, m, f, ft, t):
        x = torch.cat((m, f, ft), dim=1)
        x, cache = self.downs(x, t)
        x = self.mid(x, t)
        x = self.ups(x, t, cache)
        x = self.conv(x)
        return x


            