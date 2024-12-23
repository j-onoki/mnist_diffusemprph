import torch
import math
import torch.nn.functional as f
import matplotlib.pyplot as plt
import loss
import torch.nn as nn
step = 1000
s = 0.008

#alfa, bataの計算
def alfa_bata_cos(T, s):
    t = torch.arange(0, T, 1)
    t = (t/T + s)/(1 + s)*torch.pi/2
    ft = torch.cos(t)
    ft = ft*ft
    alfaTBar = ft/ft[0]
    bataTBar = 1 - alfaTBar
    alfaT = torch.roll(alfaTBar, 1)
    alfaT[0] = 1
    alfaT = alfaTBar/alfaT 
    bataT = 1 - alfaT
    return alfaTBar, bataTBar, alfaT, bataT


#tの埋め込み
def embeddings(t, dmodel):

    embeddings = torch.exp(- 2*math.log(10000)/dmodel*(torch.arange(dmodel/2)))
    embeddings = torch.reshape(embeddings, (1, embeddings.size()[0]))
    if type(t) == int:
        t = torch.tensor([[t]])
    else:
        t = torch.reshape(t, (t.size()[0], 1))
    embeddings = torch.mm(t.float().to('cuda'), embeddings.to('cuda'))
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) 

    return embeddings

def crossCorrelation(mphi, f):
    mphimean = torch.mean(mphi, dim=2, keepdim=True)
    mphimean = torch.mean(mphimean, dim=3, keepdim=True)
    fmean = torch.mean(f, dim=2, keepdim=True)
    fmean = torch.mean(fmean, dim=3, keepdim=True)
    mphi = mphi - mphimean
    f = f - fmean
    a = torch.sum(mphi*f, dim=2)
    a = torch.sum(a, dim=2)
    b1 = torch.sum(mphi*mphi, dim=2)
    b1 = torch.sqrt(torch.sum(b1, dim=2))
    b2 = torch.sum(f*f, dim=2)
    b2 = torch.sqrt(torch.sum(b2, dim=2))
    b = b1*b2
    cc = a/(b+1e-5)
    loss = torch.mean(cc)
    return loss

#損失関数
def criterion(epsilonhat, epsilon, mphi, f, phi):
    ramda = 1 
    ramdaphi = 1
    g = loss.gradientLoss()
    diffusion_loss = torch.nn.functional.mse_loss(epsilon, epsilonhat)
    f = (f+1)/2
    cc = crossCorrelation(mphi, f)
    regist_loss = -cc*cc + ramdaphi*g(phi)
    l = diffusion_loss + ramda*regist_loss
    return l, diffusion_loss, cc

#拡散過程
def diffusion(x, t, device):

    datadim1 = x.size()[0]
    datadim2 = x.size()[1]
    AlfaTBar, BataTBar, AlfaT, BataT = alfa_bata_cos(step, s)
    rnd = torch.randn(size=(datadim1, datadim2)).to(device)
    xt = x*torch.sqrt(AlfaTBar[t]) + rnd*torch.sqrt(1-AlfaTBar[t])
    return xt, rnd

#逆拡散過程
def backDiffusion(model, m, f, ft, T, device):

    AlfaTBar, BataTBar, AlfaT, BataT = alfa_bata_cos(step, s)
    model.eval()
    datasize = 1
    x = ft
    datadim = x.size()[2]

    for t in range(T):
        rnd = torch.randn(size=(datasize,datadim,datadim)).to(device)
        if t == T:
            rnd = 0
        time = torch.full((datasize,1),fill_value=T-t+1)
        x = (x - (BataT[T-t]/torch.sqrt(BataTBar[T-t]))*model.G(m, f, x, time))/torch.sqrt(AlfaT[T-t]) + torch.sqrt(BataT[T-t])*rnd
        print(t+1)
    
    return x

#逆拡散過程(高周波用)
def backDiffusion2(model, x, xcond, T, device):

    AlfaTBar, BataTBar, AlfaT, BataT = alfa_bata_cos(step, s)
    model.eval()
    datasize = 1
    datadim = x.size()[2]

    for t in range(T):
        rnd = torch.randn(size=(datasize,3,datadim,datadim)).to(device)
        if t == T:
            rnd = 0
        time = torch.full((datasize,1),fill_value=T-t+1)
        x = (x - (BataT[T-t]/torch.sqrt(BataTBar[T-t]))*model(torch.cat([xcond, x], dim=1), time))/torch.sqrt(AlfaT[T-t]) + torch.sqrt(BataT[T-t])*rnd
        #print(t+1)
    
    return x


#逆拡散過程(半分隠す)
def backDiffusionCond(model, x, xcond, T, key, device):

    BataT = torch.linspace(0.001, 0.1, step).to(device)
    AlfaT = 1 - BataT
    AlfaTBar = torch.cumprod(AlfaT, 0)
    BataTBar = 1 - AlfaTBar
    model.eval()
    datasize = 1
    datadim1 = x.size()[1]

    for t in range(T):
        rnd = torch.randn(size=(datasize,28,28)).to(device)
        if t == T:
            rnd = 0
        time = torch.full((datasize,1),fill_value=T-t+1)
        if key == 0:
            x[0, 0, 0:13, :] = xcond[0, 0, 0:13, :]
        elif key == 1:
            x[0, 0, :, 14:27] = xcond[0, 0, :, 14:27]
        x = (x - (BataT[T-t]/torch.sqrt(BataTBar[T-t]))*model(x, time))/torch.sqrt(AlfaT[T-t]) + torch.sqrt(BataT[T-t])*rnd
        #print(t+1)
    
    return x


#t, epsilonをサンプリングしてxtを計算
def addNoise(images, device):

    t = torch.randint(step, size = (images.size()[0],)).to(device)
    epsilon = torch.zeros(size = (images.size()[0], 1, images.size()[2], images.size()[3]))
    output = torch.zeros(size = images.size()).to(device)
    
    for i in range(images.size()[0]):
        output[i, 0, :], epsilon[i, 0, :] = diffusion(images[i, 0, :], t[i], device)

    return output, t, epsilon

#t, epsilonをサンプリングしてxtを計算
def addNoise2(images, device):

    t = torch.randint(step, size = (images.size()[0],)).to(device)
    epsilon1 = torch.zeros(size = (images.size()[0], images.size()[2], images.size()[3]))
    epsilon2 = torch.zeros(size = (images.size()[0], images.size()[2], images.size()[3]))
    epsilon3 = torch.zeros(size = (images.size()[0], images.size()[2], images.size()[3]))

    for i in range(images.size()[0]):
        images[i, 0, :], epsilon1[i, :] = diffusion(images[i, 0, :], t[i], device)
        images[i, 1, :], epsilon2[i, :] = diffusion(images[i, 1, :], t[i], device)
        images[i, 2, :], epsilon3[i, :] = diffusion(images[i, 2, :], t[i], device)
    return images, t, epsilon1, epsilon2, epsilon3


if __name__ == '__main__':

    T = 1000
    #s = 0.0001
    t = torch.arange(0, 1000, 1)
    alfaTBar, bataTBar, alfaT, bataT = alfa_bata_cos(T, s)

    plt.figure()
    plt.plot(t, alfaTBar)
    plt.plot(t, bataTBar)
    plt.savefig('./image/alfa_bata_cos.png')

    alfaTBar = torch.cumprod(alfaT, 0)
    bataTBar = 1 - alfaTBar
    plt.figure()
    plt.plot(t, alfaTBar)
    plt.plot(t, bataTBar)
    plt.savefig('./image/alfa_bata_cos_2.png')

