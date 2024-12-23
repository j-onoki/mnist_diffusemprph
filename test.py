import model as M
import load_mnist as l
import ddpm_function as df
import torch
import matplotlib.pyplot as plt
import loss
import einops
from einops import rearrange
import STL

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = M.model().to(device)
    
    model.load_state_dict(torch.load("model1000.pth"))

    stl = STL.Dense2DSpatialTransformer()

    #MNISTデータのダウンロード
    train_images = torch.load("./data/train_3.pt")
    test_images = torch.load("./data/test_3.pt")
    train_invert= torch.load("./data/train_3_invert.pt")
    test_invert= torch.load("./data/test_3_invert.pt")

    #indexes = torch.randperm(test_invert.size()[0])
    #test_invert = test_invert[indexes]

    #データセットの作成
    train_dataset = torch.utils.data.TensorDataset(train_images, train_invert)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_invert)
    f = test_dataset[0][0].reshape(1,1,32,32).to(device)*2-1
    m = test_dataset[9][1].reshape(1,1,32,32).to(device)*2-1
    m[0, 0, 16:32, :] = 1
    
    t = 0
    ft, epsilon = df.diffusion(f[0, 0], t, device)

    with torch.no_grad(): 
        epsilonhat, mphi, phi = model(m, f, ft.reshape(1, 1, 32, 32), t)
        fhat = df.backDiffusion(model, m, f, ft.reshape(1, 1, 32, 32), t, device)
    
    #print(phi.size())
    phiInv1 = stl(-phi[:,0].reshape(1, 1, 32, 32), phi)
    phiInv2 = stl(-phi[:,1].reshape(1, 1, 32, 32), phi)
    phiInv = torch.cat((phiInv1, phiInv2), dim=1)
    phi2 = torch.zeros(1, 1, 32, 32).to(device)
    phi = torch.cat((phi2, phi), dim=1)
    phi = rearrange(phi, "b c h w -> h w (b c)")

    #画素値を正規化
    phi[:, :, 1] = (phi[:, :, 1] - torch.min(phi[:, :, 1]))/(torch.max(phi[:, :, 1])-torch.min(phi[:, :, 1]))
    phi[:, :, 2] = (phi[:, :, 2] - torch.min(phi[:, :, 2]))/(torch.max(phi[:, :, 2])-torch.min(phi[:, :, 2]))

    f = (f+1)/2
    ft = (ft+1)/2
    fhat = (fhat+1)/2
    m = (m+1)/2
    
    mhat = stl(mphi, phiInv)
    phiInv = torch.cat((phi2, phiInv), dim=1)
    phiInv = rearrange(phiInv, "b c h w -> h w (b c)")
    #画素値を正規化
    phiInv[:, :, 1] = (phiInv[:, :, 1] - torch.min(phiInv[:, :, 1]))/(torch.max(phiInv[:, :, 1])-torch.min(phiInv[:, :, 1]))
    phiInv[:, :, 2] = (phiInv[:, :, 2] - torch.min(phiInv[:, :, 2]))/(torch.max(phiInv[:, :, 2])-torch.min(phiInv[:, :, 2]))
   
    
    plt.figure()
    plt.imshow(f.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_f.png')


    plt.figure()
    plt.imshow(ft.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_ft.png')


    plt.figure()
    plt.imshow(fhat.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_fhat.png')

    plt.figure()
    plt.imshow(m.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_m.png')

    plt.figure()
    plt.imshow(mphi.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_mphi.png')

    plt.figure()
    plt.imshow(epsilonhat.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_epsilonhat.png')

    plt.figure()
    plt.imshow(phi.cpu())
    plt.savefig('./image/test_phi.png')
    
    plt.figure()
    plt.imshow(phiInv.cpu())
    plt.savefig('./image/test_phiInv.png')
    
    plt.figure()
    plt.imshow(mhat.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_mhat.png')