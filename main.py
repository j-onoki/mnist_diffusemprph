import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import model as m
import load_mnist as l
import ddpm_learning
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #GPUが使えるか確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    #モデルのインスタンス化
    model = m.model().to(device)
    print(model)

    #MNISTデータのダウンロード
    train_images = torch.load("./data/train_3.pt")
    test_images = torch.load("./data/test_3.pt")
    train_invert = torch.load("./data/train_3_invert.pt")
    test_invert = torch.load("./data/test_3_invert.pt")

    #反転した方をランダムにシャッフル
    indexes = torch.randperm(train_invert.size()[0])
    train_invert = train_invert[indexes]
    indexes = torch.randperm(test_invert.size()[0])
    test_invert = test_invert[indexes]

    #データセットの作成
    train_dataset = torch.utils.data.TensorDataset(train_images, train_invert)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_invert)

    #ミニバッチの作成
    train_loader, test_loader = l.loader_MNIST(train_dataset, test_dataset)

    #最適化法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.01)

    num_epochs = 1000
    train_loss_list, test_loss_list = ddpm_learning.lerning(model, train_loader, test_loader, optimizer, num_epochs, device)

    plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('./image/loss1.png')

    #モデルを保存する。
    torch.save(model.state_dict(), "model.pth")