from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

#データのロード
def load_MNIST():
    #訓練データ
    train_dataset = datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.Compose([transforms.ToTensor()]),
                                            download = True)
    #検証データ
    test_dataset = datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.Compose([transforms.ToTensor()]),
                                            download = True)
    return train_dataset, test_dataset

#ミニバッチの作成
def loader_MNIST(train_dataset, test_dataset):

    batch_size = 100

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    return train_loader, test_loader
