import load_mnist as l
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f
from torchvision import transforms as transforms

#前処理用の関数
def conv(image, kernel):
    w, h = image.size()[0], image.size()[1]
    s = kernel.size()[0]
    output = torch.zeros((w//2, h//2))
    for i in range(w//2):
        for j in range(h//2):
            output[i, j] = torch.sum(image[i*2:i*2+2, j*2:j*2+2]*kernel)

    return output

if __name__ == '__main__':

    train = torch.load("./data/train.pt") 
    train_invert = torch.load("./data/train_invert.pt") 
    train_label = torch.load("./data/train_label.pt") 
    test = torch.load("./data/test.pt")
    test_invert = torch.load("./data/test_invert.pt") 
    test_label = torch.load("./data/test_label.pt")

    train_3 = torch.zeros(6131, 32, 32)
    train_3_invert = torch.zeros(6131, 32, 32)

    # transform = transforms.RandomInvert(p=1)
    
    # train_invert = torch.zeros(10000, 32, 32)

    # for i in range(10000):
    #     train_invert[i] = transform(train[i].reshape(1, 32, 32))
    #     print(i)

    # train_invert = torch.zeros(60000, 32, 32)

    j=0
    for i in range(60000):
        if train_label[i]==3:
            train_3[j] = train[i] 
            train_3_invert[j] = train_invert[i]
            j+=1
        print(i)

    print(j)
    torch.save(train_3_invert, "./data/train_3_invert.pt")
    torch.save(train_3, "./data/train_3.pt")

    plt.figure()
    plt.imshow(train_3[1009].cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/train_3.png')