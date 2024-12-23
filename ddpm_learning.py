import torch
import ddpm_function as df
from tqdm import tqdm

#1epoch分の訓練を行う関数
def train_model(model, train_loader, optimizer, device):

    train_loss = 0.0
    train_dloss = 0.0
    train_rloss = 0.0

    # 学習モデルに変換
    model.train()

    for i, (f, m) in enumerate(tqdm(train_loader)):

        f, m = f.to(device), m.to(device)
        f = f.reshape(f.size()[0], 1, 32, 32)*2 - 1 #range(0,1)->(-1,1)
        m = m.reshape(m.size()[0], 1, 32, 32)*2 - 1

        ft, t, epsilon = df.addNoise(f, device)

        # 勾配を初期化
        optimizer.zero_grad()

        # 推論(順伝播)
        epsilonhat, mphi, phi = model(m, f, ft, t)

        # 損失の算出
        loss, dloss, rloss = df.criterion(epsilonhat, epsilon.to(device), mphi, f, phi)
        
        # 誤差逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()
        train_dloss += dloss.item()
        train_rloss += rloss.item()
    
    # lossの平均値を取る
    train_loss = train_loss / len(train_loader)
    train_dloss = train_dloss / len(train_loader)
    train_rloss = train_rloss / len(train_loader)
    
    return train_loss, train_dloss, train_rloss

#モデル評価を行う関数
def test_model(model, test_loader, optimizer, device):

    test_loss = 0.0
    test_dloss = 0.0
    test_rloss = 0.0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化

        for i, (f, m) in enumerate(tqdm(test_loader)):

            f, m = f.to(device), m.to(device)
            f = f.reshape(f.size()[0], 1, 32, 32)*2 - 1
            m = m.reshape(m.size()[0], 1, 32, 32)*2 - 1

            ft, t, epsilon = df.addNoise(f, device)
            epsilonhat, mphi, phi = model(m, f, ft, t)
            loss, dloss, rloss = df.criterion(epsilonhat, epsilon.to(device), mphi, f, phi)
            
            test_loss += loss.item()
            test_dloss += dloss.item()
            test_rloss += rloss.item()


    # lossの平均値を取る
    test_loss = test_loss / len(test_loader)
    test_dloss = test_dloss / len(test_loader)
    test_rloss = test_rloss / len(test_loader)

    return test_loss, test_dloss, test_rloss

def lerning(model, train_loader, test_loader, optimizer, num_epochs, device):

    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs+1, 1):

        train_loss, train_dloss, train_rloss = train_model(model, train_loader, optimizer, device)
        test_loss, test_dloss, test_rloss = test_model(model, test_loader, optimizer, device)
        
        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))
        print("epoch : {}, train_dloss : {:.5f}, test_dloss : {:.5f}" .format(epoch, train_dloss, test_dloss))
        print("epoch : {}, train_rloss : {:.5f}, test_rloss : {:.5f}" .format(epoch, train_rloss, test_rloss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if epoch%100 == 0:
            filename = 'model' + str(epoch) + '.pth'
            torch.save(model.state_dict(), filename)
    return train_loss_list, test_loss_list
