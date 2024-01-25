import math
import random
import os
import numpy as np
import sys
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from CCCC.BCCA import BCCA_small,BCCA_base,BCCA_large

class MyDataSet(Dataset):
    def __init__(self,datas,labels):
        self.datas = torch.from_numpy(datas).float()
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, item):
        data=self.datas[item]
        label=self.labels[item]
        return data,label
    # 手动规定批处理函数，使批处理时的格式符合预期，而不是自动
    @staticmethod
    def  collate_fn(batch):
        datas,labels=tuple(zip(*batch)) #zip(*)解压 返回元组
        datas=torch.stack(datas,dim=0)
        labels=torch.as_tensor(labels)
        return datas,labels

def train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        datas,labels=data
        sample_num += datas.shape[0]
        cuda_datas = datas.to(device)
        labels=F.one_hot(labels,2).float()
        cuda_labels = labels.to(device)
        pred = model(cuda_datas)
        loss = loss_function(pred, cuda_labels)
        # 梯度反向传播
        loss.backward()
        accu_loss += loss.detach()
        # 提取预测维度上的结果????
        p=torch.max(pred.cpu(), dim=1)[1]
        l=torch.max(labels.cpu(), dim=1)[1]
        accu_num += torch.eq(p,l).sum()
        # 进度条打印格式
        data_loader.desc = "[train epoch {}] loss: {:.7f}, acc: {:.7f}".format(epoch,accu_loss.item()/(step + 1),accu_num.item()/sample_num)
        # 防止梯度消失
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        datas, labels = data
        sample_num += datas.shape[0]
        pred = model(datas.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        # ===================================================================================================#
        for i in range(pred_classes.shape[0]):
            TP += ((pred_classes[i] == 1) and (labels[i] == 1)).cpu().sum()
            TN += ((pred_classes[i] == 0) and (labels[i] == 0)).cpu().sum()
            FN += ((pred_classes[i] == 0) and (labels[i] == 1)).cpu().sum()
            FP += ((pred_classes[i] == 1) and (labels[i] == 0)).cpu().sum()
        data_loader.desc = "*****[valid epoch {}] " \
                           "loss: {:.7f}, acc: {:.7f}".format(epoch, accu_loss.item() / (step + 1),
                                                              accu_num.item() / sample_num)
    #print(f'TP:{TP}  TN:{TN}  FN:{FN}  FP:{FP}')
    dict={'TP':TP.item() , 'TN':TN.item(), 'FN':FN.item(),  'FP':FP.item()}
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,dict

def main(model,data,label,type):
    rand=random.seed(0)
    batch_size=64
    epochs=300
    lr=0.0001
    lrf=0.01

    #读取数据，划分数据集和测试集
    data_train, data_test, val_train, val_test = train_test_split(data, label, test_size=0.1,random_state=rand,shuffle=True)
    #制作dataset
    train_dataset = MyDataSet(datas=data_train,labels=val_train)
    val_dataset = MyDataSet(datas=data_test, labels=val_test)
    # pin_memory=True 如果您的数据元素是自定义类型，或者您collate_fn返回的批次是自定义类型,数据加载器将在返回之前将张量复制到 CUDA 固定内存中
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, pin_memory=True,collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, pin_memory=True,collate_fn=val_dataset.collate_fn)

    # #AdamW优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    # 将每个参数组的学习率设置为初始 lr 乘以给定函数。
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    np_log = np.zeros((epochs, 2))
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,optimizer=optimizer,data_loader=train_loader,device=device,epoch=epoch)
        # validate
        val_loss, val_acc,dict = evaluate(model=model,data_loader=val_loader,device=device,epoch=epoch)
        np_log[epoch, 0] = train_acc
        np_log[epoch, 1] = val_acc
        scheduler.step()
        if (epoch+1)%50==0:
            np.save("./BCCA_A_{}.npy".format(type),np_log)
            np.save('./BCCA_TFPN_A_{}.npy'.format(type), dict)
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                  'lr': scheduler.state_dict()}
    torch.save(checkpoint, "./BCCA_A_{}.pth".format(type))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)

    data = np.load("../finaldata/deap_npdata/eegmatrix3ds.npy")
    #valence = np.load("../finaldata/deap_npdata/valence.npy")
    arousal = np.load("../finaldata/deap_npdata/arousal.npy")
    #arousal_valence = np.load("../finaldata/deap_npdata/valabels.npy")
    model = BCCA_large(num_classes=2).to(device)
    main(model=model,data=data,label=arousal,type='large')