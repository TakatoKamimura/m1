import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import models_pooling_padding as Model
from dataset import dataset_padding as Data
import os
import numpy as np
import math

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
class AverageMeter(object):#損失の推移を確認する用のクラス
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=8):#valにlossをいれる
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def acc(input, labels):
    bs = input.size(0)
    # print(bs)
    sigmoid = nn.Sigmoid()
    output=sigmoid(input)
    output = (output >= 0.5).float()  # 0.5以上を1に、それ未満を0に変換
    output = output.view(-1)  # テンソルをフラットにする
    labels = labels.view(-1)  # テンソルをフラットにする
    acc = torch.sum(output == labels).item()  # 正しい予測数を計算
    return acc/8
def acc2(input, labels):
    bs = input.size(0)
    # print(bs)
    sigmoid = nn.Sigmoid()
    output=sigmoid(input)
    output = list(output)
    preds = []
    for out in output:
        if out >= 0.5:
            preds.append(1)
        else:
            preds.append(0)
    preds = torch.Tensor(preds)
    device=labels.device
    preds=preds.to(device)

    acc = preds.eq(labels).sum().item()
    return acc / 8

def train(num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    model=Model.BERT_A()
    model.to(device)
    # model=Model.BERT_B()
    v_loss=[]
    v_acc=[]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.fc.parameters(), lr=1e-3)
    # bert_top_params = []
    # for name, param in model.named_parameters():
    #     if "bert.pooler" in name:
    #         bert_top_params.append(param)
    # optimizer.add_param_group({'params':bert_top_params,'lr':1e-3})
    # Train_dataset,test_dataset=torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    train_ratio = 0.8
    val_ratio = 0.1

    train_size = math.ceil(len(dataset) * train_ratio)
    val_size = math.ceil(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size]
    )
    with tqdm(range(num_epoch)) as epoch_bar:
        for epoch in epoch_bar:
            l_s=0
            a_s=0
            train_loss=AverageMeter()
            train_acc = AverageMeter()
            val_loss=AverageMeter()
            val_acc=AverageMeter()
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            # train_dataset,val_dataset=torch.utils.data.random_split(Train_dataset, [int(len(Train_dataset)*0.9), len(Train_dataset)-int(len(Train_dataset)*0.9)])
            data_loader = DataLoader(train_dataset,batch_size=8,shuffle=True, drop_last=True)
            model.train().to(device)
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False) as batch_bar:
                for i, (batch, label,mask) in batch_bar:
                    # batch = list(batch)#タプルをリストに
                    #print(batch)
                    batch=batch.to(device)
                    label=label.view(-1,1)
                    label=label.to(device)
                    mask=mask.to(device)
                    optimizer.zero_grad()#勾配の初期化
                    output = model(batch,attention_mask=mask)#順伝搬
                    loss = criterion(output, label)#損失の計算
                    loss.backward()#誤差逆伝搬
                    optimizer.step()#重みの更新
                    train_loss.update(loss,8)
                    #output = sigmoid(output.detach())
                    #print(output)
                    a=acc(output,label)
                    # print(a)
                    train_acc.update(a*8, 8)
                    batch_bar.set_postfix(OrderedDict(loss=train_loss.val, acc=train_acc.val))
            #評価
            data_loader = DataLoader(val_dataset,batch_size=8,shuffle=True, drop_last=True)
            model.eval().to(device)
            s=0
            a_s=0
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False) as batch_bar:
                for i, (batch, label,mask) in batch_bar:
                    label=label.view(-1,1)
                    label=label.to(device)
                    batch=batch.to(device)
                    mask=mask.to(device)
                    output = model(batch,attention_mask=mask)#順伝搬
                    loss = criterion(output, label)#損失の計算
                    s+=loss.item()
                    val_loss.update(loss,8)
                    a=acc(output,label)
                    # print(a)
                    a_s+=a
                    val_acc.update(a, 8)
                    batch_bar.set_postfix(OrderedDict(loss=val_loss.val, acc=val_acc.val))
            v_loss.append(s/len(data_loader))
            v_acc.append(a_s/len(data_loader))
            torch.save(model.to('cpu').state_dict(), 'Weight/'+str(epoch+1)+'kuzuha_kirinukich_Wrime無し統合_batch8_val改善_pooling.pth')
        print(v_loss)
        print(v_acc)
        Min=v_loss.index(min(v_loss))+1

        g=[]
        for i in range(num_epoch):
            g.append(i+1)
        x = np.array(g)
        y = np.array(v_loss)
        y1=np.array(v_acc)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss_sum")
        plt.plot(x, y, color = "red", marker = "o", label = "val_loss")
        plt.legend()
        plt.savefig("loss_acc_png\\Wrime無し統合_batch8_val改善_pooling_valtrain_loss.png")
        plt.show()

        plt.title("acc")
        plt.xlabel("epoch")
        plt.ylabel("acc_sum")
        plt.plot(x, y1, color = "blue", marker = "o", label = "val_acc")
        plt.legend()
        plt.savefig("loss_acc_png\\Wrime無し統合_batch8_val改善_pooling_valtrain_acc.png")
        plt.show()
    return test_dataset,Min


# def test(test_dataset,Min):
#     accuracy=0
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # dataset = Data.MyDataset()
#     model = Model.BERT_A()
#     model.load_state_dict(torch.load('Weight/'+str(Min)+'kuzuha_kirinukich_end_padding.pth'))
#     model.eval()
#     model.to(device)
#     # Train_dataset,test_dataset=torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
#     with tqdm(range(1)) as epoch_bar:
#         for epoch in epoch_bar:
#             epoch_bar.set_description("[Epoch %d]" % (epoch))
#             data_loader = DataLoader(test_dataset,batch_size=1,shuffle=True, drop_last=True)
#             with tqdm(enumerate(data_loader),
#                       total=len(data_loader),
#                       leave=False) as batch_bar:
#                 for i, (batch, label,mask) in batch_bar:
#                     # batch = list(batch)#タプルをリストに
#                     #print(batch)
#                     label=label.view(-1,1)
#                     label=label.to(device)
#                     batch=batch.to(device)
#                     mask=mask.to(device)
#                     output = model(batch,attention_mask=mask)#順伝搬
#                     a=acc2(output,label)
#                     accuracy+=a
#     return accuracy/len(test_dataset)


def test(test_dataset, Min):
    sigmoid=nn.Sigmoid()
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model.BERT_A()
    # model.load_state_dict(torch.load('Weight/'+str(Min)+'kuzuha_kirinukich_end_padding_truncation.pth'))
    model.load_state_dict(torch.load('Weight/'+str(Min)+'kuzuha_kirinukich_Wrime無し統合_batch8_val改善_pooling.pth'))
    model.eval()
    model.to(device)

    with tqdm(range(1)) as epoch_bar:
        for epoch in epoch_bar:
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
            
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False) as batch_bar:
                for i, (batch, label, mask) in batch_bar:
                    label = label.view(-1, 1)
                    label = label.to(device)
                    batch = batch.to(device)
                    mask = mask.to(device)

                    output = model(batch, attention_mask=mask)
                    output=sigmoid(output)
                    if output.item()>=0.5:
                        output=1
                    else:
                        output=0

                    output=torch.tensor(output)
                    tp = torch.sum((output == 1) & (label == 1)).item()

                    # FP（False Positive）: 予測が1で、実際のラベルは0
                    fp= torch.sum((output == 1) & (label == 0)).item()

                    # TN（True Negative）: 予測が0で、実際のラベルも0
                    tn= torch.sum((output == 0) & (label == 0)).item()

                    # FN（False Negative）: 予測が0で、実際のラベルは1
                    fn= torch.sum((output == 0) & (label == 1)).item()


                    true_positive += tp
                    false_positive += fp
                    true_negative += tn
                    false_negative += fn

    return true_positive, false_positive, true_negative, false_negative
test_dataset,min=train(50)
print(min)
tp, fp, tn, fn = test(test_dataset, min)

precision = tp / (tp + fp)
specificity = tn / (tn + fp)
recall = tp / (tp + fn)

print(f'True Positive: {tp}')
print(f'False Positive: {fp}')
print(f'True Negative: {tn}')
print(f'False Negative: {fn}')
print(f'Precision: {precision}')
print(f'Specificity: {specificity}')
print(f'Recall: {recall}')
