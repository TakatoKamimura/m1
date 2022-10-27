import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict

from models import models as Model
from dataset import dataset as Data

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

    def update(self, val, n=1):#valにlossをいれる
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
    output = list(output)
    print(output[0])
    preds = []
    for out in output:
        if out >= 0.5:
            preds.append(1)
        else:
            preds.append(0)
    print('ラベル：',end='')
    print(labels)
    print('予測：',end='')
    print(preds)
    preds = torch.Tensor(preds)
    acc = preds.eq(labels).sum().item()
    return acc / bs

def train(num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    train_dataset,test_dataset=torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    # print(len(train_dataset))
    data_loader = DataLoader(train_dataset,batch_size=1,shuffle=True, drop_last=True)
    model=Model.BERT_A()
    # model=Model.BERT_B()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.fc.parameters(), lr=1e-3)

    with tqdm(range(num_epoch)) as epoch_bar:
        for epoch in epoch_bar:
            train_loss=AverageMeter()
            train_acc = AverageMeter()
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            model.train()
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False) as batch_bar:
                for i, (batch, label) in batch_bar:
                    # batch = list(batch)#タプルをリストに
                    #print(batch)
                    label=label.view(-1,1)
                    optimizer.zero_grad()#勾配の初期化
                    output = model(batch)#順伝搬
                    loss = criterion(output, label)#損失の計算
                    loss.backward()#誤差逆伝搬
                    optimizer.step()#重みの更新
                    train_loss.update(loss,1)
                    #output = sigmoid(output.detach())
                    #print(output)
                    a=acc(output,label)
                    print(a)
                    train_acc.update(a, 1)
                    batch_bar.set_postfix(OrderedDict(loss=train_loss.val, acc=train_acc.val))
            # model.eval()
            # with torch.no_grad():
            #     with tqdm(enumerate(data_loader),
            #             total=len(data_loader),
            #             leave=False) as batch_bar:
            #         for i, (batch, label) in batch_bar:
            #             output=model(batch)
            #             loss=criterion(output,label)
            #             a=acc(output,label)
            #             print(a)
                

            print(f"train_loss:avg{train_loss.avg}")
            print(f"train_acc:avg{train_acc.avg}")

train(1)


