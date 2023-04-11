import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import models as Model
from dataset import movie_dataset as Data
import os
import numpy as np
import pandas as pd


def pred(num_epoch):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    # train_dataset,test_dataset=torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    # data_loader = DataLoader(train_dataset,batch_size=1,shuffle=True, drop_last=True)
    model=Model.BERT_A()
    model.load_state_dict(torch.load('Weight/90epoch.pth'))
    # model=Model.BERT_B()
    # v_loss=[]
    # v_acc=[]
    sigmoid = nn.Sigmoid()
    pred=[]
    df=pd.read_csv('C:\\Users\\admin\\Desktop\\m1\\ひなーの3.csv',encoding='utf-8')
    with tqdm(range(num_epoch)) as epoch_bar:
        for epoch in epoch_bar:
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            data_loader = DataLoader(dataset,batch_size=1,shuffle=False, drop_last=True)
            model.train()
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False) as batch_bar:
                for i, (batch) in batch_bar:
                    # batch = list(batch)#タプルをリストに
                    #print(batch)
                    output = model(batch)#順伝搬
                    output = sigmoid(output)
                    output = list(output)
                    for out in output:
                        if out >= 0.5:
                            pred.append(1)
                        else:
                            pred.append(0)
    pred=pd.DataFrame(pred)
    df['予測値']=pred
    print(len(pred))
    print(pred)
    print(df)
    df.to_csv('new_hinano3.csv',encoding='utf-8',index=False)

                

pred(1)
