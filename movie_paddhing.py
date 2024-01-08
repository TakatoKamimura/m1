# 学習済みのモデルを使って、出力を新たなexcelファイルに保存するためのプログラム
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import model_padding as Model
from dataset import movie_dataset_paddhing as Data
import os
import numpy as np
import pandas as pd


def pred(num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    # train_dataset,test_dataset=torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    # data_loader = DataLoader(train_dataset,batch_size=1,shuffle=True, drop_last=True)
    model=Model.BERT_A()
    model.to(device)
    model.load_state_dict(torch.load('Weight\\37kuzuha_kirinukich_Wrim無し統合_batch8_val改善_使うやつ.pth'))
    # model=Model.BERT_B()
    # v_loss=[]
    # v_acc=[]
    sigmoid = nn.Sigmoid()
    pred=[]
    df=pd.read_csv('C:\\Users\\admin\\Desktop\\m1\\textchat_from_youtube\\lYJE1CBf_2o(kuzuha_vcc).csv',encoding='utf-8')
    with tqdm(range(num_epoch)) as epoch_bar:
        for epoch in epoch_bar:
            epoch_bar.set_description("[Epoch %d]" % (epoch))
            data_loader = DataLoader(dataset,batch_size=10,shuffle=False, drop_last=True)
            model.eval()
            with tqdm(enumerate(data_loader),
                      total=len(data_loader),
                      leave=False,ncols=50) as batch_bar:
                for i, (batch,mask) in batch_bar:
                    # batch = list(batch)#タプルをリストに
                    #print(batch)
                    batch=batch.to(device)
                    mask=mask.to(device)
                    output = model(batch,mask)#順伝搬
                    output = sigmoid(output)
                    output = list(output)
                    for out in output:
                        if out >= 0.5:
                            pred.append(1)
                            # print(1)
                        else:
                            pred.append(0)
                            # print(0)
    pred=pd.DataFrame(pred)
    df['予測値']=pred
    print(len(pred))
    print(pred)
    print(df)
    df.to_csv('textchat_from_youtube\\lYJE1CBf_2o_37kuzuha_kirinukich_Wrim無し統合_batch8_val改善_使うやつ',encoding='utf-8',index=False)

                

pred(1)
