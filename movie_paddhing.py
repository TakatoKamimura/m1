# 学習済みのモデルを使って、出力を新たなexcelファイルに保存するためのプログラム
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import model_Linear as Model
from dataset import movie_dataset_paddhing as Data
import os
import numpy as np
import pandas as pd


def pred():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Data.MyDataset()
    model=Model.BERT_A()
    model.to(device)
    model.load_state_dict(torch.load('モデルのパラメータファイルのパス'))
    sigmoid = nn.Sigmoid()
    pred=[]
    df=pd.read_csv('取得したテキストチャットのcsvファイルのパス',encoding='utf-8')
    with tqdm(range(1)) as epoch_bar:
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
    df.to_csv('推定結果のcsvの保存先のパス',encoding='utf-8',index=False)

                

pred()
