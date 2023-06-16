import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("textchat_from_youtube\\lYJE1CBf_2o(kuzuha_vcc).csv",usecols=['コメント','時間'])):
        self.comment = df["コメント"]
        self.time=df["時間"]
        # self.comment=self.comment[23:].reset_index(drop=True)
        # self.comment=self.comment[:18489-23].reset_index(drop=True)
        # self.time=self.time[23:].reset_index(drop=True)
        # self.time=self.time[:18489-23].reset_index(drop=True)

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        comment = str(comment)  
        return comment
