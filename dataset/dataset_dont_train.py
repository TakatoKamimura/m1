import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("学習に使うテキストチャットのcsvファイルのパス",usecols=['コメント','時間'])):
        self.comment = df["コメント"]
        self.time=df["時間"]

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        comment = str(comment)  
        return comment
