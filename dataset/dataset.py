import pandas as pd
import torch
from torch.utils.data import DataLoader


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("ひなーの2.csv",usecols=['コメント'])):
        self.comment = df["コメント"]

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        comment = str(comment)  
        if "w" in comment or "ｗ" in comment or "草" in comment:
            label = torch.ones([1])
        else:
            label = torch.zeros([1])
        return comment, label
