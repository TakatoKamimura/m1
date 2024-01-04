import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer
import re

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("textchat_from_youtube\\Wrime無し.csv",usecols=['コメント','ラベル'])):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # self.comment = df["コメント"]
        self.comment = []
        self.mask=[]
        for comment in df["コメント"]:
            comment=str(comment)
            # 特殊文字（\W）の連続を削除
            comment = re.sub(r"(\W)\1+", r"\1", comment)
            # 文字の連続を最大2文字までに制限
            comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)
            bert_tokens=self.tokenizer.encode(comment)
            # bert_tokens=self.tokenizer.encode_plus(comment, padding=True, return_attention_mask=True, truncation=True)
            # self.mask.append(bert_tokens['attention_mask'])
            bert_tokens=torch.tensor(bert_tokens)
            self.comment.append(bert_tokens)
            
        # self.label=df["ラベル"]
        self.labels = torch.tensor(df["ラベル"].astype('float32'))


    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        label=self.labels[idx] 
        return comment, label
