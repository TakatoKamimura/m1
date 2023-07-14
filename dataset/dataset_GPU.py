import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("textchat_from_youtube\\JmOGWt-XjzI(葛葉切り抜き集用).csv",usecols=['コメント','ラベル'])):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # self.comment = df["コメント"]
        self.comment = []

        for comment in df["コメント"]:
            bert_tokens=self.tokenizer.encode(str(comment))
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
