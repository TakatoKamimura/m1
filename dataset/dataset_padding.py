import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer
import re

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("textchat_from_youtube\\wrime無し_統合_前処理完了.csv", usecols=['コメント','ラベル'])):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.comment = []
        self.input_ids = []
        self.mask = []
        for comment in df["コメント"]:
            comment = str(comment)
            comment = re.sub(r"(\W)\1+", r"\1", comment)
            comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)
            bert_tokens = self.tokenizer.encode_plus(comment, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
            self.input_ids.append(bert_tokens['input_ids'])
            mask = torch.tensor(bert_tokens['attention_mask'])
            self.mask.append(mask)
            bert_tokens = torch.tensor(bert_tokens['input_ids'])
            self.comment.append(bert_tokens)
        # self.max_length = max_length  # この行は不要
        self.labels = torch.tensor(df["ラベル"].astype('float32'))

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        label = self.labels[idx]
        mask = self.mask[idx]
        return comment, label, mask
