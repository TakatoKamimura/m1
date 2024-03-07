import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer
import re

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("推定を行いたいテキストチャットのcsvファイルのパス", usecols=['コメント'])):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
        self.comment = []
        self.mask = []
        for comment in df["コメント"]:
            comment = str(comment)
            comment = re.sub(r"(\W)\1+", r"\1", comment)
            comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)
            bert_tokens = self.tokenizer.encode_plus(comment, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
            mask = torch.tensor(bert_tokens['attention_mask'])
            self.mask.append(mask)
            bert_tokens = torch.tensor(bert_tokens['input_ids'])
            self.comment.append(bert_tokens)

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        mask = self.mask[idx]
        return comment, mask
