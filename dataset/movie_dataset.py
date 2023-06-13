import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df=pd.read_csv("C:\\Users\\admin\\Desktop\\m1\\textchat_from_youtube\\lYJE1CBf_2o(kuzuha_vcc).csv",usecols=['コメント'])):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.comment = df["コメント"]

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, idx):
        comment = self.comment[idx]
        comment = str(comment)  
        # if "w" in comment or "ｗ" in comment or "草" or "くさ" in comment:
        #     label = torch.ones([1])
        # else:
        #     label = torch.zeros([1])
        bert_tokens = self.tokenizer.encode(comment)
        bert_tokens=torch.tensor(bert_tokens)

        return bert_tokens