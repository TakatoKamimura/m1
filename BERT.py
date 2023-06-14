from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#トークン化とBERTモデルの定義
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
head = nn.Linear(768,1)
# model_bert.to(device)

def calc_embedding(text):
  batch_size = len(text)
  bert_tokens = tokenizer(text, return_tensors="pt", padding=True)
  bert_tokens.input_ids
  ids = bert_tokens["input_ids"]
  tokens_tensor = torch.tensor(ids).reshape(batch_size, -1)

  with torch.no_grad():
    output = model_bert(tokens_tensor)["pooler_output"]
    # output = head(output)
  return output.numpy()

text = []
text.append('私は父です')
output = calc_embedding(text)
print(output)
# print(torch.ones([1]))

