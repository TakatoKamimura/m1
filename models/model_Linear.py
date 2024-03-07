from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertModel
import torch
import torch.nn as nn


class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.bert=BertModel.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
        self.fc=nn.Linear(768,1)

        
    def forward(self, input,attention_mask) -> torch.Tensor:
        output = self.bert(input,attention_mask=attention_mask).last_hidden_state[:, 0, :]
        output = self.fc(output)#順伝搬の出力
        return output