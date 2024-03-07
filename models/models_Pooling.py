from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertJapaneseTokenizer,BertModel
import torch
import torch.nn as nn


class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.bert=BertModel.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
        self.fc=nn.Linear(768,1)

    def forward(self, input, attention_mask) -> torch.Tensor:
        hidden_states = self.bert(input, attention_mask=attention_mask).last_hidden_state
        # Attention Maskを使ってパディングされた部分を無視して平均プーリング
        sum_hidden_states = (hidden_states * attention_mask.unsqueeze(-1).float()).sum(1)
        sum_attention_mask = attention_mask.sum(1)
        # 各サンプルごとに平均を取る
        avg_pooled_output = sum_hidden_states / sum_attention_mask.unsqueeze(-1)  # ゼロ除算を避けるためにmin=1を追加

        output = self.fc(avg_pooled_output)
        return output