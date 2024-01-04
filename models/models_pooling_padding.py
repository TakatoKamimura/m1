from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertJapaneseTokenizer,BertModel
import torch
import torch.nn as nn


class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        # self.bert=BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',num_labels = 2)
        self.fc=nn.Linear(768,1)

        
    # def forward(self, input,attention_mask) -> torch.Tensor:
    #     # bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
    #     # input.input_ids
    #     # ids = input["input_ids"]
    #     # tokens_tensor = input.reshape(1, -1)
    #     # output = self.bert(input).last_hidden_state[:, 0, :]
        
    #     hidden_states = self.bert(input,attention_mask=attention_mask).last_hidden_state
    #     pooled_output = torch.mean(hidden_states, dim=1)
    #     output = self.fc(pooled_output)#順伝搬の出力
    #     return output
    
    def forward(self, input, attention_mask) -> torch.Tensor:
        hidden_states = self.bert(input, attention_mask=attention_mask).last_hidden_state
        # Attention Maskを使ってパディングされた部分を無視して平均プーリング
        sum_hidden_states = (hidden_states * attention_mask.unsqueeze(-1).float()).sum(1)
        sum_attention_mask = attention_mask.sum(1)
        # 各サンプルごとに平均を取る
        avg_pooled_output = sum_hidden_states / sum_attention_mask.unsqueeze(-1)  # ゼロ除算を避けるためにmin=1を追加

        output = self.fc(avg_pooled_output)
        return output

class BERT_B(nn.Module): 
    def __init__(self):
        super().__init__()
        # self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        self.bert=BertModel.from_pretrained('cl-tohoku/bert-large-japanese')
        # self.bert=BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',num_labels = 2)
        self.fc=nn.Linear(1024,1)
        
    def forward(self, input) -> torch.Tensor:
        # bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
        # input.input_ids
        # ids = input["input_ids"]
        # tokens_tensor = input.reshape(1, -1)
        output = self.bert(input).last_hidden_state[:, 0, :]
        output = self.fc(output)#順伝搬の出力
        return output
    