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
        
    def forward(self, input) -> torch.Tensor:
        # bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
        # input.input_ids
        # ids = input["input_ids"]
        # tokens_tensor = input.reshape(1, -1)
        output = self.bert(input).last_hidden_state[:, 0, :]
        output = self.fc(output)#順伝搬の出力
        # sigmoid = nn.Sigmoid()
        # output=sigmoid(output)#シグモイドに通す
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
        # sigmoid = nn.Sigmoid()
        # output=sigmoid(output)#シグモイドに通す
        return output
    