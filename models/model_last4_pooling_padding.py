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
        
        
    # def forward(self, input) -> torch.Tensor:
    #     # bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
    #     # input.input_ids
    #     # ids = input["input_ids"]
    #     # tokens_tensor = input.reshape(1, -1)
    #     output = self.bert(input).last_hidden_state[:, 0, :]
    #     output = self.fc(output)#順伝搬の出力
    #     return output
    
    def forward(self, input,mask) -> torch.Tensor:
        # BERTの出力を取得
        print(input.size())
        outputs = self.bert(input,mask)
        hidden_states=outputs.last_hidden_state
        cls_vectors = hidden_states[:, -4:, :][:, :, 0, :]
        
        print(hidden_states.size())
        exit()
        # 最終4層のCLSトークンを取り出す
        cls_tokens = hidden_states[:, -4, 0, :]
        print(cls_tokens)
        print(cls_tokens.size())
        # 平均プーリングを行う
        pooled_output = torch.mean(cls_tokens, dim=1)

        hidden_states = outputs.last_hidden_state[-4:][:,0,:]
        print(hidden_states.size())
        print(hidden_states.size())
        exit()
        # 平均プーリングを行う
        pooled_output = torch.mean(hidden_states, dim=0)
        # 全結合層に入力して順伝搬の出力を得る
        output = self.fc(pooled_output)
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
    