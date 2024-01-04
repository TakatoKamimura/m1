from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertJapaneseTokenizer,BertModel
import torch
import torch.nn as nn


# class BERT_A(nn.Module): 
#     def __init__(self):
#         super().__init__()
#         self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
#         # self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
#         # self.bert=BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',num_labels = 2)
#         self.conv1d = nn.Conv1d(in_channels=768, out_channels=200, kernel_size=3)
#         self.ReLU=nn.ReLU()
#         self.avg_pooling = nn.AvgPool1d(3)
#         self.fc=nn.Linear(200,1)

        
#     def forward(self, input,mask) -> torch.Tensor:
#         # bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
#         # input.input_ids
#         # ids = input["input_ids"]
#         # tokens_tensor = input.reshape(1, -1)
#         # output = self.bert(input).last_hidden_state[:, 0, :]
        
#         output = self.bert(input,mask).last_hidden_state
#         print(output.size())
#         output=output.permute(0,2,1)

#         output = self.conv1d(output)
#         print(output.size())
#         exit()
#         output= self.ReLU(output)
#         output = self.avg_pooling(output).squeeze(2)
#         output = self.fc(output)#順伝搬の出力
#         return output
class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=200, kernel_size=3)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(200, 1)

    def forward(self, input, mask) -> torch.Tensor:
        output = self.bert(input, mask).last_hidden_state
        output = self.conv1d(output.permute(0, 2, 1))
        output = self.ReLU(output)       
        output = torch.mean(output,dim=2)
        output = self.fc(output)
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