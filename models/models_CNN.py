from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertJapaneseTokenizer,BertModel
import torch
import torch.nn as nn

class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.bert=BertModel.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
        self.cnn1 = nn.Conv1d(100, 150, kernel_size=2, padding=0)
        self.linear=nn.Linear(767,1)
    def forward(self, input_ids,mask,):
        relu=nn.ReLU()
        outputs = self.bert(input_ids, mask)
        last_hidden_state = outputs['last_hidden_state']
        cnn_embeddings = relu(self.cnn1(last_hidden_state))
        logits, _ = torch.max(cnn_embeddings, 1)
        logits=self.linear(logits)
        return logits
