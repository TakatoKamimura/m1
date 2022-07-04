# from transformers import BertJapaneseTokenizer,BertForSequenceClassification, AdamW, BertConfig,BertModel
from transformers import BertJapaneseTokenizer,BertModel
import torch
import torch.nn as nn


class BERT_A(nn.Module): 
    def __init__(self):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.bert=BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.fc=nn.Linear(768,1)
        
    def forward(self, input) -> torch.Tensor:
        batch_size=len(input)
        bert_tokens = self.tokenizer(input, return_tensors="pt", padding=True)
        bert_tokens.input_ids
        ids = bert_tokens["input_ids"]
        tokens_tensor = ids.reshape(batch_size, -1)
        output = self.bert(tokens_tensor)["pooler_output"]
        output = self.fc(output)
        return output
    
    