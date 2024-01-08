from transformers import BertTokenizer

# モデルの読み込み
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertTokenizer.from_pretrained(model_name)

# パディングを含むテキストのエンコード
text = ""
# encoded_text = tokenizer.encode(text, padding=True, truncation=True)
# print(encoded_text)
encoded_text = tokenizer.encode_plus(text, max_length=10, padding='max_length', return_attention_mask=True, truncation=True)
encoded_text=encoded_text['input_ids']
# トークンの表示
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(tokens)

