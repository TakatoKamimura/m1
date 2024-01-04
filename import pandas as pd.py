import pandas as pd
from transformers import BertJapaneseTokenizer

# CSVファイルのパス
csv_path = 'textchat_from_youtube\\end.csv'

# CSVファイルの読み込み
df = pd.read_csv(csv_path)

# BERTトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# ラベルが1の場合のコメントのみ選択
label_1_comments = df[df['ラベル'] == 1]['コメント']

# 最大トークン長を初期化
max_token_length = 0
Sum=0
count=0
# ラベルが1のコメントをトークン化し、最大トークン長を更新
for comment in label_1_comments:
    count+=1
    tokens = tokenizer.encode(comment, return_tensors='pt')
    token_length = tokens.size(1)
    Sum+=token_length
    if token_length > max_token_length:
        max_token_length = token_length

# 結果を表示
print(f'最大トークン長: {max_token_length}')
print(f'平均:{Sum/count}')
# import pandas as pd
# from transformers import BertJapaneseTokenizer
# import numpy as np

# # CSVファイルのパス
# csv_path = 'textchat_from_youtube\\end.csv'

# # CSVファイルの読み込み
# df = pd.read_csv(csv_path)

# # BERTトークナイザーのロード
# tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# # ラベルが1の場合のコメントのみ選択
# label_1_comments = df[df['ラベル'] == 0]['コメント']

# # トークンの長さを計算
# token_lengths = [len(tokenizer.encode(comment, return_tensors='pt')[0]) for comment in label_1_comments]
# sorted_token_lengths = sorted(token_lengths)
# # 75パーセンタイルを計算
# percentile_75 = np.percentile(token_lengths, 75)

# # 結果を表示
# print(f'75パーセンタイル: {percentile_75:.2f}')
