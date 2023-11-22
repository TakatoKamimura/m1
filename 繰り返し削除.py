import pandas as pd
import re

# 新しいDataFrameを作成
new_df = pd.DataFrame(columns=["コメント", "ラベル"])

# 元のDataFrameを読み込む
df = pd.read_csv("textchat_from_youtube\\usingWrime.csv", usecols=['コメント', 'ラベル'],encoding='utf-8')

# 元のDataFrameからコメント列とラベル列を取得
for index, row in df.iterrows():
    comment = str(row["コメント"])

    # 特殊文字（\W）の連続を削除
    comment = re.sub(r"(\W)\1+", r"\1", comment)

    # 文字の連続を最大2文字までに制限
    comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)

    # 新しいDataFrameに追加
    new_df = pd.concat([new_df, pd.DataFrame({"コメント": [comment], "ラベル": [row["ラベル"]]})], ignore_index=True)

# 新しいCSVファイルに保存
new_df.to_csv("新しいファイル.csv", index=False, encoding='utf-8-sig')
