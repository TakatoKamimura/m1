import pandas as pd
import re

# 新しいDataFrameを作成
new_df = pd.DataFrame(columns=["時間", "コメント","ラベル"])

# 元のDataFrameを読み込む
df = pd.read_csv("JmOGWt-XjzI(葛葉切り抜き集用)_wを1に.csv", usecols=['コメント', '時間','ラベル'],encoding='utf-8')

# 元のDataFrameからコメント列とラベル列を取得
for index, row in df.iterrows():
    comment = str(row["コメント"])

    # 特殊文字（\W）の連続を削除
    comment = re.sub(r"(\W)\1+", r"\1", comment)

    # 文字の連続を最大2文字までに制限
    comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)

    # 新しいDataFrameに追加
    new_df = pd.concat([new_df, pd.DataFrame({"時間": [row["時間"]],"コメント": [comment],"ラベル":[row["ラベル"]]})], ignore_index=True)

# 新しいCSVファイルに保存
new_df.to_csv("JmOGWt-XjzI(葛葉切り抜き集用)_繰り返し削除.csv", index=False, encoding='utf-8-sig')
