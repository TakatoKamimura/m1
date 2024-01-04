import pandas as pd
import re

# 新しいDataFrameを作成
new_df = pd.DataFrame(columns=["コメント", "ラベル"])

# 元のDataFrameを読み込む
df = pd.read_csv("concat.csv", usecols=['コメント', 'ラベル'],encoding='utf-8')

# 元のDataFrameからコメント列とラベル列を取得
for index, row in df.iterrows():
    comment = str(row["コメント"])

    # 特殊文字（\W）の連続を削除
    comment = re.sub(r"(\W)\1+", r"\1", comment)

    # 文字の連続を最大2文字までに制限
    comment = re.sub(r"(\w)\1{2,}", r"\1\1", comment)

    # 新しいDataFrameに追加
    new_df = pd.concat([new_df, pd.DataFrame({"コメント": [comment], "ラベル": [row["ラベル"]]})], ignore_index=True)


df = new_df

# ラベルが1のコメントのセットを作成
label_1_set = set(df[df['ラベル'] == 1]['コメント'])

# ラベルが0でかつセットに含まれていないコメントを抽出
filtered_df_label_0 = df[(df['ラベル'] == 0) & (~df['コメント'].isin(label_1_set))]

# ラベルが1のコメントも含めた新しいCSVファイルにUTF-8で保存する（実際のファイルパスに適応してください）
output_file_path = 'wrime無し_統合_0から1を削除.csv'
pd.concat([df[df['ラベル'] == 1], filtered_df_label_0]).to_csv(output_file_path, index=False, encoding='utf-8')
