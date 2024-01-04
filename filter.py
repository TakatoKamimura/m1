import pandas as pd

# 1つ目のCSVファイルのパス
csv1_path = 'textchat_from_youtube\\wrime無し.csv'

# 2つ目のCSVファイルのパス
csv2_path = 'JmOGWt-XjzI(葛葉切り抜き集用)_前処理完了_1のみ.csv'

# 1つ目のCSVファイルを読み込み
df1 = pd.read_csv(csv1_path)

# 2つ目のCSVファイルを読み込み
df2 = pd.read_csv(csv2_path)

# 2つのデータフレームを結合
merged_df = pd.concat([df1, df2], ignore_index=True)

# 結合したデータフレームを新しいCSVファイルに保存
merged_csv_path = '結合したファイル.csv'
merged_df.to_csv(merged_csv_path, index=False)

