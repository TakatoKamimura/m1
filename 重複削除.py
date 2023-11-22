import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv("C:\\Users\\admin\\Desktop\\m1\\新しいファイル.csv")

# 重複した行を削除する
df_no_duplicate = df.drop_duplicates(subset='コメント')

# 新しいCSVファイルに保存する
df_no_duplicate.to_csv("new_file.csv", index=False,encoding='utf-8-sig')
