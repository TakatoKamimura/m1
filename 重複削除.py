import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv("wrime無し_統合_0から1を削除.csv")

# 重複した行を削除する
df_no_duplicate = df.drop_duplicates(subset='コメント')

# 新しいCSVファイルに保存する
df_no_duplicate.to_csv("wrime無し_統合_前処理完了.csv", index=False,encoding='utf-8-sig')
