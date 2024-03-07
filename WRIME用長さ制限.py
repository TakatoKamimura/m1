import pandas as pd
import re

# CSVファイルを読み込む
df = pd.read_csv("new_file_downsampled.csv")

# 処理と保存を行う関数
def process_comment(comment):
    # 正規表現を使用して感嘆詞までの部分を取得
    match = re.search(r'^.*?[！？。]', comment)

    if match:
        result = match.group(0)
        # 空白を削除
        result = result.strip()
        return result
    else:
        # 感嘆詞が見つからなかった場合、そのまま取得
        return comment.strip()

# 'コメント' 列に対して処理を適用
df['コメント'] = df['コメント'].apply(process_comment)

# 新しいCSVファイルに保存する
df.to_csv("end.csv", index=False)
