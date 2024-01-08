import pandas as pd
import mojimoji  # mojimojiパッケージを利用

# CSVファイルを読み込む
df = pd.read_csv("")

# 各セルの値に対して半角文字を全角文字に変換する関数
def convert_to_full_width(text):
    return mojimoji.han_to_zen(text)

# データフレームの各セルに対して変換を適用
df = df.applymap(convert_to_full_width)

# 新しいCSVファイルに保存する
df.to_csv("converted_file.csv", index=False)
