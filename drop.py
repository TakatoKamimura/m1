import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv("textchat_from_youtube\\重複と連続の削除.csv")

# 0のラベルの数を取得
label_0_count = len(df[df['ラベル'] == 0])

# 1のラベルの数を取得
label_1_count = len(df[df['ラベル'] == 1])

# ラベルが0の行をランダムに選択するマスクを作成
mask_label_0 = (df['ラベル'] == 0) & (np.random.rand(len(df)) < (label_1_count / label_0_count))

# マスクに基づいてデータを選択
df_downsampled_label_0 = df[mask_label_0]

# 1のラベルの行を追加
df_downsampled_label_1 = df[df['ラベル'] == 1]

# 新しいデータフレームを作成
df_downsampled = pd.concat([df_downsampled_label_0, df_downsampled_label_1])

# 新しいCSVファイルに保存する
df_downsampled.to_csv("new_file_downsampled.csv", index=False)
