import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv("入力csvファイルのパス")

# 0のラベルの数を取得
label_0_count = len(df[df['ラベル'] == 0])

# 1のラベルの数を取得
label_1_count = len(df[df['ラベル'] == 1])

# ラベルが1の行をランダムに選択するマスクを作成 (ラベル1の方が多い場合)
mask_label_1 = (df['ラベル'] == 1) & (np.random.rand(len(df)) < (label_0_count / label_1_count))

# マスクに基づいてデータを選択
df_downsampled_label_1 = df[mask_label_1]

# 0のラベルの行を追加
df_downsampled_label_0 = df[df['ラベル'] == 0]

# 新しいデータフレームを作成
df_downsampled = pd.concat([df_downsampled_label_0, df_downsampled_label_1])

# 新しいCSVファイルに保存する
df_downsampled.to_csv("出力csvファイルのパス", index=False)
