import pandas as pd
import matplotlib.pyplot as plt
from editor import editor as ed
# データの読み込み
df = pd.read_csv('new_hinano4.csv')

# timestamp列をdatetime型に変換してインデックスに設定
df['時間'] = pd.to_datetime(df['時間'], unit='s')
df.set_index('時間', inplace=True)

counts = df['予測値'].resample('10s').sum()

center_idx = counts.argmax()
start_idx = max(0, center_idx - 10)
end_idx = min(len(counts), center_idx + 10)
counts = counts.iloc[start_idx:end_idx]

# プロット
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(counts.index, counts.values)
ax.set_xlabel('time')
ax.set_ylabel('count')
plt.show()



# CSVファイルを読み込み
df = pd.read_csv('new_hinano4.csv')

# 時間をdatetime型に変換
df['時間'] = pd.to_datetime(df['時間'], unit='s')

# 時間を5秒ごとの区間に分割
df['時間'] = pd.cut(df['時間'], pd.date_range(start=df['時間'].iloc[0], end=df['時間'].iloc[-1] + pd.Timedelta(seconds=10), freq='10s'))

# 区間ごとに1と予測されたデータの数を集計
interval_counts = df[df['予測値'] == 1].groupby('時間').size()
interval_counts_sorted = interval_counts.sort_values(ascending=False)
# 結果を出力
print(interval_counts_sorted)

editor=ed(5570,5630)
a=[]
a.append(ed.cut(5570,5630))
b=ed.concatenate(a)
ed.save(b)

