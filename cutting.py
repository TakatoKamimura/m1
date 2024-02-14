import pandas as pd
import matplotlib.pyplot as plt
from editor import editor as ed

# 区間を統合するためのメソッド
def merge_intervals(intervals):
    intervals.sort()  # 区間リストを昇順にソート
    merged_intervals = [intervals[0]]  # 最初の区間を追加

    for current_interval in intervals[1:]:
        last_interval = merged_intervals[-1]  # 最後の統合済み区間を取得

        if current_interval[0] <= last_interval[1]:
            # 現在の区間と最後の統合済み区間が重複している場合、統合する
            last_interval[1] = max(last_interval[1], current_interval[1])
        else:
            # 重複していない場合、現在の区間を追加
            merged_intervals.append(current_interval)

    return merged_intervals



# CSVファイルを読み込み
df = pd.read_csv('読み込みたいcsvファイルのパス（csvの中身には時間、予測値のカラムが必要です）')

# 時間をdatetime型に変換
df['時間'] = pd.to_datetime(df['時間'], unit='s')

# 時間を10秒ごとの区間に分割
df['時間'] = pd.cut(df['時間'], pd.date_range(start=df['時間'].iloc[0], end=df['時間'].iloc[-1] + pd.Timedelta(seconds=10), freq='10s'))

# 区間ごとに1と予測されたデータの数を集計
interval_counts = df[df['予測値'] == 1].groupby('時間').size()
total_counts = df.groupby('時間').size()

# 割合を計算し、新しい列に追加
interval_ratios = interval_counts*(interval_counts / total_counts)
interval_counts_sorted = interval_ratios.sort_values(ascending=False)

# 結果を出力
print(interval_counts_sorted)
section=[]
a=0  #ハイライト動画に含める区間数
for interval_info, count in interval_counts_sorted.items():
    if a>142:
        break
    start = interval_info.left.time()
    end = interval_info.right.time()
    start_seconds = start.hour * 3600 + start.minute * 60 + start.second-20
    end_seconds = end.hour * 3600 + end.minute * 60 + end.second
    section.append([start_seconds,end_seconds])
    section=merge_intervals(section)
    a+=1

section.sort()
total_length = sum(end - start for start, end in section)
print(total_length)
print(section)
list_as_string = ",".join([str(sublist) for sublist in section])

# テキストファイルに書き込み（切り抜かれた区間をtxtファイルに出力するための文です．必要であれば使ってください．）
# with open("lYJE1CBf_2o_28kuzuha_kirinukich_Wrime無し_batch8_val改善.txt", "w") as file:
#     file.write(list_as_string)

a=[]
for i,v in enumerate(section):
    editor=ed(v[0],v[1])
    a.append(ed.cut(v[0],v[1]))
b=ed.concatenate(a)
ed.save(b)

