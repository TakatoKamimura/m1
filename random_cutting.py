import pandas as pd
import matplotlib.pyplot as plt
from editor import editor as ed
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

import random

# 選択した区間の合計時間が8分になるまでランダムに区間を選択する
total_length = 0
section = []
d={}
while total_length < 8 * 60:
    # 0~239までの乱数を発生させる
    random_index = random.randint(2, 239)
    if random_index in d:
        continue
    else:
        d[random_index]=1
    # それ以外の場合、ランダムに選択された区間の前後20秒を合わせた30秒の区間を選択する
    
    start_seconds = random_index*10-20
    end_seconds = random_index*10+10
    section.append([start_seconds, end_seconds])
    section=merge_intervals(section)
    t=sum(end - start for start, end in section)
    if t>=480:
        break

print(section)
print(sum(end - start for start, end in section))
list_as_string = ",".join([str(sublist) for sublist in section])

# テキストファイルに書き込み
with open("出力ファイルのパス", "w") as file:
    file.write(list_as_string)
print(total_length)
a=[]
for i,v in enumerate(section):
    editor=ed(v[0],v[1])
    a.append(ed.cut(v[0],v[1]))
b=ed.concatenate(a)
ed.save(b)

