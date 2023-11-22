# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from editor import editor as ed
# # # # データの読み込み
# # # df = pd.read_csv('new_lYJE1CBf_2o(kuzuha_vcc_葛葉ch).csv')

# # # # timestamp列をdatetime型に変換してインデックスに設定
# # # df['時間'] = pd.to_datetime(df['時間'], unit='s')
# # # df.set_index('時間', inplace=True)

# # # counts = df['予測値'].resample('10s').sum()

# # # center_idx = counts.argmax()
# # # start_idx = max(0, center_idx - 10)
# # # end_idx = min(len(counts), center_idx + 10)
# # # counts = counts.iloc[start_idx:end_idx]

# # # # プロット
# # # fig, ax = plt.subplots(figsize=(12, 6))
# # # ax.plot(counts.index, counts.values)
# # # ax.set_xlabel('time')
# # # ax.set_ylabel('count')
# # # plt.show()

# # def merge_intervals(intervals):
# #     intervals.sort()  # 区間リストを昇順にソート
# #     merged_intervals = [intervals[0]]  # 最初の区間を追加

# #     for current_interval in intervals[1:]:
# #         last_interval = merged_intervals[-1]  # 最後の統合済み区間を取得

# #         if current_interval[0] <= last_interval[1]:
# #             # 現在の区間と最後の統合済み区間が重複している場合、統合する
# #             last_interval[1] = max(last_interval[1], current_interval[1])
# #         else:
# #             # 重複していない場合、現在の区間を追加
# #             merged_intervals.append(current_interval)

# #     return merged_intervals



# # # CSVファイルを読み込み
# # df = pd.read_csv('textchat_from_youtube\\new_lYJE1CBf_2o(kuzuha_vcc_葛葉切り抜きch).csv')

# # # 時間をdatetime型に変換
# # df['時間'] = pd.to_datetime(df['時間'], unit='s')

# # # 時間を10秒ごとの区間に分割
# # df['時間'] = pd.cut(df['時間'], pd.date_range(start=df['時間'].iloc[0], end=df['時間'].iloc[-1] + pd.Timedelta(seconds=10), freq='10s'))

# # # 区間ごとに1と予測されたデータの数を集計
# # interval_counts = df[df['予測値'] == 1].groupby('時間').size()
# # interval_counts_sorted = interval_counts.sort_values(ascending=False)
# # # 結果を出力
# # print(interval_counts_sorted)
# # section=[]
# # a=0
# # for interval_info, count in interval_counts_sorted.items():
# #     if a>142:
# #         break
# #     start = interval_info.left.time()
# #     end = interval_info.right.time()
# #     start_seconds = start.hour * 3600 + start.minute * 60 + start.second-20
# #     end_seconds = end.hour * 3600 + end.minute * 60 + end.second
# #     section.append([start_seconds,end_seconds])
# #     section=merge_intervals(section)
# #     a+=1

# # # print(section)
# # section.sort()
# # # print(section)
# # total_length = sum(end - start for start, end in section)

# # print(section)
# # list_as_string = ",".join([str(sublist) for sublist in section])

# # # テキストファイルに書き込み
# # with open("output.txt", "w") as file:
# #     file.write(list_as_string)
# # print(total_length)
# # exit()
# # a=[]
# # for i,v in enumerate(section):
# #     editor=ed(v[0],v[1])
# #     a.append(ed.cut(v[0],v[1]))
# # # editor=ed(60*165,60*165+30)
# # # a.append(ed.cut(60*165,60*165+30))
# # b=ed.concatenate(a)
# # ed.save(b)


# import pandas as pd
# import matplotlib.pyplot as plt
# from editor import editor as ed
# # # データの読み込み
# # df = pd.read_csv('new_lYJE1CBf_2o(kuzuha_vcc_葛葉ch).csv')

# # # timestamp列をdatetime型に変換してインデックスに設定
# # df['時間'] = pd.to_datetime(df['時間'], unit='s')
# # df.set_index('時間', inplace=True)

# # counts = df['予測値'].resample('10s').sum()

# # center_idx = counts.argmax()
# # start_idx = max(0, center_idx - 10)
# # end_idx = min(len(counts), center_idx + 10)
# # counts = counts.iloc[start_idx:end_idx]

# # # プロット
# # fig, ax = plt.subplots(figsize=(12, 6))
# # ax.plot(counts.index, counts.values)
# # ax.set_xlabel('time')
# # ax.set_ylabel('count')
# # plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from editor import editor as ed
def merge_intervals(intervals,time):
    intervals.sort()  # 区間リストを昇順にソート
    merged_intervals = [intervals[0]]  # 最初の区間を追加

    for current_interval in intervals[1:]:
        last_interval = merged_intervals[-1]  # 最後の統合済み区間を取得

        if current_interval[0] <= last_interval[1]:
            # 現在の区間と最後の統合済み区間が重複している場合、統合する
            time-=30
            time-=last_interval[1]-last_interval[0]
            last_interval[1] = max(last_interval[1], current_interval[1])
            time+=last_interval[1]-last_interval[0]
        else:
            # 重複していない場合、現在の区間を追加
            merged_intervals.append(current_interval)

    return merged_intervals,time



# CSVファイルを読み込み
df = pd.read_csv('textchat_from_youtube\\new_lYJE1CBf_2o(kuzuha_vcc_葛葉切り抜きch)_usingwrime.csv')

# 時間をdatetime型に変換
df['時間'] = pd.to_datetime(df['時間'], unit='s')

# 時間を10秒ごとの区間に分割
df['時間'] = pd.cut(df['時間'], pd.date_range(start=df['時間'].iloc[0], end=df['時間'].iloc[-1] + pd.Timedelta(seconds=10), freq='10s'))

# 区間ごとに1と予測されたデータの数を集計
interval_counts = df[df['予測値'] == 1].groupby('時間').size()
interval_counts_sorted = interval_counts.sort_values(ascending=False)
# 結果を出力
print(interval_counts_sorted)
section=[]
a=0

#全体から均等に区間をとれるように区間の合計値用の変数を用意
time_1=0
time_2=0
time_3=0
time_4=0
time_5=0
time_6=0
time_7=0
time_8=0
time_9=0
time_10=0


for interval_info, count in interval_counts_sorted.items():
    # if a>10:
    #     break
    if time_1>300 and time_2>300 and time_3>300 and time_4>300 and time_5>300 and time_6>300 and time_7>300 and time_8>300 and time_9>300 and time_10>300:
        break
    start = interval_info.left.time()
    end = interval_info.right.time()
    start_seconds = start.hour * 3600 + start.minute * 60 + start.second-20
    end_seconds = end.hour * 3600 + end.minute * 60 + end.second

    if end_seconds<=2200:
        if time_1>300:
            continue
        time_1+=30
        section.append([start_seconds,end_seconds])
        section,time_1=merge_intervals(section,time_1)
    elif end_seconds<=4400:
        if time_2>300:
            continue
        time_2+=30
        section.append([start_seconds,end_seconds])
        section,time_2=merge_intervals(section,time_2)
    elif end_seconds<=6600:
        if time_3>300:
            continue
        time_3+=30
        section.append([start_seconds,end_seconds])
        section,time_3=merge_intervals(section,time_3)
    elif end_seconds<=8800:
        if time_4>300:
            continue
        time_4+=30
        section.append([start_seconds,end_seconds])
        section,time_4=merge_intervals(section,time_4)
    elif end_seconds<=11000:
        if time_5>300:
            continue
        time_5+=30
        section.append([start_seconds,end_seconds])
        section,time_5=merge_intervals(section,time_5)
    elif end_seconds<=13200:
        if time_6>300:
            continue
        time_6+=30
        section.append([start_seconds,end_seconds])
        section,time_6=merge_intervals(section,time_6)
    elif end_seconds<=15400:
        if time_7>300:
            continue
        time_7+=30
        section.append([start_seconds,end_seconds])
        section,time_7=merge_intervals(section,time_7)
    elif end_seconds<=17600:
        if time_8>300:
            continue
        time_8+=30
        section.append([start_seconds,end_seconds])
        section,time_8=merge_intervals(section,time_8)
    elif end_seconds<=19800:
        if time_9>300:
            continue
        time_9+=30
        section.append([start_seconds,end_seconds])
        section,time_9=merge_intervals(section,time_9)
    elif end_seconds<=22000:
        if time_10>300:
            continue
        time_10+=30
        section.append([start_seconds,end_seconds])
        section,time_10=merge_intervals(section,time_10)
    print("time_1",time_1, "time_2", time_2,"time_3",time_3,"time_4",time_4,"time_5",time_5,"time_6", time_6,"time7",time_7,"time_8",time_8,"time_9",time_9, "time_10", time_10)


    a+=1

# print(section)
section.sort()
# print(section)
total_length = sum(end - start for start, end in section)

print(section)
list_as_string = ",".join([str(sublist) for sublist in section])

# テキストファイルに書き込み
with open("output.txt", "w") as file:
    file.write(list_as_string)
print(total_length)
a=[]
for i,v in enumerate(section):
    editor=ed(v[0],v[1])
    a.append(ed.cut(v[0],v[1]))

b=ed.concatenate(a)
ed.save(b)


# # データの読み込み
# df = pd.read_csv('new_lYJE1CBf_2o(kuzuha_vcc_葛葉ch).csv')

# # timestamp列をdatetime型に変換してインデックスに設定
# df['時間'] = pd.to_datetime(df['時間'], unit='s')
# df.set_index('時間', inplace=True)

# counts = df['予測値'].resample('10s').sum()

# center_idx = counts.argmax()
# start_idx = max(0, center_idx - 10)
# end_idx = min(len(counts), center_idx + 10)
# counts = counts.iloc[start_idx:end_idx]

# # プロット
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(counts.index, counts.values)
# ax.set_xlabel('time')
# ax.set_ylabel('count')
# plt.show()

# def merge_intervals(intervals):
#     intervals.sort()  # 区間リストを昇順にソート
#     merged_intervals = [intervals[0]]  # 最初の区間を追加

#     for current_interval in intervals[1:]:
#         last_interval = merged_intervals[-1]  # 最後の統合済み区間を取得

#         if current_interval[0] <= last_interval[1]:
#             # 現在の区間と最後の統合済み区間が重複している場合、統合する
#             last_interval[1] = max(last_interval[1], current_interval[1])
#         else:
#             # 重複していない場合、現在の区間を追加
#             merged_intervals.append(current_interval)

#     return merged_intervals



# # CSVファイルを読み込み
# df = pd.read_csv('textchat_from_youtube\\new_lYJE1CBf_2o(kuzuha_vcc_葛葉切り抜きch)_usingwrime.csv')

# # 時間をdatetime型に変換
# df['時間'] = pd.to_datetime(df['時間'], unit='s')

# # 時間を10秒ごとの区間に分割
# df['時間'] = pd.cut(df['時間'], pd.date_range(start=df['時間'].iloc[0], end=df['時間'].iloc[-1] + pd.Timedelta(seconds=10), freq='10s'))

# # 区間ごとに1と予測されたデータの数を集計
# interval_counts = df[df['予測値'] == 1].groupby('時間').size()
# interval_counts_sorted = interval_counts.sort_values(ascending=False)
# # 結果を出力
# print(interval_counts_sorted)
# section=[]
# a=0
# for interval_info, count in interval_counts_sorted.items():
#     if a>142:
#         break
#     start = interval_info.left.time()
#     end = interval_info.right.time()
#     start_seconds = start.hour * 3600 + start.minute * 60 + start.second-20
#     end_seconds = end.hour * 3600 + end.minute * 60 + end.second
#     section.append([start_seconds,end_seconds])
#     section=merge_intervals(section)
#     a+=1

# # print(section)
# section.sort()
# # print(section)
# total_length = sum(end - start for start, end in section)

# print(section)
# list_as_string = ",".join([str(sublist) for sublist in section])

# # テキストファイルに書き込み
# with open("output.txt", "w") as file:
#     file.write(list_as_string)
# print(total_length)
# a=[]
# for i,v in enumerate(section):
#     editor=ed(v[0],v[1])
#     a.append(ed.cut(v[0],v[1]))
# # editor=ed(60*165,60*165+30)
# # a.append(ed.cut(60*165,60*165+30))
# b=ed.concatenate(a)
# ed.save(b)

