def find_common_intervals(A, B):
    common_intervals = []
    
    for interval_A in A:
        for interval_B in B:
            common_start = max(interval_A[0], interval_B[0])
            common_end = min(interval_A[1], interval_B[1])

            if common_start <= common_end:
                common_interval = [common_start, common_end]
                common_intervals.append(common_interval)

    common_sum = 0
    for interval in common_intervals:
        common_sum += interval[1] - interval[0]

    return common_intervals, common_sum

# AとBが複数の区間を持つ場合を定義
A = [[10, 20], [30, 40]]
B = [[15, 25], [5, 15],[35,90]]

# 共通部分とその合計を求める
common_intervals, common_sum = find_common_intervals(A, B)

if common_intervals:
    print("共通部分の区間:", common_intervals)
    print("共通部分の合計:", common_sum)
else:
    print("共通部分はありません")
