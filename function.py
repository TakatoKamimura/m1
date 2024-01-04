import numpy as np
import matplotlib.pyplot as plt

# ReLU関数の定義
def relu(x):
    return np.maximum(0, x)

# x軸の値を生成（-1から1まで、0.25刻みに変更）
x_values = np.arange(-1, 1.25, 0.25)

# ReLU関数の計算
relu_values = relu(x_values)

# グラフの描画
plt.figure(figsize=(6, 4))

# ReLU関数のグラフ
plt.plot(x_values, relu_values, linestyle='-', label='ReLU', color='blue')
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.ylim(0, 1)
plt.xlim(-1, 1 )
plt.grid(True)
plt.yticks(np.arange(0, 1.1, 0.1))

# x軸の目盛りを手動で設定
plt.xticks(np.arange(-1, 1.1, 0.25))
# 画像として保存
plt.savefig('ReLU_plot.png')

# 表示
plt.show()
