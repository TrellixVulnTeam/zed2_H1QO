import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-100, 100, 201)
y = 0.1 * x ** 2 + 35 * x + 70 + np.random.randn(201) * 45

# 近似式
ref1 = np.polyfit(x, y, 1)
ref2 = np.polyfit(x, y, 2)

# 近似式の計算
y1 = np.poly1d(ref1)(x)
y2 = np.poly1d(ref2)(x)

# ベクトルの準備
one = np.ones_like(x)
X1 = np.stack([x, one], 1)
X2 = np.stack([x * x, x, one], 1)

# 行列演算での係数算出
invX1 = np.linalg.inv(np.dot(X1.T, X1))
X1tY = np.dot(X1.T, y)
ans1 = np.dot(invX1, X1tY)

invX2 = np.linalg.inv(np.dot(X2.T, X2))
X2tY = np.dot(X2.T, y)
ans2 = np.dot(invX2, X2tY)

# グラフに表示
plt.scatter(x, y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
