from sklearn.datasets import make_regression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest

# 回帰ダミーデータを作成する
X, Y, coef = make_regression(
    random_state=12,
    n_samples=100,
    n_features=1,
    n_informative=1,
    noise=10.0,
    bias=-0.0,
    coef=True,
)

print("X =", X[:5])
print("Y =", Y[:5])
print("coef =", coef)

# 外れ値を追加する
X = np.concatenate([X, np.array([[2.2], [2.3], [2.4]])])
Y = np.append(Y, [2.2, 2.3, 2.4])

# X座標の値とY座標の値がばらばらなので、一つの変数にまとめる
X_train = np.concatenate([X, Y[:, np.newaxis]], 1)

# 生データすべてをプロットする
plt.figure(figsize=(20, 4))
plt.subplot(1, 4, 1)
plt.title("raw data")
plt.plot(X, Y, "bo")

# IsolationForestインスタンスを作成する
clf = IsolationForest(
    contamination='auto', behaviour='new', max_features=2, random_state=42
)

# 学習用データを学習させる
clf.fit(X_train)

# 検証用データを分類する
y_pred = clf.predict(X_train)

# IsolationForest は 正常=1 異常=-1 として結果を返す
# 外れ値と判定したデータを赤色でプロットする
plt.subplot(1, 4, 2)
plt.title("outlier data(auto)")
plt.scatter(
    X_train[y_pred == -1, 0],
    X_train[y_pred == -1, 1],
    c='r',
)

# 外れ値スコアを算出する
outlier_score = clf.decision_function(X_train)
# 外れ値スコアの閾値を設定する
THRETHOLD = -0.08
# 外れ値スコア以下のインデックスを取得する
predicted_outlier_index = np.where(outlier_score < THRETHOLD)

# 外れ値と判定したデータを緑色でプロットする
predicted_outlier = X_train[predicted_outlier_index]
plt.subplot(1, 4, 3)
plt.title("outlier data(manual)")
plt.scatter(
    predicted_outlier[:, 0],
    predicted_outlier[:, 1],
    c='g',
)
plt.show()