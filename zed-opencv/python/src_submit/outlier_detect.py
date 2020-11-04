from sklearn.ensemble import IsolationForest

def outlier_detection(X_train):
    # IsolationForestインスタンスを作成する
    clf = IsolationForest(
        contamination='auto', behaviour='new', max_features=2, random_state=42
    )

    # 学習用データを学習させる
    clf.fit(X_train)

    # 検証用データを分類する
    y_pred = clf.predict(X_train)
    return y_pred
