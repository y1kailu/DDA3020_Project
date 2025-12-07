import pandas as pd
import numpy as np
import lightgbm as lgb

USE_TOP_10_ONLY = False     # ★ 关闭 top10 过滤，避免 split() 报错
WINDOW_SIZE = 10
EPOCHS = 300
LGBM_LEAVES = 128
LEARNING_RATE = 0.05


def create_lgbm_features(df, window):
    df = df.copy()

    if "investment_id" not in df.columns:
        raise ValueError("❌ create_lgbm_features() requires 'investment_id' column.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for c in ["forward_returns", "market_forward_excess_returns"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    for col in numeric_cols:
        df[f"{col}_lag_1"] = df.groupby("investment_id")[col].shift(1)

        shifted = df.groupby("investment_id")[col].shift(1)

        df[f"{col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"{col}_roll_std_{window}"] = shifted.rolling(window).std()
        df[f"{col}_roll_max_{window}"] = shifted.rolling(window).max()
        df[f"{col}_roll_min_{window}"] = shifted.rolling(window).min()

    return df



def run(X_train, y_train, X_test, params=None):
    print("=" * 60)
    print(">>> LightGBM: Training & Predicting (Fixed Version)")
    print("=" * 60)

    # -------- Convert numpy to DataFrame --------
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    # -------- Add investment_id if missing --------
    if "investment_id" not in X_train.columns:
        X_train["investment_id"] = 0
    if "investment_id" not in X_test.columns:
        X_test["investment_id"] = 0

    # -------- Feature Engineering --------
    train_feat = create_lgbm_features(X_train, WINDOW_SIZE)
    train_feat = train_feat.dropna()
    y_train_aligned = y_train.loc[train_feat.index]

    print(f"Training shape after FE: {train_feat.shape}")

    # -------- Default Parameters --------
    if params is None:
        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": LGBM_LEAVES,
            "learning_rate": LEARNING_RATE,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        }

    # -------- Train --------
    dtrain = lgb.Dataset(train_feat, label=y_train_aligned)
    model = lgb.train(params, dtrain, num_boost_round=EPOCHS)

    # -------- Create test features --------
    tail = X_train.groupby("investment_id").tail(WINDOW_SIZE)
    concat = pd.concat([tail, X_test], axis=0)
    test_feat_all = create_lgbm_features(concat, WINDOW_SIZE)
    test_feat_all = test_feat_all.dropna()

    test_feat = test_feat_all.loc[X_test.index]

    # -------- Align columns --------
    for col in train_feat.columns:
        if col not in test_feat.columns:
            test_feat[col] = 0
    test_feat = test_feat[train_feat.columns]

    # -------- Predict --------
    preds = model.predict(test_feat)

    return preds
