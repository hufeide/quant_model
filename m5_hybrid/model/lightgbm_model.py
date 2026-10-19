#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightGBM 表格模型：交叉特征 + 滞后特征（store-item 粒度）。

数据行来自 data_process.build_lightgbm_rows。
"""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from common import SEED, PRED_LEN

# 不参与建模的元数据列（时间/标识），必须从特征中剔除，避免泄漏或冗余。
META_COLS = {
    "target", "store_id", "state_id", "item_id",
    "origin", "target_idx", "origin_date", "target_date",
}


def train_lightgbm(train_df, val_df):
    if lgb is None:
        raise ImportError("请先安装 lightgbm: pip install lightgbm")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "seed": SEED,
    }

    feat_cols = [c for c in train_df.columns if c not in META_COLS]
    # 类别特征：实体标识 + 日历/预测偏移（均为整数且语义为类别）。
    cat_cols = [
        "entity_idx", "wday", "month", "year", "horizon",
        "target_wday", "target_month", "target_year",
    ]
    cat_cols = [c for c in cat_cols if c in feat_cols]

    lgb_train = lgb.Dataset(
        train_df[feat_cols], train_df["target"],
        categorical_feature=cat_cols,
    )
    lgb_val = lgb.Dataset(
        val_df[feat_cols], val_df["target"],
        reference=lgb_train, categorical_feature=cat_cols,
    )

    bst = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=[lgb_val], callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return bst, feat_cols


def lightgbm_predict(bst, feat_cols, test_df):
    pred_log = bst.predict(test_df[feat_cols])
    return np.expm1(np.maximum(pred_log, 0))  # 还原 log1p


def split_lightgbm(df, n_days, seq_len, pred_len, train_frac=0.70, val_frac=0.85):
    """按目标区间把全量 LightGBM 行切分为 train/val/test，与 make_origins 完全对齐。

    df 必须含 `origin`（首目标日全局下标）与 `target_idx`（目标日全局下标）。
    窗口最后目标日 target_end = origin + pred_len - 1。
      train: target_end < T_train
      val  : origin >= T_train 且 target_end < T_val
      test : origin >= T_val 且 target_end < n_days
    跨边界的窗口统一归入更晚的切分（val/test），绝不进入 train，避免目标泄漏。
    """
    T_train = int(n_days * train_frac)
    T_val = int(n_days * val_frac)
    target_end = df["origin"] + pred_len - 1

    out = df.copy()
    out["_split"] = "test"
    val_mask = (df["origin"] >= T_train) & (target_end < T_val)
    out.loc[val_mask, "_split"] = "val"
    train_mask = target_end < T_train
    out.loc[train_mask, "_split"] = "train"

    train_df = out.loc[out["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df = out.loc[out["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    test_df = out.loc[out["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
    return train_df, val_df, test_df
