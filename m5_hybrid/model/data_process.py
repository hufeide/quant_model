#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 数据处理：store-item 粒度、价格首现周切除、价格前向填充及陈旧度特征。

输出：
  outputs/panel_storeitem.parquet（没有 pyarrow 时降级为 csv）
  outputs/matrices_storeitem.npz

约定：
  * 每个 (store_id, item_id) 是独立序列，不跨店聚合。
  * 某实体有价格记录时，切除其价格首现周起始日之前的销量。
  * 对价格缺口只做前向填充，不做反向填充。
  * 完全没有价格记录的实体保留在原始面板中，但 sales_valid 全为 False，
    下游窗口生成器会跳过该实体，避免把无基准价的销量混入训练。
  * price_norm / days_norm 的统计量只在训练时间段拟合，默认训练时间段为全局日期的前 70%。
  * make_windows() 返回所有时间起点的样本，并附 target_end（预测区间右边界，左闭右开）；
    切分时要确保训练样本 target_end <= train_end，避免目标区间跨入验证集。
"""

from pathlib import Path
import zipfile
import subprocess
import warnings

import numpy as np
import pandas as pd

# ============================================================
# 1. 配置
# ============================================================
SEED = 42
N_ITEMS = 50
TRAIN_FRACTION = 0.70
# 仅保留「价格首现」在训练期之前的实体，确保所有模型（尤其 PatchTST/iTransformer
# 需要较长连续有效历史）拥有共同的有效建模区间，避免个别晚上市商品把公共有效起点
# 拖到训练滑窗之后，导致没有任何可用窗口。单位：天（从 d_1 起算）。
EARLY_CUTOFF_DAYS = 365
STATES = ["CA", "TX", "WI"]

BASE_DIR = Path(__file__).resolve().parent          # 本文件所在目录（model/）
PROJECT_DIR = BASE_DIR.parent                        # 项目根目录（m5_hybrid/，数据实际所在处）
DATA_DIR = PROJECT_DIR / "data" / "m5"
OUT_DIR = PROJECT_DIR / "outputs"


# ============================================================
# 2. 下载数据
# ============================================================
def download_data():
    """若所需 M5 文件不存在，则使用 Kaggle CLI 下载并解压。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]
    if all((DATA_DIR / name).exists() for name in required):
        print("M5 数据文件已存在。")
        return

    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "m5-forecasting-accuracy",
            "-p", str(DATA_DIR),
        ],
        check=True,
    )
    for path in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(DATA_DIR)

    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"下载后仍缺少文件：{missing}。请检查 Kaggle 权限、下载结果和解压目录。"
        )


def _safe_mean_std(values, fallback_mean=0.0, fallback_std=1.0):
    """计算有限值的均值和标准差；样本为空或标准差过小时使用兜底值。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback_mean), float(fallback_std)

    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(mean):
        mean = float(fallback_mean)
    if not np.isfinite(std) or std < 1e-5:
        std = float(fallback_std)
    if not np.isfinite(std) or std < 1e-5:
        std = 1.0
    return mean, std


def _validate_window_args(history, horizon, stride):
    if history <= 0:
        raise ValueError("history 必须大于 0。")
    if horizon <= 0:
        raise ValueError("horizon 必须大于 0。")
    if stride <= 0:
        raise ValueError("stride 必须大于 0。")


# ============================================================
# 3. 主流程：构建 store-item 级面板
# ============================================================
def prepare_data(
    n_items=N_ITEMS,
    stores=None,
    seed=SEED,
    save=True,
    train_fraction=TRAIN_FRACTION,
):
    """构建 store-item 粒度数据。

    返回字典的主要键：
      panel              长表，每行一个 (store_id, item_id, date)
      entities           [(store_id, state_id, item_id), ...]
      dates              全局日期轴 (n_days,)
      sales              (n_days, n_entities)，无效/首现周前销量为 NaN
      sales_valid        (n_days, n_entities)，真正可用于建模的掩码
      price              前向填充价格矩阵；无价格或首现周前为 NaN
      price_norm         使用训练期统计量归一化后的价格；缺失值哨兵为 0
      price_avail        当日是否有真实价格记录（前向填充值为 0）
      days_since_price   距离最近真实价格周起始日的天数；从未有价格为 9999
      days_norm          训练期归一化的 log1p(days_since_price)；哨兵值为 0
      snap               SNAP 状态特征矩阵，包含整个日期轴
      cal_feat           (n_days, 4)：归一化 wday/month/year 与 event_flag
      cal_raw            (n_days, 4)：原始 wday/month/year 与 event_flag
      entity_static      store 独热 + state 独热
      start_day          每条实体序列的价格首现日下标；无价格实体为 0（但其 valid 全 False）
      train_end          训练区间右边界，Python 切片语义 [0, train_end)
    """
    if not isinstance(n_items, (int, np.integer)) or n_items <= 0:
        raise ValueError("n_items 必须是正整数。")
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction 必须在 0 和 1 之间。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [
        DATA_DIR / "sales_train_validation.csv",
        DATA_DIR / "calendar.csv",
        DATA_DIR / "sell_prices.csv",
    ]
    missing_paths = [str(p) for p in required_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "缺少 M5 数据文件：" + ", ".join(missing_paths) + "。请先调用 download_data()。"
        )

    sales_df = pd.read_csv(required_paths[0])
    calendar = pd.read_csv(required_paths[1])
    prices = pd.read_csv(required_paths[2])

    required_sales_meta_cols = {"item_id", "store_id", "state_id"}
    missing_sales_meta = required_sales_meta_cols - set(sales_df.columns)
    if missing_sales_meta:
        raise ValueError(f"销量文件缺少必要列：{sorted(missing_sales_meta)}")

    required_price_cols = {"item_id", "store_id", "wm_yr_wk", "sell_price"}
    missing_price_cols = required_price_cols - set(prices.columns)
    if missing_price_cols:
        raise ValueError(f"sell_prices.csv 缺少必要列：{sorted(missing_price_cols)}")
    if prices[list(required_price_cols)].isna().any().any():
        raise ValueError("sell_prices.csv 的 item/store/week/price 必要列中存在缺失值。")

    # 3.1 日历与日期轴准备
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("sales_train_validation.csv 中没有 d_* 销量列。")
    day_nums = np.array([int(c[2:]) for c in day_cols], dtype=np.int64)
    order = np.argsort(day_nums)
    day_cols = [day_cols[i] for i in order]

    required_calendar_cols = {"d", "date", "wm_yr_wk", "wday", "month", "year"}
    missing_calendar_cols = required_calendar_cols - set(calendar.columns)
    if missing_calendar_cols:
        raise ValueError(f"calendar.csv 缺少必要列：{sorted(missing_calendar_cols)}")

    calendar = calendar.copy()
    calendar["d_num"] = calendar["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
    calendar = (
        calendar.sort_values("d_num")
        .drop_duplicates("d_num", keep="first")
        .reset_index(drop=True)
    )
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")

    if calendar["d"].duplicated().any():
        raise ValueError("calendar.csv 中 d 列不是唯一键。")
    week_start = calendar.groupby("wm_yr_wk", sort=False)["date"].min()

    # 3.2 选定店铺与商品
    all_stores = sorted(sales_df["store_id"].dropna().unique().tolist())
    if stores is None:
        selected_stores = all_stores
    else:
        requested_stores = list(dict.fromkeys(stores))
        selected_stores = sorted(s for s in requested_stores if s in all_stores)
        ignored = sorted(set(requested_stores) - set(selected_stores))
        if ignored:
            print(f"警告：以下店铺不在销量数据中，已忽略：{ignored}")
    if not selected_stores:
        raise ValueError("没有可用的店铺。请检查 stores 参数。")

    sales_df = sales_df[sales_df["store_id"].isin(selected_stores)].copy()
    if sales_df.empty:
        raise ValueError("过滤店铺后没有销量数据。")

    rng = np.random.default_rng(seed)
    available_items = np.array(sorted(sales_df["item_id"].dropna().unique()))
    if available_items.size == 0:
        raise ValueError("所选店铺没有可用 item_id。")
    n_choose = min(int(n_items), len(available_items))
    chosen_items = sorted(
        rng.choice(available_items, size=n_choose, replace=False).tolist()
    )

    sampled = sales_df.loc[
        sales_df["item_id"].isin(chosen_items),
        ["item_id", "store_id", "state_id", *day_cols],
    ].copy()
    if sampled.duplicated(["store_id", "item_id"]).any():
        raise ValueError("销量数据中存在重复的 (store_id, item_id) 记录。")

    # 3.3 转长表，不跨店聚合
    long = sampled.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype(np.float32)
    if long["sales"].isna().any() or (long["sales"] < 0).any():
        raise ValueError("销量含 NaN 或负值，请先检查源数据。")

    # 3.4 合并日历
    calendar_cols = [
        c for c in [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        if c in calendar.columns
    ]
    calendar_sub = calendar[calendar_cols].copy()
    if calendar_sub["d"].duplicated().any():
        raise ValueError("calendar 子表的 d 不是唯一键。")
    long = long.merge(
        calendar_sub,
        on="d",
        how="left",
        validate="many_to_one",
    )
    if long["date"].isna().any() or long["wm_yr_wk"].isna().any():
        raise ValueError("部分销量日期无法在 calendar 中找到 date/wm_yr_wk。")

    # 3.5 SNAP 按实体所属州映射
    snap_choices = []
    snap_conditions = []
    for state in STATES:
        col = f"snap_{state}"
        if col not in long.columns:
            raise ValueError(f"calendar.csv 缺少 {col} 列。")
        snap_conditions.append(long["state_id"].eq(state).to_numpy())
        snap_choices.append(pd.to_numeric(long[col], errors="coerce").to_numpy(dtype=float))
    long["snap_active"] = np.select(
        snap_conditions,
        snap_choices,
        default=np.nan,
    ).astype(np.float32)
    if long["snap_active"].isna().any():
        unknown_states = sorted(long.loc[long["snap_active"].isna(), "state_id"].unique().tolist())
        raise ValueError(f"存在不支持的州或 SNAP 缺失：{unknown_states}")

    # 3.6 合并门店级价格；同时按选定店铺过滤，避免混入其他店铺统计。
    price_cols = ["item_id", "store_id", "wm_yr_wk", "sell_price"]
    prices_sub = prices.loc[
        prices["item_id"].isin(chosen_items)
        & prices["store_id"].isin(selected_stores),
        price_cols,
    ].copy()
    prices_sub["sell_price"] = pd.to_numeric(prices_sub["sell_price"], errors="raise").astype(np.float32)
    price_key = ["item_id", "store_id", "wm_yr_wk"]
    if prices_sub.duplicated(price_key).any():
        raise ValueError("sell_prices.csv 存在重复的 (item_id, store_id, wm_yr_wk) 键。")

    long = long.merge(
        prices_sub,
        on=price_key,
        how="left",
        validate="many_to_one",
    )

    # 3.7 切除价格首现周起始日之前的数据
    first_price_wk = (
        prices_sub.loc[prices_sub["sell_price"].notna()]
        .groupby(["store_id", "item_id"])["wm_yr_wk"]
        .min()
        .rename("first_price_wk")
    )
    first_price = first_price_wk.reset_index()
    first_price["first_price_date"] = first_price["first_price_wk"].map(week_start)
    if first_price["first_price_date"].isna().any():
        bad_weeks = first_price.loc[
            first_price["first_price_date"].isna(), "first_price_wk"
        ].drop_duplicates().tolist()
        raise ValueError(f"以下价格周在 calendar 中没有对应周起始日：{bad_weeks[:10]}")

    long = long.merge(
        first_price,
        on=["store_id", "item_id"],
        how="left",
        validate="many_to_one",
    )
    keep_mask = long["first_price_date"].isna() | (long["date"] >= long["first_price_date"])
    n_before_cut = len(long)
    long = long.loc[keep_mask].copy()
    print(
        f"价格首现周切除：移除 {n_before_cut - len(long)} 行；"
        f"选定商品/店铺组合中有价格记录的实体数={len(first_price)}。"
    )
    if long.empty:
        raise ValueError("价格首现周切除后没有剩余数据。")

    # 3.7b 仅保留价格首现足够早的实体，确保所有模型拥有共同的有效建模区间。
    # first_price_date 是价格首现「周起始日」，有效起点 = 该日下标；要求其在
    # EARLY_CUTOFF_DAYS 之内，使公共有效起点远早于训练滑窗，避免无可用窗口。
    _first_d_date = calendar.set_index("d").loc[day_cols[0], "date"]
    _cutoff = pd.Timestamp(_first_d_date) + pd.Timedelta(days=EARLY_CUTOFF_DAYS)
    _keep_ent = first_price.loc[
        first_price["first_price_date"] <= _cutoff, ["store_id", "item_id"]
    ].drop_duplicates()
    long = long.merge(_keep_ent, on=["store_id", "item_id"], how="inner")
    if long.empty:
        raise ValueError(
            "剔除晚上市实体后没有剩余数据，请调大 EARLY_CUTOFF_DAYS 或扩大 N_ITEMS。"
        )

    # 3.8 前向填充价格 + 按最近真实价格周起始日计算陈旧度
    long = long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    group_cols = ["store_id", "item_id"]
    long["price_nonnull"] = long["sell_price"].notna()
    long["sell_price"] = long.groupby(group_cols, sort=False)["sell_price"].ffill()

    long["week_start_date"] = long["wm_yr_wk"].map(week_start)
    if long["week_start_date"].isna().any():
        raise ValueError("部分销量日期的 wm_yr_wk 无法映射到周起始日。")
    long["_masked_week_start"] = long["week_start_date"].where(long["price_nonnull"])
    long["last_price_date"] = long.groupby(group_cols, sort=False)["_masked_week_start"].ffill()
    age = (long["date"] - long["last_price_date"]).dt.days
    long["days_since_price"] = age.fillna(9999).astype(np.int32)
    long["price_available"] = long["price_nonnull"].astype(np.float32)

    # 3.9 实体及完整全局日期轴。日期轴基于原始销量数据，不受实体切除起点影响。
    ent = (
        long[["store_id", "state_id", "item_id"]]
        .drop_duplicates()
        .sort_values(["store_id", "item_id"])
        .reset_index(drop=True)
    )
    entities = list(zip(ent["store_id"], ent["state_id"], ent["item_id"]))
    entity_keys = list(zip(ent["store_id"], ent["item_id"]))
    n_entities = len(entities)
    if n_entities == 0:
        raise ValueError("没有构建出任何实体。")

    # 仅保留销量文件中确实存在的日历日期（通常是 d_1 至 d_1913）。
    sales_calendar = calendar.loc[calendar["d"].isin(day_cols)].copy()
    if sales_calendar.empty:
        raise ValueError("销量日期与 calendar 没有交集。")
    max_sales_date = sales_calendar["date"].max()
    all_dates = pd.DatetimeIndex(
        calendar.loc[calendar["date"] <= max_sales_date, "date"]
        .drop_duplicates()
        .sort_values()
    )
    if len(all_dates) == 0:
        raise ValueError("全局日期轴为空。")

    if long.duplicated(["date", "store_id", "item_id"]).any():
        raise ValueError("切除后长表存在重复的 (date, store_id, item_id) 记录。")

    target_columns = pd.MultiIndex.from_tuples(
        entity_keys, names=["store_id", "item_id"]
    )

    def to_wide_frame(column):
        wide = long.pivot(
            index="date",
            columns=["store_id", "item_id"],
            values=column,
        )
        return wide.reindex(index=all_dates, columns=target_columns)

    sales_mat = to_wide_frame("sales").to_numpy(dtype=np.float32)
    price_mat = to_wide_frame("sell_price").to_numpy(dtype=np.float32)
    price_avail_mat = to_wide_frame("price_available").to_numpy(dtype=np.float32)
    price_avail_mat = np.nan_to_num(price_avail_mat, nan=0.0, posinf=0.0, neginf=0.0)
    days_mat = to_wide_frame("days_since_price").to_numpy(dtype=np.float32)

    # SNAP 是未来已知日历变量，所有日期都构造出来，不因价格切除而变成 NaN。
    calendar_by_date = calendar.drop_duplicates("date").set_index("date").reindex(all_dates)
    snap_mat = np.zeros((len(all_dates), n_entities), dtype=np.float32)
    for j, (_store, state, _item) in enumerate(entities):
        col = f"snap_{state}"
        if col not in calendar_by_date.columns:
            raise ValueError(f"calendar.csv 缺少 {col} 列。")
        vals = pd.to_numeric(calendar_by_date[col], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(vals).all():
            raise ValueError(f"{col} 在日期轴上存在缺失值。")
        snap_mat[:, j] = vals

    # 起点映射。无价格实体保持 start_day=0，但之后其 sales_valid 全为 False。
    first_date_map = {
        (row.store_id, row.item_id): pd.Timestamp(row.first_price_date)
        for row in first_price.itertuples(index=False)
    }
    first_dates = [first_date_map.get(key, pd.NaT) for key in entity_keys]
    start_day = np.zeros(n_entities, dtype=np.int64)
    for j, fdate in enumerate(first_dates):
        if pd.notna(fdate):
            start_day[j] = int(all_dates.searchsorted(pd.Timestamp(fdate)))

    # 有效样本必须同时满足：销量存在、前向填充后价格存在、在首现价格日之后。
    sales_valid = np.isfinite(sales_mat) & np.isfinite(price_mat)
    for j, fdate in enumerate(first_dates):
        if pd.notna(fdate):
            sales_valid[:start_day[j], j] = False
        else:
            # 完全没有任何价格的实体不参与模型窗口，但仍保留原始数据便于审计。
            sales_valid[:, j] = False

    # 3.10 只用训练区间拟合价格与价格陈旧度归一化参数
    n_days = len(all_dates)
    train_end = min(max(int(np.floor(n_days * float(train_fraction))), 1), n_days)

    global_price_fit_mask = (
        sales_valid[:train_end]
        & (price_avail_mat[:train_end] > 0.5)
        & np.isfinite(price_mat[:train_end])
    )
    global_price_values = price_mat[:train_end][global_price_fit_mask]
    global_price_mean, global_price_std = _safe_mean_std(
        global_price_values, fallback_mean=0.0, fallback_std=1.0
    )

    global_age_fit_mask = (
        sales_valid[:train_end]
        & np.isfinite(days_mat[:train_end])
        & (days_mat[:train_end] >= 0)
        & (days_mat[:train_end] < 9000)
    )
    global_age_values = np.log1p(days_mat[:train_end][global_age_fit_mask])
    global_age_mean, global_age_std = _safe_mean_std(
        global_age_values, fallback_mean=0.0, fallback_std=1.0
    )

    price_norm_mat = np.zeros_like(price_mat, dtype=np.float32)
    days_norm_mat = np.zeros_like(days_mat, dtype=np.float32)

    for j in range(n_entities):
        s0 = int(start_day[j])
        fit_stop = max(s0, min(train_end, n_days))

        # 价格：用训练期真实观测价格拟合；变换时保留整个有效区间的前向填充值。
        p_fit_mask = (
            sales_valid[s0:fit_stop, j]
            & (price_avail_mat[s0:fit_stop, j] > 0.5)
            & np.isfinite(price_mat[s0:fit_stop, j])
        )
        p_fit = price_mat[s0:fit_stop, j][p_fit_mask]
        if p_fit.size:
            p_mean, p_std = _safe_mean_std(
                p_fit, fallback_mean=global_price_mean, fallback_std=global_price_std
            )
        else:
            # 晚于训练切点才有价格的实体使用训练集总体统计量，绝不借用未来价格拟合。
            p_mean, p_std = global_price_mean, global_price_std

        if s0 < n_days:
            pj = price_mat[s0:, j]
            pnorm = (pj - p_mean) / p_std
            pnorm = np.where(np.isfinite(pnorm), pnorm, 0.0)
            price_norm_mat[s0:, j] = pnorm.astype(np.float32)

        # 陈旧度：log1p(age) 的均值/方差也只在训练期有效区间估计。
        d_fit = days_mat[s0:fit_stop, j]
        d_fit_mask = (
            sales_valid[s0:fit_stop, j]
            & np.isfinite(d_fit)
            & (d_fit >= 0)
            & (d_fit < 9000)
        )
        d_fit_log = np.log1p(d_fit[d_fit_mask])
        if d_fit_log.size:
            d_mean, d_std = _safe_mean_std(
                d_fit_log, fallback_mean=global_age_mean, fallback_std=global_age_std
            )
        else:
            d_mean, d_std = global_age_mean, global_age_std

        if s0 < n_days:
            dj = days_mat[s0:, j]
            dlog = np.log1p(np.maximum(dj, 0.0))
            dnorm = (dlog - d_mean) / d_std
            # 9999 是“从未出现价格”的哨兵，不应变成一个巨大连续特征。
            dnorm = np.where((dj >= 9000) | (~np.isfinite(dnorm)), 0.0, dnorm)
            days_norm_mat[s0:, j] = dnorm.astype(np.float32)

    # 3.11 日历特征：保留原始值，模型特征使用固定的归一化口径。
    required_calendar_feature_cols = ["wday", "month", "year"]
    cal_aligned = calendar_by_date
    if cal_aligned[required_calendar_feature_cols].isna().any().any():
        raise ValueError("日期轴上的 wday/month/year 未能完全对齐 calendar。")

    wday_raw = pd.to_numeric(cal_aligned["wday"], errors="raise").to_numpy(dtype=np.float32)
    month_raw = pd.to_numeric(cal_aligned["month"], errors="raise").to_numpy(dtype=np.float32)
    year_raw = pd.to_numeric(cal_aligned["year"], errors="raise").to_numpy(dtype=np.float32)

    event_parts = []
    for col in ("event_name_1", "event_name_2", "event_type_1", "event_type_2"):
        if col in cal_aligned.columns:
            event_parts.append(cal_aligned[col].notna().to_numpy(dtype=np.float32))
        else:
            event_parts.append(np.zeros(n_days, dtype=np.float32))
    event_flag = np.maximum.reduce(event_parts).astype(np.float32)

    cal_feat = np.stack(
        [
            (wday_raw - 1.0) / 6.0,
            (month_raw - 1.0) / 12.0,
            (year_raw - 2011.0) / 10.0,
            event_flag,
        ],
        axis=-1,
    ).astype(np.float32)
    cal_raw = np.stack([wday_raw, month_raw, year_raw, event_flag], axis=-1).astype(np.float32)

    # 3.12 静态特征：所选店铺独热 + 三州独热
    store_list = sorted(selected_stores)
    store_idx = {store: i for i, store in enumerate(store_list)}
    state_idx = {state: i for i, state in enumerate(STATES)}
    entity_static = np.zeros((n_entities, len(store_list) + len(STATES)), dtype=np.float32)
    for j, (store, state, _item) in enumerate(entities):
        if state not in state_idx:
            raise ValueError(f"发现不在 STATES={STATES} 中的州：{state}")
        entity_static[j, store_idx[store]] = 1.0
        entity_static[j, len(store_list) + state_idx[state]] = 1.0

    # 3.13 长表，保留原始特征、归一化特征、有效标记和实体索引。
    panel_cols = [
        "store_id", "state_id", "item_id", "date", "d", "sales",
        "sell_price", "price_available", "days_since_price", "snap_active",
        "first_price_date",
    ]
    panel = long[panel_cols].copy()
    key_to_idx = {key: idx for idx, key in enumerate(entity_keys)}
    panel["entity_idx"] = [key_to_idx[(st, it)] for st, it in zip(panel["store_id"], panel["item_id"])]
    date_to_idx = {pd.Timestamp(date): idx for idx, date in enumerate(all_dates)}
    di = panel["date"].map(date_to_idx).to_numpy(dtype=np.int64)
    ei = panel["entity_idx"].to_numpy(dtype=np.int64)
    panel["price_norm"] = price_norm_mat[di, ei]
    panel["days_norm"] = days_norm_mat[di, ei]
    panel["sales_valid"] = sales_valid[di, ei]
    panel["wday"] = np.rint(wday_raw[di]).astype(np.int16)
    panel["month"] = np.rint(month_raw[di]).astype(np.int16)
    panel["year"] = np.rint(year_raw[di]).astype(np.int16)
    panel["event_flag"] = event_flag[di].astype(np.int8)

    # 3.14 打印与保存
    print("实体数 (店×商品):", n_entities)
    print("日期数:", n_days, "范围:", all_dates.min().date(), "至", all_dates.max().date())
    print(f"训练期归一化截止：索引 [0, {train_end})，train_fraction={train_fraction:.3f}")
    print("有效格数:", int(sales_valid.sum()), "/", sales_valid.size)
    no_price_entities = int((~np.isfinite(price_mat).any(axis=0)).sum())
    if no_price_entities:
        print(f"提示：{no_price_entities} 个实体全程没有价格，不会生成训练窗口。")

    if save:
        parquet_path = OUT_DIR / "panel_storeitem.parquet"
        csv_path = OUT_DIR / "panel_storeitem.csv"
        try:
            panel.to_parquet(parquet_path, index=False)
            if csv_path.exists():
                csv_path.unlink()
            print("已保存：", parquet_path)
        except Exception as exc:
            panel.to_csv(csv_path, index=False)
            print(f"Parquet 保存失败（{exc}），已降级保存：{csv_path}")

        np.savez_compressed(
            OUT_DIR / "matrices_storeitem.npz",
            dates=all_dates.to_numpy(dtype="datetime64[ns]"),
            sales=sales_mat,
            sales_valid=sales_valid,
            price=price_mat,
            price_norm=price_norm_mat,
            price_avail=price_avail_mat,
            days_since_price=days_mat,
            days_norm=days_norm_mat,
            snap=snap_mat,
            cal_feat=cal_feat,
            cal_raw=cal_raw,
            entity_static=entity_static,
            start_day=start_day,
            train_end=np.asarray(train_end, dtype=np.int64),
            train_fraction=np.asarray(float(train_fraction), dtype=np.float32),
        )
        print("已保存：", OUT_DIR / "matrices_storeitem.npz")

    return {
        "panel": panel,
        "entities": entities,
        "dates": all_dates,
        "sales": sales_mat,
        "sales_valid": sales_valid,
        "price": price_mat,
        "price_norm": price_norm_mat,
        "price_avail": price_avail_mat,
        "days_since_price": days_mat,
        "days_norm": days_norm_mat,
        "snap": snap_mat,
        "cal_feat": cal_feat,
        "cal_raw": cal_raw,
        "entity_static": entity_static,
        "start_day": start_day,
        "train_end": train_end,
        "train_fraction": float(train_fraction),
    }


# ============================================================
# 4. 构造序列模型窗口
# ============================================================
def make_windows(
    prepared,
    history=112,
    horizon=28,
    stride=7,
    use_cov=True,
    log_sales=True,
):
    """逐实体构造序列样本，并跳过历史/预测区间含无效日期的窗口。

    每个 sample 包含：
      x_sales: (history,)
      cov_hist: (history, D)，use_cov=False 时 D=0
      cov_fut: (horizon, D)，use_cov=False 时 D=0
      y: (horizon,)
      entity_idx, origin

    origin 是首个预测日的全局下标。返回全部时间起点样本；切分训练/验证/测试时，
    训练期样本必须满足 target_end <= train_end；target_end 是预测区间右边界（左闭右开）。
    不要随机拆分高度重叠的窗口。
    """
    _validate_window_args(history, horizon, stride)

    dates = prepared["dates"]
    n_days = len(dates)
    sales = np.asarray(prepared["sales"], dtype=np.float32)
    valid = np.asarray(prepared["sales_valid"], dtype=bool)
    cal = np.asarray(prepared["cal_feat"], dtype=np.float32)
    snap = np.asarray(prepared["snap"], dtype=np.float32)
    price_norm = np.asarray(prepared["price_norm"], dtype=np.float32)
    price_avail = np.asarray(prepared["price_avail"], dtype=np.float32)
    days_norm = np.asarray(prepared["days_norm"], dtype=np.float32)
    start_day = np.asarray(prepared["start_day"], dtype=np.int64)
    n_ent = sales.shape[1]

    if valid.shape != sales.shape:
        raise ValueError("sales_valid 的形状必须与 sales 一致。")
    if snap.shape != sales.shape:
        raise ValueError("snap 的形状必须与 sales 一致。")

    # 每日协变量：(n_days, n_entities, 8)
    cal_b = np.broadcast_to(cal[:, None, :], (n_days, n_ent, 4))
    cov_all = np.stack(
        [
            price_norm,
            cal_b[:, :, 0],
            cal_b[:, :, 1],
            cal_b[:, :, 2],
            snap,
            cal_b[:, :, 3],
            price_avail,
            days_norm,
        ],
        axis=-1,
    ).astype(np.float32)

    if log_sales:
        with np.errstate(invalid="ignore", divide="ignore"):
            target_sales = np.log1p(np.maximum(sales, 0.0)).astype(np.float32)
    else:
        target_sales = sales.copy()

    # 整体训练期统计量仅作为晚上市实体的兜底；不使用训练切点之后的数据。
    train_end = int(prepared.get("train_end", max(1, int(n_days * TRAIN_FRACTION))))
    train_end = min(max(train_end, 1), n_days)
    global_fit_mask = valid[:train_end] & np.isfinite(target_sales[:train_end])
    global_fit_values = target_sales[:train_end][global_fit_mask]
    global_mu, global_sd = _safe_mean_std(global_fit_values, 0.0, 1.0)

    samples = []
    for j in range(n_ent):
        s0 = int(start_day[j])
        fit_stop = min(train_end, n_days)
        local_values = target_sales[s0:fit_stop, j]
        local_mask = valid[s0:fit_stop, j] & np.isfinite(local_values)
        local_values = local_values[local_mask]
        if local_values.size:
            mu, sd = _safe_mean_std(local_values, global_mu, global_sd)
        else:
            mu, sd = global_mu, global_sd

        if not np.isfinite(sd) or sd < 1e-5:
            sd = 1.0
        svals = ((target_sales[:, j] - mu) / sd).astype(np.float32)
        # 明确将无效销量格保留为 NaN，防止未来误把哨兵值送入模型。
        svals[~valid[:, j]] = np.nan

        first_origin = s0 + history
        final_origin = n_days - horizon
        for origin in range(first_origin, final_origin + 1, stride):
            hist_slice = slice(origin - history, origin)
            fut_slice = slice(origin, origin + horizon)
            if not valid[hist_slice, j].all() or not valid[fut_slice, j].all():
                continue

            x_s = svals[hist_slice]
            y = svals[fut_slice]
            if not np.isfinite(x_s).all() or not np.isfinite(y).all():
                continue

            if use_cov:
                ch = cov_all[hist_slice, j, :]
                cf = cov_all[fut_slice, j, :]
                if not np.isfinite(ch).all() or not np.isfinite(cf).all():
                    continue
            else:
                ch = np.empty((history, 0), dtype=np.float32)
                cf = np.empty((horizon, 0), dtype=np.float32)

            samples.append(
                {
                    "x_sales": x_s.astype(np.float32, copy=False),
                    "cov_hist": ch.astype(np.float32, copy=False),
                    "cov_fut": cf.astype(np.float32, copy=False),
                    "y": y.astype(np.float32, copy=False),
                    "entity_idx": int(j),
                    "origin": int(origin),
                    "target_end": int(origin + horizon),
                }
            )

    cov_dim = cov_all.shape[-1] if use_cov else 0
    print(
        f"构造序列窗口：{len(samples)} 个 (entity×起点) 样本，"
        f"history={history}, horizon={horizon}, cov_dim={cov_dim}"
    )
    return samples, cov_dim


# ============================================================
# 5. 构造 LightGBM 表格行
# ============================================================
def build_lightgbm_rows(prepared, history=56, horizon=28, stride=7, origins=None):
    """构造 store-item 粒度的直接多步 LightGBM 训练行。

    这里 o 是历史窗口起点，L=o+history-1 是最后一个已观测日；预测目标为
    L+1 至 L+horizon。每个 horizon 行共享截至 L 的历史特征，并加入目标日已知协变量。
    对 target_price 的使用依赖于“预测期价格已知”的实验假设。
    """
    _validate_window_args(history, horizon, stride)

    sales = np.asarray(prepared["sales"], dtype=np.float32)
    valid = np.asarray(prepared["sales_valid"], dtype=bool)
    dates = pd.DatetimeIndex(prepared["dates"])
    price = np.asarray(prepared["price"], dtype=np.float32)
    snap = np.asarray(prepared["snap"], dtype=np.float32)
    price_avail = np.asarray(prepared["price_avail"], dtype=np.float32)
    price_norm = np.asarray(prepared["price_norm"], dtype=np.float32)
    days_since_price = np.asarray(prepared["days_since_price"], dtype=np.float32)
    days_norm = np.asarray(prepared["days_norm"], dtype=np.float32)
    entities = prepared["entities"]
    n_days, n_ent = sales.shape

    if "cal_raw" in prepared:
        cal_raw = np.asarray(prepared["cal_raw"], dtype=np.float32)
        wday_r = np.rint(cal_raw[:, 0]).astype(int)
        month_r = np.rint(cal_raw[:, 1]).astype(int)
        year_r = np.rint(cal_raw[:, 2]).astype(int)
        event_r = np.rint(cal_raw[:, 3]).astype(int)
    else:
        # 与旧版 prepared 结构兼容。
        cal = np.asarray(prepared["cal_feat"], dtype=np.float32)
        wday_r = np.rint(cal[:, 0] * 6 + 1).astype(int)
        month_r = np.rint(cal[:, 1] * 12 + 1).astype(int)
        year_r = np.rint(cal[:, 2] * 10 + 2011).astype(int)
        event_r = np.rint(cal[:, 3]).astype(int)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_sales = np.log1p(np.maximum(sales, 0.0)).astype(np.float32)

    rows = []
    for j in range(n_ent):
        s0 = int(prepared["start_day"][j])
        store, state, item = entities[j]
        sl = log_sales[:, j]
        pr = price[:, j]
        sn = snap[:, j]
        pav = price_avail[:, j]
        pnorm = price_norm[:, j]
        age = days_since_price[:, j]
        agenorm = days_norm[:, j]

        # o 表示历史窗口起点，不是预测起点；修正原代码额外跳过 history 天的问题。
        last_start = n_days - history - horizon
        if origins is None:
            o_iter = range(s0, last_start + 1, stride)
        else:
            o_iter = [int(x) for x in np.asarray(origins).ravel()]
        for o in o_iter:
            if o < s0 or o > last_start:
                continue
            L = o + history - 1
            full_stop = o + history + horizon
            if not valid[o:full_stop, j].all():
                continue

            def lag(k):
                # 相对预测起点，lag_1 是最后一个已观测日 L；lag_7 是 L-6。
                idx = L - (k - 1)
                if idx < s0 or not valid[idx, j] or not np.isfinite(sl[idx]):
                    return 0.0
                return float(sl[idx])

            def roll(k):
                lo = max(s0, L - k + 1)
                vals = sl[lo:L + 1]
                vals = vals[np.isfinite(vals)]
                return float(vals.mean()) if vals.size else 0.0

            price_window = pr[max(s0, L - 27):L + 1]
            price_window = price_window[np.isfinite(price_window)]
            price_28_mean = float(price_window.mean()) if price_window.size else 0.0
            year_ago_idx = L - 364
            year_ago = (
                float(sl[year_ago_idx])
                if year_ago_idx >= s0 and valid[year_ago_idx, j] and np.isfinite(sl[year_ago_idx])
                else 0.0
            )

            cw, cm, cy, ce = wday_r[L], month_r[L], year_r[L], event_r[L]
            base = {
                "entity_idx": int(j),
                "store_id": store,
                "state_id": state,
                "item_id": item,
                "lag_1": lag(1),
                "lag_7": lag(7),
                "lag_14": lag(14),
                "lag_21": lag(21),
                "lag_28": lag(28),
                "roll_mean_7": roll(7),
                "roll_mean_14": roll(14),
                "roll_mean_28": roll(28),
                "roll_std_28": float(np.std(sl[max(s0, L - 27):L + 1])),
                "price_last": float(pr[L]),
                "price_28_mean": price_28_mean,
                "price_change": float(pr[L]) - price_28_mean,
                "year_ago": year_ago,
                "wday": int(cw),
                "month": int(cm),
                "year": int(cy),
                "snap": float(sn[L]),
                "event_flag": int(ce),
            }

            for h in range(1, horizon + 1):
                target_idx = L + h
                row = dict(base)
                row.update(
                    {
                        # 时间元数据用于按目标日期切分，建模时不要作为特征输入。
                        "origin": int(L + 1),
                        "target_idx": int(target_idx),
                        "origin_date": dates[L + 1],
                        "target_date": dates[target_idx],
                        "horizon": int(h),
                        "target_wday": int(wday_r[target_idx]),
                        "target_month": int(month_r[target_idx]),
                        "target_snap": float(sn[target_idx]),
                        "target_price": float(pr[target_idx]),
                        "target_price_norm": float(pnorm[target_idx]),
                        "target_year": int(year_r[target_idx]),
                        "target_event_flag": int(event_r[target_idx]),
                        "target_price_available": float(pav[target_idx]),
                        "target_days_since_price": float(age[target_idx]),
                        "target_days_norm": float(agenorm[target_idx]),
                        "target": float(sl[target_idx]),
                    }
                )
                rows.append(row)

    if not rows:
        print("构造 LightGBM 行：0 行。请检查数据、history/horizon 或 sales_valid 掩码。")
        return pd.DataFrame(
            columns=[
                "entity_idx", "store_id", "state_id", "item_id",
                "origin", "target_idx", "origin_date", "target_date", "horizon",
                "target_wday", "target_month", "target_snap", "target_price", "target",
            ]
        )

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    finite_rows = np.isfinite(df[numeric_cols].to_numpy(dtype=np.float64)).all(axis=1)
    before = len(df)
    df = df.loc[finite_rows].reset_index(drop=True)
    if len(df) < before:
        print(f"  丢弃 {before - len(df)} 行包含 NaN/Inf 特征的样本。")

    print(
        f"构造 LightGBM 行：{len(df)} 行（{n_ent} 个实体，history={history}，"
        f"horizon={horizon}，stride={stride}）。"
    )
    return df


if __name__ == "__main__":
    download_data()
    prepared = prepare_data()
    samples, cov_dim = make_windows(prepared)
    lgb_rows = build_lightgbm_rows(prepared)
    print(
        "完成。实体数:", len(prepared["entities"]),
        "序列样本数:", len(samples),
        "协变量维数:", cov_dim,
        "LightGBM 行数:", len(lgb_rows),
    )
