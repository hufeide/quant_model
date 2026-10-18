#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5: sales-only baseline vs. multi-covariate PatchTST -> iTransformer-style model.

Expected files:
  data/m5/sales_train_validation.csv
  data/m5/calendar.csv
  data/m5/sell_prices.csv

The covariate model uses historical sales + historical covariates in PatchTST,
then cross-item attention, and conditions each forecast day on known future
calendar/price covariates. All scalers are fitted on the training time range only.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable


# ============================================================
# 0. Environment-driven configuration.
#    Every hyper-parameter / architecture switch can be overridden through an
#    environment variable so that the same source can be swept by experiment_loop.py
#    without code duplication. Defaults reproduce the original behaviour.
# ============================================================
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None and value != "" else default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 1. Configuration
# ============================================================
SEED = _env_int("M5_SEED", 42)
N_ITEMS = _env_int("M5_N_ITEMS", 50)
HISTORY = _env_int("M5_HISTORY", 112)
HORIZON = _env_int("M5_HORIZON", 28)
PATCH_LEN = _env_int("M5_PATCH_LEN", 16)
PATCH_STRIDE = _env_int("M5_PATCH_STRIDE", 8)
D_MODEL = _env_int("M5_D_MODEL", 64)
N_HEADS = _env_int("M5_N_HEADS", 8)
N_TEMPORAL_LAYERS = _env_int("M5_TEMPORAL_LAYERS", 2)
DROPOUT = _env_float("M5_DROPOUT", 0.1)
BATCH_SIZE = _env_int("M5_BATCH_SIZE", 32)
EPOCHS = _env_int("M5_EPOCHS", 50)
PATIENCE = _env_int("M5_PATIENCE", 8)
LEARNING_RATE = _env_float("M5_LR", 3e-3)
WEIGHT_DECAY = _env_float("M5_WEIGHT_DECAY", 1e-5)
TRAIN_RATIO = _env_float("M5_TRAIN_RATIO", 0.70)
VAL_RATIO = _env_float("M5_VAL_RATIO", 0.15)
EVAL_STRIDE = _env_int("M5_EVAL_STRIDE", 7)  # rolling-origin validation/testing; append final origin as well

# Architecture / training switches (controlled by experiment_loop.py).
COV_FUSION = _env_str("M5_COV_FUSION", "add")   # "add" (original) | "gate" | "proj"
LOSS_TYPE = _env_str("M5_LOSS", "huber")        # "mse" | "huber" | "mae"
LR_SCHEDULE = _env_str("M5_LR_SCHEDULE", "false") == "true"  # cosine learning-rate decay
GRAD_CLIP = _env_float("M5_GRAD_CLIP", 1.0)
ITEM_EMBED = _env_str("M5_ITEM_EMBED", "false") == "true"  # per-item learned embedding
ENSEMBLE = _env_str("M5_ENSEMBLE", "true") == "true"       # blend sales_only+multi_feature
ARCH = _env_str("M5_ARCH", "v2")                # "v1" (original) | "v2" (fixed: per-patch cross-item + horizon self-attn + future-conditioned query)
PRICE_MASK = _env_str("M5_PRICE_MASK", "false") == "true"  # mask unreliable carried-forward future prices

if N_ITEMS <= 0:
    raise ValueError("M5_N_ITEMS must be positive")
if HISTORY <= 0 or HORIZON <= 0:
    raise ValueError("M5_HISTORY and M5_HORIZON must be positive")
if PATCH_LEN <= 0 or PATCH_STRIDE <= 0 or HISTORY < PATCH_LEN:
    raise ValueError("Require 0 < PATCH_LEN <= HISTORY and PATCH_STRIDE > 0")
if D_MODEL <= 0 or N_HEADS <= 0 or D_MODEL % N_HEADS != 0:
    raise ValueError("D_MODEL must be positive and divisible by N_HEADS")
if BATCH_SIZE <= 0 or EPOCHS <= 0 or PATIENCE <= 0:
    raise ValueError("BATCH_SIZE, EPOCHS, and PATIENCE must be positive")
if LEARNING_RATE <= 0 or WEIGHT_DECAY < 0 or GRAD_CLIP <= 0:
    raise ValueError("Learning rate and gradient clip must be positive; weight decay cannot be negative")
if not (0.0 < TRAIN_RATIO < 1.0 and 0.0 < VAL_RATIO < 1.0 and TRAIN_RATIO + VAL_RATIO < 1.0):
    raise ValueError("TRAIN_RATIO and VAL_RATIO must be positive and sum to less than 1")
if EVAL_STRIDE <= 0:
    raise ValueError("M5_EVAL_STRIDE must be positive")
if COV_FUSION not in {"add", "gate", "proj"}:
    raise ValueError("M5_COV_FUSION must be one of: add, gate, proj")
if LOSS_TYPE not in {"mse", "huber", "mae"}:
    raise ValueError("M5_LOSS must be one of: mse, huber, mae")
if ARCH not in {"v1", "v2"}:
    raise ValueError("M5_ARCH must be either v1 or v2")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "m5"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility without forcing deterministic algorithms that can fail on some GPUs.


def d_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Cannot extract day number from {value!r}")
    return int(match.group(1))


def make_event_dummies(calendar: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create day-level one-hot event features from M5 calendar categories."""
    out = calendar.copy()
    dummy_cols: list[str] = []
    event_cols = [
        "event_name_1", "event_type_1", "event_name_2", "event_type_2"
    ]
    for col in event_cols:
        if col not in out.columns:
            continue
        values = out[col].fillna("").astype(str).str.strip()
        categories = sorted(v for v in values.unique() if v and v.lower() != "nan")
        for category in categories:
            # Keep feature names stable and filesystem/CSV friendly.
            slug = re.sub(r"[^0-9A-Za-z]+", "_", category).strip("_") or "event"
            name = f"{col}__{slug}"
            out[name] = (values == category).astype(np.float32)
            dummy_cols.append(name)
    return out, dummy_cols


def load_and_prepare() -> dict:
    for path in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the M5 CSV files and put them in {DATA_DIR.resolve()}"
            )

    print("Loading M5 CSV files...")
    sales = pd.read_csv(SALES_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    prices = pd.read_csv(PRICES_FILE)

    day_cols = sorted(
        [c for c in sales.columns if re.fullmatch(r"d_\d+", str(c))],
        key=d_number,
    )
    if not day_cols:
        raise ValueError("No d_1, d_2, ... sales columns found in sales CSV")

    # Deterministic item sample; keep all available stores for each selected item.
    rng = np.random.default_rng(SEED)
    available = np.array(sorted(sales["item_id"].dropna().unique()))
    chosen = sorted(rng.choice(available, size=min(N_ITEMS, len(available)), replace=False).tolist())
    selected = sales.loc[
        sales["item_id"].isin(chosen),
        [c for c in ["item_id", "store_id", "state_id", *day_cols] if c in sales.columns],
    ].copy()
    if "store_id" not in selected.columns or "state_id" not in selected.columns:
        raise ValueError("Expected store_id and state_id in sales_train_validation.csv")

    pd.Series(chosen, name="item_id").to_csv(OUT_DIR / "sampled_items.csv", index=False)
    print(f"Sampled {len(chosen)} items and {selected['store_id'].nunique()} stores")

    # Store-level sales -> item-level total daily sales.
    long = selected.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype(np.float32)
    if long["sales"].isna().any() or (long["sales"] < 0).any():
        raise ValueError("Sales contain missing values or negative values; inspect raw CSV before continuing")
    item_daily = long.groupby(["item_id", "d"], as_index=False)["sales"].sum()
    sales_wide = (
        item_daily.pivot(index="d", columns="item_id", values="sales")
        .reindex(index=day_cols, columns=chosen)
    )
    if sales_wide.isna().any().any():
        bad = int(sales_wide.isna().sum().sum())
        raise ValueError(f"Sales matrix has {bad} missing item/day cells; do not silently fill them with zero")
    sales_raw = sales_wide.to_numpy(dtype=np.float32)  # [T_sales, N_items]

    # Calendar may include the 28 future forecast days. Keep all d rows, sorted by numeric index.
    calendar = calendar.copy()
    calendar["d"] = calendar["d"].astype(str)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar["day_num"] = calendar["d"].map(d_number)
    if calendar["d"].duplicated().any():
        duplicate_days = calendar.loc[calendar["d"].duplicated(), "d"].head().tolist()
        raise ValueError(f"calendar.csv has duplicate d keys, e.g. {duplicate_days}")
    if calendar["day_num"].duplicated().any():
        raise ValueError("calendar.csv contains multiple d labels for the same numeric day")
    calendar = calendar.sort_values("day_num").reset_index(drop=True)
    expected_day_nums = np.arange(1, len(calendar) + 1)
    if not np.array_equal(calendar["day_num"].to_numpy(), expected_day_nums):
        raise ValueError("calendar.csv d values must be contiguous from d_1 without gaps")
    if not set(day_cols).issubset(set(calendar["d"])):
        raise ValueError("Some sales day columns are missing from calendar.csv")
    calendar["wm_yr_wk"] = pd.to_numeric(calendar["wm_yr_wk"], errors="coerce")

    # Calendar encodings: cyclical features are more sensible than feeding category IDs as numbers.
    if "wday" in calendar.columns:
        wday = pd.to_numeric(calendar["wday"], errors="coerce")
    else:
        wday = calendar["date"].dt.dayofweek.astype(float) + 1.0
    wday = wday.fillna(calendar["date"].dt.dayofweek.astype(float) + 1.0)
    calendar["dow_sin"] = np.sin(2.0 * np.pi * (wday.to_numpy() - 1.0) / 7.0)
    calendar["dow_cos"] = np.cos(2.0 * np.pi * (wday.to_numpy() - 1.0) / 7.0)

    month = pd.to_numeric(calendar.get("month", calendar["date"].dt.month), errors="coerce")
    month = month.fillna(calendar["date"].dt.month).to_numpy(dtype=float)
    calendar["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    calendar["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    iso_week = calendar["date"].dt.isocalendar().week.astype(int).to_numpy()
    calendar["week_sin"] = np.sin(2.0 * np.pi * (iso_week - 1.0) / 52.1775)
    calendar["week_cos"] = np.cos(2.0 * np.pi * (iso_week - 1.0) / 52.1775)
    calendar["year"] = pd.to_numeric(calendar.get("year", calendar["date"].dt.year), errors="coerce")
    calendar["year"] = calendar["year"].fillna(calendar["date"].dt.year).astype(float)

    # wm_yr_wk is primarily the price-join key. This ordered index uses its chronology without
    # treating a code like 11552 -> 11601 as a large numeric jump.
    unique_weeks = sorted(calendar["wm_yr_wk"].dropna().unique().tolist())
    week_to_idx = {week: i for i, week in enumerate(unique_weeks)}
    calendar["retail_week_index"] = calendar["wm_yr_wk"].map(week_to_idx).astype(float)

    # Preserve the original region flags, plus an aggregated SNAP share aligned with the item-level target.
    for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
        if snap_col not in calendar:
            calendar[snap_col] = 0.0
        calendar[snap_col] = pd.to_numeric(calendar[snap_col], errors="coerce").fillna(0.0).astype(float)

    store_state = selected[["store_id", "state_id"]].drop_duplicates()
    if store_state["store_id"].duplicated().any():
        raise ValueError("A store maps to more than one state in the selected sales rows")
    state_store_counts = store_state["state_id"].value_counts().to_dict()
    n_stores = max(1, len(store_state))
    calendar["snap_active_store_share"] = 0.0
    snap_cols_by_state = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    for state, snap_col in snap_cols_by_state.items():
        calendar["snap_active_store_share"] += (
            state_store_counts.get(state, 0) / n_stores
        ) * calendar[snap_col]

    # Known calendar events are one-hot encoded. These are covariates, not sales labels.
    calendar, event_dummy_cols = make_event_dummies(calendar)

    # Item-store pairs ensure prices are only aggregated over the stores whose sales were aggregated.
    item_store = selected[["item_id", "store_id"]].drop_duplicates()
    prices = prices.copy()
    prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], errors="coerce")
    prices_sub = prices.merge(item_store, on=["item_id", "store_id"], how="inner", validate="many_to_one")
    price_key = ["item_id", "store_id", "wm_yr_wk"]
    if prices_sub.duplicated(price_key).any():
        raise ValueError("Duplicate rows found in sell_prices for item/store/week")
    prices_sub["sell_price"] = pd.to_numeric(prices_sub["sell_price"], errors="coerce")
    store_count_by_item = item_store.groupby("item_id")["store_id"].nunique()
    price_weekly = (
        prices_sub.groupby(["item_id", "wm_yr_wk"], as_index=False)
        .agg(
            sell_price_mean=("sell_price", "mean"),
            sell_price_std=("sell_price", "std"),
            priced_store_count=("sell_price", "count"),
        )
    )
    price_weekly["expected_store_count"] = price_weekly["item_id"].map(store_count_by_item).astype(float)
    price_weekly["price_coverage"] = (
        price_weekly["priced_store_count"] / price_weekly["expected_store_count"].clip(lower=1.0)
    ).clip(0.0, 1.0)
    # A single available price gives undefined std; std=0 plus low coverage conveys that uncertainty.
    price_weekly["sell_price_std"] = price_weekly["sell_price_std"].fillna(0.0)

    # Cross product: one feature row per selected item and every calendar day (including future dates).
    calendar_feature_cols = [
        "d", "date", "day_num", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
        "retail_week_index", "snap_CA", "snap_TX", "snap_WI", "snap_active_store_share",
        *event_dummy_cols,
    ]
    calendar_feature_cols = [c for c in calendar_feature_cols if c in calendar.columns]
    cal_features = calendar[calendar_feature_cols].copy()
    item_df = pd.DataFrame({"item_id": chosen})
    item_df["_key"] = 1
    cal_features["_key"] = 1
    feature_panel = item_df.merge(cal_features, on="_key", how="inner").drop(columns="_key")

    # Recalculate SNAP share per item using only the stores contributing to that item's sales.
    # This is more accurate than applying one global state mix to every item.
    item_store_state = selected[["item_id", "store_id", "state_id"]].drop_duplicates()
    item_store_counts = item_store_state.groupby("item_id")["store_id"].nunique().clip(lower=1)
    feature_panel["snap_active_store_share"] = 0.0
    for state, snap_col in {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}.items():
        state_counts = (
            item_store_state.loc[item_store_state["state_id"] == state]
            .groupby("item_id")["store_id"].nunique()
        )
        state_share = state_counts.div(item_store_counts).to_dict()
        feature_panel["snap_active_store_share"] += (
            feature_panel["item_id"].map(state_share).fillna(0.0).to_numpy()
            * feature_panel[snap_col].to_numpy(dtype=np.float32)
        )

    feature_panel = feature_panel.merge(
        price_weekly[["item_id", "wm_yr_wk", "sell_price_mean", "sell_price_std", "price_coverage"]],
        on=["item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    feature_panel = feature_panel.merge(
        item_daily, on=["item_id", "d"], how="left", validate="one_to_one"
    )
    item_order = {item: i for i, item in enumerate(chosen)}
    feature_panel["item_order"] = feature_panel["item_id"].map(item_order)
    feature_panel = feature_panel.sort_values(["day_num", "item_order"]).reset_index(drop=True)

    # Train/val/test boundaries are computed on sales dates only; no future targets are fabricated.
    n_sales_days = len(day_cols)
    train_end = int(n_sales_days * TRAIN_RATIO)
    val_end = int(n_sales_days * (TRAIN_RATIO + VAL_RATIO))
    if train_end < HISTORY + HORIZON:
        raise ValueError("Training split must contain at least HISTORY + HORIZON days")
    if val_end - train_end < HORIZON:
        raise ValueError("Validation split must contain at least one full forecast horizon")
    if n_sales_days - val_end < HORIZON:
        raise ValueError("Test split must contain at least one full forecast horizon")

    # Price feature handling is causal: forward-fill only from earlier dates, never backward-fill.
    # price_coverage remains the observed coverage on that day/week; price_age_days tells the
    # model how stale the carried-forward price is. If an item has no prior known price, use a
    # neutral zero sentinel together with coverage=0 and a capped age sentinel.
    model_frame = feature_panel.copy()
    model_frame["price_coverage"] = pd.to_numeric(
        model_frame["price_coverage"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    model_frame["sell_price_mean"] = pd.to_numeric(
        model_frame["sell_price_mean"], errors="coerce"
    )
    model_frame["sell_price_std"] = pd.to_numeric(
        model_frame["sell_price_std"], errors="coerce"
    )
    model_frame = model_frame.sort_values(["item_id", "day_num"]).reset_index(drop=True)

    last_price_day = model_frame["day_num"].where(model_frame["price_coverage"] > 0.0)
    last_price_day = last_price_day.groupby(model_frame["item_id"]).ffill()
    model_frame["price_age_days"] = (
        model_frame["day_num"] - last_price_day
    ).fillna(365.0).clip(0.0, 365.0)

    for col in ["sell_price_mean", "sell_price_std"]:
        model_frame[col] = model_frame.groupby("item_id", sort=False)[col].ffill()
        model_frame[col] = model_frame[col].fillna(0.0).clip(lower=0.0)
    model_frame["log_sell_price_mean"] = np.log1p(model_frame["sell_price_mean"])
    model_frame["log_sell_price_std"] = np.log1p(model_frame["sell_price_std"])
    model_frame["log_price_age_days"] = np.log1p(model_frame["price_age_days"])

    # Save a transparent panel. sales is NaN after the final observed sales day by design;
    # the price columns show observed/causally carried values, while future inputs are rebuilt
    # at each forecast origin by future_covariates_as_of().
    panel_export_cols = [c for c in model_frame.columns if c != "item_order"]
    model_frame[panel_export_cols].to_csv(OUT_DIR / "m5_panel_multifeature.csv", index=False)

    # Raw panel is saved for inspection. Future sales are NaN because they are unknown by definition.
    # Actual model covariates. wm_yr_wk joins price data and also informs retail_week_index;
    # weekday/wday and month become cyclical features instead of arbitrary integer IDs.
    covariate_cols = [
        "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
        "year", "retail_week_index", "snap_CA", "snap_TX", "snap_WI",
        "snap_active_store_share", "log_sell_price_mean", "log_sell_price_std",
        "price_coverage", "log_price_age_days", *event_dummy_cols,
    ]
    covariate_cols = [c for c in covariate_cols if c in model_frame.columns]
    if not covariate_cols:
        raise ValueError("No covariate columns were constructed")

    model_frame[covariate_cols] = model_frame[covariate_cols].replace([np.inf, -np.inf], np.nan)
    # All non-price covariates should be finite; fill unexpected missing values using TRAIN-only medians.
    train_rows_mask = model_frame["day_num"] <= train_end
    for col in covariate_cols:
        if model_frame[col].isna().any():
            med = model_frame.loc[train_rows_mask, col].median()
            if pd.isna(med):
                med = 0.0
            model_frame[col] = model_frame[col].fillna(float(med))

    # Standardize continuous covariates only; fit statistics on training dates only.
    continuous_cols = [
        c for c in ["year", "retail_week_index", "log_sell_price_mean", "log_sell_price_std", "log_price_age_days"]
        if c in covariate_cols
    ]
    scaler_info = {}
    train_feature_rows = model_frame.loc[train_rows_mask, covariate_cols]
    for col in continuous_cols:
        mu = float(train_feature_rows[col].mean())
        sigma = float(train_feature_rows[col].std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1.0
        model_frame[col] = (model_frame[col] - mu) / sigma
        scaler_info[col] = {"mean": mu, "std": sigma}

    # Re-establish the required day-major ordering after item-wise causal price filling.
    # This is essential before reshaping rows to [day, item, feature].
    model_frame = model_frame.sort_values(["day_num", "item_order"]).reset_index(drop=True)

    # Convert day x item x feature. Ordering is deterministic: day_num then chosen-item order.
    full_calendar_days = calendar["d"].tolist()
    n_all_days = len(full_calendar_days)
    if n_all_days < len(day_cols) + HORIZON:
        raise ValueError(
            f"calendar.csv has {n_all_days - len(day_cols)} future day(s), but HORIZON={HORIZON}. "
            "Provide calendar rows for all required future dates before training."
        )
    cov = model_frame[covariate_cols].to_numpy(dtype=np.float32).reshape(
        n_all_days, len(chosen), len(covariate_cols)
    )
    # Check the reshape ordering and alignment explicitly.
    if model_frame.groupby("day_num").size().nunique() != 1 or model_frame.groupby("day_num").size().iloc[0] != len(chosen):
        raise ValueError("Feature panel is not a complete day x item grid")

    # Align sales target days with calendar order d_1, d_2, ... (M5 convention).
    day_to_pos = {d: i for i, d in enumerate(full_calendar_days)}
    sales_positions = [day_to_pos[d] for d in day_cols]
    if sales_positions != list(range(len(day_cols))):
        # Do not silently misalign if the calendar starts with unexpected IDs/order.
        raise ValueError("Sales d-columns are not aligned to the first chronological calendar positions")

    # log1p target and fit target scaler on train dates only, per item.
    sales_log = np.log1p(np.maximum(sales_raw, 0.0)).astype(np.float32)
    y_mean = sales_log[:train_end].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = sales_log[:train_end].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    sales_scaled = ((sales_log - y_mean) / y_std).astype(np.float32)

    # Forecast origins are first target indices: history is [origin-HISTORY, origin), target [origin, origin+HORIZON).
    max_origin = n_sales_days - HORIZON
    train_origins = list(range(HISTORY, train_end - HORIZON + 1))
    val_origins = list(range(train_end, val_end - HORIZON + 1, EVAL_STRIDE))
    test_origins = list(range(val_end, max_origin + 1, EVAL_STRIDE))
    last_val_origin = val_end - HORIZON
    if last_val_origin >= train_end and last_val_origin not in val_origins:
        val_origins.append(last_val_origin)
    if not test_origins or test_origins[-1] != max_origin:
        test_origins.append(max_origin)
    val_origins = sorted(set(o for o in val_origins if train_end <= o <= val_end - HORIZON))
    test_origins = sorted(set(o for o in test_origins if val_end <= o <= max_origin))
    if not train_origins or not val_origins or not test_origins:
        raise ValueError("Empty train/validation/test origins; check split settings")

    dates_sales = pd.to_datetime(calendar.set_index("d").loc[day_cols, "date"]).reset_index(drop=True)
    print(f"Items={len(chosen)}, known sales days={n_sales_days}, all calendar days={n_all_days}")
    print(f"Features ({len(covariate_cols)}): {covariate_cols}")
    print(f"Origins: train={len(train_origins)}, val={len(val_origins)}, test={len(test_origins)}")

    return {
        "chosen": chosen,
        "sales_raw": sales_raw,
        "sales_scaled": sales_scaled,
        "cov": cov,
        "covariate_cols": covariate_cols,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_origins": train_origins,
        "val_origins": val_origins,
        "test_origins": test_origins,
        "dates_sales": dates_sales,
        "dates_calendar": pd.to_datetime(calendar["date"]).reset_index(drop=True),
        "calendar_days": full_calendar_days,
        "day_cols": day_cols,
        "scaler_info": scaler_info,
        "price_feature_indices": [covariate_cols.index(c) for c in [
            "log_sell_price_mean", "log_sell_price_std", "price_coverage"
        ] if c in covariate_cols],
        "price_coverage_index": covariate_cols.index("price_coverage") if "price_coverage" in covariate_cols else None,
        "price_age_index": covariate_cols.index("log_price_age_days") if "log_price_age_days" in covariate_cols else None,
        "price_age_scaler": scaler_info.get("log_price_age_days"),
        "train_end": train_end,
        "val_end": val_end,
        "n_sales_days": n_sales_days,
    }


def future_covariates_as_of(
    cov: np.ndarray,
    origin: int,
    horizon: int,
    price_feature_indices: list[int],
    price_coverage_index: int | None,
    price_age_index: int | None = None,
    price_age_scaler: dict | None = None,
) -> np.ndarray:
    """Use known calendar covariates and price data available strictly before origin.

    Target-week observed prices in the complete M5 CSV are deliberately discarded. For each
    item, the latest pre-origin price with positive coverage is carried forward. The carried
    price's historical coverage is preserved and its age is increased for each forecast day.
    """
    if origin < 0 or horizon <= 0:
        raise ValueError("origin must be non-negative and horizon must be positive")
    if origin + horizon > len(cov):
        raise ValueError(
            f"Calendar covariates end at {len(cov)} days, but origin={origin} and "
            f"horizon={horizon} require {origin + horizon} days. Extend calendar.csv."
        )
    future = cov[origin:origin + horizon].copy()
    if not price_feature_indices or price_coverage_index is None:
        return future

    # With no prior history, mask price values instead of leaking any target-week price.
    if origin == 0:
        future[:, :, price_feature_indices] = 0.0
        future[:, :, price_coverage_index] = 0.0
        if price_age_index is not None:
            log_age = np.full(horizon, np.log1p(365.0), dtype=np.float32)
            if price_age_scaler:
                log_age = (log_age - float(price_age_scaler["mean"])) / float(price_age_scaler["std"])
            future[:, :, price_age_index] = log_age[:, None]
        return future

    history = cov[:origin]
    for item_idx in range(cov.shape[1]):
        coverage_history = history[:, item_idx, price_coverage_index]
        valid = np.flatnonzero(coverage_history > 0.0)
        if len(valid):
            source_day = int(valid[-1])
            source_features = history[source_day, item_idx, price_feature_indices].copy()
            source_coverage = float(history[source_day, item_idx, price_coverage_index])
            ages = np.minimum(365.0, origin + np.arange(horizon, dtype=np.float32) - source_day)
        else:
            # Price has never been observed in the available history: retain a masked sentinel.
            source_day = origin - 1
            source_features = history[source_day, item_idx, price_feature_indices].copy()
            source_coverage = 0.0
            ages = np.full(horizon, 365.0, dtype=np.float32)

        if PRICE_MASK and source_coverage < 0.5:
            source_features = np.zeros_like(source_features)
            source_coverage = 0.0
        future[:, item_idx, price_feature_indices] = source_features[None, :]
        future[:, item_idx, price_coverage_index] = source_coverage

        if price_age_index is not None:
            log_age = np.log1p(ages).astype(np.float32)
            if price_age_scaler:
                scale = max(float(price_age_scaler["std"]), 1e-6)
                log_age = (log_age - float(price_age_scaler["mean"])) / scale
            future[:, item_idx, price_age_index] = log_age
    return future


class WindowDataset(Dataset):
    def __init__(
        self,
        sales_scaled: np.ndarray,
        cov: np.ndarray,
        origins: Iterable[int],
        history: int,
        horizon: int,
        price_feature_indices: list[int] | None = None,
        price_coverage_index: int | None = None,
        price_age_index: int | None = None,
        price_age_scaler: dict | None = None,
    ):
        self.sales = np.asarray(sales_scaled, dtype=np.float32)
        self.cov = np.asarray(cov, dtype=np.float32)
        self.origins = list(origins)
        self.history = history
        self.horizon = horizon
        self.price_feature_indices = price_feature_indices or []
        self.price_coverage_index = price_coverage_index
        self.price_age_index = price_age_index
        self.price_age_scaler = price_age_scaler

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int):
        origin = self.origins[i]
        x_sales = torch.from_numpy(
            self.sales[origin - self.history:origin].copy()
        )  # [T_hist, N]
        x_cov_hist = torch.from_numpy(
            self.cov[origin - self.history:origin].copy()
        )  # [T_hist, N, F]
        y_sales = torch.from_numpy(
            self.sales[origin:origin + self.horizon].copy()
        )  # [H, N]
        future_cov = future_covariates_as_of(
            self.cov,
            origin,
            self.horizon,
            self.price_feature_indices,
            self.price_coverage_index,
            self.price_age_index,
            self.price_age_scaler,
        )
        x_cov_future = torch.from_numpy(future_cov)  # [H, N, F]
        return x_sales, x_cov_hist, y_sales, x_cov_future


class PatchTSTThenITransformer(nn.Module):
    """Patch temporal encoder -> cross-item attention -> horizon-to-history attention.

    Each item is encoded as a sequence of temporal patches. An iTransformer-style
    cross-item attention block models relationships between item tokens. Every forecast
    horizon then attends to the item's full set of encoded history patches, preventing
    the sales-only baseline from collapsing all historical temporal detail to one vector.
    This is iTransformer-inspired, not a literal reproduction of the original model.
    """
    def __init__(
        self,
        history: int,
        horizon: int,
        n_covariates: int,
        use_covariates: bool,
        patch_len: int = PATCH_LEN,
        patch_stride: int = PATCH_STRIDE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        temporal_layers: int = N_TEMPORAL_LAYERS,
        dropout: float = DROPOUT,
        cov_fusion: str = COV_FUSION,
        n_items: int | None = None,
        item_embed: bool = ITEM_EMBED,
        arch: str = ARCH,
    ):
        super().__init__()
        if history < patch_len:
            raise ValueError("history must be >= patch_len")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.history = history
        self.horizon = horizon
        self.use_covariates = use_covariates
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.cov_fusion = cov_fusion
        self.item_embed_flag = bool(item_embed)
        self.arch = arch
        self.n_input_channels = 1 + (n_covariates if use_covariates else 0)
        n_patches = 1 + (history - patch_len) // patch_stride

        self.patch_proj = nn.Linear(patch_len * self.n_input_channels, d_model)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1, n_patches, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.item_norm1 = nn.LayerNorm(d_model)
        self.item_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.item_norm2 = nn.LayerNorm(d_model)
        self.item_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        if self.item_embed_flag and n_items is not None:
            self.item_embed = nn.Embedding(n_items, d_model)
        else:
            self.item_embed = None

        self.horizon_embedding = nn.Parameter(torch.randn(1, horizon, 1, d_model) * 0.02)
        self.history_query_norm = nn.LayerNorm(d_model)
        self.history_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.history_attn_norm = nn.LayerNorm(d_model)

        # v2 only: horizon self-attention lets the 28 forecast days exchange information
        # (retail daily sales have strong intra-window autocorrelation that v1 ignored).
        self.horizon_query_norm = nn.LayerNorm(d_model)
        self.horizon_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.horizon_norm = nn.LayerNorm(d_model)

        self.cov_gate = None
        self.cov_proj2 = None
        if use_covariates:
            self.future_cov_proj = nn.Sequential(
                nn.Linear(n_covariates, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if cov_fusion == "gate":
                self.cov_gate = nn.Linear(2 * d_model, d_model)
            elif cov_fusion == "proj":
                self.cov_proj2 = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
            else:
                self.cov_gate = None
                self.cov_proj2 = None
        else:
            self.future_cov_proj = None
            self.cov_gate = None
            self.cov_proj2 = None
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _condition_query(self, query: torch.Tensor, future_emb: torch.Tensor | None) -> torch.Tensor:
        """Fuse future covariates exactly once, before history attention."""
        if future_emb is None:
            return query
        if self.cov_fusion == "gate":
            gate = torch.sigmoid(self.cov_gate(torch.cat([query, future_emb], dim=-1)))
            update = gate * future_emb
        elif self.cov_fusion == "proj":
            update = self.cov_proj2(future_emb)
        else:
            update = future_emb
        return query + update

    def forward(
        self,
        x_sales: torch.Tensor,
        x_cov_hist: torch.Tensor | None = None,
        x_cov_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_sales: [B,T,N]; x_cov_hist: [B,T,N,F]; x_cov_future: [B,H,N,F]
        if self.use_covariates:
            if x_cov_hist is None or x_cov_future is None:
                raise ValueError("Covariate model requires historical and future covariate tensors")
            x = torch.cat([x_sales.unsqueeze(-1), x_cov_hist], dim=-1)
        else:
            x = x_sales.unsqueeze(-1)

        # Channel-independent temporal patches per item (shared by v1 and v2).
        x = x.permute(0, 2, 1, 3).contiguous()  # [B,N,T,C]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()  # [B,N,P,L,C]
        b, n, p, patch_len, channels = patches.shape
        patches = patches.reshape(b, n, p, patch_len * channels)
        patch_tokens = self.patch_proj(patches) + self.temporal_pos[:, :, :p, :]
        patch_tokens = patch_tokens.reshape(b * n, p, -1)
        patch_tokens = self.temporal_norm(self.temporal_encoder(patch_tokens))
        patch_tokens = patch_tokens.reshape(b, n, p, -1)  # [B,N,P,D]

        if self.arch == "v1":
            # ---- original architecture (kept for A/B comparison) ----
            # A pooled item representation is used only for cross-item exchange.
            item_repr = patch_tokens.mean(dim=2)  # [B,N,D]
            if self.item_embed is not None:
                item_ids = torch.arange(n, device=item_repr.device).unsqueeze(0).expand(b, -1)
                item_repr = item_repr + self.item_embed(item_ids)
            q = self.item_norm1(item_repr)
            attended, _ = self.item_attention(q, q, q, need_weights=False)
            item_repr = item_repr + attended
            item_repr = item_repr + self.item_ffn(self.item_norm2(item_repr))

            query = item_repr.unsqueeze(1) + self.horizon_embedding[:, :self.horizon]  # [B,H,N,D]
            future_emb = self.future_cov_proj(x_cov_future) if self.use_covariates else None
            query = self._condition_query(query, future_emb)
            query_bn = query.permute(0, 2, 1, 3).reshape(b * n, self.horizon, -1)
            history_bn = patch_tokens.reshape(b * n, p, -1)
            history_attended, _ = self.history_attention(
                self.history_query_norm(query_bn), history_bn, history_bn, need_weights=False
            )
            query_bn = self.history_attn_norm(query_bn + history_attended)
            rep = query_bn.reshape(b, n, self.horizon, -1).permute(0, 2, 1, 3).contiguous()
            pred = self.forecast_head(rep).squeeze(-1)  # [B,H,N]
            return pred

        # ---- v2 (fixed) ----
        # Inject item identity into every patch so cross-item mixing knows which item.
        if self.item_embed is not None:
            item_ids = torch.arange(n, device=patch_tokens.device).unsqueeze(0).expand(b, -1)
            patch_tokens = patch_tokens + self.item_embed(item_ids).unsqueeze(2)  # [B,N,1,D]

        # Cross-item attention performed PER PATCH: this preserves the temporal axis
        # instead of collapsing the whole series to a mean before item mixing.
        pt = patch_tokens.permute(0, 2, 1, 3).reshape(b * p, n, -1)  # [B*P, N, D]
        att, _ = self.item_attention(self.item_norm1(pt), pt, pt, need_weights=False)
        pt = pt + att
        pt = pt + self.item_ffn(self.item_norm2(pt))
        enriched = pt.reshape(b, p, n, -1).permute(0, 2, 1, 3).contiguous()  # [B,N,P,D]

        # Future covariates condition the look-back query (future-aware attention),
        # instead of being added as a late, weakly-coupled residual.
        if self.use_covariates:
            future_emb = self.future_cov_proj(x_cov_future)  # [B,H,N,D]
        else:
            future_emb = None

        query = self.horizon_embedding[:, :self.horizon].expand(b, self.horizon, n, -1).contiguous()  # [B,H,N,D]
        if self.item_embed is not None:
            query = query + self.item_embed(item_ids).unsqueeze(1)  # [B,N,1,D] -> [B,H,N,D]
        query = self._condition_query(query, future_emb)
        query_bn = query.permute(0, 2, 1, 3).reshape(b * n, self.horizon, -1)  # [B*N,H,D]
        history_bn = enriched.reshape(b * n, p, -1)
        history_attended, _ = self.history_attention(
            self.history_query_norm(query_bn), history_bn, history_bn, need_weights=False
        )
        query_bn = self.history_attn_norm(query_bn + history_attended)

        # Horizon self-attention: let the 28 forecast days exchange information, so the
        # intra-window autocorrelation (weekends, adjacent-day inertia) is modelled.
        h_att, _ = self.horizon_attention(
            self.horizon_query_norm(query_bn), query_bn, query_bn, need_weights=False
        )
        query_bn = self.horizon_norm(query_bn + h_att)

        rep = query_bn.reshape(b, n, self.horizon, -1).permute(0, 2, 1, 3).contiguous()  # [B,H,N,D]

        pred = self.forecast_head(rep).squeeze(-1)  # [B,H,N]
        return pred

def inverse_target(x: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """From standardized log1p target back to non-negative sales."""
    log_values = x * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    return np.maximum(np.expm1(log_values), 0.0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_covariates: bool,
    device: torch.device,
    model_path: Path,
    loss_type: str = LOSS_TYPE,
    lr_schedule: bool = LR_SCHEDULE,
    grad_clip: float = GRAD_CLIP,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, EPOCHS), eta_min=LEARNING_RATE * 0.01
        )
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x_sales, x_cov_hist, y_sales, x_cov_future in train_loader:
            x_sales = x_sales.to(device)
            y_sales = y_sales.to(device)
            if use_covariates:
                x_cov_hist = x_cov_hist.to(device)
                x_cov_future = x_cov_future.to(device)
                pred = model(x_sales, x_cov_hist, x_cov_future)
            else:
                pred = model(x_sales)
            loss = criterion(pred, y_sales)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_sales, x_cov_hist, y_sales, x_cov_future in val_loader:
                x_sales = x_sales.to(device)
                y_sales = y_sales.to(device)
                if use_covariates:
                    pred = model(x_sales, x_cov_hist.to(device), x_cov_future.to(device))
                else:
                    pred = model(x_sales)
                val_losses.append(float(criterion(pred, y_sales).cpu()))
        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        print(f"  epoch {epoch:03d} | train={mean_train:.5f} | val={mean_val:.5f}")

        if mean_val < best_val - 1e-5:
            best_val = mean_val
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping at epoch {epoch}; best validation={best_val:.5f}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    return model


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    use_covariates: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for x_sales, x_cov_hist, y_sales, x_cov_future in loader:
        x_sales = x_sales.to(device)
        if use_covariates:
            pred = model(x_sales, x_cov_hist.to(device), x_cov_future.to(device))
        else:
            pred = model(x_sales)
        preds.append(pred.cpu().numpy())
        trues.append(y_sales.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def metric_row(
    name: str,
    evaluation: str,
    pred_z: np.ndarray,
    true_z: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict:
    pred = inverse_target(pred_z, y_mean, y_std)
    true = inverse_target(true_z, y_mean, y_std)
    error = pred - true
    wape = float(np.abs(error).sum() / max(np.abs(true).sum(), 1e-8))
    return {
        "model": name,
        "evaluation": evaluation,
        "n_windows": int(pred.shape[0]),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE": wape,
    }


def metric_row_sales(
    name: str,
    evaluation: str,
    pred_sales: np.ndarray,
    true_sales: np.ndarray,
) -> dict:
    """Metrics for already inverse-transformed, non-negative sales predictions."""
    pred_sales = np.maximum(np.asarray(pred_sales, dtype=np.float64), 0.0)
    true_sales = np.maximum(np.asarray(true_sales, dtype=np.float64), 0.0)
    error = pred_sales - true_sales
    return {
        "model": name,
        "evaluation": evaluation,
        "n_windows": int(pred_sales.shape[0]),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE": float(np.abs(error).sum() / max(np.abs(true_sales).sum(), 1e-8)),
    }


def make_train_loader(dataset: Dataset, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


@torch.no_grad()
def predict_future_28_days(
    model: nn.Module,
    data: dict,
    device: torch.device,
) -> np.ndarray:
    """Predict the 28 days after the final day with observed sales."""
    origin = data["n_sales_days"]
    if origin < HISTORY:
        raise ValueError("Not enough observed sales history for the future forecast")
    if origin + HORIZON > len(data["cov"]):
        raise ValueError(
            f"calendar.csv contains only {len(data['cov']) - origin} days after the last observed "
            f"sales day; {HORIZON} future days are required. Use a calendar with at least "
            f"{HORIZON} additional dates."
        )

    x_sales = torch.from_numpy(
        data["sales_scaled"][origin - HISTORY:origin].copy()
    ).unsqueeze(0).to(device)
    x_cov_hist = torch.from_numpy(
        data["cov"][origin - HISTORY:origin].copy()
    ).unsqueeze(0).to(device)
    future_cov = future_covariates_as_of(
        data["cov"],
        origin,
        HORIZON,
        data["price_feature_indices"],
        data["price_coverage_index"],
        data["price_age_index"],
        data["price_age_scaler"],
    )
    x_cov_future = torch.from_numpy(future_cov).unsqueeze(0).to(device)

    model.eval()
    if model.use_covariates:
        pred_z = model(x_sales, x_cov_hist, x_cov_future).cpu().numpy()
    else:
        pred_z = model(x_sales).cpu().numpy()
    return inverse_target(pred_z, data["y_mean"], data["y_std"])[0]


def main() -> None:
    seed_everything(SEED)
    data = load_and_prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        "Future price policy: each forecast origin uses the last price information available "
        "before that origin; target-week realized prices are not supplied to the model."
    )

    dataset_args = {
        "price_feature_indices": data["price_feature_indices"],
        "price_coverage_index": data["price_coverage_index"],
        "price_age_index": data["price_age_index"],
        "price_age_scaler": data["price_age_scaler"],
    }
    train_ds = WindowDataset(
        data["sales_scaled"], data["cov"], data["train_origins"], HISTORY, HORIZON, **dataset_args
    )
    val_ds = WindowDataset(
        data["sales_scaled"], data["cov"], data["val_origins"], HISTORY, HORIZON, **dataset_args
    )
    test_ds = WindowDataset(
        data["sales_scaled"], data["cov"], data["test_origins"], HISTORY, HORIZON, **dataset_args
    )

    # Additional evaluation with non-overlapping 28-day windows. This avoids repeatedly
    # counting the same target date, though the final remainder may be left unevaluated.
    max_origin = data["n_sales_days"] - HORIZON
    nonoverlap_origins = list(range(data["val_end"], max_origin + 1, HORIZON))
    if not nonoverlap_origins:
        raise ValueError("No non-overlapping test window is available for the selected split")
    nonoverlap_ds = WindowDataset(
        data["sales_scaled"], data["cov"], nonoverlap_origins, HISTORY, HORIZON, **dataset_args
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    nonoverlap_loader = DataLoader(nonoverlap_ds, batch_size=BATCH_SIZE, shuffle=False)

    model_specs = [("sales_only", False), ("multi_feature", True)]
    metrics: list[dict] = []
    outputs: dict[str, np.ndarray] = {}
    nonoverlap_outputs: dict[str, np.ndarray] = {}
    models: dict[str, nn.Module] = {}
    test_true_z: np.ndarray | None = None
    nonoverlap_true_z: np.ndarray | None = None

    for name, use_cov in model_specs:
        print(f"\nTraining {name} (use_covariates={use_cov})")
        # Reset model initialization RNG and use the same deterministic shuffle sequence per model.
        seed_everything(SEED)
        model = PatchTSTThenITransformer(
            history=HISTORY,
            horizon=HORIZON,
            n_covariates=len(data["covariate_cols"]),
            use_covariates=use_cov,
            cov_fusion=COV_FUSION,
            n_items=len(data["chosen"]),
            item_embed=ITEM_EMBED,
            arch=ARCH,
        )
        train_loader = make_train_loader(train_ds, SEED)
        model = train_model(
            model,
            train_loader,
            val_loader,
            use_cov,
            device,
            OUT_DIR / f"{name}_best.pt",
            loss_type=LOSS_TYPE,
            lr_schedule=LR_SCHEDULE,
            grad_clip=GRAD_CLIP,
        )
        models[name] = model

        pred_z, true_z_i = predict_dataset(model, test_loader, use_cov, device)
        if test_true_z is None:
            test_true_z = true_z_i.copy()
        else:
            np.testing.assert_allclose(test_true_z, true_z_i, rtol=0.0, atol=1e-7)
        outputs[name] = pred_z
        metrics.append(
            metric_row(name, "rolling_overlapping", pred_z, true_z_i, data["y_mean"], data["y_std"])
        )

        pred_non, true_non_i = predict_dataset(model, nonoverlap_loader, use_cov, device)
        if nonoverlap_true_z is None:
            nonoverlap_true_z = true_non_i.copy()
        else:
            np.testing.assert_allclose(nonoverlap_true_z, true_non_i, rtol=0.0, atol=1e-7)
        nonoverlap_outputs[name] = pred_non
        metrics.append(
            metric_row(name, "non_overlapping", pred_non, true_non_i, data["y_mean"], data["y_std"])
        )

    if ENSEMBLE and "sales_only" in outputs and "multi_feature" in outputs:
        # Blend forecasts in the original sales scale, not in standardized log space.
        sales_only_sales = inverse_target(outputs["sales_only"], data["y_mean"], data["y_std"])
        multi_feature_sales = inverse_target(outputs["multi_feature"], data["y_mean"], data["y_std"])
        true_test_sales = inverse_target(test_true_z, data["y_mean"], data["y_std"])
        ens_sales = 0.5 * (sales_only_sales + multi_feature_sales)
        metrics.append(metric_row_sales("ensemble", "rolling_overlapping", ens_sales, true_test_sales))

        sales_only_non = inverse_target(nonoverlap_outputs["sales_only"], data["y_mean"], data["y_std"])
        multi_feature_non = inverse_target(nonoverlap_outputs["multi_feature"], data["y_mean"], data["y_std"])
        true_non_sales = inverse_target(nonoverlap_true_z, data["y_mean"], data["y_std"])
        ens_non_sales = 0.5 * (sales_only_non + multi_feature_non)
        metrics.append(metric_row_sales("ensemble", "non_overlapping", ens_non_sales, true_non_sales))

    assert test_true_z is not None and nonoverlap_true_z is not None
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\nTest metrics (rolling windows may overlap; non_overlapping rows use disjoint 28-day windows):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Plot and export latest historical test window. These rows include actuals and are not
    # the same as the genuinely future 28-day forecasts exported below.
    last_origin = data["test_origins"][-1]
    pred_last = {
        name: inverse_target(pred_arr[-1:], data["y_mean"], data["y_std"])[0]
        for name, pred_arr in outputs.items()
    }
    if ENSEMBLE and "sales_only" in pred_last and "multi_feature" in pred_last:
        pred_last["ensemble"] = 0.5 * (pred_last["sales_only"] + pred_last["multi_feature"])
    true_last = inverse_target(test_true_z[-1:], data["y_mean"], data["y_std"])[0]
    forecast_dates = pd.to_datetime(
        data["dates_sales"].iloc[last_origin:last_origin + HORIZON]
    ).to_list()
    chosen = data["chosen"]

    rows = []
    for h, dt in enumerate(forecast_dates):
        for j, item in enumerate(chosen):
            row = {"date": dt, "item_id": item, "actual_sales": float(true_last[h, j])}
            for name in pred_last:
                row[f"{name}_forecast"] = float(pred_last[name][h, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "latest_test_28day_forecast.csv", index=False)

    n_plot = min(6, len(chosen))
    fig, axes = plt.subplots(n_plot, 1, figsize=(13, max(3, 2.7 * n_plot)), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(forecast_dates, true_last[:, j], label="Actual", linewidth=2)
        ax.plot(forecast_dates, pred_last["sales_only"][:, j], label="Sales-only baseline", linestyle="--")
        ax.plot(forecast_dates, pred_last["multi_feature"][:, j], label="Multi-feature hybrid", linestyle="-")
        if "ensemble" in pred_last:
            ax.plot(forecast_dates, pred_last["ensemble"][:, j], label="Ensemble", linestyle=":")
        ax.set_title(str(chosen[j]))
        ax.set_ylabel("Sales")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(ncol=4 if "ensemble" in pred_last else 3, fontsize=9)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Latest historical test window: sales-only vs multi-feature hybrid", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_six_items.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(forecast_dates, true_last.sum(axis=1), label="Actual total", linewidth=2)
    ax.plot(forecast_dates, pred_last["sales_only"].sum(axis=1), label="Sales-only baseline", linestyle="--")
    ax.plot(forecast_dates, pred_last["multi_feature"].sum(axis=1), label="Multi-feature hybrid")
    if "ensemble" in pred_last:
        ax.plot(forecast_dates, pred_last["ensemble"].sum(axis=1), label="Ensemble", linestyle=":")
    ax.set_title("Aggregate sales of sampled items — latest historical test window")
    ax.set_ylabel("Total sales")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Actual future forecast beyond the last observed sales date, with no future sales labels.
    future_dates = pd.to_datetime(
        data["dates_calendar"].iloc[data["n_sales_days"]:data["n_sales_days"] + HORIZON]
    ).to_list()
    future_predictions = {
        name: predict_future_28_days(model, data, device)
        for name, model in models.items()
    }
    if ENSEMBLE and "sales_only" in future_predictions and "multi_feature" in future_predictions:
        future_predictions["ensemble"] = 0.5 * (
            future_predictions["sales_only"] + future_predictions["multi_feature"]
        )
    future_rows = []
    for h, dt in enumerate(future_dates):
        for j, item in enumerate(chosen):
            row = {"date": dt, "item_id": item}
            for name in future_predictions:
                row[f"{name}_forecast"] = float(future_predictions[name][h, j])
            future_rows.append(row)
    pd.DataFrame(future_rows).to_csv(OUT_DIR / "future_28day_forecast.csv", index=False)

    fig, axes = plt.subplots(n_plot, 1, figsize=(13, max(3, 2.7 * n_plot)), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(future_dates, future_predictions["sales_only"][:, j], label="Sales-only baseline", linestyle="--")
        ax.plot(future_dates, future_predictions["multi_feature"][:, j], label="Multi-feature hybrid")
        if "ensemble" in future_predictions:
            ax.plot(future_dates, future_predictions["ensemble"][:, j], label="Ensemble", linestyle=":")
        ax.set_title(str(chosen[j]))
        ax.set_ylabel("Forecast sales")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(ncol=3 if "ensemble" in future_predictions else 2, fontsize=9)
    axes[-1].set_xlabel("Future date")
    fig.suptitle("Forecast for the 28 days after the last observed sales date", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "future_28day_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Store run metadata and feature scaling values for auditable inference/reproduction.
    metadata = {
        "seed": SEED,
        "sampled_items": data["chosen"],
        "history": HISTORY,
        "horizon": HORIZON,
        "train_end_index_exclusive": data["train_end"],
        "validation_end_index_exclusive": data["val_end"],
        "n_sales_days": data["n_sales_days"],
        "covariate_columns": data["covariate_cols"],
        "covariate_scalers": data["scaler_info"],
        "target_scaler": {
            "mean_by_item": data["y_mean"].reshape(-1).tolist(),
            "std_by_item": data["y_std"].reshape(-1).tolist(),
        },
        "future_price_policy": "carry forward only the latest price features available before each forecast origin; target-week realized prices are not passed; log_price_age_days records staleness",
        "price_aggregation": "simple mean/std over selected stores with observed prices; not sales-weighted",
        "covariate_fusion": COV_FUSION,
        "architecture": ARCH,
        "item_embedding": ITEM_EMBED,
        "test_metrics_note": "rolling_overlapping metrics count repeated target dates; non_overlapping metrics use disjoint 28-day windows",
    }
    with (OUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")
    print("  m5_panel_multifeature.csv")
    print("  metrics.csv")
    print("  latest_test_28day_forecast.csv")
    print("  future_28day_forecast.csv  (actual future forecast; no actual sales labels)")
    print("  forecast_six_items.png / forecast_total.png / future_28day_forecast.png")
    print("  sales_only_best.pt / multi_feature_best.pt / run_metadata.json")


if __name__ == "__main__":
    main()
