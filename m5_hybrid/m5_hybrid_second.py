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
import random
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 1. Configuration
# ============================================================
SEED = 42
N_ITEMS = 50
HISTORY = 112
HORIZON = 28
PATCH_LEN = 16
PATCH_STRIDE = 8
D_MODEL = 64
N_HEADS = 4
N_TEMPORAL_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EVAL_STRIDE = 7  # rolling-origin validation/testing; append final origin as well

# 锚定到脚本所在目录，使调试时无论从哪个 CWD 启动都能正确找到数据/输出
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
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["day_num"] = calendar["d"].map(d_number)
    calendar = calendar.sort_values("day_num").drop_duplicates("d").reset_index(drop=True)
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
    if train_end <= HISTORY + HORIZON or val_end >= n_sales_days:
        raise ValueError("Not enough dates for the configured history/horizon/splits")

    # Price missingness is retained in price_coverage; price mean is imputed using TRAIN-only medians.
    model_frame = feature_panel.copy()
    train_time_mask = model_frame["day_num"] <= train_end
    item_medians = {}
    for item in chosen:
        mask = (model_frame["item_id"] == item) & train_time_mask
        median = model_frame.loc[mask, "sell_price_mean"].median()
        item_medians[item] = float(median) if pd.notna(median) else np.nan
    global_train_median = model_frame.loc[train_time_mask, "sell_price_mean"].median()
    if pd.isna(global_train_median):
        global_train_median = 0.0
    for item in chosen:
        mask = model_frame["item_id"] == item
        fill_value = item_medians[item]
        if not np.isfinite(fill_value):
            fill_value = float(global_train_median)
        model_frame.loc[mask, "sell_price_mean"] = model_frame.loc[mask, "sell_price_mean"].fillna(fill_value)
    model_frame["sell_price_std"] = model_frame["sell_price_std"].fillna(0.0)
    model_frame["price_coverage"] = model_frame["price_coverage"].fillna(0.0).clip(0.0, 1.0)
    model_frame["log_sell_price_mean"] = np.log1p(model_frame["sell_price_mean"].clip(lower=0))
    model_frame["log_sell_price_std"] = np.log1p(model_frame["sell_price_std"].clip(lower=0))

    # Raw panel is saved for inspection. Future sales are NaN because they are unknown by definition.
    feature_panel.drop(columns=["item_order"], errors="ignore").to_csv(
        OUT_DIR / "m5_panel_multifeature.csv", index=False
    )

    # Actual model covariates. wm_yr_wk joins price data and also informs retail_week_index;
    # weekday/wday and month become cyclical features instead of arbitrary integer IDs.
    covariate_cols = [
        "dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos",
        "year", "retail_week_index", "snap_CA", "snap_TX", "snap_WI",
        "snap_active_store_share", "log_sell_price_mean", "log_sell_price_std",
        "price_coverage", *event_dummy_cols,
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
        c for c in ["year", "retail_week_index", "log_sell_price_mean", "log_sell_price_std"]
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

    # Convert day x item x feature. Ordering is deterministic: day_num then chosen-item order.
    full_calendar_days = calendar["d"].tolist()
    n_all_days = len(full_calendar_days)
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
        "day_cols": day_cols,
        "scaler_info": scaler_info,
    }


class WindowDataset(Dataset):
    def __init__(self, sales_scaled: np.ndarray, cov: np.ndarray, origins: Iterable[int], history: int, horizon: int):
        self.sales = torch.from_numpy(sales_scaled.astype(np.float32))
        self.cov = torch.from_numpy(cov.astype(np.float32))
        self.origins = list(origins)
        self.history = history
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int):
        origin = self.origins[i]
        x_sales = self.sales[origin - self.history:origin]       # [T_hist, N]
        x_cov = self.cov[origin - self.history:origin]            # [T_hist, N, F]
        y_sales = self.sales[origin:origin + self.horizon]        # [H, N]
        y_cov = self.cov[origin:origin + self.horizon]            # [H, N, F], known future covariates
        return x_sales, x_cov, y_sales, y_cov


class PatchTSTThenITransformer(nn.Module):
    """Patch temporal encoder -> cross-item attention -> covariate-conditioned forecast head.

    The cross-item block is iTransformer-style variable attention, not a literal reproduction
    of the original iTransformer architecture.
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
    ):
        super().__init__()
        if history < patch_len:
            raise ValueError("history must be >= patch_len")
        self.history = history
        self.horizon = horizon
        self.use_covariates = use_covariates
        self.patch_len = patch_len
        self.patch_stride = patch_stride
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

        self.horizon_embedding = nn.Parameter(torch.randn(1, horizon, 1, d_model) * 0.02)
        if use_covariates:
            self.future_cov_proj = nn.Sequential(
                nn.Linear(n_covariates, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.future_cov_proj = None
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

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

        # Channel-independent temporal processing: each item's history is patched over time.
        x = x.permute(0, 2, 1, 3).contiguous()  # [B,N,T,C]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        # unfold result: [B,N,P,C,patch_len] -> [B,N,P,patch_len,C]
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()
        b, n, p, patch_len, channels = patches.shape
        patches = patches.view(b, n, p, patch_len * channels)
        z = self.patch_proj(patches) + self.temporal_pos[:, :, :p, :]
        z = z.reshape(b * n, p, -1)
        z = self.temporal_encoder(z)
        z = self.temporal_norm(z).mean(dim=1).reshape(b, n, -1)  # [B,N,D]

        # iTransformer-style cross-variable attention: items are tokens.
        q = self.item_norm1(z)
        attended, _ = self.item_attention(q, q, q, need_weights=False)
        z = z + attended
        z = z + self.item_ffn(self.item_norm2(z))

        # Each future day receives its own known calendar/price features.
        rep = z.unsqueeze(1) + self.horizon_embedding[:, : self.horizon]
        if self.use_covariates:
            future_emb = self.future_cov_proj(x_cov_future)  # [B,H,N,D]
            rep = rep + future_emb
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
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

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
        print(f"  epoch {epoch:03d} | train MSE={mean_train:.5f} | val MSE={mean_val:.5f}")

        if mean_val < best_val - 1e-5:
            best_val = mean_val
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping at epoch {epoch}; best validation MSE={best_val:.5f}")
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


def metric_row(name: str, pred_z: np.ndarray, true_z: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> dict:
    pred = inverse_target(pred_z, y_mean, y_std)
    true = inverse_target(true_z, y_mean, y_std)
    error = pred - true
    wape = float(np.abs(error).sum() / max(np.abs(true).sum(), 1e-8))
    return {
        "model": name,
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE": wape,
    }


def main() -> None:
    seed_everything(SEED)
    data = load_and_prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = WindowDataset(data["sales_scaled"], data["cov"], data["train_origins"], HISTORY, HORIZON)
    val_ds = WindowDataset(data["sales_scaled"], data["cov"], data["val_origins"], HISTORY, HORIZON)
    test_ds = WindowDataset(data["sales_scaled"], data["cov"], data["test_origins"], HISTORY, HORIZON)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model_specs = [
        ("sales_only", False),
        ("multi_feature", True),
    ]
    metrics = []
    outputs = {}

    for name, use_cov in model_specs:
        print(f"\nTraining {name} (use_covariates={use_cov})")
        # Reset seed before each model to make comparisons less sensitive to incidental RNG order.
        seed_everything(SEED)
        model = PatchTSTThenITransformer(
            history=HISTORY,
            horizon=HORIZON,
            n_covariates=len(data["covariate_cols"]),
            use_covariates=use_cov,
        )
        model = train_model(
            model, train_loader, val_loader, use_cov, device, OUT_DIR / f"{name}_best.pt"
        )
        pred_z, true_z = predict_dataset(model, test_loader, use_cov, device)
        outputs[name] = pred_z
        metrics.append(metric_row(name, pred_z, true_z, data["y_mean"], data["y_std"]))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\nTest metrics over rolling forecast windows (overlapping targets are counted repeatedly):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Plot the latest rolling test forecast, which is fully contained in known sales history.
    last_origin = data["test_origins"][-1]
    pred_last = {
        name: inverse_target(pred_arr[-1:], data["y_mean"], data["y_std"])[0]
        for name, pred_arr in outputs.items()
    }
    true_last = inverse_target(true_z[-1:], data["y_mean"], data["y_std"])[0]
    forecast_dates = pd.to_datetime(data["dates_sales"].iloc[last_origin:last_origin + HORIZON]).to_list()
    chosen = data["chosen"]
    actual_last = true_last  # [H, N]

    # One row per item/date for reproducible downstream analysis.
    rows = []
    for h, dt in enumerate(forecast_dates):
        for j, item in enumerate(chosen):
            row = {"date": dt, "item_id": item, "actual_sales": float(actual_last[h, j])}
            for name in outputs:
                row[f"{name}_forecast"] = float(pred_last[name][h, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "latest_28day_forecast.csv", index=False)

    # Forecast plot for the first six sampled products.
    n_plot = min(6, len(chosen))
    fig, axes = plt.subplots(n_plot, 1, figsize=(13, max(3, 2.7 * n_plot)), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(forecast_dates, actual_last[:, j], label="Actual", linewidth=2)
        ax.plot(forecast_dates, pred_last["sales_only"][:, j], label="Sales-only baseline", linestyle="--")
        ax.plot(forecast_dates, pred_last["multi_feature"][:, j], label="Multi-feature hybrid", linestyle="-")
        ax.set_title(str(chosen[j]))
        ax.set_ylabel("Sales")
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Latest 28-day forecast: sales-only vs. multi-feature PatchTST → cross-item attention", y=1.002)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_six_items.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Sum across the 50 products to compare aggregate demand.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(forecast_dates, actual_last.sum(axis=1), label="Actual total", linewidth=2)
    ax.plot(forecast_dates, pred_last["sales_only"].sum(axis=1), label="Sales-only baseline", linestyle="--")
    ax.plot(forecast_dates, pred_last["multi_feature"].sum(axis=1), label="Multi-feature hybrid")
    ax.set_title("Aggregate sales of sampled products — latest 28-day forecast")
    ax.set_ylabel("Total sales")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved outputs to: {OUT_DIR.resolve()}")
    print("  m5_panel_multifeature.csv")
    print("  metrics.csv")
    print("  latest_28day_forecast.csv")
    print("  forecast_six_items.png")
    print("  forecast_total.png")
    print("  sales_only_best.pt / multi_feature_best.pt")


if __name__ == "__main__":
    main()
