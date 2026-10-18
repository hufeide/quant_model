#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stream fundamental + technical stock model (TSMixer + MLP).

This script provides a complete CSV-based research pipeline:
- point-in-time as-of alignment of fundamental snapshots;
- trading-calendar reindexing between each symbol's first/last observed bars;
- technical feature engineering without backward-looking operations;
- purged chronological train/validation/test split;
- train-only robust scaling and missing-value imputation;
- technical TSMixer branch + fundamental MLP branch + gated fusion;
- Huber regression + within-date pairwise ranking loss;
- validation RankIC early stopping and CSV/JSON/checkpoint outputs.

Important: this is a research baseline, not a production execution/backtest engine.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


# -------------------------------
# Reproducibility and helpers
# -------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred over peak speed for this research baseline.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def require_columns(df: pd.DataFrame, cols: Sequence[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source} 缺少必要字段: {missing}; 当前字段: {list(df.columns)}")


def safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    valid = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out.loc[valid] = np.log(a.loc[valid] / b.loc[valid])
    return out


# -------------------------------
# Data loading and PIT alignment
# -------------------------------

BAR_REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume"]
BAR_NUMERIC = ["open", "high", "low", "close", "volume", "amount", "turnover"]


def load_bars(path: str, date_col: str = "date", symbol_col: str = "symbol") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={symbol_col: str})
    require_columns(df, [date_col, symbol_col, "open", "high", "low", "close", "volume"], "bars.csv")
    df = df.rename(columns={date_col: "date", symbol_col: "symbol"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["date", "symbol"])
    df = df[df["symbol"] != ""]

    keep = ["date", "symbol"] + [c for c in BAR_NUMERIC if c in df.columns]
    df = df[keep].copy()
    for col in [c for c in keep if c not in ("date", "symbol")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.duplicated(["symbol", "date"]).any():
        examples = df.loc[df.duplicated(["symbol", "date"], keep=False), ["symbol", "date"]].head(10)
        raise ValueError(
            "bars.csv 每只股票每个交易日只能有一条记录。请先处理重复行，例如:\n"
            + examples.to_string(index=False)
        )
    if df.empty:
        raise ValueError("bars.csv 没有有效数据")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_fundamentals(
    path: str,
    date_col: str = "available_date",
    symbol_col: str = "symbol",
    requested_cols: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load PIT fundamental snapshots. `date_col` must be the date the data became usable."""
    df = pd.read_csv(path, dtype={symbol_col: str})
    require_columns(df, [date_col, symbol_col], "fundamentals.csv")
    df = df.rename(columns={date_col: "available_date", symbol_col: "symbol"})
    df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["available_date", "symbol"])
    df = df[df["symbol"] != ""]

    key_cols = {"available_date", "symbol"}
    if requested_cols:
        fund_cols = [c.strip() for c in requested_cols.split(",") if c.strip()]
        require_columns(df, fund_cols, "fundamentals.csv / --fund-cols")
        if any(c in key_cols for c in fund_cols):
            raise ValueError("--fund-cols 只能指定数值型基本面特征，不能包含日期或股票代码字段")
    else:
        # Conservative auto-detection: skip identifiers and period/date metadata.
        forbidden_tokens = ("date", "time", "period", "quarter", "report", "year", "month", "code", "id")
        fund_cols = []
        for col in df.columns:
            if col in key_cols:
                continue
            lower = col.lower()
            if any(token in lower for token in forbidden_tokens):
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
                fund_cols.append(col)

    for col in fund_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not fund_cols:
        raise ValueError(
            "未检测到基本面数值特征。请在 fundamentals.csv 中提供 ROE、PE 等数值列，"
            "或用 --fund-cols roe,pe,pb 显式指定。"
        )

    df = df[["available_date", "symbol"] + fund_cols].copy()
    # Multiple full snapshots for a symbol on the same usable date are ambiguous.
    # Keeping the final input row is deterministic, but the source should ideally resolve this upstream.
    if df.duplicated(["symbol", "available_date"]).any():
        warnings.warn(
            "fundamentals.csv 存在同一股票同一 available_date 多条记录；代码会按文件顺序保留最后一条。"
            "请尽可能提前合并为每个可用日期一条 PIT 快照。",
            RuntimeWarning,
        )
        df = df.drop_duplicates(["symbol", "available_date"], keep="last")
    return df.sort_values(["symbol", "available_date"]).reset_index(drop=True), fund_cols


def reindex_each_symbol_to_market_calendar(bars: pd.DataFrame) -> pd.DataFrame:
    """Insert missing market sessions inside each ticker's observed life as NaN price rows.

    The market calendar is derived from all dates present in bars.csv. The raw open remains NaN
    on inserted suspension/missing rows, so entry/exit labels on those dates are dropped instead
    of pretending that the stock traded at its stale close.
    """
    calendar = pd.DatetimeIndex(sorted(bars["date"].dropna().unique()))
    parts: List[pd.DataFrame] = []
    numeric_cols = [c for c in bars.columns if c not in ("date", "symbol")]
    for symbol, group in bars.groupby("symbol", sort=False):
        group = group.sort_values("date").set_index("date")
        symbol_calendar = calendar[(calendar >= group.index.min()) & (calendar <= group.index.max())]
        group = group.reindex(symbol_calendar)
        group.index.name = "date"
        group["symbol"] = symbol
        # Preserve raw OHLC values for labels; missing volume/amount on inserted rows mean zero activity.
        for col in ("volume", "amount"):
            if col in group.columns:
                group[col] = group[col].fillna(0.0)
        parts.append(group.reset_index())
    out = pd.concat(parts, ignore_index=True)
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def merge_fundamentals_asof(
    bars: pd.DataFrame,
    fundamentals: pd.DataFrame,
    raw_fund_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """For each symbol/date, use only the latest fundamental snapshot available on/before that date."""
    output_cols = [f"fund__{c}" for c in raw_fund_cols]
    fund_parts = []
    by_symbol = {sym: grp.sort_values("available_date") for sym, grp in fundamentals.groupby("symbol", sort=False)}

    for symbol, bar_group in bars.groupby("symbol", sort=False):
        left = bar_group.sort_values("date").copy()
        right = by_symbol.get(symbol)
        if right is None or right.empty:
            for c in output_cols:
                left[c] = np.nan
            left["_fund_available_date"] = pd.NaT
            fund_parts.append(left)
            continue

        right = right[["available_date"] + list(raw_fund_cols)].copy()
        right = right.rename(columns={c: f"fund__{c}" for c in raw_fund_cols})
        right = right.rename(columns={"available_date": "_fund_available_date"})
        merged = pd.merge_asof(
            left.sort_values("date"),
            right.sort_values("_fund_available_date"),
            left_on="date",
            right_on="_fund_available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        # Carry forward a missing individual feature from the prior known snapshot, never from the future.
        merged[output_cols] = merged[output_cols].ffill()
        fund_parts.append(merged)

    out = pd.concat(fund_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    age = (out["date"] - out["_fund_available_date"]).dt.days.astype(float)
    out["fund__age_days"] = age.clip(lower=0, upper=5000)
    out["fund__has_data"] = out["_fund_available_date"].notna().astype(float)
    final_fund_cols = output_cols + ["fund__age_days", "fund__has_data"]
    return out, final_fund_cols


# -------------------------------
# Causal technical features + labels
# -------------------------------

def add_technical_features_one_symbol(group: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    g = group.sort_values("date").copy()

    # For feature construction only, forward-fill a missing close during a missing/suspended session.
    # Raw open is untouched and is still used for label construction.
    close = pd.to_numeric(g["close"], errors="coerce").where(lambda x: x > 0).ffill()
    open_raw = pd.to_numeric(g["open"], errors="coerce")
    open_feat = open_raw.where(open_raw > 0, close)
    high_raw = pd.to_numeric(g["high"], errors="coerce")
    low_raw = pd.to_numeric(g["low"], errors="coerce")
    high = high_raw.where(high_raw > 0, close)
    low = low_raw.where(low_raw > 0, close)
    prev_close = close.shift(1)

    feat: Dict[str, pd.Series] = {}
    log_close = np.log(close.where(close > 0))
    ret1 = log_close.diff()
    feat["tech_ret_1"] = ret1
    for period in (2, 5, 10, 20, 60):
        feat[f"tech_ret_{period}"] = log_close - log_close.shift(period)

    feat["tech_overnight_gap"] = safe_log_ratio(open_feat, prev_close)
    feat["tech_intraday_ret"] = safe_log_ratio(close, open_feat)
    feat["tech_range_pct"] = (high - low) / close.replace(0, np.nan)
    denom = (high - low).replace(0, np.nan)
    feat["tech_close_location"] = ((close - low) / denom).clip(0, 1)

    for window in (5, 20, 60):
        min_periods = max(3, window // 2)
        feat[f"tech_volatility_{window}"] = ret1.rolling(window, min_periods=min_periods).std(ddof=0)
        ma = close.rolling(window, min_periods=min_periods).mean()
        feat[f"tech_ma_gap_{window}"] = close / ma.replace(0, np.nan) - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=10).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    feat["tech_rsi_14"] = rsi / 100.0

    volume = pd.to_numeric(g["volume"], errors="coerce").fillna(0).clip(lower=0)
    log_volume = np.log1p(volume)
    for period in (1, 5, 20):
        feat[f"tech_volume_log_change_{period}"] = log_volume - log_volume.shift(period)
    feat["tech_volume_z20"] = (
        log_volume - log_volume.rolling(20, min_periods=10).mean()
    ) / log_volume.rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)

    if "amount" in g.columns:
        amount = pd.to_numeric(g["amount"], errors="coerce").fillna(0).clip(lower=0)
        log_amount = np.log1p(amount)
        for period in (1, 5, 20):
            feat[f"tech_amount_log_change_{period}"] = log_amount - log_amount.shift(period)
        amt_mean = log_amount.rolling(20, min_periods=10).mean()
        amt_std = log_amount.rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)
        feat["tech_amount_z20"] = (log_amount - amt_mean) / amt_std

    if "turnover" in g.columns:
        feat["tech_turnover"] = pd.to_numeric(g["turnover"], errors="coerce")
        turn = pd.to_numeric(g["turnover"], errors="coerce")
        feat["tech_turnover_change_5"] = turn - turn.shift(5)

    for name, values in feat.items():
        g[name] = pd.to_numeric(values, errors="coerce")

    feature_cols = list(feat.keys())
    return g, feature_cols


def build_features_and_labels(
    bars: pd.DataFrame,
    fundamentals: pd.DataFrame,
    raw_fund_cols: Sequence[str],
    horizon: int,
    reindex_calendar: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if horizon < 1:
        raise ValueError("--horizon 必须 >= 1")
    bars = reindex_each_symbol_to_market_calendar(bars) if reindex_calendar else bars.copy()
    merged, fund_cols = merge_fundamentals_asof(bars, fundamentals, raw_fund_cols)

    tech_parts: List[pd.DataFrame] = []
    tech_cols: Optional[List[str]] = None
    for _, group in merged.groupby("symbol", sort=False):
        result, cols = add_technical_features_one_symbol(group)
        tech_parts.append(result)
        if tech_cols is None:
            tech_cols = cols
        elif cols != tech_cols:
            raise RuntimeError("内部错误：不同股票生成的技术特征不一致")
    df = pd.concat(tech_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    tech_cols = tech_cols or []

    # Signal formed using information through close(t); enter at open(t+1), exit at open(t+horizon+1).
    target_parts = []
    for _, group in df.groupby("symbol", sort=False):
        g = group.sort_values("date").copy()
        entry_open = pd.to_numeric(g["open"], errors="coerce").shift(-1)
        exit_open = pd.to_numeric(g["open"], errors="coerce").shift(-(horizon + 1))
        valid = (entry_open > 0) & (exit_open > 0) & np.isfinite(entry_open) & np.isfinite(exit_open)
        target = pd.Series(np.nan, index=g.index, dtype=float)
        target.loc[valid] = np.log(exit_open.loc[valid] / entry_open.loc[valid])
        g["target"] = target
        g["label_end_date"] = g["date"].shift(-(horizon + 1))
        target_parts.append(g)
    df = pd.concat(target_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Avoid infinities caused by invalid/zero data; imputation is fit on training dates later.
    all_feature_cols = tech_cols + list(fund_cols)
    for col in all_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df, tech_cols, list(fund_cols)


# -------------------------------
# Train-only robust transform
# -------------------------------

@dataclass
class RobustStats:
    median: List[float]
    center: List[float]
    scale: List[float]


def fit_robust_stats(values: np.ndarray) -> RobustStats:
    """Median imputation + IQR scaling, fitted only on training-period rows."""
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("特征矩阵必须是二维且至少有一列")
    n_features = values.shape[1]
    medians = np.zeros(n_features, dtype=np.float64)
    centers = np.zeros(n_features, dtype=np.float64)
    scales = np.ones(n_features, dtype=np.float64)
    for j in range(n_features):
        col = values[:, j].astype(np.float64, copy=False)
        col = col[np.isfinite(col)]
        if col.size == 0:
            medians[j] = centers[j] = 0.0
            scales[j] = 1.0
            continue
        med = float(np.median(col))
        q25, q75 = np.percentile(col, [25, 75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = float(np.std(col))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        medians[j] = med
        centers[j] = med
        scales[j] = scale
    return RobustStats(median=medians.tolist(), center=centers.tolist(), scale=scales.tolist())


def apply_robust_stats(values: np.ndarray, stats: RobustStats, clip: float = 10.0) -> np.ndarray:
    x = values.astype(np.float64, copy=True)
    med = np.asarray(stats.median, dtype=np.float64)
    center = np.asarray(stats.center, dtype=np.float64)
    scale = np.asarray(stats.scale, dtype=np.float64)
    x[~np.isfinite(x)] = np.take(med, np.where(~np.isfinite(x))[1])
    x = (x - center) / scale
    x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


# -------------------------------
# Sample construction / chronological purged split
# -------------------------------

def create_sample_table(
    df: pd.DataFrame,
    seq_len: int,
) -> pd.DataFrame:
    if seq_len < 2:
        raise ValueError("--seq-len 建议至少为 2")
    sample_parts: List[pd.DataFrame] = []

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        positions = np.arange(seq_len - 1, len(g), dtype=np.int64)
        if positions.size == 0:
            continue
        targets = g["target"].to_numpy(dtype=np.float64)
        end_dates = pd.to_datetime(g["label_end_date"]).to_numpy()
        valid = np.isfinite(targets[positions]) & ~pd.isna(end_dates[positions])
        positions = positions[valid]
        if positions.size == 0:
            continue
        part = pd.DataFrame({
            "symbol": str(symbol),
            "pos": positions,
            "date": g.loc[positions, "date"].to_numpy(),
            "label_end_date": g.loc[positions, "label_end_date"].to_numpy(),
            "target": targets[positions].astype(np.float32),
        })
        sample_parts.append(part)

    if not sample_parts:
        raise ValueError(
            "没有可训练样本。请检查数据长度、序列窗口、未来开盘价是否缺失，以及股票数量。"
        )
    samples = pd.concat(sample_parts, ignore_index=True)
    samples["date"] = pd.to_datetime(samples["date"])
    samples["label_end_date"] = pd.to_datetime(samples["label_end_date"])
    dates = np.sort(samples["date"].unique())
    date_to_id = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    samples["date_id"] = samples["date"].map(date_to_id).astype(np.int64)
    samples = samples.sort_values(["date", "symbol"]).reset_index(drop=True)
    return samples


def chronological_purged_split(
    samples: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    if not (0.5 <= train_fraction < 0.9):
        raise ValueError("--train-fraction 应在 [0.5, 0.9) 之间")
    if not (0.05 <= val_fraction < 0.4):
        raise ValueError("--val-fraction 应在 [0.05, 0.4) 之间")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction 必须小于 1")

    dates = np.sort(samples["date"].unique())
    n_dates = len(dates)
    i_train = int(n_dates * train_fraction)
    i_val = int(n_dates * (train_fraction + val_fraction))
    if i_train < 2 or i_val <= i_train or i_val >= n_dates:
        raise ValueError(f"有效交易日只有 {n_dates} 天，无法按当前比例切分")

    train_end = pd.Timestamp(dates[i_train - 1])
    val_start = pd.Timestamp(dates[i_train])
    val_end = pd.Timestamp(dates[i_val - 1])
    test_start = pd.Timestamp(dates[i_val])

    # Purge labels that reach into the next split. Labels are only admitted if their exit date
    # lies strictly before the next split's first signal date.
    train_mask = (samples["date"] <= train_end) & (samples["label_end_date"] < val_start)
    val_mask = (
        (samples["date"] >= val_start)
        & (samples["date"] <= val_end)
        & (samples["label_end_date"] < test_start)
    )
    test_mask = samples["date"] >= test_start

    train = samples.loc[train_mask].copy().reset_index(drop=True)
    val = samples.loc[val_mask].copy().reset_index(drop=True)
    test = samples.loc[test_mask].copy().reset_index(drop=True)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "切分后某个集合没有样本。请提供更长历史数据，或缩短 --horizon / --seq-len，"
            "或调整 --train-fraction 与 --val-fraction。"
        )
    boundaries = {
        "nominal_train_cutoff_date": str(train_end.date()),
        "train_actual_first_signal_date": str(pd.Timestamp(train["date"].min()).date()),
        "train_actual_last_signal_date_after_purge": str(pd.Timestamp(train["date"].max()).date()),
        "nominal_validation_start": str(val_start.date()),
        "nominal_validation_end": str(val_end.date()),
        "validation_actual_first_signal_date": str(pd.Timestamp(val["date"].min()).date()),
        "validation_actual_last_signal_date_after_purge": str(pd.Timestamp(val["date"].max()).date()),
        "test_start": str(test_start.date()),
        "test_end_signal_date": str(pd.Timestamp(test["date"].max()).date()),
        "purge_rule": "label_end_date < next_split_first_signal_date",
    }
    return train, val, test, boundaries


class SequenceDataset(Dataset):
    def __init__(self, feature_groups: Dict[str, Tuple[np.ndarray, np.ndarray]], samples: pd.DataFrame, target_scale: float = 1.0):
        self.feature_groups = feature_groups
        self.samples = samples.reset_index(drop=True).copy()
        self.target_scale = float(target_scale) if float(target_scale) > 1e-12 else 1.0
        self.symbols = self.samples["symbol"].astype(str).to_numpy()
        self.positions = self.samples["pos"].to_numpy(dtype=np.int64)
        self.targets = self.samples["target"].to_numpy(dtype=np.float32)
        self.date_ids = self.samples["date_id"].to_numpy(dtype=np.int64)
        self.seq_len = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        symbol = self.symbols[idx]
        pos = int(self.positions[idx])
        tech, fund = self.feature_groups[symbol]
        start = pos - (self._seq_len - 1)
        # Copy each window so tensors do not share a writable NumPy view.
        x_tech = torch.from_numpy(np.array(tech[start:pos + 1], dtype=np.float32, copy=True))
        x_fund = torch.from_numpy(np.array(fund[pos], dtype=np.float32, copy=True))
        # Predict standardized forward returns for a better-balanced regression/ranking objective.
        y = torch.tensor(self.targets[idx] / self.target_scale, dtype=torch.float32)
        date_id = torch.tensor(self.date_ids[idx], dtype=torch.long)
        return x_tech, x_fund, y, date_id, torch.tensor(idx, dtype=torch.long)

    def set_seq_len(self, seq_len: int) -> None:
        self._seq_len = int(seq_len)


class DateGroupedBatchSampler(Sampler[List[int]]):
    """Batch samples by date so the ranking loss compares stocks from the same cross-section."""
    def __init__(self, dataset: SequenceDataset, batch_size: int, shuffle: bool):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        by_date: Dict[int, List[int]] = {}
        for i, d in enumerate(dataset.date_ids):
            by_date.setdefault(int(d), []).append(i)
        self.by_date = by_date

    def __iter__(self):
        dates = list(self.by_date.keys())
        if self.shuffle:
            random.shuffle(dates)
        for date_id in dates:
            indices = list(self.by_date[date_id])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(v) / self.batch_size) for v in self.by_date.values())


# -------------------------------
# Model: TSMixer + fundamental MLP + gated fusion
# -------------------------------

class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1, expansion: int = 2):
        super().__init__()
        self.norm_time = nn.LayerNorm(d_model)
        self.norm_feature = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.temporal_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * expansion, seq_len),
        )
        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm_time(x).transpose(1, 2)
        z = self.temporal_mixer(z).transpose(1, 2)
        x = x + self.dropout(z)
        z = self.feature_mixer(self.norm_feature(x))
        x = x + self.dropout(z)
        return x


class TechnicalEncoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 64, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.Sequential(*[
            TSMixerBlock(seq_len=seq_len, d_model=d_model, dropout=dropout)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError(f"technical input should be [B, {self.seq_len}, F], got {tuple(x.shape)}")
        x = self.input_proj(x)
        x = self.blocks(x)
        last = self.out_norm(x[:, -1, :])
        mean = self.out_norm(x.mean(dim=1))
        return self.pool_proj(torch.cat([last, mean], dim=-1))


class FundamentalEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStreamTSMixer(nn.Module):
    def __init__(
        self,
        tech_features: int,
        fund_features: int,
        seq_len: int,
        d_model: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tech_encoder = TechnicalEncoder(tech_features, seq_len, d_model, depth, dropout)
        self.fund_encoder = FundamentalEncoder(fund_features, d_model, dropout)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.alpha_head = nn.Linear(d_model, 1)

    def forward(self, x_tech: torch.Tensor, x_fund: torch.Tensor, return_aux: bool = False):
        h_tech = self.tech_encoder(x_tech)
        h_fund = self.fund_encoder(x_fund)
        gate = self.gate(torch.cat([h_tech, h_fund], dim=-1))
        mixed = gate * h_tech + (1.0 - gate) * h_fund
        fused_input = torch.cat([
            mixed,
            h_tech,
            h_fund,
            h_tech * h_fund,
            gate * h_tech,
        ], dim=-1)
        fused = self.fusion(fused_input)
        alpha = self.alpha_head(fused).squeeze(-1)
        if return_aux:
            return {"alpha": alpha, "gate": gate,
                    "technical_embedding": h_tech, "fundamental_embedding": h_fund}
        return alpha


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    """Pairwise ranking loss. Batch sampler ensures a batch contains one date only."""
    n = pred.numel()
    if n < 2:
        return pred.new_zeros(())
    total_pairs = n * (n - 1) // 2
    n_pairs = min(max_pairs, total_pairs)
    i = torch.randint(0, n, (n_pairs,), device=pred.device)
    j = torch.randint(0, n, (n_pairs,), device=pred.device)
    keep = i != j
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return pred.new_zeros(())
    delta_y = target[i] - target[j]
    valid = delta_y.abs() > 1e-8
    if not valid.any():
        return pred.new_zeros(())
    delta_pred = pred[i[valid]] - pred[j[valid]]
    sign = delta_y[valid].sign()
    return F.softplus(-sign * delta_pred).mean()


# -------------------------------
# Evaluation
# -------------------------------

def compute_metrics(pred_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    required = {"date", "target", "pred"}
    if not required.issubset(pred_df.columns):
        raise ValueError(f"compute_metrics expects columns {required}")
    d = pred_df.dropna(subset=["target", "pred"]).copy()
    if d.empty:
        return {"n_samples": 0}, pd.DataFrame(columns=["date", "rank_ic", "top_bucket_return", "bottom_bucket_return", "top_minus_bottom"])

    err = d["pred"].to_numpy(dtype=float) - d["target"].to_numpy(dtype=float)
    daily_rows = []
    for date, g in d.groupby("date", sort=True):
        if len(g) < 3 or g["pred"].nunique() < 2 or g["target"].nunique() < 2:
            rank_ic = np.nan
        else:
            rank_ic = g["pred"].corr(g["target"], method="spearman")
        n = len(g)
        k = max(1, int(math.ceil(n * 0.1)))
        top = g.nlargest(k, "pred")["target"].mean()
        bottom = g.nsmallest(k, "pred")["target"].mean()
        daily_rows.append({
            "date": date,
            "rank_ic": rank_ic,
            "top_bucket_return": float(top),
            "bottom_bucket_return": float(bottom),
            "top_minus_bottom": float(top - bottom),
            "n_stocks": n,
        })
    daily = pd.DataFrame(daily_rows)
    ic = daily["rank_ic"].replace([np.inf, -np.inf], np.nan).dropna()
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    ic_mean = float(ic.mean()) if len(ic) else float("nan")
    metrics = {
        "n_samples": int(len(d)),
        "n_days": int(daily["date"].nunique()),
        "mae_prediction_vs_forward_return": float(np.mean(np.abs(err))),
        "rmse_prediction_vs_forward_return": float(np.sqrt(np.mean(err ** 2))),
        "rank_ic_mean": ic_mean,
        "rank_ic_std": ic_std,
        "rank_ic_ir_unannualized": float(ic_mean / ic_std) if np.isfinite(ic_std) and ic_std > 0 else float("nan"),
        "top_decile_mean_forward_return": float(daily["top_bucket_return"].mean()),
        "bottom_decile_mean_forward_return": float(daily["bottom_bucket_return"].mean()),
        "top_minus_bottom_mean_forward_return": float(daily["top_minus_bottom"].mean()),
    }
    return metrics, daily


def predict_dataset(
    model: nn.Module,
    dataset: SequenceDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, float]:
    sampler = DateGroupedBatchSampler(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=(device.type == "cuda"))
    pred_values = np.full(len(dataset), np.nan, dtype=np.float32)
    loss_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for x_tech, x_fund, y, date_id, sample_idx in loader:
            x_tech = x_tech.to(device, non_blocking=True)
            x_fund = x_fund.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x_tech, x_fund)
            loss = F.smooth_l1_loss(pred, y, reduction="sum")
            loss_sum += float(loss.item())
            count += int(y.numel())
            pred_values[sample_idx.numpy()] = (pred.detach().cpu().numpy() * dataset.target_scale).astype(np.float32)
    out = dataset.samples.copy()
    out["pred"] = pred_values
    out = out.rename(columns={"target": "target"})
    return out, (loss_sum / max(count, 1))


def make_loader(dataset: SequenceDataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    sampler = DateGroupedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=(device.type == "cuda"))


# -------------------------------
# Main training pipeline
# -------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-share two-stream model: technical TSMixer + fundamental MLP + gated fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bars", required=True, help="日频 OHLCV CSV 路径")
    parser.add_argument("--fundamentals", required=True, help="PIT 基本面快照 CSV 路径")
    parser.add_argument("--out-dir", default="runs/two_stream_tsmixer", help="输出目录")
    parser.add_argument("--bar-date-col", default="date", help="bars 日期字段名")
    parser.add_argument("--symbol-col", default="symbol", help="两个 CSV 的股票代码字段名")
    parser.add_argument("--fund-date-col", default="available_date", help="基本面可用日期字段名，必须为信息可用日期")
    parser.add_argument("--fund-cols", default="", help="基本面数值列，以逗号分隔；为空则自动检测")
    parser.add_argument("--horizon", type=int, default=5, help="持有周期 H 个交易日；信号日收盘后，下一交易日开盘入场")
    parser.add_argument("--seq-len", type=int, default=60, help="技术面历史窗口长度")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="时间切分中训练时间比例")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="时间切分中验证时间比例；剩余为测试期")
    parser.add_argument("--epochs", type=int, default=50, help="最大训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=512, help="每批同一交易日的股票样本数上限")
    parser.add_argument("--d-model", type=int, default=64, help="隐藏维度")
    parser.add_argument("--depth", type=int, default=3, help="TSMixer block 数量")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--rank-weight", type=float, default=0.10, help="排序损失权重")
    parser.add_argument("--patience", type=int, default=8, help="验证 RankIC 连续多少轮无改善则早停")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="早停最小改善阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--no-calendar-reindex", action="store_true", help="不补齐股票生命周期内缺失的市场交易日（不推荐）")
    return parser.parse_args(argv)


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, object]:
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 PyTorch 检测不到可用 CUDA")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    print(f"[INFO] device={device}")

    bars = load_bars(args.bars, date_col=args.bar_date_col, symbol_col=args.symbol_col)
    funds, raw_fund_cols = load_fundamentals(
        args.fundamentals,
        date_col=args.fund_date_col,
        symbol_col=args.symbol_col,
        requested_cols=args.fund_cols or None,
    )
    print(f"[INFO] bars rows={len(bars):,}, symbols={bars['symbol'].nunique():,}, dates={bars['date'].nunique():,}")
    print(f"[INFO] fundamental feature columns={raw_fund_cols}")

    df, tech_cols, fund_cols = build_features_and_labels(
        bars, funds, raw_fund_cols, horizon=args.horizon,
        reindex_calendar=not args.no_calendar_reindex,
    )
    print(f"[INFO] engineered rows={len(df):,}, technical features={len(tech_cols)}, fundamental features={len(fund_cols)}")
    del bars, funds
    gc.collect()

    # Create sample metadata (not duplicated feature windows) to obtain split boundaries.
    samples = create_sample_table(df, args.seq_len)
    train_samples, val_samples, test_samples, boundaries = chronological_purged_split(
        samples, args.train_fraction, args.val_fraction
    )
    train_end_date = pd.Timestamp(train_samples["date"].max())
    train_rows_mask = df["date"] <= train_end_date
    target_scale = float(train_samples["target"].std(ddof=0))
    if not np.isfinite(target_scale) or target_scale < 1e-8:
        target_scale = 1.0
    print(f"[INFO] target scale (train-only std)={target_scale:.6g}")
    if int(train_rows_mask.sum()) == 0:
        raise RuntimeError("无法获得训练期特征行")

    # Fit imputers/scalers using training-period rows only, then transform all dates.
    tech_stats = fit_robust_stats(df.loc[train_rows_mask, tech_cols].to_numpy(dtype=np.float64))
    fund_stats = fit_robust_stats(df.loc[train_rows_mask, fund_cols].to_numpy(dtype=np.float64))
    tech_all = apply_robust_stats(df[tech_cols].to_numpy(dtype=np.float64), tech_stats)
    fund_all = apply_robust_stats(df[fund_cols].to_numpy(dtype=np.float64), fund_stats)

    # Assign normalized arrays to each symbol in the same chronological row order used by `pos`.
    scaled_df = df[["symbol", "date"]].copy()
    scaled_df["_row_id"] = np.arange(len(df), dtype=np.int64)
    scaled_group_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for symbol, idx in scaled_df.groupby("symbol", sort=False).groups.items():
        row_ids = np.asarray(list(idx), dtype=np.int64)
        # df is globally sorted symbol/date; group order is chronological.
        scaled_group_arrays[str(symbol)] = (tech_all[row_ids], fund_all[row_ids])

    # The same positions in samples refer to per-symbol rows. Replace raw feature arrays with scaled ones.
    dataset_groups = scaled_group_arrays
    # Release wide intermediate frames before training; large universes can contain millions of rows.
    del tech_all, fund_all, scaled_df, df, scaled_group_arrays
    gc.collect()

    train_ds = SequenceDataset(dataset_groups, train_samples, target_scale=target_scale)
    val_ds = SequenceDataset(dataset_groups, val_samples, target_scale=target_scale)
    test_ds = SequenceDataset(dataset_groups, test_samples, target_scale=target_scale)
    for ds in (train_ds, val_ds, test_ds):
        ds.set_seq_len(args.seq_len)

    print(f"[INFO] split={boundaries}")
    print(f"[INFO] samples train={len(train_ds):,}, validation={len(val_ds):,}, test={len(test_ds):,}")
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("训练、验证或测试样本为空")

    model_config = {
        "tech_features": len(tech_cols),
        "fund_features": len(fund_cols),
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "depth": args.depth,
        "dropout": args.dropout,
    }
    model = TwoStreamTSMixer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        loader = make_loader(train_ds, args.batch_size, shuffle=True, device=device)
        for x_tech, x_fund, y, date_id, sample_idx in loader:
            x_tech = x_tech.to(device, non_blocking=True)
            x_fund = x_fund.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_tech, x_fund)
            loss_reg = F.smooth_l1_loss(pred, y)
            loss_rank = pairwise_rank_loss(pred, y)
            loss = loss_reg + args.rank_weight * loss_rank
            if not torch.isfinite(loss):
                raise RuntimeError(f"第 {epoch} 轮出现非有限 loss，请检查输入数据与归一化")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(y.numel())
            train_count += int(y.numel())

        val_pred_df, val_reg_loss = predict_dataset(model, val_ds, args.batch_size, device)
        val_metrics, _ = compute_metrics(val_pred_df)
        val_rank_ic = val_metrics.get("rank_ic_mean", float("nan"))
        # Prefer the rank metric for stock selection; fall back to negative regression loss if undefined.
        score = val_rank_ic if np.isfinite(val_rank_ic) else -val_reg_loss
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss_sum / max(train_count, 1)),
            "val_huber_loss": float(val_reg_loss),
            "val_rank_ic_mean": float(val_rank_ic) if np.isfinite(val_rank_ic) else float("nan"),
            "val_rank_ic_ir_unannualized": float(val_metrics.get("rank_ic_ir_unannualized", float("nan"))),
        }
        history.append(row)
        print(
            f"[EPOCH {epoch:03d}] train_loss={row['train_loss']:.6f} "
            f"val_huber={row['val_huber_loss']:.6f} "
            f"val_RankIC={row['val_rank_ic_mean']:.5f}"
        )

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"[INFO] early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("训练没有产生可用的最佳模型状态")
    model.load_state_dict(best_state)

    # Final model is evaluated once on the held-out test split.
    val_pred_df, _ = predict_dataset(model, val_ds, args.batch_size, device)
    test_pred_df, _ = predict_dataset(model, test_ds, args.batch_size, device)
    val_metrics, val_daily = compute_metrics(val_pred_df)
    test_metrics, test_daily = compute_metrics(test_pred_df)

    # Use explicit date values and retain raw forward return labels for reporting.
    val_pred_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    val_daily.to_csv(out_dir / "validation_daily_metrics.csv", index=False)
    test_daily.to_csv(out_dir / "test_daily_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "tech_feature_columns": list(tech_cols),
        "fundamental_feature_columns": list(fund_cols),
        "technical_robust_stats": asdict(tech_stats),
        "fundamental_robust_stats": asdict(fund_stats),
        "sequence_length": args.seq_len,
        "horizon": args.horizon,
        "target_scale_train_std": target_scale,
        "best_epoch": best_epoch,
        "split_boundaries": boundaries,
        "notes": (
            "Prediction model only. For live inference, reproduce feature engineering and PIT as-of joins exactly; "
            "do not fit normalization on new data. Predictions are not trade instructions."
        ),
    }
    torch.save(checkpoint, out_dir / "best_model.pt")

    report = {
        "device": str(device),
        "best_epoch": best_epoch,
        "model_config": model_config,
        "horizon_sessions": args.horizon,
        "sequence_length": args.seq_len,
        "target_scale_train_std": target_scale,
        "technical_features": list(tech_cols),
        "fundamental_features": list(fund_cols),
        "split_boundaries": boundaries,
        "sample_counts": {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "interpretation_warning": (
            "Top/bottom decile results are averages of overlapping forward H-session labels, not a complete "
            "portfolio backtest. They do not include fees, taxes, slippage, suspensions, limit-up/down fills, "
            "market impact, or portfolio holding overlap. Do not interpret them as realized strategy Sharpe/PnL."
        ),
        "arguments": vars(args),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=json_default)

    print("\n[TEST METRICS]")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2, default=json_default))
    print(f"\n[INFO] artifacts saved to: {out_dir.resolve()}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        train_and_evaluate(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
