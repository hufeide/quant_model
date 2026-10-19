#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 销量预测：修正版对照实验
==========================
模型：
  1. LightGBM：逐预测步长直接预测，使用历史滞后/滚动特征 + 目标日已知协变量
  2. Transformer：历史编码器 + 未来已知协变量预测头，一次输出未来 28 天

数据：
  data/m5/sales_train_validation.csv
  data/m5/calendar.csv
  data/m5/sell_prices.csv

注意：
- 固定随机种子抽取 50 个 item_id，并将这些商品跨门店销量求和。
- 这是“抽样商品汇总序列”的实验，不是完整 M5 官方 30,490 条 item-store 序列。
- 本脚本报告抽样汇总序列的 RMSSE/MAE/RMSE/WAPE；不能将其称为官方 WRMSSE。
- 对历史回测，目标值只使用 sales_train_validation.csv 中真实存在的销量日期。
- 未来日历特征可提前获知；未来价格只使用 sell_prices.csv 中对应预测周已经存在的价格记录。
- 验证/测试窗口的历史可以使用预测起点之前的真实销量；目标区间严格位于相应 split 内。
"""

from pathlib import Path
import math
import random
import zipfile
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# ============================================================
# 1. 配置
# ============================================================

SEED = 42
N_ITEMS = 50
SEQ_LEN = 56
PRED_LEN = 28
STRIDE = 7

D_MODEL = 128
N_HEADS = 8
LAYERS = 4
DROPOUT = 0.15

BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
PATIENCE = 6
NUM_WORKERS = 0

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "m5"
OUT_DIR = BASE_DIR / "outputs2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 下载数据
# ============================================================

def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        "sales_train_validation.csv",
        "calendar.csv",
        "sell_prices.csv",
    ]
    if all((DATA_DIR / f).exists() for f in required):
        print("M5 数据文件已存在。")
        return

    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "m5-forecasting-accuracy",
                "-p", str(DATA_DIR),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "未找到 kaggle 命令。请先安装 Kaggle API 并配置凭据，"
            "或手动将三个 CSV 文件放入 data/m5/。"
        ) from exc

    for path in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(DATA_DIR)

    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"下载后仍缺少文件：{missing}")


# ============================================================
# 3. 数据处理
# ============================================================

def _event_features(calendar):
    """事件特征：事件是否发生、事件名称/类型是否出现；不使用目标销量。"""
    out = pd.DataFrame(index=calendar.index)
    for col in ("event_name_1", "event_name_2"):
        if col in calendar:
            out[f"{col}_flag"] = calendar[col].notna().astype(np.float32)
        else:
            out[f"{col}_flag"] = 0.0
    for col in ("event_type_1", "event_type_2"):
        if col in calendar:
            out[f"{col}_flag"] = calendar[col].notna().astype(np.float32)
        else:
            out[f"{col}_flag"] = 0.0
    return out


def prepare_data():
    """
    返回：
      sales: (n_sales_days, n_items) 聚合销量
      hist_cov: (n_sales_days, n_items, n_hist_features)
      future_cov: (n_calendar_days, n_items, n_future_features)
      dates: 销量真实日期
      calendar_dates: 完整日历日期
      chosen: 抽样商品 ID
      item_scale: 每商品训练期原始销量差分均方根尺度
      split: 训练/验证时间边界
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sales_df = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    day_num = np.array([int(c[2:]) for c in day_cols])
    order = np.argsort(day_num)
    day_cols = [day_cols[i] for i in order]
    day_num = day_num[order]

    # 固定种子抽样商品。保留原始 item_id，模型内部用 0..N-1 编码。
    rng = np.random.default_rng(SEED)
    available = np.array(sorted(sales_df["item_id"].unique()))
    chosen = sorted(
        rng.choice(available, size=min(N_ITEMS, len(available)), replace=False).tolist()
    )
    sampled = sales_df.loc[
        sales_df["item_id"].isin(chosen),
        ["item_id", "store_id", "state_id", *day_cols],
    ].copy()

    # 宽表按 d_1, d_2... 的数值顺序，而不是字符串排序。
    sampled = sampled.set_index(["item_id", "store_id", "state_id"])
    raw_sales = sampled[day_cols].to_numpy(dtype=np.float32)
    raw_sales = np.maximum(raw_sales, 0.0)
    # (item-store, day) -> (day, item-store) -> 跨门店按 item 汇总
    row_items = sampled.index.get_level_values("item_id").to_numpy()
    n_days = len(day_cols)
    sales_mat = np.zeros((n_days, len(chosen)), dtype=np.float32)
    item_to_idx = {item: i for i, item in enumerate(chosen)}
    for row_i, item in enumerate(row_items):
        sales_mat[:, item_to_idx[item]] += raw_sales[row_i, :]

    # calendar 用 d 的数值索引排序，并将真实销量日期与 calendar 明确对齐。
    calendar = calendar.copy()
    calendar["d_num"] = calendar["d"].str.replace("d_", "", regex=False).astype(int)
    calendar = calendar.sort_values("d_num").drop_duplicates("d_num")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar_by_num = calendar.set_index("d_num")
    missing_calendar = sorted(set(day_num.tolist()) - set(calendar_by_num.index.tolist()))
    if missing_calendar:
        raise ValueError(f"calendar 缺少销量日期 d_num：{missing_calendar[:10]}")

    sales_dates = pd.DatetimeIndex(calendar_by_num.loc[day_num, "date"].to_numpy())
    if not sales_dates.is_monotonic_increasing or sales_dates.has_duplicates:
        raise ValueError("销量日期映射到 calendar 后不是唯一且严格递增的日期。")

    # 日历特征覆盖完整 calendar，供历史输入及未来 28 天已知特征使用。
    cal = calendar.reset_index(drop=True).copy()
    cal["date"] = pd.to_datetime(cal["date"])
    dow = pd.to_numeric(cal["wday"], errors="coerce").fillna(1).to_numpy(np.float32)
    month = cal["date"].dt.month.to_numpy(np.float32)
    dayofyear = cal["date"].dt.dayofyear.to_numpy(np.float32)
    year = cal["date"].dt.year.to_numpy(np.float32)

    # M5 的 wday 编码是 1..7；转成 0..6 后做周期编码。
    dow0 = (dow - 1.0) % 7.0
    event_df = _event_features(cal)
    event_values = event_df.to_numpy(dtype=np.float32)

    cal_features = np.column_stack([
        np.sin(2 * np.pi * dow0 / 7.0),
        np.cos(2 * np.pi * dow0 / 7.0),
        np.sin(2 * np.pi * (month - 1.0) / 12.0),
        np.cos(2 * np.pi * (month - 1.0) / 12.0),
        np.sin(2 * np.pi * (dayofyear - 1.0) / 365.25),
        np.cos(2 * np.pi * (dayofyear - 1.0) / 365.25),
        (year - 2011.0) / 10.0,
        event_values,
    ]).astype(np.float32)

    # SNAP：先在 item-store 日级构造，再跨门店计算覆盖比例。
    # 缺失州标记视为 0；M5 数据中的州标记通常完整。
    snap_cols = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    store_state = sales_df[["item_id", "store_id", "state_id"]].drop_duplicates()
    store_state = store_state[store_state["item_id"].isin(chosen)].copy()
    state_snap = {}
    for state, col in snap_cols.items():
        if col in cal:
            state_snap[state] = pd.to_numeric(cal[col], errors="coerce").fillna(0).to_numpy(np.float32)
        else:
            state_snap[state] = np.zeros(len(cal), dtype=np.float32)

    snap_share = np.zeros((len(cal), len(chosen)), dtype=np.float32)
    for item, item_idx in item_to_idx.items():
        stores = store_state[store_state["item_id"] == item]
        if len(stores) == 0:
            continue
        vectors = []
        for state in stores["state_id"].unique():
            vectors.append(state_snap.get(state, np.zeros(len(cal), dtype=np.float32)))
        if vectors:
            snap_share[:, item_idx] = np.mean(np.stack(vectors), axis=0)

    # 价格表按 item/store/week 合并到完整 calendar。
    # 价格可用性单独保留，不能把缺失简单解释为真实零价格。
    prices = prices[prices["item_id"].isin(chosen)].copy()
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    if prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).any():
        raise ValueError("sell_prices.csv 的 item_id/store_id/wm_yr_wk 键重复。")

    # 对每个商品和日历周，在有价格的门店上计算均价；此处价格源只来自 sell_prices.csv。
    calendar_price = cal[["d_num", "date", "wm_yr_wk"]].copy()
    price_rows = []
    for item in chosen:
        p_item = prices[prices["item_id"] == item]
        if p_item.empty:
            price_rows.append(np.full(len(cal), np.nan, dtype=np.float32))
            continue
        weekly = p_item.groupby("wm_yr_wk")["sell_price"].mean()
        vals = calendar_price["wm_yr_wk"].map(weekly).to_numpy(dtype=np.float32)
        price_rows.append(vals)
    price_raw = np.stack(price_rows, axis=1)  # (n_calendar_days, n_items)
    price_available = np.isfinite(price_raw).astype(np.float32)

    # 价格的因果填充：只向前填充，绝不 backward-fill。
    price_filled = pd.DataFrame(price_raw).ffill().to_numpy(dtype=np.float32)
    # 对序列开始前从未有价格的项使用 0 sentinel，并保留 available/missing 标记。
    price_filled = np.where(np.isfinite(price_filled), price_filled, 0.0).astype(np.float32)

    # 训练区间统计量，只由训练期价格估计。先计算时间边界。
    n_sales_days = len(day_cols)
    t_train = int(n_sales_days * 0.70)
    t_val = int(n_sales_days * 0.85)
    train_price = price_filled[:t_train]
    train_avail = price_available[:t_train] > 0
    price_mean = np.zeros(len(chosen), dtype=np.float32)
    price_std = np.ones(len(chosen), dtype=np.float32)
    for j in range(len(chosen)):
        vals = train_price[train_avail[:, j], j]
        if len(vals):
            price_mean[j] = float(vals.mean())
            std = float(vals.std())
            price_std[j] = std if std > 1e-5 else 1.0
        else:
            price_mean[j] = 0.0
            price_std[j] = 1.0
    price_norm = ((price_filled - price_mean[None, :]) / price_std[None, :]).astype(np.float32)
    # 没有已知价格的日期使用中性数值，由 availability 特征区分。
    price_norm = np.where(price_available > 0, price_norm, 0.0).astype(np.float32)

    # 未来协变量：日历、事件、SNAP、价格、价格有效性。
    future_cov_base = np.concatenate([
        np.broadcast_to(cal_features[:, None, :],
                        (len(cal), len(chosen), cal_features.shape[1])),
        snap_share[:, :, None],
        price_norm[:, :, None],
        price_available[:, :, None],
    ], axis=-1).astype(np.float32)

    # 历史特征：log 销量 + 已知协变量。
    # 对历史价格同样使用只向前填充的价格序列。
    hist_cov = np.concatenate([
        np.log1p(sales_mat)[:, :, None],
        future_cov_base[:n_sales_days],
    ], axis=-1).astype(np.float32)

    # 用训练期原始销量的一阶差分均方根作为 RMSSE 分母尺度。
    train_sales = sales_mat[:t_train]
    denom = np.mean(np.diff(train_sales, axis=0) ** 2, axis=0) if len(train_sales) > 1 else np.ones(len(chosen))
    item_scale = np.sqrt(np.maximum(denom, 1e-8)).astype(np.float32)
    # 防止个别近恒定序列的分母退化（→ 1e-4）导致 RMSSE 数值爆炸：
    # 以各序列尺度的 5% 分位为下界（至少为 1e-2）。与 tranformer.py 一致。
    scale_floor = max(float(np.percentile(item_scale, 5)), 1e-2)
    item_scale = np.maximum(item_scale, scale_floor).astype(np.float32)

    print(f"抽样商品数：{len(chosen)}")
    print(f"销量日期数：{n_sales_days}")
    print(f"完整日历日期数：{len(cal)}")
    print(f"销量矩阵形状 [日期, 商品]：{sales_mat.shape}")
    print(f"销量日期范围：{sales_dates.min().date()} 至 {sales_dates.max().date()}")
    print(f"训练/验证边界索引：{t_train}/{t_val}")
    print(f"历史特征形状：[天, 商品, 特征] = {hist_cov.shape}")
    print(f"未来协变量形状：[日历天, 商品, 特征] = {future_cov_base.shape}")

    assert hist_cov.shape[0] == sales_mat.shape[0]
    assert future_cov_base.shape[0] == len(cal)
    assert np.all(np.isfinite(hist_cov))
    assert np.all(np.isfinite(future_cov_base))
    assert np.all(sales_mat >= 0)

    return {
        "chosen": chosen,
        "sales": sales_mat,
        "hist_cov": hist_cov,
        "future_cov": future_cov_base,
        "dates": sales_dates,
        "calendar_dates": pd.DatetimeIndex(cal["date"]),
        "calendar": cal,
        "item_scale": item_scale,
        "split": (t_train, t_val),
        "item_to_idx": item_to_idx,
        "cal_features_for_lgb": cal_features,
        "snap_share": snap_share,
        "price_norm": price_norm,
        "price_available": price_available,
    }


# ============================================================
# 4. 时间窗口：目标区间严格处于 split 内
# ============================================================

def make_origins(n_days, seq_len=SEQ_LEN, pred_len=PRED_LEN, stride=STRIDE):
    t_train = int(n_days * 0.70)
    t_val = int(n_days * 0.85)
    # origin 是历史窗口起点；预测目标从 origin+seq_len 开始。
    origins = np.arange(0, n_days - seq_len - pred_len + 1, stride, dtype=int)

    train = origins[origins + seq_len + pred_len <= t_train]
    val = origins[
        (origins + seq_len >= t_train)
        & (origins + seq_len + pred_len <= t_val)
    ]
    test = origins[
        (origins + seq_len >= t_val)
        & (origins + seq_len + pred_len <= n_days)
    ]

    # 确保验证/测试各自有覆盖；不跨过对应目标区间边界。
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"窗口不足：train={len(train)}, val={len(val)}, test={len(test)}。"
            "请检查数据长度或调整窗口/切分比例。"
        )
    return train, val, test, (t_train, t_val)


def make_non_overlapping_origins(origins, pred_len=PRED_LEN):
    """从候选起点中选取目标区间不重叠的窗口，便于报告独立 28 天块的指标。"""
    selected = []
    last_target_end = -1
    for s in origins:
        target_start = int(s) + SEQ_LEN
        target_end = target_start + pred_len
        if target_start >= last_target_end:
            selected.append(int(s))
            last_target_end = target_end
    return np.asarray(selected, dtype=int)


# ============================================================
# 5. 构造 Transformer 数据
# ============================================================

class WindowDataset(Dataset):
    def __init__(self, hist, future, item_idx, y, scale):
        self.hist = torch.from_numpy(hist.astype(np.float32))
        self.future = torch.from_numpy(future.astype(np.float32))
        self.item_idx = torch.from_numpy(item_idx.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.scale = torch.from_numpy(scale.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.hist[idx], self.future[idx], self.item_idx[idx], self.y[idx], self.scale[idx]


def build_transformer_arrays(data, origins):
    sales = data["sales"]
    hist_cov = data["hist_cov"]
    future_cov = data["future_cov"]
    item_scale = data["item_scale"]
    n_days, n_items = sales.shape
    hist_list, fut_list, item_list, y_list, scale_list = [], [], [], [], []

    for origin in origins:
        target_start = int(origin) + SEQ_LEN
        target_end = target_start + PRED_LEN
        if target_end > n_days:
            continue
        hist = hist_cov[origin:target_start]  # (SEQ_LEN, item, hist_features)
        fut = future_cov[target_start:target_end]  # (PRED_LEN, item, future_features)
        target = sales[target_start:target_end]  # (PRED_LEN, item)
        for item_idx in range(n_items):
            hist_list.append(hist[:, item_idx, :])
            fut_list.append(fut[:, item_idx, :])
            item_list.append(item_idx)
            y_list.append(target[:, item_idx])
            scale_list.append(item_scale[item_idx])

    return (
        np.stack(hist_list).astype(np.float32),
        np.stack(fut_list).astype(np.float32),
        np.asarray(item_list, dtype=np.int64),
        np.stack(y_list).astype(np.float32),
        np.asarray(scale_list, dtype=np.float32).reshape(-1, 1),
    )


# ============================================================
# 6. Transformer：历史编码 + 未来已知协变量
# ============================================================

class M5TransformerForecaster(nn.Module):
    def __init__(
        self,
        hist_dim,
        future_dim,
        num_items,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=LAYERS,
        dropout=DROPOUT,
        pred_len=PRED_LEN,
        seq_len=SEQ_LEN,
        item_emb_dim=16,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除。")

        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.item_proj = nn.Linear(item_emb_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.hist_norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 未来协变量逐预测日编码，历史摘要与未来特征融合后逐日输出。
        self.future_proj = nn.Sequential(
            nn.Linear(future_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.pred_len = pred_len

    def forward(self, hist, future, item_idx):
        # hist: [B, L, Fh], future: [B, H, Ff]
        item = self.item_proj(self.item_emb(item_idx)).unsqueeze(1)
        z = self.hist_proj(hist) + item + self.pos_embedding
        z = self.hist_norm(self.encoder(z))
        # 结合末端状态和均值摘要，兼顾近期变化与整体水平。
        context = self.context_proj(torch.cat([z[:, -1, :], z.mean(dim=1)], dim=-1))
        # 将历史摘要广播到未来每一天。
        context = context.unsqueeze(1).expand(-1, self.pred_len, -1)
        fut = self.future_proj(future)
        pred_input = torch.cat([context, fut], dim=-1)
        # Softplus 约束预测销量非负。
        return torch.nn.functional.softplus(self.head(pred_input).squeeze(-1))


def scaled_mse_loss(pred, true, scale):
    """按训练期原始销量差分 RMS 尺度归一化的 MSE；不是官方 WRMSSE。"""
    scale = torch.clamp(scale, min=1e-4)
    return torch.mean(((pred - true) / scale) ** 2)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def predict_transformer(model, loader):
    model.eval()
    ys, preds, scales, items = [], [], [], []
    with torch.no_grad():
        for hist, future, item_idx, y, scale in loader:
            hist = hist.to(DEVICE)
            future = future.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            pred = model(hist, future, item_idx).cpu().numpy()
            ys.append(y.numpy())
            preds.append(pred)
            scales.append(scale.numpy())
            items.append(item_idx.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(scales),
        np.concatenate(items),
    )


def compute_metrics(y_true, y_pred, scale):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1, 1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-8))
    rmsse_per_window = np.sqrt(np.mean((err / np.maximum(scale, 1e-8)) ** 2, axis=1))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "RMSSE": float(np.mean(rmsse_per_window)),
    }


def train_transformer(model, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = min(total_steps - 1, len(train_loader) * WARMUP_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for hist, future, item_idx, y, scale in train_loader:
            hist = hist.to(DEVICE)
            future = future.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            y = y.to(DEVICE)
            scale = scale.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(hist, future, item_idx)
            loss = scaled_mse_loss(pred, y, scale)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        yv, pv, sv, _ = predict_transformer(model, val_loader)
        val_metrics = compute_metrics(yv, pv, sv)
        logs.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_RMSSE": val_metrics["RMSSE"],
            "val_MAE": val_metrics["MAE"],
            "val_WAPE": val_metrics["WAPE"],
        })
        print(
            f"Transformer | Epoch {epoch:02d}/{EPOCHS} "
            f"| train_loss={np.mean(losses):.5f} "
            f"| val_RMSSE={val_metrics['RMSSE']:.4f} "
            f"| val_WAPE={val_metrics['WAPE']:.4f}"
        )

        if val_metrics["RMSSE"] < best_val:
            best_val = val_metrics["RMSSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Transformer early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(logs).to_csv(OUT_DIR / "training_Transformer.csv", index=False)
    return model


# ============================================================
# 7. LightGBM：逐预测日使用目标日已知协变量
# ============================================================

def build_lightgbm_rows(data, origins):
    sales = data["sales"]
    hist_cov = data["hist_cov"]
    future_cov = data["future_cov"]
    n_days, n_items = sales.shape
    log_sales = np.log1p(sales)
    rows = []

    for origin in origins:
        origin = int(origin)
        hist_end = origin + SEQ_LEN - 1
        target_start = origin + SEQ_LEN
        target_end = target_start + PRED_LEN
        if target_end > n_days:
            continue

        for item_idx, item_id in enumerate(data["chosen"]):
            sl = log_sales[:, item_idx]
            # 所有历史特征只使用截至 hist_end 的数据。
            def lag(k):
                idx = hist_end - k + 1
                return float(sl[idx]) if idx >= 0 else 0.0

            def roll_mean(k):
                lo = max(0, hist_end - k + 1)
                return float(np.mean(sl[lo:hist_end + 1]))

            def roll_std(k):
                lo = max(0, hist_end - k + 1)
                return float(np.std(sl[lo:hist_end + 1]))

            hist_price = hist_cov[:hist_end + 1, item_idx, -2]
            hist_price_avail = hist_cov[:hist_end + 1, item_idx, -1]
            # 从 hist_cov 中取价格与 availability 的索引，末两列分别是 price_norm/availability。
            base = {
                "item_idx": item_idx,
                "lag_1": lag(1),
                "lag_7": lag(7),
                "lag_14": lag(14),
                "lag_21": lag(21),
                "lag_28": lag(28),
                "roll_mean_7": roll_mean(7),
                "roll_mean_14": roll_mean(14),
                "roll_mean_28": roll_mean(28),
                "roll_std_28": roll_std(28),
                "year_ago": float(sl[hist_end - 364]) if hist_end - 364 >= 0 else 0.0,
                "hist_price_last": float(hist_price[-1]) if len(hist_price) else 0.0,
                "hist_price_available": float(hist_price_avail[-1]) if len(hist_price_avail) else 0.0,
            }

            for h in range(PRED_LEN):
                target_idx = target_start + h
                # 目标日协变量：完整日历/价格表中当日可用的特征。
                f = future_cov[target_idx, item_idx]
                row = dict(base)
                row["horizon"] = h + 1
                for j, value in enumerate(f):
                    row[f"future_cov_{j}"] = float(value)
                row["target"] = float(log_sales[target_idx, item_idx])
                rows.append(row)

    return pd.DataFrame(rows)


def train_lightgbm(train_df, val_df):
    if lgb is None:
        raise ImportError("请先安装 LightGBM：pip install lightgbm")

    feature_cols = [c for c in train_df.columns if c != "target"]
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 80,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": -1,
    }
    train_set = lgb.Dataset(train_df[feature_cols], train_df["target"])
    val_set = lgb.Dataset(val_df[feature_cols], val_df["target"], reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1500,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(80, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    return model, feature_cols


def predict_lightgbm(model, feature_cols, df):
    pred_log = model.predict(df[feature_cols], num_iteration=model.best_iteration)
    return np.maximum(np.expm1(pred_log), 0.0)


# ============================================================
# 8. 评估组织与可视化
# ============================================================

def evaluate_windowed_predictions(y_true, y_pred, scales, label):
    metrics = compute_metrics(y_true, y_pred, scales)
    metrics["model"] = label
    return metrics


def reshape_lgb_predictions(pred, n_origins, n_items):
    # 构造顺序固定为 origin -> item -> horizon。
    return pred.reshape(n_origins * n_items, PRED_LEN)


def plot_forecasts(data, origin, transformer_pred, lgb_pred):
    chosen = data["chosen"]
    sales = data["sales"]
    start = int(origin) + SEQ_LEN
    end = start + PRED_LEN
    dates = data["dates"][start:end]
    actual = sales[start:end]

    # Transformer / LightGBM 输入均按 item 为连续块排列。
    trans = transformer_pred.reshape(len(chosen), PRED_LEN).T
    lgbp = lgb_pred.reshape(len(chosen), PRED_LEN).T

    n_plot = min(6, len(chosen))
    fig, axes = plt.subplots(n_plot, 1, figsize=(13, 2.8 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        axes[i].plot(dates, actual[:, i], label="Actual", linewidth=2)
        axes[i].plot(dates, trans[:, i], label="Transformer", linewidth=1.5)
        axes[i].plot(dates, lgbp[:, i], label="LightGBM", linewidth=1.5)
        axes[i].set_title(str(chosen[i]))
        axes[i].legend()
    axes[-1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_six_items.png", dpi=160)
    plt.close(fig)


def evaluate_by_horizon(y_true, y_pred, scale, model_name):
    rows = []
    for start in range(0, PRED_LEN, 7):
        end = min(start + 7, PRED_LEN)
        yt = y_true[:, start:end]
        yp = y_pred[:, start:end]
        sc = scale
        err = yp - yt
        rmsse = float(np.mean(np.sqrt(np.mean((err / np.maximum(sc, 1e-8)) ** 2, axis=1))))
        rows.append({
            "model": model_name,
            "horizon": f"{start + 1}-{end}",
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "WAPE": float(np.sum(np.abs(err)) / max(np.sum(np.abs(yt)), 1e-8)),
            "RMSSE": rmsse,
        })
    return rows


# ============================================================
# 9. 主流程
# ============================================================

def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        DATA_DIR / "sales_train_validation.csv",
        DATA_DIR / "calendar.csv",
        DATA_DIR / "sell_prices.csv",
    ]
    if not all(p.exists() for p in required):
        download_data()

    data = prepare_data()
    n_days, n_items = data["sales"].shape
    train_o, val_o, test_o, (t_train, t_val) = make_origins(n_days)
    test_nonoverlap = make_non_overlapping_origins(test_o)

    print(f"Windows: train={len(train_o)}, val={len(val_o)}, test_rolling={len(test_o)}, test_nonoverlap={len(test_nonoverlap)}")
    print(f"设备：{DEVICE}")

    # ---------- Transformer ----------
    tr_arrays = build_transformer_arrays(data, train_o)
    va_arrays = build_transformer_arrays(data, val_o)
    te_arrays = build_transformer_arrays(data, test_o)
    te_nonoverlap_arrays = build_transformer_arrays(data, test_nonoverlap)

    train_loader = DataLoader(
        WindowDataset(*tr_arrays),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        WindowDataset(*va_arrays),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        WindowDataset(*te_arrays),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_nonoverlap_loader = DataLoader(
        WindowDataset(*te_nonoverlap_arrays),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    model = M5TransformerForecaster(
        hist_dim=tr_arrays[0].shape[-1],
        future_dim=tr_arrays[1].shape[-1],
        num_items=n_items,
    )
    print("Transformer 参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = train_transformer(model, train_loader, val_loader)

    yt, pt, st, item_indices = predict_transformer(model, test_loader)
    trans_metrics = evaluate_windowed_predictions(yt, pt, st, "Transformer")
    print("Transformer test rolling:", trans_metrics)
    torch.save(model.state_dict(), OUT_DIR / "Transformer.pt")

    ytn, ptn, stn, _ = predict_transformer(model, test_nonoverlap_loader)
    trans_nonoverlap_metrics = evaluate_windowed_predictions(
        ytn, ptn, stn, "Transformer_nonoverlap"
    )

    # ---------- LightGBM ----------
    if lgb is None:
        print("未安装 LightGBM，跳过 LightGBM。安装命令：pip install lightgbm")
        lgb_model = None
        lgb_metrics = None
    else:
        print("\n构建 LightGBM 表格特征 ...")
        lgb_train_df = build_lightgbm_rows(data, train_o)
        lgb_val_df = build_lightgbm_rows(data, val_o)
        lgb_test_df = build_lightgbm_rows(data, test_o)
        lgb_test_nonoverlap_df = build_lightgbm_rows(data, test_nonoverlap)

        lgb_model, feat_cols = train_lightgbm(lgb_train_df, lgb_val_df)
        pred_lgb = predict_lightgbm(lgb_model, feat_cols, lgb_test_df)
        n_test_samples = len(test_o) * n_items
        y_lgb = lgb_test_df["target"].to_numpy()
        y_lgb = np.expm1(y_lgb).reshape(n_test_samples, PRED_LEN)
        p_lgb = reshape_lgb_predictions(pred_lgb, len(test_o), n_items)
        scale_lgb = np.tile(data["item_scale"], len(test_o)).reshape(-1, 1)
        lgb_metrics = evaluate_windowed_predictions(y_lgb, p_lgb, scale_lgb, "LightGBM")
        print("LightGBM test rolling:", lgb_metrics)
        lgb_model.save_model(str(OUT_DIR / "LightGBM.txt"))

        pred_lgb_n = predict_lightgbm(lgb_model, feat_cols, lgb_test_nonoverlap_df)
        y_lgb_n = np.expm1(lgb_test_nonoverlap_df["target"].to_numpy()).reshape(
            len(test_nonoverlap) * n_items, PRED_LEN
        )
        p_lgb_n = reshape_lgb_predictions(pred_lgb_n, len(test_nonoverlap), n_items)
        scale_lgb_n = np.tile(data["item_scale"], len(test_nonoverlap)).reshape(-1, 1)
        lgb_nonoverlap_metrics = evaluate_windowed_predictions(
            y_lgb_n, p_lgb_n, scale_lgb_n, "LightGBM_nonoverlap"
        )

    # ---------- 指标报告 ----------
    result_rows = [trans_metrics, trans_nonoverlap_metrics]
    if lgb is not None and lgb_metrics is not None:
        result_rows.extend([lgb_metrics, lgb_nonoverlap_metrics])
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n========== 指标对比 ==========")
    print(results.to_string(index=False))

    # 按预测步长分组报告（滚动测试窗口）。
    horizon_rows = evaluate_by_horizon(yt, pt, st, "Transformer")
    if lgb is not None and lgb_metrics is not None:
        horizon_rows.extend(evaluate_by_horizon(y_lgb, p_lgb, scale_lgb, "LightGBM"))
    pd.DataFrame(horizon_rows).to_csv(OUT_DIR / "metrics_by_horizon.csv", index=False)

    # 绘制测试集中最后一个合法窗口。
    final_origin = int(test_o[-1])
    final_arrays = build_transformer_arrays(data, np.asarray([final_origin]))
    final_loader = DataLoader(
        WindowDataset(*final_arrays), batch_size=len(final_arrays[0]), shuffle=False
    )
    _, final_trans, _, _ = predict_transformer(model, final_loader)

    if lgb is not None and lgb_metrics is not None:
        final_df = build_lightgbm_rows(data, np.asarray([final_origin]))
        final_lgb = predict_lightgbm(lgb_model, feat_cols, final_df)
        plot_forecasts(data, final_origin, final_trans, final_lgb)
    else:
        # LightGBM 缺失时仍输出 Transformer 图。
        chosen = data["chosen"]
        start = final_origin + SEQ_LEN
        dates = data["dates"][start:start + PRED_LEN]
        actual = data["sales"][start:start + PRED_LEN]
        pred = final_trans.reshape(len(chosen), PRED_LEN).T
        fig, axes = plt.subplots(min(6, len(chosen)), 1, figsize=(13, 15), sharex=True)
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            ax.plot(dates, actual[:, i], label="Actual")
            ax.plot(dates, pred[:, i], label="Transformer")
            ax.set_title(str(chosen[i]))
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_six_items.png", dpi=160)
        plt.close(fig)

    print("\n完成。输出目录：", OUT_DIR.resolve())
    print("输出：metrics.csv、metrics_by_horizon.csv、training_Transformer.csv、Transformer.pt、forecast_six_items.png")
    print("说明：当前数据是 50 个商品跨门店汇总序列，指标不是官方 M5 WRMSSE。")


if __name__ == "__main__":
    main()
