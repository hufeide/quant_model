#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 复现实验的通用配置、工具与数据集（被各模型模块共享）。

数据一律来自同目录的 data_process 模块（store-item 粒度）。
"""

import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 路径与基础常量复用 data_process，避免到处重复定义。
from data_process import DATA_DIR, OUT_DIR, BASE_DIR, SEED, N_ITEMS


# ============================================================
# 1. 配置（与 tranformer.md 一致）
# ============================================================

SEQ_LEN = 56          # 输入历史窗口（md: 56 天）
PRED_LEN = 28         # 预测步长（md: 28 天）
STRIDE = 7            # 滑窗步长（md: 7，避免样本过度重叠）

# Transformer 超参（取自 md 第 3 节）
D_MODEL = 128
N_HEADS = 8
LAYERS = 4
D_EMB_ITEM = 16
DROPOUT = 0.1

# 训练
BATCH_SIZE = 64
EPOCHS = 20
LR_Peak = 2e-4
WARMUP_EPOCHS = 5
PATIENCE = 5

# PatchTST / iTransformer 超参（取自各自原脚本，保持可比；与上方 Transformer 互不干扰）
HISTORY2 = 112        # 两模型回看历史（原脚本 HISTORY=112）
PT_CONTEXT = HISTORY2 + PRED_LEN   # 输入上下文长度：历史 112 + 预测 28（含未来协变量）
PATCH_LEN = 16        # 必须定义：PatchTSTEncoder 默认参数在类定义时求值
PATCH_STRIDE = 8
PT_D_MODEL = 64
PT_N_HEADS = 4
PT_LAYERS = 2
PT_DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 时间轴切分（按目标区间，保证不泄漏）
# ============================================================

def make_origins(n_days, seq_len, pred_len, stride):
    """
    按「时间轴」切分，保证 train/val/test 的目标区间 [s+seq, s+seq+pred) 互不重叠，
    且不越过对应 split 边界（目标不泄漏）。历史允许跨越边界。

      T_train = 70% 天；T_val = 85% 天
      train: 目标结束 <= T_train
      val  : 历史末端 >= T_train 且 目标结束 <= T_val
      test : 历史末端 >= T_val 且 目标结束 <= n_days
    """
    T_train = int(n_days * 0.70)
    T_val = int(n_days * 0.85)
    origins = np.arange(0, n_days - seq_len - pred_len + 1, stride)

    # 目标区间严格落在对应 split 内；历史允许跨越边界（不泄漏目标）。
    train = origins[origins + seq_len + pred_len <= T_train]
    val = origins[
        (origins + seq_len >= T_train) & (origins + seq_len + pred_len <= T_val)
    ]
    test = origins[
        (origins + seq_len >= T_val) & (origins + seq_len + pred_len <= n_days)
    ]
    return train, val, test, (T_train, T_val)


# ============================================================
# 3. 序列归一化损失（md 第 4.1 节：RMSSE 风格）
# ============================================================

def rmsse_loss(pred, true, scale):
    """scale: (B,1) 各序列训练期原始销量差分 RMS；返回 squared-RMSSE（指标里开方）。"""
    scale = torch.clamp(scale, min=1e-4)
    diff = (pred - true) / scale
    return torch.mean(diff ** 2)


def get_cosine_schedule_with_warmup(optimizer, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        prog = float(step - num_warmup) / float(max(1, num_total - num_warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 4. 评估指标
# ============================================================

def compute_metrics(y_true, y_pred, scale_per_sample):
    """
    y_true, y_pred: (B, PRED) 原始销量
    scale_per_sample: (B,1) 各序列归一化因子
    返回 MAE / RMSE / WAPE / 平均 RMSSE（md 采用的官方指标风格）
    """
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-8))

    # 逐样本 RMSSE 再平均
    sc = scale_per_sample.reshape(-1, 1)
    per = np.sqrt(np.mean(((y_pred - y_true) / sc) ** 2, axis=1))
    rmsse = float(np.mean(per))
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "RMSSE": rmsse}


@torch.no_grad()
def transformer_predict(model, loader):
    model.eval()
    preds, trues, scales = [], [], []
    for X, future, item_idx, y, sc in loader:
        X = X.to(DEVICE)
        future = future.to(DEVICE)
        item_idx = item_idx.to(DEVICE)
        p = model(X, future, item_idx).cpu().numpy()
        preds.append(p)
        trues.append(y.numpy())
        scales.append(sc.numpy())
    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(scales),
    )


# ============================================================
# 5. 数据集
# ============================================================

class WindowDataset(Dataset):
    """Transformer 用：返回 (X, future, item_idx, y, scale)。"""

    def __init__(self, X, future, item_idx, y, scale):
        self.X = X
        self.future = future
        self.item_idx = item_idx
        self.y = y
        self.scale = scale

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i]),
            torch.tensor(self.future[i]),
            torch.tensor(self.item_idx[i]),
            torch.tensor(self.y[i]),
            torch.tensor(self.scale[i]),
        )


class SalesSeriesDataset(Dataset):
    """纯销量序列窗口：x=[history, N]，y=[horizon, N]。"""

    def __init__(self, values, origins, history, horizon):
        self.values = values
        self.origins = origins
        self.history = history
        self.horizon = horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        o = self.origins[idx]
        x = self.values[o - self.history:o]
        y = self.values[o:o + self.horizon]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class CovariateSeriesDataset(Dataset):
    """销量 + 协变量序列窗口，供 PatchTST / iTransformer 使用。
    返回 x=[win, N, D]（win = history + horizon）：
      - 通道 0：逐日销量（已 log1p 标准化），未来段（最后 horizon 天）置 0 以屏蔽目标；
      - 通道 1..：协变量（价格/日历/SNAP/事件/价格可用性），历史与未来段均提供（不泄漏销量）。
    y=[horizon, N]：未来 horizon 天销量（标准化空间，与 patchTS/itranformer 一致）。
    """

    def __init__(self, sales, cov, origins, history, horizon):
        self.sales = sales            # [n_days, n_items]（标准化后销量）
        self.cov = cov                # [n_days, n_items, D_cov]
        self.origins = origins
        self.history = history
        self.horizon = horizon
        self.win = history + horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        o = self.origins[idx]
        ctx_sales = self.sales[o - self.history:o + self.horizon]   # [win, N]
        ctx_cov = self.cov[o - self.history:o + self.horizon]       # [win, N, D_cov]
        x = np.concatenate([ctx_sales[..., None], ctx_cov], axis=-1)  # [win, N, D]
        x[self.history:, :, 0] = 0.0    # 屏蔽未来销量，仅保留协变量可见
        y = self.sales[o:o + self.horizon]                           # [horizon, N]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ============================================================
# 6. 全局标准化 log-sales 缩放（PatchTST / iTransformer 用）
# ============================================================

def fit_log_scaler(train_raw):
    # 对非负销量先 log1p，再按商品标准化（仅用训练期）。
    # 注意：晚上市实体的首现日前销量为 NaN，须用 nan 感知聚合，否则该实体
    # 的 mean/std 会变成 NaN，进而整列 values 变 NaN 导致模型 loss 非有限。
    logged = np.log1p(np.maximum(train_raw, 0))
    mean = np.nanmean(logged, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(logged, axis=0, keepdims=True).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-5] = 1.0
    return mean, std


def log_transform(raw, mean, std):
    return ((np.log1p(np.maximum(raw, 0)) - mean) / std).astype(np.float32)


def log_inverse(x, mean, std):
    return np.maximum(np.expm1(x * std + mean), 0)


# ============================================================
# 7. RMSSE 逐实体尺度（与 build_transformer_data / LightGBM 共用）
# ============================================================

def compute_rmsse_scale(prepared):
    """逐实体（store-item）训练期有效「原始销量」的一阶差分 RMS，带 5% 分位下界。"""
    sales = np.asarray(prepared["sales"], dtype=np.float32)
    valid = np.asarray(prepared["sales_valid"], dtype=bool)
    start_day = np.asarray(prepared["start_day"], dtype=np.int64)
    n_days, n_ent = sales.shape
    train_end = int(prepared.get("train_end", int(n_days * 0.70)))

    scale = np.zeros(n_ent, dtype=np.float32)
    for j in range(n_ent):
        s0 = int(start_day[j])
        seg = slice(max(s0, 0), train_end)
        sv = sales[seg, j][valid[seg, j]]
        if sv.size > 1:
            denom = np.mean(np.diff(sv) ** 2)
        else:
            denom = 1.0
        scale[j] = np.sqrt(max(denom, 1e-8))

    if np.any(scale > 0):
        floor = max(float(np.percentile(scale[scale > 0], 5)), 1e-2)
    else:
        floor = 1e-2
    return np.maximum(scale, floor).astype(np.float32)


# ============================================================
# 8. 绘制对比
# ============================================================

def plot_compare(dates, y_true, preds, item_ids):
    import matplotlib.pyplot as plt

    actual = y_true[0]  # (PRED, n_items)

    plt.figure(figsize=(13, 5))
    for name, p in preds.items():
        plt.plot(dates, p[0, :, 0], label=name, linewidth=1.8)
    plt.plot(dates, actual[:, 0], label="Actual", linewidth=2.5, marker="o")
    plt.title(f"28-Day Forecast: {item_ids[0]}")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forecast_item.png", dpi=160)
    plt.close()

    n_plot = min(6, len(item_ids))
    fig, axes = plt.subplots(n_plot, 1, figsize=(13, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        axes[i].plot(dates, actual[:, i], label="Actual", linewidth=2)
        for name, p in preds.items():
            axes[i].plot(dates, p[0, :, i], label=name)
        axes[i].set_title(item_ids[i])
        axes[i].legend()
    axes[-1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forecast_six_items.png", dpi=160)
    plt.close(fig)
