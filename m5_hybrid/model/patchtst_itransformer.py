#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PatchTST / iTransformer 对比模型（store-item 粒度，带协变量）。

销量与协变量由 run.py 用 data_process 的产出组装后送入 CovariateSeriesDataset。
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    HISTORY2, PT_CONTEXT, PATCH_LEN, PATCH_STRIDE, PT_D_MODEL, PT_N_HEADS,
    PT_LAYERS, PT_DROPOUT, PRED_LEN, BATCH_SIZE, EPOCHS, PATIENCE, DEVICE,
    CovariateSeriesDataset, fit_log_scaler, log_transform, log_inverse, OUT_DIR,
)


# ============================================================
# 1. PatchTST
# ============================================================

class PatchTSTEncoder(nn.Module):
    """输入 [B,T,N,D]：每个商品 n 在时刻 t 的特征向量 [销量, +协变量]。
    仅对销量通道做实例归一化（用历史段，与原始 PatchTST 行为一致），协变量原样拼接，
    再沿时间维切 Patch，输出 [B,N,D_model]。"""

    def __init__(self, history=HISTORY2, d_model=PT_D_MODEL, n_heads=PT_N_HEADS,
                 patch_len=PATCH_LEN, stride=PATCH_STRIDE, dropout=PT_DROPOUT,
                 layers=PT_LAYERS, d_input=1, hist_len=None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.hist_len = hist_len if hist_len is not None else history
        n_patches = math.ceil((history - patch_len) / stride) + 1
        self.patch_embed = nn.Linear(patch_len * d_input, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, N, D]，通道 0 为销量，1: 为协变量
        B, T, N, D = x.shape
        sales = x[..., 0:1]                         # [B, T, N, 1]
        cov = x[..., 1:]                            # [B, T, N, D-1]
        # 仅用历史段对销量做实例归一化（与原始 PatchTST 行为一致）
        hist_sales = sales[:, :self.hist_len]
        mean = hist_sales.mean(dim=1, keepdim=True)
        std = hist_sales.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        sales_n = (sales - mean) / std
        # 未来销量输入原本是占位 0；标准化后必须再次置零，避免产生人为常数信号。
        sales_n = sales_n.clone()
        sales_n[:, self.hist_len:] = 0.0
        x = torch.cat([sales_n, cov], dim=-1)       # [B, T, N, D]
        x = x.transpose(1, 2)                       # [B, N, T, D]
        # 右侧补齐，使最后一个 patch 覆盖序列末端，避免丢掉最后几天。
        remainder = (T - self.patch_len) % self.stride
        pad_len = (self.stride - remainder) % self.stride
        if pad_len:
            x = torch.cat([x, x.new_zeros(B, N, pad_len, D)], dim=2)
        patches = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        b, n, p, length, d = patches.shape
        tokens = patches.reshape(b * n, p, length * d)
        tokens = self.patch_embed(tokens)
        tokens = tokens + self.pos_embed[:, :p, :]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        # 均值捕捉全局模式，最后一个 patch 强调近期状态。
        pooled = torch.cat([tokens.mean(dim=1), tokens[:, -1, :]], dim=-1)
        item_repr = pooled.reshape(b, n, -1)
        return item_repr


class PatchTSTBaseline(nn.Module):
    def __init__(self, history=HISTORY2, horizon=PRED_LEN, d_input=1,
                 hist_len=None, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", PT_D_MODEL)
        dropout = kwargs.get("dropout", PT_DROPOUT)
        self.temporal = PatchTSTEncoder(
            history=history, d_model=d_model,
            n_heads=kwargs.get("n_heads", PT_N_HEADS),
            patch_len=kwargs.get("patch_len", PATCH_LEN),
            stride=kwargs.get("stride", PATCH_STRIDE),
            dropout=dropout, layers=kwargs.get("layers", PT_LAYERS),
            d_input=d_input, hist_len=hist_len,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # 模型内部按每条序列历史段实例归一化；预测后必须还原到输入的
        # 全局标准化 log-sales 空间，否则会把实例标准化空间误当作训练标签空间。
        sales = x[..., 0]
        hist = sales[:, :self.temporal.hist_len]
        mean = hist.mean(dim=1)                         # [B,N]
        std = hist.std(dim=1, unbiased=False).clamp_min(1e-4)
        h = self.temporal(x)
        pred = self.head(h).transpose(1, 2)             # [B,H,N]
        return pred * std.unsqueeze(1) + mean.unsqueeze(1)


# ============================================================
# 2. iTransformer
# ============================================================

class iTransformer(nn.Module):
    """Inverted Transformer（带协变量）：输入 [B,T,N,D]，通道 0 为销量，1: 为协变量。
    对每个商品把其 [T,D] 时间序列（销量实例归一化后 + 协变量）压扁为单个 token，
    在变量（商品）维度做自注意力，输出 [B,H,N]。"""

    def __init__(self, history=HISTORY2, horizon=PRED_LEN, n_vars=None,
                 d_model=PT_D_MODEL, n_heads=PT_N_HEADS,
                 dropout=PT_DROPOUT, layers=PT_LAYERS,
                 d_input=1, hist_len=None):
        super().__init__()
        self.d_input = d_input
        self.hist_len = hist_len if hist_len is not None else history
        self.value_embed = nn.Linear(history * d_input, d_model)
        if n_vars is not None:
            self.var_pos_embed = nn.Parameter(torch.zeros(1, n_vars, d_model))
        else:
            self.var_pos_embed = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # x: [B, T, N, D]
        B, T, N, D = x.shape
        sales = x[..., 0:1]                         # [B, T, N, 1]
        cov = x[..., 1:]                            # [B, T, N, D-1]
        # 仅用历史段对销量做实例归一化（与原始 iTransformer 行为一致）
        hist_sales = sales[:, :self.hist_len]
        mean = hist_sales.mean(dim=1, keepdim=True)
        std = hist_sales.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        sales_n = (sales - mean) / std
        # 未来销量输入原本是占位 0；标准化后必须再次置零，避免产生人为常数信号。
        sales_n = sales_n.clone()
        sales_n[:, self.hist_len:] = 0.0
        x = torch.cat([sales_n, cov], dim=-1)       # [B, T, N, D]
        x = x.permute(0, 2, 1, 3).reshape(B, N, T * D)
        tokens = self.value_embed(x)               # [B,N,d_model]
        if self.var_pos_embed is not None:
            tokens = tokens + self.var_pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        y = self.head(tokens)                      # [B,N,H]，实例归一化空间
        # 标签仍处于全局标准化 log-sales 空间，需反变换实例归一化。
        y = y * std[:, 0, :, 0].unsqueeze(-1) + mean[:, 0, :, 0].unsqueeze(-1)
        return y.transpose(1, 2)                   # [B,H,N]


# ============================================================
# 3. 训练与评估
# ============================================================

def train_patch_model(model, train_loader, val_loader, mean, std, model_name):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    logs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{model_name} loss 非有限值: {loss.item()}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        # 验证（反归一化后比较 MAE，与 patchTS/itranformer 原脚本一致）
        model.eval()
        vs, vp = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                vs.append(y.numpy())
                vp.append(model(x).cpu().numpy())
        yt = log_inverse(np.concatenate(vs), mean, std)
        yp = log_inverse(np.concatenate(vp), mean, std)
        val_mae = float(np.mean(np.abs(yt - yp)))
        logs.append({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "val_mae": val_mae,
        })
        print(f"{model_name} | Epoch {epoch:02d}/{EPOCHS} | "
              f"Train MSE={np.mean(losses):.5f} | Val MAE={val_mae:.4f}")
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    import pandas as pd
    pd.DataFrame(logs).to_csv(OUT_DIR / f"training_{model_name}.csv", index=False)
    return model


@torch.no_grad()
def eval_patch_model(model, loader, mean, std):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        ys.append(y.numpy())
        ps.append(model(x).cpu().numpy())
    yt = log_inverse(np.concatenate(ys), mean, std)
    yp = log_inverse(np.concatenate(ps), mean, std)
    return yt, yp                               # 均为 [n_windows, H, N]
