#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer 模型：纯编码器 + 可学习位置编码 + 序列归一化损失（RMSSE 风格）。

数据来自 data_process（store-item 粒度）。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    SEQ_LEN, PRED_LEN, D_MODEL, N_HEADS, LAYERS, D_EMB_ITEM, DROPOUT,
    BATCH_SIZE, EPOCHS, LR_Peak, WARMUP_EPOCHS, PATIENCE, DEVICE, OUT_DIR,
    rmsse_loss, get_cosine_schedule_with_warmup, compute_metrics,
    transformer_predict, WindowDataset, compute_rmsse_scale,
)


# ============================================================
# 1. 模型
# ============================================================

class M5TransformerForecaster(nn.Module):
    """
    特征融合：
      历史数值特征 -> 线性投影到 d_model；
      实体(商品) embedding -> 融合向量，广播加到每个历史时间步；
      叠加可学习位置编码；
      TransformerEncoder -> 结合「末端状态 + 均值」摘要；
      历史摘要与「未来已知协变量」逐预测日融合 -> 回归头输出 28 天（softplus 保证非负）。
    """

    def __init__(
        self,
        hist_dim,
        future_dim,
        num_items,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=LAYERS,
        dropout=DROPOUT,
        d_emb_item=D_EMB_ITEM,
        pred_len=PRED_LEN,
        seq_len=SEQ_LEN,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除。")

        self.input_proj = nn.Linear(hist_dim, d_model)

        self.item_emb = nn.Embedding(num_items, d_emb_item)
        self.emb_fusion = nn.Linear(d_emb_item, d_model)

        # 可学习位置编码（序列长度固定为 SEQ_LEN，不需要正弦外推）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.hist_norm = nn.LayerNorm(d_model)

        # 结合末端状态与均值，兼顾近期变化与整体水平
        self.context_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # 未来协变量逐预测日编码
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

    def forward(self, x, future, item_idx):
        # x: (B, SEQ, hist_dim); future: (B, PRED, future_dim)
        x = self.input_proj(x)

        emb = self.emb_fusion(self.item_emb(item_idx)).unsqueeze(1)  # (B,1,D)
        x = x + emb + self.pos_embedding

        x = self.hist_norm(self.encoder(x))
        context = self.context_proj(
            torch.cat([x[:, -1, :], x.mean(dim=1)], dim=-1)
        )                                                 # (B, D)
        context = context.unsqueeze(1).expand(-1, self.pred_len, -1)
        fut = self.future_proj(future)                    # (B, PRED, D)
        pred_input = torch.cat([context, fut], dim=-1)     # (B, PRED, 2D)
        # softplus 约束预测销量非负
        return torch.nn.functional.softplus(self.head(pred_input).squeeze(-1))


# ============================================================
# 2. 构造 Transformer 训练样本（store-item 粒度，带有效性守卫）
# ============================================================

def build_transformer_data(prepared, origins):
    """
    返回 X(B,SEQ,hist_feat) / future(B,PRED,fut_feat) / item_idx(B) / y(B,PRED) / scale(B,1)。
    - 价格/日历/SNAP/事件 等协变量直接取自 data_process 的归一化结果，仅在训练期拟合。
    - 逐实体 RMSSE 尺度取自 compute_rmsse_scale（训练期有效原始销量的一阶差分 RMS）。
    - 窗口内销量/价格存在无效（价格首现周前或NaN）时跳过，避免污染训练。
    通道顺序：
      hist_feat(8): log销量, 价格归一, wday, month, year, SNAP, 事件, 价格可用性
      fut_feat(7) : 价格归一, wday, month, year, SNAP, 事件, 价格可用性  （不含销量）
    """
    sales = np.asarray(prepared["sales"], dtype=np.float32)         # (n_days, n_ent)
    valid = np.asarray(prepared["sales_valid"], dtype=bool)
    price_norm = np.asarray(prepared["price_norm"], dtype=np.float32)
    snap = np.asarray(prepared["snap"], dtype=np.float32)
    cal = np.asarray(prepared["cal_feat"], dtype=np.float32)         # (n_days, 4)
    price_avail = np.asarray(prepared["price_avail"], dtype=np.float32)
    start_day = np.asarray(prepared["start_day"], dtype=np.int64)
    n_days, n_ent = sales.shape

    with np.errstate(invalid="ignore"):
        log_sales = np.log1p(np.maximum(sales, 0.0)).astype(np.float32)

    cal_bc = np.broadcast_to(cal[:, None, :], (n_days, n_ent, 4)).astype(np.float32)

    # 历史特征：log 销量 + 价格 + 日历 + SNAP + 事件 + 价格可用性
    hist_feat = np.stack(
        [
            log_sales,
            price_norm,
            cal_bc[:, :, 0],
            cal_bc[:, :, 1],
            cal_bc[:, :, 2],
            snap,
            cal_bc[:, :, 3],
            price_avail,
        ],
        axis=-1,
    )  # (n_days, n_ent, 8)

    # 未来协变量（目标区间已知，不泄漏未来销量）
    fut_feat = np.stack(
        [
            price_norm,
            cal_bc[:, :, 0],
            cal_bc[:, :, 1],
            cal_bc[:, :, 2],
            snap,
            cal_bc[:, :, 3],
            price_avail,
        ],
        axis=-1,
    )  # (n_days, n_ent, 7)

    scale = compute_rmsse_scale(prepared).reshape(1, n_ent)         # (1, n_ent)

    Xs, Fs, Ys, I, S = [], [], [], [], []
    for s in origins:
        hist_slice = slice(s, s + SEQ_LEN)
        fut_slice = slice(s + SEQ_LEN, s + SEQ_LEN + PRED_LEN)
        for j in range(n_ent):
            # 有效性守卫：窗口内销量/价格必须全有效，否则跳过该 (实体,原点)。
            if not (valid[hist_slice, j].all() and valid[fut_slice, j].all()):
                continue
            if not (np.isfinite(hist_feat[hist_slice, j]).all()
                    and np.isfinite(fut_feat[fut_slice, j]).all()):
                continue
            Xs.append(hist_feat[hist_slice, j])
            Fs.append(fut_feat[fut_slice, j])
            Ys.append(log_sales[fut_slice, j])
            I.append(j)
            S.append(scale[0, j])

    if not Xs:
        raise ValueError("build_transformer_data：没有可用样本，请检查 sales_valid 与 origins。")

    X = np.stack(Xs).astype(np.float32)         # (B, SEQ, 8)
    F = np.stack(Fs).astype(np.float32)         # (B, PRED, 7)
    y = np.stack(Ys).astype(np.float32)         # (B, PRED)
    item_idx = np.array(I, dtype=np.int64)
    sc = np.array(S, dtype=np.float32).reshape(-1, 1)
    return X, F, item_idx, y, sc


# ============================================================
# 3. 训练
# ============================================================

def train_transformer(model, train_loader, val_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_Peak, weight_decay=1e-4
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_val = float("inf")
    best_state = None
    bad = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for X, future, item_idx, y, sc in train_loader:
            X = X.to(DEVICE)
            future = future.to(DEVICE)
            item_idx = item_idx.to(DEVICE)
            y = y.to(DEVICE)
            sc = sc.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X, future, item_idx)
            loss = rmsse_loss(pred, y, sc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        yt, pt, st = transformer_predict(model, val_loader)
        val = compute_metrics(yt, pt, st)
        logs.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_rmsse": val["RMSSE"],
            "val_mae": val["MAE"],
        })
        print(
            f"Transformer | Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss={np.mean(losses):.5f} | "
            f"Val RMSSE={val['RMSSE']:.4f} | "
            f"Val MAE={val['MAE']:.4f}"
        )

        if val["RMSSE"] < best_val:
            best_val = val["RMSSE"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    import pandas as pd
    pd.DataFrame(logs).to_csv(OUT_DIR / "training_Transformer.csv", index=False)
    return model
