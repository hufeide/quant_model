#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 四模型对照实验入口（store-item 粒度）。

数据统一来自 data_process；本文件负责编排四个模型的训练、评估、对比与绘图。
运行：python run.py   （在 model/ 目录下；数据缺失会自动触发下载）
"""

import numpy as np
import pandas as pd
import torch

from common import (
    SEED, SEQ_LEN, PRED_LEN, STRIDE, N_ITEMS, DEVICE, OUT_DIR, DATA_DIR, BATCH_SIZE,
    HISTORY2, PT_CONTEXT, PT_D_MODEL, PT_N_HEADS, PT_LAYERS, PATCH_LEN, PATCH_STRIDE, PT_DROPOUT,
    make_origins, compute_metrics, plot_compare, transformer_predict,
    WindowDataset, CovariateSeriesDataset, fit_log_scaler, log_transform,
    compute_rmsse_scale, seed_everything,
)
from data_process import download_data, prepare_data, build_lightgbm_rows
from transformer_model import M5TransformerForecaster, build_transformer_data, train_transformer
from lightgbm_model import train_lightgbm, lightgbm_predict, split_lightgbm
from patchtst_itransformer import (
    PatchTSTBaseline, iTransformer, train_patch_model, eval_patch_model,
)
from torch.utils.data import DataLoader


def build_cov(prepared):
    """组装 PatchTST / iTransformer 的协变量张量 (n_days, n_ent, 7)。

    通道顺序与 Transformer 的 fut_feat 完全一致（均为已知未来信息，不泄漏销量）：
      价格归一, wday, month, year, SNAP, 事件, 价格可用性
    """
    n_days, n_ent = prepared["sales"].shape
    cal = np.asarray(prepared["cal_feat"], dtype=np.float32)   # (n_days, 4)
    cal_b = np.broadcast_to(cal[:, None, :], (n_days, n_ent, 4)).astype(np.float32)
    cov = np.stack([
        prepared["price_norm"],
        cal_b[..., 0],
        cal_b[..., 1],
        cal_b[..., 2],
        prepared["snap"],
        cal_b[..., 3],
        prepared["price_avail"],
    ], axis=-1).astype(np.float32)
    return cov


def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not all(
        (DATA_DIR / f).exists() for f in [
            "sales_train_validation.csv", "calendar.csv", "sell_prices.csv",
        ]
    ):
        download_data()

    prepared = prepare_data()
    dates = prepared["dates"]
    n_days, n_ent = prepared["sales"].shape
    entities = prepared["entities"]

    train_o, val_o, test_o, (T_train, T_val) = make_origins(
        n_days, SEQ_LEN, PRED_LEN, STRIDE
    )
    if min(len(train_o), len(val_o), len(test_o)) == 0:
        raise ValueError("滑窗切分后没有足够的训练/验证/测试窗口。")

    print(f"Windows: train={len(train_o)}, val={len(val_o)}, test={len(test_o)}")

    # ---------- Transformer ----------
    Xtr, Ftr, itr, ytr, str_ = build_transformer_data(prepared, train_o)
    Xva, Fva, iva, yva, sva = build_transformer_data(prepared, val_o)
    Xte, Fte, ite, yte, ste = build_transformer_data(prepared, test_o)

    train_loader = DataLoader(
        WindowDataset(Xtr, Ftr, itr, ytr, str_), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        WindowDataset(Xva, Fva, iva, yva, sva), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        WindowDataset(Xte, Fte, ite, yte, ste), batch_size=BATCH_SIZE, shuffle=False
    )

    model = M5TransformerForecaster(
        hist_dim=Xtr.shape[-1], future_dim=Ftr.shape[-1], num_items=n_ent
    )
    print("Transformer 参数量:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = train_transformer(model, train_loader, val_loader)

    yt, pt, st = transformer_predict(model, test_loader)
    transformer_metrics = compute_metrics(yt, pt, st)
    transformer_metrics["model"] = "Transformer(归一化损失+可学习位置编码)"
    print("Transformer Test:", transformer_metrics)

    import torch
    torch.save(model.state_dict(), OUT_DIR / "Transformer.pt")

    # ---------- LightGBM ----------
    print("\n训练 LightGBM ...")
    lgb_df = build_lightgbm_rows(prepared, history=SEQ_LEN, horizon=PRED_LEN, stride=STRIDE)
    lgb_train_df, lgb_val_df, lgb_test_df = split_lightgbm(
        lgb_df, n_days, SEQ_LEN, PRED_LEN
    )
    print(f"  LightGBM 切分: train={len(lgb_train_df)}, val={len(lgb_val_df)}, "
          f"test={len(lgb_test_df)}")

    bst, feat_cols = train_lightgbm(lgb_train_df, lgb_val_df)

    yte_lgb = lightgbm_predict(bst, feat_cols, lgb_test_df)   # (n_test_rows,)
    # 还原为 (B, PRED)：行按 (实体, 窗口, h) 顺序生成，每 PRED 行 = 一个样本。
    n_test_samples = len(lgb_test_df) // PRED_LEN
    y_pred_lgb = yte_lgb.reshape(n_test_samples, PRED_LEN)
    y_true_lgb = lgb_test_df["target"].to_numpy().reshape(n_test_samples, PRED_LEN)
    # 逐样本 RMSSE 尺度：取每样本首行的实体标识对应的逐实体尺度。
    ent_per_sample = lgb_test_df["entity_idx"].to_numpy().reshape(n_test_samples, PRED_LEN)[:, 0]
    scale_lgb = compute_rmsse_scale(prepared)[ent_per_sample].reshape(-1, 1)
    lightgbm_metrics = compute_metrics(y_true_lgb, y_pred_lgb, scale_lgb)
    lightgbm_metrics["model"] = "LightGBM(交叉特征+滞后特征)"
    print("LightGBM Test:", lightgbm_metrics)

    bst.save_model(str(OUT_DIR / "LightGBM.txt"))

    # ---------- PatchTST / iTransformer ----------
    # 对齐到与 Transformer 相同的目标区间：以 p = s+SEQ_LEN 为「预测起始日」原点。
    # store-item 粒度下，价格首现周前销量可能为 NaN，须过滤掉窗口内存在无效格的起点，
    # 否则协变量序列的样本会含 NaN 导致训练/评估失败（Transformer/LightGBM 各自已有守卫）。
    valid_all = prepared["sales_valid"]

    def window_valid(o):
        return bool(valid_all[o - HISTORY2:o + PRED_LEN, :].all())

    pstarts_train = np.array(
        [o for o in (train_o + SEQ_LEN) if o >= HISTORY2 and window_valid(o)]
    )
    pstarts_val = np.array(
        [o for o in (val_o + SEQ_LEN) if o >= HISTORY2 and window_valid(o)]
    )
    pstarts_test = np.array(
        [o for o in (test_o + SEQ_LEN) if o >= HISTORY2 and window_valid(o)]
    )
    if min(len(pstarts_train), len(pstarts_val), len(pstarts_test)) == 0:
        raise ValueError("PatchTST/iTransformer 没有可用的有效窗口，请检查 sales_valid。")

    # 取最后一个「目标区间有效」的测试原点作为绘图窗口，保证三类模型在同一窗口上对比。
    cands = [o for o in test_o if (o + SEQ_LEN) >= HISTORY2 and window_valid(o + SEQ_LEN)]
    final_origin = int(cands[-1]) if cands else int(test_o[-1])

    raw = prepared["sales"].astype(np.float32)              # (n_days, n_ent)
    T_train_days = int(n_days * 0.70)
    mean, std = fit_log_scaler(raw[:T_train_days])
    values = log_transform(raw, mean, std)                  # (n_days, n_ent)

    cov = build_cov(prepared)
    D_INPUT = cov.shape[-1] + 1                             # 销量通道 + 协变量通道

    patch_models = {
        "PatchTST": PatchTSTBaseline(
            history=PT_CONTEXT, horizon=PRED_LEN, d_input=D_INPUT, hist_len=HISTORY2,
            d_model=PT_D_MODEL, n_heads=PT_N_HEADS, layers=PT_LAYERS,
            patch_len=PATCH_LEN, stride=PATCH_STRIDE, dropout=PT_DROPOUT,
        ),
        "iTransformer": iTransformer(
            history=PT_CONTEXT, horizon=PRED_LEN, n_vars=n_ent,
            d_input=D_INPUT, hist_len=HISTORY2,
            d_model=PT_D_MODEL, n_heads=PT_N_HEADS, layers=PT_LAYERS,
            dropout=PT_DROPOUT,
        ),
    }

    entity_scale = compute_rmsse_scale(prepared)            # (n_ent,)
    all_metrics = [transformer_metrics, lightgbm_metrics]
    patch_preds = {}
    final_origin_plot = final_origin + SEQ_LEN             # 对齐 test_dates 起点
    if final_origin_plot < HISTORY2 or final_origin_plot + PRED_LEN > n_days:
        raise ValueError("PatchTST/iTransformer 最后预测窗口越界。")

    for name, m in patch_models.items():
        print("\n训练:", name)
        dl_tr = DataLoader(
            CovariateSeriesDataset(values, cov, pstarts_train, HISTORY2, PRED_LEN),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        dl_va = DataLoader(
            CovariateSeriesDataset(values, cov, pstarts_val, HISTORY2, PRED_LEN),
            batch_size=1, shuffle=False,
        )
        dl_te = DataLoader(
            CovariateSeriesDataset(values, cov, pstarts_test, HISTORY2, PRED_LEN),
            batch_size=1, shuffle=False,
        )
        dl_fin = DataLoader(
            CovariateSeriesDataset(values, cov, np.array([final_origin_plot]), HISTORY2, PRED_LEN),
            batch_size=1, shuffle=False,
        )

        m = train_patch_model(m, dl_tr, dl_va, mean, std, name)
        yt_p, yp_p = eval_patch_model(m, dl_te, mean, std)
        # yt_p/yp_p 形状 (n_test, H, N)；每个 (窗口, 实体) 视为一个长度为 H 的序列样本
        B = yt_p.shape[0] * yt_p.shape[2]
        yt_f = yt_p.reshape(B, PRED_LEN)
        yp_f = yp_p.reshape(B, PRED_LEN)
        sc_f = np.tile(entity_scale, yt_p.shape[0]).reshape(B, 1)
        met = compute_metrics(yt_f, yp_f, sc_f)
        met["model"] = name
        met["parameters"] = sum(
            p.numel() for p in m.parameters() if p.requires_grad
        )
        all_metrics.append(met)
        print(name, "Test:", {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()
        })

        _, yp_last = eval_patch_model(m, dl_fin, mean, std)   # (1, H, N)
        patch_preds[name] = yp_last
        torch.save(m.state_dict(), OUT_DIR / (name + ".pt"))

    # ---------- 对比 ----------
    results = pd.DataFrame(all_metrics)[["model", "RMSSE", "MAE", "RMSE", "WAPE"]]
    results.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n========== 对比结果 ==========")
    print(results.to_string(index=False))

    # ---------- 绘图（取测试集最后一个窗口） ----------
    Xf, Ff, itf, yf, sf = build_transformer_data(prepared, np.array([final_origin]))
    floader = DataLoader(
        WindowDataset(Xf, Ff, itf, yf, sf), batch_size=len(Xf), shuffle=False
    )
    _, ptf, _ = transformer_predict(model, floader)

    last_df = build_lightgbm_rows(prepared, history=SEQ_LEN, horizon=PRED_LEN,
                                  stride=STRIDE, origins=np.array([final_origin]))
    pl = lightgbm_predict(bst, feat_cols, last_df).reshape(n_ent, PRED_LEN)

    # 统一成 (1, PRED, n_ent) 便于绘图。
    yf_plot = yf.T.reshape(1, PRED_LEN, n_ent)
    preds = {
        "Transformer": ptf.T.reshape(1, PRED_LEN, n_ent),
        "LightGBM": pl.T.reshape(1, PRED_LEN, n_ent),
        **patch_preds,
    }
    test_dates = pd.date_range(
        start=dates[final_origin + SEQ_LEN],
        periods=PRED_LEN, freq="D",
    )
    plot_compare(test_dates, yf_plot, preds, entities)

    print("\n完成。结果目录:", OUT_DIR.resolve())
    print("绘图文件: forecast_item.png / forecast_six_items.png")


if __name__ == "__main__":
    main()
