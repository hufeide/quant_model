
"""
M5 销量预测复现（对照实验）
============================
依据 tranformer.md 的方法论，复现并对比两类模型：

  1. LightGBM      —— 交叉特征 + 滞后特征（表格型基线）
  2. Transformer   —— 纯编码器 + 可学习位置编码 + 序列归一化损失（RMSSE 风格）

数据管线沿用仓库中 patchTS.py / itranformer.py 的 50 商品抽样方案，
但按 md 描述改为：输入窗口 56 天、预测 28 天、滑窗步长 7，
销量做 log1p 变换，引入日历 / SNAP / 价格 / 促销（事件）等特征。

说明：md 原文针对 item-store 级 30490 条序列；此处为可直接运行，
改为 50 个商品（跨门店聚合）的抽样，方法学保持一致。
"""

from pathlib import Path
import random
import zipfile
import subprocess
import math

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

# 锚定到脚本所在目录
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "m5"
OUT_DIR = BASE_DIR / "outputs"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少文件: {missing}。请检查 Kaggle 下载结果。"
        )


# ============================================================
# 3. 抽样 50 个商品，构建宽表与逐日特征
# ============================================================

def prepare_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    # 按 d 的数值顺序排序（避免字符串排序 d_10 < d_2 导致错位）
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_num = np.array([int(c[2:]) for c in day_cols])
    order = np.argsort(day_num)
    day_cols = [day_cols[i] for i in order]

    # 固定种子，抽取相同的 50 个商品
    rng = np.random.default_rng(SEED)
    available = np.array(sorted(sales["item_id"].unique()))
    chosen = sorted(
        rng.choice(
            available, size=min(N_ITEMS, len(available)), replace=False
        ).tolist()
    )

    sampled = sales.loc[
        sales["item_id"].isin(chosen),
        ["item_id", "store_id", "state_id", *day_cols],
    ].copy()

    long = sampled.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype(np.float32)
    assert long["sales"].notna().all() and (long["sales"] >= 0).all()

    # 日历按 d 数值排序，并与销量日期明确对齐
    calendar = calendar.copy()
    calendar["d_num"] = calendar["d"].str.replace("d_", "", regex=False).astype(int)
    calendar = calendar.sort_values("d_num").drop_duplicates("d_num")
    calendar["date"] = pd.to_datetime(calendar["date"])

    calendar_cols = [
        c for c in [
            "d", "date", "wm_yr_wk",
            "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1",
            "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ] if c in calendar.columns
    ]
    calendar_sub = calendar[calendar_cols].copy()
    assert calendar_sub["d"].is_unique

    long = long.merge(calendar_sub, on="d", how="left", validate="many_to_one")
    if long["date"].isna().any():
        raise ValueError("部分销量日期没有匹配到 calendar。")

    # 按州取对应 SNAP 标记
    snap = np.select(
        [
            long["state_id"].eq("CA"),
            long["state_id"].eq("TX"),
            long["state_id"].eq("WI"),
        ],
        [
            long["snap_CA"].to_numpy(float),
            long["snap_TX"].to_numpy(float),
            long["snap_WI"].to_numpy(float),
        ],
        default=np.nan,
    ).astype(np.float32)
    long["snap_active"] = snap

    # 合并门店级价格，并标记当日是否有价格（缺失不能当成真实 0）
    prices_sub = prices.loc[
        prices["item_id"].isin(chosen),
        ["item_id", "store_id", "wm_yr_wk", "sell_price"],
    ].copy()
    price_key = ["item_id", "store_id", "wm_yr_wk"]
    assert not prices_sub.duplicated(price_key).any()

    long = long.merge(prices_sub, on=price_key, how="left", validate="many_to_one")
    long["price_available"] = long["sell_price"].notna().astype(np.float32)

    # 汇总为商品级每日面板（跨门店求和销量，均价 / 价格可用性 / SNAP 占比）
    cal_feats = [c for c in calendar_cols if c != "d"]
    group_cols = ["item_id", "d", *cal_feats]
    panel = (
        long.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            sales=("sales", "sum"),
            sell_price_mean=("sell_price", "mean"),
            price_available=("price_available", "max"),
            snap_active_store_share=("snap_active", "mean"),
        )
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )

    assert not panel.duplicated(["item_id", "d"]).any()

    # 宽表：[日期, 商品]
    wide_sales = panel.pivot(index="d", columns="item_id", values="sales")
    wide_price = panel.pivot(index="d", columns="item_id", values="sell_price_mean")
    wide_price_avail = panel.pivot(
        index="d", columns="item_id", values="price_available"
    )
    wide_snap = panel.pivot(
        index="d", columns="item_id", values="snap_active_store_share"
    )

    day_to_date = calendar.set_index("d")["date"]
    for w in (wide_sales, wide_price, wide_price_avail, wide_snap):
        w.index = pd.to_datetime(w.index.map(day_to_date))
        w.sort_index(inplace=True)

    wide_sales = wide_sales.reindex(columns=chosen).fillna(0)
    wide_price = wide_price.reindex(columns=chosen).fillna(0)
    wide_price_avail = wide_price_avail.reindex(columns=chosen).fillna(0)
    wide_snap = wide_snap.reindex(columns=chosen).fillna(0)

    sales_mat = wide_sales.to_numpy(dtype=np.float32)   # (n_days, n_items)
    price_mat = wide_price.to_numpy(dtype=np.float32)
    price_avail_mat = wide_price_avail.to_numpy(dtype=np.float32)
    snap_mat = wide_snap.to_numpy(dtype=np.float32)

    # 关键修正：用「销量真实日期」对齐日历，保证 cal_feat 与 sales_mat 逐行一致
    cal_by_date = calendar.set_index("date")
    cal_aligned = cal_by_date.reindex(wide_sales.index)
    if cal_aligned[["wday", "month", "year"]].isna().any().any():
        raise ValueError("销量日期未能完全对齐 calendar。")

    wday = pd.to_numeric(cal_aligned["wday"], errors="coerce").fillna(1).to_numpy(np.float32)
    month = pd.to_numeric(cal_aligned["month"], errors="coerce").fillna(1).to_numpy(np.float32)
    year = pd.to_numeric(cal_aligned["year"], errors="coerce").fillna(2011).to_numpy(np.float32)

    # 任一事件列非空即标记；覆盖 name_1/2 与 type_1/2
    ev_parts = []
    for col in ("event_name_1", "event_name_2", "event_type_1", "event_type_2"):
        if col in cal_aligned:
            ev_parts.append(cal_aligned[col].notna().to_numpy(np.float32))
        else:
            ev_parts.append(np.zeros(len(cal_aligned), dtype=np.float32))
    event_flag = np.maximum.reduce(ev_parts).astype(np.float32)

    # 归一化（与 M5 实际编码一致）：wday 1-7 -> 0-1
    cal_df = np.stack([
        (wday - 1.0) / 6.0,
        (month - 1.0) / 12.0,
        (year - 2011.0) / 10.0,
        event_flag,
    ], axis=-1).astype(np.float32)   # (n_days, 4): wday, month, year, event_flag

    print("抽样商品数:", len(chosen))
    print("日期数:", sales_mat.shape[0])
    print("面板形状 [日期, 商品]:", sales_mat.shape)
    print("日期范围:", wide_sales.index.min(), "至", wide_sales.index.max())

    return {
        "chosen": chosen,
        "sales": sales_mat,
        "price": price_mat,
        "price_avail": price_avail_mat,
        "snap": snap_mat,
        "cal_feat": cal_df,         # (n_days, 4): wday, month, year, event_flag
        "dates": wide_sales.index,
    }


# ============================================================
# 4. 构造滑窗样本
# ============================================================
#
# 与 md 一致：窗口 56 天 / 预测 28 天 / 步长 7。
# 每个样本对应一个 (商品, 起点) 组合。

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


def build_transformer_data(data, origins):
    """
    返回 X(B,SEQ,hist_feat) / future(B,PRED,fut_feat) / item_idx(B) / y(B,PRED) / scale(B,1)。
    - 价格归一化只用训练期（前 70% 天）统计量，且对缺失价格置 0 并给可用性标记。
    - RMSSE 分母尺度用「原始销量」训练期一阶差分 RMS（与目标同空间）。
    - 未来协变量用目标区间已知的日历 / 价格 / SNAP（不泄漏未来销量）。
    """
    sales = data["sales"]          # (n_days, n_items)
    price = data["price"]
    price_avail = data["price_avail"]
    snap = data["snap"]
    cal = data["cal_feat"]          # (n_days, 4)：wday, month, year, event_flag
    n_days, n_items = sales.shape

    log_sales = np.log1p(np.maximum(sales, 0))

    # 价格归一化：仅用训练期有价格的日子估计均值/标准差（避免泄漏 + 排除缺失）。
    hist_cut = int(n_days * 0.70)
    price_mask = price_avail[:hist_cut] > 0
    price_train = np.where(price_mask, price[:hist_cut], np.nan)
    price_mean = np.nanmean(price_train, axis=0)
    price_std = np.nanstd(price_train, axis=0)
    # 训练期完全/几乎无价格的商品：均值置 0、标准差置 1，避免除零产生 NaN
    # （否则该商品在 val/test 有价格时 price_norm 会变成 NaN，污染预测）。
    price_mean = np.where(np.isnan(price_mean), 0.0, price_mean)
    price_std = np.where(np.isnan(price_std) | (price_std < 1e-5), 1.0, price_std)

    price_norm = (price - price_mean[None, :]) / price_std[None, :]
    # 无价格日：归一化值置 0；并清掉任何残留 NaN。
    price_norm = np.where(price_avail > 0, price_norm, 0.0)
    price_norm = np.nan_to_num(price_norm, nan=0.0).astype(np.float32)

    cal_bc = np.broadcast_to(
        cal[:, None, :], (n_days, n_items, cal.shape[1])
    ).astype(np.float32)

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
    )  # (n_days, n_items, 8)

    # 未来协变量（目标区间已知）：价格 + 日历 + SNAP + 事件 + 价格可用性
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
    )  # (n_days, n_items, 7)

    # RMSSE 分母：训练期「原始销量」逐日一阶差分均方根（与目标同空间）。
    train_sales = sales[:hist_cut]
    denom = (
        np.mean(np.diff(train_sales, axis=0) ** 2, axis=0)
        if train_sales.shape[0] > 1
        else np.ones(n_items)
    )
    scale = np.sqrt(np.maximum(denom, 1e-8)).astype(np.float32)  # (n_items,)
    # 防止个别近恒定序列的分母退化（→ 1e-4）导致 RMSSE 数值爆炸：
    # 以各序列尺度的 5% 分位为下界（至少为 1e-2）。
    scale_floor = max(float(np.percentile(scale, 5)), 1e-2)
    scale = np.maximum(scale, scale_floor).reshape(1, n_items)

    Xs, Fs, Ys, I, S = [], [], [], [], []
    for s in origins:
        hist = hist_feat[s:s + SEQ_LEN]                       # (SEQ, n_items, 8)
        fut = fut_feat[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN]     # (PRED, n_items, 7)
        fut_sales = sales[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN]  # (PRED, n_items)
        for it in range(n_items):
            Xs.append(hist[:, it, :])
            Fs.append(fut[:, it, :])
            Ys.append(fut_sales[:, it])
            I.append(it)
            S.append(scale[0, it])

    X = np.stack(Xs).astype(np.float32)         # (B, SEQ, 8)
    F = np.stack(Fs).astype(np.float32)         # (B, PRED, 7)
    y = np.stack(Ys).astype(np.float32)         # (B, PRED)
    item_idx = np.array(I, dtype=np.int64)
    sc = np.array(S, dtype=np.float32).reshape(-1, 1)
    return X, F, item_idx, y, sc


class WindowDataset(Dataset):
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


def build_lightgbm_rows(data, origins):
    """
    为每个 (商品, 窗口, 预测日 h) 构造一行表格特征：
    滞后特征 + 滚动统计 + 价格特征 + 日历 + 交叉特征(item_id) + 预测日偏移 h。
    目标为 log1p(未来第 h 天销量)，缓解长尾偏态。
    """
    sales = data["sales"]
    price = data["price"]
    snap = data["snap"]
    cal = data["cal_feat"]
    n_days, n_items = sales.shape
    log_sales = np.log1p(np.maximum(sales, 0))

    rows = []
    for s in origins:
        L = s + SEQ_LEN - 1  # 历史最后一天索引
        # 整段目标区间必须落在有效销量天内，避免 reshape 错位
        if L + PRED_LEN > n_days:
            continue
        for it in range(n_items):
            sl = log_sales[:, it]
            pr = price[:, it]
            sn = snap[:, it]

            def lag(k):
                return sl[L - k] if L - k >= 0 else 0.0

            def roll(k):
                lo = max(0, L - k + 1)
                return sl[lo:L + 1].mean()

            price_28 = pr[max(0, L - 27):L + 1].mean()
            year_ago = sl[L - 364] if L - 364 >= 0 else 0.0
            c = cal[L]

            base = {
                "item_id": it,
                "lag_1": lag(1),
                "lag_7": lag(7),
                "lag_14": lag(14),
                "lag_21": lag(21),
                "lag_28": lag(28),
                "roll_mean_7": roll(7),
                "roll_mean_14": roll(14),
                "roll_mean_28": roll(28),
                "roll_std_28": sl[max(0, L - 27):L + 1].std(),
                "price_last": pr[L],
                "price_28_mean": price_28,
                "price_change": pr[L] - price_28,
                "year_ago": year_ago,
                "wday": int(c[0]),
                "month": int(c[1]),
                "year": int(c[2]),
                "snap": sn[L],
                "event_flag": int(c[3]),
            }

            for h in range(1, PRED_LEN + 1):
                r = dict(base)
                r["horizon"] = h                       # 交叉特征：预测日偏移
                r["target"] = log_sales[L + h, it]     # 目标
                rows.append(r)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 5. Transformer 模型（md 第 3 节：纯编码器 + 可学习位置编码）
# ============================================================

class M5TransformerForecaster(nn.Module):
    """
    特征融合：
      历史数值特征 -> 线性投影到 d_model；
      商品 embedding -> 融合向量，广播加到每个历史时间步；
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
# 6. 序列归一化损失（md 第 4.1 节：RMSSE 风格）
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
# 7. 评估指标
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
# 8. 训练（Transformer）
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

    pd.DataFrame(logs).to_csv(OUT_DIR / "training_Transformer.csv", index=False)
    return model


# ============================================================
# 9. LightGBM 训练与预测（md 第 5.3 节：交叉特征 + 滞后特征）
# ============================================================

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

    feat_cols = [
        c for c in train_df.columns
        if c not in ("target",)
    ]
    cat_cols = ["item_id", "wday", "month", "year", "horizon"]

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


# ============================================================
# 10. 绘制对比
# ============================================================

def plot_compare(dates, y_true, preds, item_ids):
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


# ============================================================
# 11. 主流程
# ============================================================

def main():
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not all(
        (DATA_DIR / f).exists() for f in [
            "sales_train_validation.csv",
            "calendar.csv",
            "sell_prices.csv",
        ]
    ):
        download_data()

    data = prepare_data()
    chosen = data["chosen"]
    n_days, n_items = data["sales"].shape

    train_o, val_o, test_o, _ = make_origins(
        n_days, SEQ_LEN, PRED_LEN, STRIDE
    )
    if min(len(train_o), len(val_o), len(test_o)) == 0:
        raise ValueError("滑窗切分后没有足够的训练/验证/测试窗口。")

    print(f"Windows: train={len(train_o)}, val={len(val_o)}, test={len(test_o)}")

    # ---------- Transformer ----------
    Xtr, Ftr, itr, ytr, str_ = build_transformer_data(data, train_o)
    Xva, Fva, iva, yva, sva = build_transformer_data(data, val_o)
    Xte, Fte, ite, yte, ste = build_transformer_data(data, test_o)

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
        hist_dim=Xtr.shape[-1], future_dim=Ftr.shape[-1], num_items=n_items
    )
    print("Transformer 参数量:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = train_transformer(model, train_loader, val_loader)

    yt, pt, st = transformer_predict(model, test_loader)
    transformer_metrics = compute_metrics(yt, pt, st)
    transformer_metrics["model"] = "Transformer(归一化损失+可学习位置编码)"
    print("Transformer Test:", transformer_metrics)

    torch.save(model.state_dict(), OUT_DIR / "Transformer.pt")

    # ---------- LightGBM ----------
    print("\n训练 LightGBM ...")
    lgb_train_df = build_lightgbm_rows(data, train_o)
    lgb_val_df = build_lightgbm_rows(data, val_o)
    lgb_test_df = build_lightgbm_rows(data, test_o)

    bst, feat_cols = train_lightgbm(lgb_train_df, lgb_val_df)

    yte_lgb = lightgbm_predict(bst, feat_cols, lgb_test_df)  # (N_test_rows,)
    # 还原为 (B, PRED) 形状：test 行按 (window, item, h) 顺序生成
    n_samples_test = len(test_o) * n_items
    y_pred_lgb = yte_lgb.reshape(n_samples_test, PRED_LEN)
    y_true_lgb = yte.reshape(n_samples_test, PRED_LEN)
    scale_lgb = ste[:n_samples_test]
    lightgbm_metrics = compute_metrics(y_true_lgb, y_pred_lgb, scale_lgb)
    lightgbm_metrics["model"] = "LightGBM(交叉特征+滞后特征)"
    print("LightGBM Test:", lightgbm_metrics)

    bst.save_model(str(OUT_DIR / "LightGBM.txt"))

    # ---------- 对比 ----------
    results = pd.DataFrame([transformer_metrics, lightgbm_metrics])[
        ["model", "RMSSE", "MAE", "RMSE", "WAPE"]
    ]
    results.to_csv(OUT_DIR / "metrics.csv", index=False)
    print("\n========== 对比结果 ==========")
    print(results.to_string(index=False))

    # ---------- 绘图（取测试集最后一个窗口） ----------
    final_origin = int(test_o[-1])
    # 重新用单窗口构造 Transformer 预测，对齐 LightGBM 最后一个窗口
    Xf, Ff, itf, yf, sf = build_transformer_data(
        data, np.array([final_origin])
    )
    floader = DataLoader(
        WindowDataset(Xf, Ff, itf, yf, sf), batch_size=len(Xf), shuffle=False
    )
    _, ptf, _ = transformer_predict(model, floader)

    # LightGBM 最后一个窗口
    last_df = build_lightgbm_rows(data, np.array([final_origin]))
    pl = lightgbm_predict(bst, feat_cols, last_df).reshape(n_items, PRED_LEN)

    # 统一成 (1, PRED, n_items) 便于绘图。
    # yf/ptf/pl 的形状均为 (n_items, PRED_LEN)，需要先转置再 reshape，
    # 否则 reshape(1, PRED_LEN, n_items) 会按错误顺序把不同商品的点混在一起。
    yf_plot = yf.T.reshape(1, PRED_LEN, n_items)
    preds = {
        "Transformer": ptf.T.reshape(1, PRED_LEN, n_items),
        "LightGBM": pl.T.reshape(1, PRED_LEN, n_items),
    }
    test_dates = pd.date_range(
        start=data["dates"][final_origin + SEQ_LEN],
        periods=PRED_LEN, freq="D",
    )
    plot_compare(test_dates, yf_plot, preds, chosen)

    print("\n完成。结果目录:", OUT_DIR.resolve())
    print("绘图文件: forecast_item.png / forecast_six_items.png")


if __name__ == "__main__":
    main()
