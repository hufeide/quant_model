
from pathlib import Path
import random
import zipfile
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 配置
# ============================================================

SEED = 42
N_ITEMS = 50
HISTORY = 112
HORIZON = 28

D_MODEL = 64
N_HEADS = 4
PATCH_LEN = 16
STRIDE = 8
DROPOUT = 0.1
LAYERS = 2

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 5

# 锚定到脚本所在目录，使调试时无论从哪个 CWD 启动都能正确找到数据/输出
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

    missing = [
        f for f in required
        if not (DATA_DIR / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"缺少文件: {missing}。请检查 Kaggle 下载结果。"
        )


# ============================================================
# 3. 抽样 50 个商品，处理销售、价格和日历
# ============================================================

def prepare_data():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sales_path = DATA_DIR / "sales_train_validation.csv"
    calendar_path = DATA_DIR / "calendar.csv"
    price_path = DATA_DIR / "sell_prices.csv"

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(price_path)

    day_cols = [
        c for c in sales.columns
        if c.startswith("d_")
    ]

    # 固定种子，确保每次抽取相同的商品
    rng = np.random.default_rng(SEED)
    available = np.array(sorted(sales["item_id"].unique()))

    chosen = sorted(
        rng.choice(
            available,
            size=min(N_ITEMS, len(available)),
            replace=False,
        ).tolist()
    )

    sampled = sales.loc[
        sales["item_id"].isin(chosen),
        ["item_id", "store_id", "state_id", *day_cols],
    ].copy()

    long_store = sampled.melt(
        id_vars=["item_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # M5 原始销量应当是数值；不把异常值静默转换成零销量
    long_store["sales"] = pd.to_numeric(
        long_store["sales"], errors="raise"
    ).astype(np.float32)

    assert long_store["sales"].notna().all()
    assert (long_store["sales"] >= 0).all()

    # ============================================================
    # 2. 准备日历数据并进行多对一连接
    # ============================================================
    calendar_cols = [
        c for c in [
            "d", "date", "wm_yr_wk",
            "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1",
            "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        if c in calendar.columns
    ]

    calendar_sub = calendar[calendar_cols].copy()
    calendar_sub["date"] = pd.to_datetime(calendar_sub["date"])

    # 每个 d 应当只对应一个日历日期
    assert calendar_sub["d"].is_unique

    long_store = long_store.merge(
        calendar_sub,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if long_store["date"].isna().any():
        raise ValueError("部分销量日期没有匹配到 calendar，请检查 d 字段。")

    # ============================================================
    # 3. 根据门店所在州，提取对应的 SNAP 标记
    # ============================================================
    conditions = [
        long_store["state_id"].eq("CA"),
        long_store["state_id"].eq("TX"),
        long_store["state_id"].eq("WI"),
    ]

    choices = [
        long_store["snap_CA"].to_numpy(dtype=float),
        long_store["snap_TX"].to_numpy(dtype=float),
        long_store["snap_WI"].to_numpy(dtype=float),
    ]

    long_store["snap_active"] = np.select(
        conditions,
        choices,
        default=np.nan,
    ).astype(np.float32)

    # ============================================================
    # 4. 先合并门店级价格，不提前丢弃 store_id
    # ============================================================
    price_cols = [
        "item_id", "store_id", "wm_yr_wk", "sell_price"
    ]

    prices_sub = prices.loc[
        prices["item_id"].isin(chosen),
        price_cols,
    ].copy()

    price_key = ["item_id", "store_id", "wm_yr_wk"]

    if prices_sub.duplicated(price_key).any():
        raise ValueError(
            "价格表的 item_id/store_id/wm_yr_wk 存在重复记录。"
        )

    long_store = long_store.merge(
        prices_sub,
        on=price_key,
        how="left",
        validate="many_to_one",
    )

    # ============================================================
    # 5. 汇总成商品级每日面板
    # ============================================================
    calendar_features = [
        c for c in calendar_cols if c != "d"
    ]

    group_cols = ["item_id", "d", *calendar_features]

    panel = (
        long_store
        .groupby(
            group_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            # 跨门店的商品每日总销量
            sales=("sales", "sum"),

            # 所有存在价格记录的门店的平均标价
            sell_price_mean=("sell_price", "mean"),

            # 门店之间的价格离散程度
            sell_price_std=("sell_price", "std"),

            # 有有效价格的门店比例
            price_coverage=(
                "sell_price",
                lambda s: s.notna().mean(),
            ),

            # 当天处于 SNAP 标记状态的门店比例
            snap_active_store_share=("snap_active", "mean"),
        )
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )

    # ============================================================
    # 6. 检查数据完整性并保存
    # ============================================================
    assert not panel.duplicated(["item_id", "d"]).any()

    expected_rows = len(chosen) * len(day_cols)

    print("抽样商品数:", len(chosen))
    print("日期数:", len(day_cols))
    print("面板行数:", len(panel))
    print("理论行数:", expected_rows)
    print("实际匹配价格覆盖率:", round(
        panel["price_coverage"].mean(), 4
    ))
    print(panel.head())


    # 保存长格式面板，便于检查及后续加入多特征模型
    panel.to_csv(
        OUT_DIR / "m5_panel.csv",
        index=False,
    )

    # 转成 [日期, 商品] 矩阵
    wide = item_sales.pivot(
        index="d",
        columns="item_id",
        values="sales",
    )

    day_to_date = calendar.set_index("d")["date"]

    wide.index = pd.to_datetime(
        wide.index.map(day_to_date)
    )

    wide = wide.sort_index()
    wide = wide.reindex(columns=chosen).fillna(0)
    wide.index.name = "date"

    wide.to_csv(OUT_DIR / "sales_wide.csv")

    print("面板形状 [日期, 商品]:", wide.shape)
    print("日期范围:", wide.index.min(), "至", wide.index.max())

    return wide, chosen


# ============================================================
# 4. 按时间切分数据，防止未来信息泄漏
# ============================================================

def chronological_split(n_time, history, horizon):
    train_boundary = int(n_time * 0.70)
    val_boundary = int(n_time * 0.85)

    origins = np.arange(
        history,
        n_time - horizon + 1,
    )

    train_origins = origins[
        origins + horizon <= train_boundary
    ]

    val_origins = origins[
        (origins >= train_boundary)
        & (origins + horizon <= val_boundary)
    ]

    test_origins = origins[
        (origins >= val_boundary)
        & (origins + horizon <= n_time)
    ]

    return (
        train_origins,
        val_origins,
        test_origins,
        train_boundary,
        val_boundary,
    )


def fit_scaler(train_raw):
    # 对非负销量先 log1p，再按商品标准化
    logged = np.log1p(np.maximum(train_raw, 0))

    mean = logged.mean(axis=0, keepdims=True)
    std = logged.std(axis=0, keepdims=True)

    std[std < 1e-5] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def transform(raw, mean, std):
    logged = np.log1p(np.maximum(raw, 0))
    return ((logged - mean) / std).astype(np.float32)


def inverse_transform(x, mean, std):
    return np.maximum(
        np.expm1(x * std + mean),
        0,
    )


class SalesDataset(Dataset):
    def __init__(self, values, origins, history, horizon):
        self.values = values
        self.origins = origins
        self.history = history
        self.horizon = horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        origin = self.origins[idx]

        # X: [T, N]
        x = self.values[
            origin - self.history:origin
        ]

        # Y: [H, N]
        y = self.values[
            origin:origin + self.horizon
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ============================================================
# 5. PatchTST 时间编码器
# ============================================================

class PatchTSTEncoder(nn.Module):
    """
    输入 [B,T,N]
    每个商品独立切 Patch，共享时间编码参数。
    输出 [B,N,D]
    """

    def __init__(
        self,
        history=HISTORY,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        dropout=DROPOUT,
        layers=LAYERS,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        n_patches = (
            (history - patch_len) // stride + 1
        )

        self.patch_embed = nn.Linear(
            patch_len,
            d_model,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x [B,T,N]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)

        x = (x - mean) / std

        # [B,T,N] -> [B,N,T]
        x = x.transpose(1, 2)

        # [B,N,T] -> [B,N,P,L]
        patches = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

        b, n, p, length = patches.shape

        # 每个商品的每个 Patch 投影为 D 维 Token
        tokens = patches.reshape(
            b * n, p, length
        )

        tokens = self.patch_embed(tokens)
        tokens = tokens + self.pos_embed[:, :p, :]

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # 聚合时间 Patch，得到每个商品的时间表示
        # [B*N,P,D] -> [B,N,D]
        item_repr = tokens.mean(dim=1).reshape(
            b, n, -1
        )

        return item_repr


# ============================================================
# 6. iTransformer 风格的变量交互编码器
# ============================================================

class InvertedVariableEncoder(nn.Module):
    """
    每个商品是一枚 Token。
    在商品维度上执行注意力，建模跨商品关联。
    """

    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        layers=LAYERS,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.encoder(x))


# ============================================================
# 7. PatchTST -> iTransformer 串联模型
# ============================================================

class PatchTSTThenITransformer(nn.Module):
    """
    输入:  [B,T,N]
    输出:  [B,H,N]

    第一步：PatchTST 编码每个商品的历史时间模式。
    第二步：iTransformer 编码商品之间的交互。
    第三步：预测每个商品未来 H 天销量。
    """

    def __init__(
        self,
        history=HISTORY,
        horizon=HORIZON,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        dropout=DROPOUT,
        layers=LAYERS,
    ):
        super().__init__()

        self.temporal = PatchTSTEncoder(
            history=history,
            d_model=d_model,
            n_heads=n_heads,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            layers=layers,
        )

        self.variable_encoder = InvertedVariableEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            layers=layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # [B,T,N] -> [B,N,D]
        h = self.temporal(x)

        # 在 N 个商品 Token 之间进行注意力交互
        h = self.variable_encoder(h)

        # [B,N,D] -> [B,N,H] -> [B,H,N]
        y = self.head(h)

        return y.transpose(1, 2)


# ============================================================
# 8. 单模型基线：PatchTST
# ============================================================

class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        history=HISTORY,
        horizon=HORIZON,
        **kwargs,
    ):
        super().__init__()

        d_model = kwargs.get("d_model", D_MODEL)
        dropout = kwargs.get("dropout", DROPOUT)

        self.temporal = PatchTSTEncoder(
            history=history,
            d_model=d_model,
            n_heads=kwargs.get("n_heads", N_HEADS),
            patch_len=kwargs.get("patch_len", PATCH_LEN),
            stride=kwargs.get("stride", STRIDE),
            dropout=dropout,
            layers=kwargs.get("layers", LAYERS),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        h = self.temporal(x)
        return self.head(h).transpose(1, 2)


# ============================================================
# 9. 单模型基线：iTransformer
# ============================================================

class ITransformerBaseline(nn.Module):
    def __init__(
        self,
        history=HISTORY,
        horizon=HORIZON,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        layers=LAYERS,
    ):
        super().__init__()

        # 将完整历史窗口投影为每个商品的 Token
        self.value_embed = nn.Linear(
            history,
            d_model,
        )

        self.variable_encoder = InvertedVariableEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            layers=layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # [B,T,N] -> [B,N,T] -> [B,N,D]
        z = self.value_embed(x.transpose(1, 2))

        z = self.variable_encoder(z)

        return self.head(z).transpose(1, 2)


# ============================================================
# 10. 指标和评估
# ============================================================

def calculate_metrics(y_true, y_pred):
    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    # 零销量较多，使用 WAPE 替代 MAPE
    wape = (
        np.sum(np.abs(error))
        / max(np.sum(np.abs(y_true)), 1e-8)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": float(wape),
    }


@torch.no_grad()
def evaluate(model, loader, mean, std):
    model.eval()

    ys = []
    preds = []

    for x, y in loader:
        x = x.to(DEVICE)

        pred = model(x).cpu().numpy()

        ys.append(y.numpy())
        preds.append(pred)

    y_scaled = np.concatenate(ys, axis=0)
    p_scaled = np.concatenate(preds, axis=0)

    y_true = inverse_transform(
        y_scaled,
        mean.reshape(1, 1, -1),
        std.reshape(1, 1, -1),
    )

    y_pred = inverse_transform(
        p_scaled,
        mean.reshape(1, 1, -1),
        std.reshape(1, 1, -1),
    )

    return (
        calculate_metrics(y_true, y_pred),
        y_true,
        y_pred,
    )


# ============================================================
# 11. 训练与早停
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    mean,
    std,
    model_name,
):
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

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

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            losses.append(loss.item())

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            mean,
            std,
        )

        val_mae = val_metrics["MAE"]

        logs.append({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "val_mae": val_mae,
            "val_rmse": val_metrics["RMSE"],
            "val_wape": val_metrics["WAPE"],
        })

        print(
            f"{model_name} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train MSE={np.mean(losses):.5f} | "
            f"Val MAE={val_mae:.4f} | "
            f"Val RMSE={val_metrics['RMSE']:.4f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(logs).to_csv(
        OUT_DIR / f"training_{model_name}.csv",
        index=False,
    )

    return model


# ============================================================
# 12. 绘制预测结果
# ============================================================

def plot_results(
    dates,
    y_true,
    predictions,
    item_ids,
):
    # 最后一个测试预测窗口
    actual = y_true[0]  # [H,N]

    # 图 1：第一个商品的预测对比
    plt.figure(figsize=(13, 5))

    for name, pred in predictions.items():
        plt.plot(
            dates,
            pred[0, :, 0],
            label=name,
            linewidth=1.8,
        )

    plt.plot(
        dates,
        actual[:, 0],
        label="Actual",
        linewidth=2.5,
        marker="o",
    )

    plt.title(f"28-Day Forecast: {item_ids[0]}")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "forecast_item.png",
        dpi=160,
    )
    plt.close()

    # 图 2：前 6 个商品的联合模型预测
    hybrid_name = "PatchTST->iTransformer"
    pred = predictions[hybrid_name][0]

    n_plot = min(6, len(item_ids))

    fig, axes = plt.subplots(
        n_plot,
        1,
        figsize=(13, 3 * n_plot),
        sharex=True,
    )

    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        axes[i].plot(
            dates,
            actual[:, i],
            label="Actual",
            linewidth=2,
        )

        axes[i].plot(
            dates,
            pred[:, i],
            label="Hybrid prediction",
        )

        axes[i].set_title(item_ids[i])
        axes[i].legend()

    axes[-1].tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "forecast_six_items.png",
        dpi=160,
    )
    plt.close(fig)

    # 图 3：50 个商品的总销量预测
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        dates,
        actual.sum(axis=1),
        label="Actual total",
        linewidth=2.5,
    )

    ax.plot(
        dates,
        pred.sum(axis=1),
        label="Hybrid forecast total",
        linewidth=2,
    )

    ax.set_title("Total Sales: Sampled 50 Items")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "forecast_total.png",
        dpi=160,
    )
    plt.close(fig)


# ============================================================
# 13. 主流程
# ============================================================

def main():
    seed_everything()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not all(
        (DATA_DIR / f).exists()
        for f in [
            "sales_train_validation.csv",
            "calendar.csv",
            "sell_prices.csv",
        ]
    ):
        download_data()

    wide, item_ids = prepare_data()

    raw = wide.to_numpy(dtype=np.float32)
    n_time, n_items = raw.shape

    train_origins, val_origins, test_origins, train_b, val_b = (
        chronological_split(
            n_time,
            HISTORY,
            HORIZON,
        )
    )

    if min(
        len(train_origins),
        len(val_origins),
        len(test_origins),
    ) == 0:
        raise ValueError("数据切分后没有足够的训练、验证或测试窗口。")

    # 仅用训练时间段拟合归一化参数
    mean, std = fit_scaler(raw[:train_b])
    values = transform(raw, mean, std)

    train_ds = SalesDataset(
        values,
        train_origins,
        HISTORY,
        HORIZON,
    )

    val_ds = SalesDataset(
        values,
        val_origins,
        HISTORY,
        HORIZON,
    )

    test_ds = SalesDataset(
        values,
        test_origins,
        HISTORY,
        HORIZON,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
    )

    print("Device:", DEVICE)
    print("Data shape:", raw.shape)
    print(
        f"Windows: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )

    models = {
        "PatchTST->iTransformer":
            PatchTSTThenITransformer(),

        "PatchTST":
            PatchTSTBaseline(),

        "iTransformer":
            ITransformerBaseline(),
    }

    results = []
    predictions = {}

    # 绘图用：最后一个测试起点的完整 28 天
    final_origin = int(test_origins[-1])

    final_ds = SalesDataset(
        values,
        np.array([final_origin]),
        HISTORY,
        HORIZON,
    )

    final_loader = DataLoader(
        final_ds,
        batch_size=1,
        shuffle=False,
    )

    test_dates = pd.date_range(
        start=wide.index[final_origin],
        periods=HORIZON,
        freq="D",
    )

    y_final = None

    for name, model in models.items():
        print("\nTraining:", name)

        model = train_model(
            model,
            train_loader,
            val_loader,
            mean,
            std,
            name.replace("->", "_"),
        )

        test_metrics, y_test, p_test = evaluate(
            model,
            test_loader,
            mean,
            std,
        )

        test_metrics["model"] = name
        test_metrics["parameters"] = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        results.append(test_metrics)

        print("Test metrics:", test_metrics)

        _, y_last, p_last = evaluate(
            model,
            final_loader,
            mean,
            std,
        )

        y_final = y_last
        predictions[name] = p_last

        torch.save(
            model.state_dict(),
            OUT_DIR / (
                name.replace("->", "_").replace(" ", "_")
                + ".pt"
            ),
        )

    metrics_df = pd.DataFrame(results)[
        ["model", "parameters", "MAE", "RMSE", "WAPE"]
    ]

    metrics_df.to_csv(
        OUT_DIR / "metrics.csv",
        index=False,
    )

    print("\n========== TEST RESULTS ==========")
    print(metrics_df.to_string(index=False))

    plot_results(
        test_dates,
        y_final,
        predictions,
        item_ids,
    )

    # 保存预测结果，方便进一步分析
    rows = []

    for i, item_id in enumerate(item_ids):
        for h, date in enumerate(test_dates):
            row = {
                "date": date,
                "item_id": item_id,
                "actual": float(y_final[0, h, i]),
            }

            for name, pred in predictions.items():
                row[name] = float(pred[0, h, i])

            rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "predictions.csv",
        index=False,
    )

    print("\n完成。结果目录:", OUT_DIR.resolve())
    print("绘图文件:")
    print(" - forecast_item.png")
    print(" - forecast_six_items.png")
    print(" - forecast_total.png")


if __name__ == "__main__":
    main()
