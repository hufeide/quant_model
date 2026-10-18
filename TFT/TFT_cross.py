
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. 配置
# ============================================================
SEED = 42
N_STOCKS = 6
N_TIMESTEPS = 500

ENCODER_LENGTH = 60
PREDICTION_LENGTH = 5
TRAIN_CUTOFF = 399
ROLLING_STEP = 5

FEATURE_COLS = ["target", "momentum", "volatility"]
QUANTILES = [0.1, 0.5, 0.9]

HIDDEN_SIZE = 32
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

pl.seed_everything(SEED, workers=True)
rng = np.random.default_rng(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("使用设备:", device)


# ============================================================
# 2. 构造示例数据
#    加入共同市场因子，使跨股票关联有可学习的来源
# ============================================================
market_return = rng.normal(0, 0.008, N_TIMESTEPS)
frames = []

for j in range(N_STOCKS):
    stock = f"S{j:02d}"

    beta = 0.5 + 0.15 * j
    idiosyncratic = rng.normal(0, 0.006, N_TIMESTEPS)

    ret = beta * market_return + idiosyncratic

    momentum = (
        pd.Series(ret)
        .rolling(10, min_periods=1)
        .mean()
        .to_numpy()
    )

    volatility = (
        pd.Series(ret)
        .rolling(20, min_periods=1)
        .std()
        .fillna(0)
        .to_numpy()
    )

    frames.append(pd.DataFrame({
        "stock": stock,
        "time_idx": np.arange(N_TIMESTEPS),
        "target": ret,
        "momentum": momentum,
        "volatility": volatility,
    }))

df = pd.concat(frames, ignore_index=True)

stocks = sorted(df["stock"].unique())
n_stocks = len(stocks)

# 构造 [时间, 股票, 特征] 的面板数据
feature_panels = []

for col in FEATURE_COLS:
    panel = df.pivot(
        index="time_idx",
        columns="stock",
        values=col,
    ).reindex(
        index=np.arange(N_TIMESTEPS),
        columns=stocks,
    )

    feature_panels.append(
        panel.to_numpy(dtype=np.float32)
    )

# X_raw: [T, N, F]
X_raw = np.stack(feature_panels, axis=-1)

# y_raw: [T, N]
y_raw = df.pivot(
    index="time_idx",
    columns="stock",
    values="target",
).reindex(
    index=np.arange(N_TIMESTEPS),
    columns=stocks,
).to_numpy(dtype=np.float32)

assert X_raw.shape == (
    N_TIMESTEPS, n_stocks, len(FEATURE_COLS)
)
assert y_raw.shape == (N_TIMESTEPS, n_stocks)


# ============================================================
# 3. 只用训练区间拟合标准化参数
# ============================================================
train_features = X_raw[: TRAIN_CUTOFF + 1]

feature_mean = train_features.mean(
    axis=(0, 1), keepdims=True
)
feature_std = train_features.std(
    axis=(0, 1), keepdims=True
)
feature_std = np.maximum(feature_std, 1e-6)

X_scaled = (X_raw - feature_mean) / feature_std

target_scale = float(
    np.std(y_raw[: TRAIN_CUTOFF + 1])
)
target_scale = max(target_scale, 1e-6)

y_scaled = y_raw / target_scale


# ============================================================
# 4. 构造训练窗口
#    每个样本同时包含所有股票
#    X: [样本, 历史60天, 股票, 特征]
#    Y: [样本, 股票, 未来5天]
# ============================================================
X_train, Y_train = [], []

# 索引规则（统一约定，防止再次泄漏）：
#   编码器窗口  : [origin - ENCODER_LENGTH + 1, origin]          # 含 origin 共 ENCODER_LENGTH 个时点
#   预测标签窗口: [origin + 1, origin + PREDICTION_LENGTH]       # 共 PREDICTION_LENGTH 个时点
# 要求所有标签时点 <= TRAIN_CUTOFF，即 origin + PREDICTION_LENGTH <= TRAIN_CUTOFF。
for origin in range(
    ENCODER_LENGTH - 1,
    TRAIN_CUTOFF - PREDICTION_LENGTH + 1,   # 上界 +1（range 不含），最大 origin = TRAIN_CUTOFF - PREDICTION_LENGTH
):
    x_start = origin - ENCODER_LENGTH + 1
    x_end = origin + 1

    y_start = origin + 1
    y_end = origin + PREDICTION_LENGTH + 1

    X_train.append(X_scaled[x_start:x_end])
    Y_train.append(y_scaled[y_start:y_end].T)

X_train = np.asarray(X_train, dtype=np.float32)
Y_train = np.asarray(Y_train, dtype=np.float32)

# 边界断言：最后一个训练样本的起点 + 预测步数不得越过 TRAIN_CUTOFF
last_origin = (ENCODER_LENGTH - 1) + len(X_train) - 1
assert last_origin + PREDICTION_LENGTH <= TRAIN_CUTOFF, \
    f"训练标签越界: last_origin={last_origin}, 标签终点={last_origin + PREDICTION_LENGTH} > TRAIN_CUTOFF={TRAIN_CUTOFF}"
assert Y_train.shape[2] == PREDICTION_LENGTH

print("训练输入:", X_train.shape)
print("训练标签:", Y_train.shape)

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(Y_train),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# ============================================================
# 5. TFT 风格时间编码器 + Cross-Asset Attention
# ============================================================
class TFTCrossAsset(nn.Module):
    def __init__(
        self,
        n_features,
        n_stocks,
        hidden_size=32,
        num_heads=4,
        prediction_length=5,
        n_quantiles=3,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size 必须能被 num_heads 整除"
            )

        self.n_features = n_features
        self.n_stocks = n_stocks
        self.hidden_size = hidden_size
        self.prediction_length = prediction_length
        self.n_quantiles = n_quantiles

        # 股票身份 embedding：为每只股票学习一个稳定的长期差异向量
        self.stock_embedding = nn.Embedding(n_stocks, hidden_size)

        # ---- A. Variable Selection Network ----
        # 为每个时点的各个变量计算选择权重
        self.variable_gate = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_features),
        )

        self.variable_projections = nn.ModuleList([
            nn.Linear(1, hidden_size)
            for _ in range(n_features)
        ])

        self.variable_norm = nn.LayerNorm(hidden_size)

        # ---- B. 时间编码器 ----
        # 每只股票共享一套时间编码器参数
        self.temporal_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # 在每只股票自身的历史窗口中做时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_norm = nn.LayerNorm(hidden_size)

        # ---- C. 跨股票注意力 ----
        # 同一个预测起点的股票互相交换信息
        self.cross_asset_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_asset_norm = nn.LayerNorm(hidden_size)

        # ---- D. 前馈网络与残差 ----
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.output_norm = nn.LayerNorm(hidden_size)

        # 输出每只股票未来5天的三个分位数
        self.prediction_head = nn.Linear(
            hidden_size,
            prediction_length * n_quantiles,
        )

    def forward(self, x, return_attention=False):
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape

        if F_in != self.n_features:
            raise ValueError("输入特征数与模型设置不一致")

        # ---- A. 变量选择 ----
        # 在最后一维计算特征权重
        weights = torch.softmax(
            self.variable_gate(x),
            dim=-1,
        )
        # weights: [B, T, N, F]

        projected = torch.stack([
            self.variable_projections[j](
                x[..., j:j + 1]
            )
            for j in range(self.n_features)
        ], dim=-2)
        # projected: [B, T, N, F, H]

        selected = (
            projected * weights.unsqueeze(-1)
        ).sum(dim=-2)
        # selected: [B, T, N, H]

        selected = self.variable_norm(selected)

        # ---- B. 每只股票的时间编码 ----
        # 转为 [B*N, T, H]
        temporal_input = (
            selected.permute(0, 2, 1, 3)
            .reshape(B * N, T, self.hidden_size)
        )

        temporal_output, _ = self.temporal_encoder(
            temporal_input
        )

        temporal_attended, _ = self.temporal_attention(
            temporal_output,
            temporal_output,
            temporal_output,
            need_weights=False,
        )

        temporal_output = self.temporal_norm(
            temporal_output + temporal_attended
        )

        # 还原为 [B, T, N, H]（T 为历史时点，N 为股票）
        temporal_output = temporal_output.reshape(
            B, N, T, self.hidden_size
        ).permute(0, 2, 1, 3)
        # [B, T, N, H]

        # ---- C. 跨股票注意力（每个历史时点都交换信息） ----
        # 注入股票身份 embedding: [N, H] -> [1, 1, N, H] 广播到 [B, T, N, H]
        stock_ids = torch.arange(N, device=x.device)
        stock_emb = self.stock_embedding(stock_ids)
        temporal_output = temporal_output + stock_emb.unsqueeze(0).unsqueeze(0)

        # 合并批与时间为 [B*T, N, H]，逐时点做跨股票注意力
        cross_input = temporal_output.reshape(
            B * T, N, self.hidden_size
        )
        cross_out, _ = self.cross_asset_attention(
            cross_input, cross_input, cross_input, need_weights=False
        )
        cross_input = self.cross_asset_norm(cross_input + cross_out)
        temporal_output = cross_input.reshape(
            B, T, N, self.hidden_size
        )
        # [B, T, N, H]

        # 取预测起点（历史窗口最后一个时点）的股票表示
        asset_repr = temporal_output[:, -1, :, :]
        # [B, N, H]

        # 跨股票注意力可视化（仅在最后时点，返回权重用于热力图）
        cross_weights = None
        if return_attention:
            _, cross_weights = self.cross_asset_attention(
                asset_repr, asset_repr, asset_repr,
                need_weights=True, average_attn_weights=False,
            )

        # ---- D. 前馈网络 ----
        asset_repr = self.output_norm(
            asset_repr + self.feed_forward(asset_repr)
        )

        # [B, N, prediction_length, n_quantiles]
        output = self.prediction_head(asset_repr)

        output = output.reshape(
            B,
            N,
            self.prediction_length,
            self.n_quantiles,
        )

        if return_attention:
            return output, cross_weights

        return output


# ============================================================
# 6. 分位数损失
# ============================================================
def quantile_loss(prediction, target):
    """
    prediction: [B, N, H, Q]
    target:     [B, N, H]
    """
    q = torch.tensor(
        QUANTILES,
        device=prediction.device,
        dtype=prediction.dtype,
    ).view(1, 1, 1, -1)

    error = target.unsqueeze(-1) - prediction

    loss = torch.maximum(
        q * error,
        (q - 1.0) * error,
    )

    return loss.mean()


# ============================================================
# 7. 训练模型一次
# ============================================================
model = TFTCrossAsset(
    n_features=len(FEATURE_COLS),
    n_stocks=n_stocks,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    prediction_length=PREDICTION_LENGTH,
    n_quantiles=len(QUANTILES),
    dropout=0.1,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_count = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(xb)
        loss = quantile_loss(prediction, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        batch_size = xb.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss={total_loss / total_count:.6f}"
    )


# ============================================================
# 8. 固定模型，按每5天预测未来5天
# ============================================================
model.eval()
all_predictions = []

max_origin = N_TIMESTEPS - PREDICTION_LENGTH - 1

forecast_origins = range(
    TRAIN_CUTOFF,
    max_origin + 1,
    ROLLING_STEP,
)

with torch.no_grad():
    for origin in forecast_origins:
        x_start = origin - ENCODER_LENGTH + 1
        x_end = origin + 1

        # 只输入截至预测起点的历史特征
        x_window = X_scaled[x_start:x_end]

        xb = torch.from_numpy(
            x_window[None, ...].astype(np.float32)
        ).to(device)

        prediction = model(xb)
        # [1, N, H, Q]

        prediction = (
            prediction[0]
            .cpu()
            .numpy()
            * target_scale
        )
        # 转回收益率原始尺度: [N, H, Q]

        for n_idx, stock in enumerate(stocks):
            for h in range(PREDICTION_LENGTH):
                target_idx = origin + h + 1

                all_predictions.append({
                    "stock": stock,
                    "time_idx": target_idx,
                    "actual": float(
                        y_raw[target_idx, n_idx]
                    ),
                    "q10": float(
                        prediction[n_idx, h, 0]
                    ),
                    "q50": float(
                        prediction[n_idx, h, 1]
                    ),
                    "q90": float(
                        prediction[n_idx, h, 2]
                    ),
                    "forecast_origin": origin,
                })

        print(
            f"预测起点 {origin} -> "
            f"预测区间 {origin + 1}～"
            f"{origin + PREDICTION_LENGTH}"
        )


# ============================================================
# 9. 检查预测结果
# ============================================================
pred_df = pd.DataFrame(all_predictions)

pred_df = pred_df.sort_values(
    ["stock", "time_idx"]
).reset_index(drop=True)

# 确保每只股票每个预测时间点只出现一次
assert not pred_df.duplicated(
    ["stock", "time_idx"]
).any()

assert pred_df["time_idx"].min() == TRAIN_CUTOFF + 1
assert pred_df["time_idx"].max() == N_TIMESTEPS - 1

print("\n预测结果前10行:")
print(pred_df.head(10))

print("\n预测记录数:", len(pred_df))


# ============================================================
# 10. 计算整体和分股票误差
# ============================================================
mae = np.mean(
    np.abs(pred_df["actual"] - pred_df["q50"])
)

rmse = np.sqrt(
    np.mean(
        (pred_df["actual"] - pred_df["q50"]) ** 2
    )
)

coverage = np.mean(
    (pred_df["actual"] >= pred_df["q10"])
    & (pred_df["actual"] <= pred_df["q90"])
)

print(f"\n整体 MAE: {mae:.6f}")
print(f"整体 RMSE: {rmse:.6f}")
print(f"q10～q90 实际覆盖率: {coverage:.2%}")

print("\n分股票误差:")
for stock, sub in pred_df.groupby("stock"):
    stock_mae = np.mean(
        np.abs(sub["actual"] - sub["q50"])
    )
    stock_rmse = np.sqrt(
        np.mean((sub["actual"] - sub["q50"]) ** 2)
    )

    print(
        f"{stock}: "
        f"MAE={stock_mae:.6f}, "
        f"RMSE={stock_rmse:.6f}"
    )


# ============================================================
# 11. 绘制实际值、预测值和分位数区间
# ============================================================
fig, axes = plt.subplots(
    n_stocks,
    1,
    figsize=(14, 3.5 * n_stocks),
    squeeze=False,
)

for i, stock in enumerate(stocks):
    ax = axes[i, 0]

    sub = pred_df[
        pred_df["stock"] == stock
    ].sort_values("time_idx")

    x = sub["time_idx"].to_numpy()

    ax.plot(
        x,
        sub["actual"].to_numpy(),
        label="Actual",
        linewidth=1.2,
    )

    ax.plot(
        x,
        sub["q50"].to_numpy(),
        label="Cross-Asset Prediction (q=0.5)",
        linestyle="--",
        linewidth=1.5,
    )

    ax.fill_between(
        x,
        sub["q10"].to_numpy(),
        sub["q90"].to_numpy(),
        alpha=0.2,
        label="q10-q90 interval",
    )

    ax.axvline(
        TRAIN_CUTOFF,
        linestyle=":",
        label="Training cutoff",
    )

    ax.set_title(f"{stock}: Actual vs Cross-Asset Forecast")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(
    "forecast_overview.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 12. 可选：查看最后一个预测批次的跨股票注意力
# ============================================================
with torch.no_grad():
    origin = TRAIN_CUTOFF
    x_window = X_scaled[
        origin - ENCODER_LENGTH + 1:origin + 1
    ]

    xb = torch.from_numpy(
        x_window[None, ...].astype(np.float32)
    ).to(device)

    _, attention = model(
        xb,
        return_attention=True,
    )

    # [B, heads, N, N]
    attention = attention[0].mean(dim=0).cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(attention, aspect="auto")
plt.colorbar(label="Attention weight")
plt.xticks(range(n_stocks), stocks, rotation=45)
plt.yticks(range(n_stocks), stocks)
plt.xlabel("Key stock")
plt.ylabel("Query stock")
plt.title("Cross-Asset Attention (averaged heads)")
plt.tight_layout()
plt.savefig(
    "cross_asset_attention.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
