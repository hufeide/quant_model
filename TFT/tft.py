
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# 1. 构造示例数据：两只股票，每只 500 个交易时点
rng = np.random.default_rng(42)
frames = []

for stock in ["A", "B"]:
    n = 500
    ret = rng.normal(0, 0.01, n)

    momentum = pd.Series(ret).rolling(
        10, min_periods=1
    ).mean()

    volatility = pd.Series(ret).rolling(
        20, min_periods=1
    ).std().fillna(0)

    frames.append(pd.DataFrame({
        "stock": stock,
        "time_idx": np.arange(n),
        "target": ret,
        "momentum": momentum,
        "volatility": volatility,
        "weekday": (np.arange(n) % 5).astype(str),
    }))

df = pd.concat(frames, ignore_index=True)

# 确保类别列类型统一
df["stock"] = df["stock"].astype(str)
df["weekday"] = df["weekday"].astype(str)

# 2. 设置历史窗口和预测窗口
encoder_length = 60
prediction_length = 5
cutoff = 399

# 训练数据：截至时点 399
train_df = df[df["time_idx"] <= cutoff].copy()

# 验证数据：保留历史数据，预测时点 400～404
valid_df = df[
    df["time_idx"] <= cutoff + prediction_length
].copy()

# 3. 定义 TFT 数据集
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="target",
    group_ids=["stock"],

    min_encoder_length=encoder_length,
    max_encoder_length=encoder_length,
    min_prediction_length=prediction_length,
    max_prediction_length=prediction_length,

    static_categoricals=["stock"],
    time_varying_known_reals=["time_idx"],
    time_varying_known_categoricals=["weekday"],
    time_varying_unknown_reals=[
        "target", "momentum", "volatility"
    ],

    target_normalizer=GroupNormalizer(groups=["stock"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    valid_df,
    predict=True,
    stop_randomization=True,
)

train_loader = training.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0,
)

val_loader = validation.to_dataloader(
    train=False,
    batch_size=64,
    num_workers=0,
)

# 4. 创建 TFT 模型
pl.seed_everything(42)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
)

# 5. 训练模型
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_checkpointing=False,
    logger=False,
)

trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

# 6. 预测，同时返回真实值和时间索引
result = tft.predict(
    val_loader,
    mode="quantiles",
    return_y=True,
    return_index=True,
)

predictions = result.output.detach().cpu().numpy()
y_true = result.y[0].detach().cpu().numpy()
index_df = result.index

print("预测张量形状:", predictions.shape)
print("真实值形状:", y_true.shape)
print("预测时间索引:")
print(index_df)

# predictions 的形状通常是：
# (股票数量, 预测步数, 分位数数量)
# 最后一维对应 [0.1, 0.5, 0.9]

# 7. 计算中位数预测的 MAE
y_pred_median = predictions[:, :, 1]
mae = np.mean(np.abs(y_true - y_pred_median))

print(f"\n验证集 MAE（中位数预测）: {mae:.6f}")

# 8. 绘制每只股票的真实值与预测值
num_samples = y_true.shape[0]

fig, axes = plt.subplots(
    num_samples,
    1,
    figsize=(12, 4 * num_samples),
    squeeze=False,
)

for i in range(num_samples):
    ax = axes[i, 0]

    # 从返回的索引中获取股票名称和预测起始时间
    stock_name = (
        str(index_df.iloc[i]["stock"])
        if "stock" in index_df.columns
        else f"样本 {i + 1}"
    )

    start_idx = (
        int(index_df.iloc[i]["time_idx"])
        if "time_idx" in index_df.columns
        else cutoff + 1
    )

    time_points = np.arange(
        start_idx,
        start_idx + y_true.shape[1],
    )

    actual = y_true[i]
    q10 = predictions[i, :, 0]
    q50 = predictions[i, :, 1]
    q90 = predictions[i, :, 2]

    # 真实值
    ax.plot(
        time_points,
        actual,
        label="Actual",
        marker="o",
        linewidth=2,
    )

    # 中位数预测
    ax.plot(
        time_points,
        q50,
        label="Predicted (q=0.5)",
        marker="x",
        linestyle="--",
        linewidth=2,
    )

    # 10%～90% 分位数区间
    ax.fill_between(
        time_points,
        q10,
        q90,
        alpha=0.2,
        label="Quantile interval (q=0.1–0.9)",
    )

    ax.set_title(f"Stock {stock_name}: Actual vs Predicted")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tft.png")
plt.show()

