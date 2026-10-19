
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


# 6. 固定模型，按每 5 个交易日滚动预测未来 5 天
# 训练截止 time_idx=399
# 预测区间 time_idx=400～499

prediction_length = 5
max_time_idx = int(df["time_idx"].max())

all_predictions = []

# 预测起点：399、404、409、...、494
for origin in range(
    cutoff,
    max_time_idx - prediction_length + 1,
    prediction_length,
):
    forecast_end = origin + prediction_length

    # 保留截至预测终点的数据，以便数据集构造完整预测窗口
    window_df = df[
        df["time_idx"] <= forecast_end
    ].copy()

    # 使用训练集的编码器、归一化器和其他设置
    # predict=True：每只股票只取当前窗口最后的 5 个预测点
    rolling_dataset = TimeSeriesDataSet.from_dataset(
        training,
        window_df,
        predict=True,
        stop_randomization=True,
    )

    rolling_loader = rolling_dataset.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0,
    )

    # 只推理，不训练
    result = tft.predict(
        rolling_loader,
        mode="quantiles",
        return_index=True,
    )

    preds = result.output.detach().cpu().numpy()
    indices = result.index

    # 分别保存每只股票未来 5 天的预测
    for i in range(len(indices)):
        stock = str(indices.iloc[i]["stock"])

        for h in range(prediction_length):
            target_idx = origin + h + 1

            actual_row = df.loc[
                (df["stock"] == stock)
                & (df["time_idx"] == target_idx),
                "target",
            ]

            if actual_row.empty:
                continue

            all_predictions.append({
                "stock": stock,
                "time_idx": target_idx,
                "actual": float(actual_row.iloc[0]),
                "q10": float(preds[i, h, 0]),
                "q50": float(preds[i, h, 1]),
                "q90": float(preds[i, h, 2]),
                "forecast_origin": origin,
            })

    print(
        f"预测起点 {origin} → "
        f"预测区间 {origin + 1}～{forecast_end}"
    )

# 7. 合并所有预测窗口
pred_df = pd.DataFrame(all_predictions)
pred_df = pred_df.sort_values(
    ["stock", "time_idx"]
).reset_index(drop=True)

print("\n预测结果前 10 行：")
print(pred_df.head(10))

print("\n预测结果总行数：", len(pred_df))

# predictions 的形状通常是：
# (股票数量, 预测步数, 分位数数量)
# 最后一维对应 [0.1, 0.5, 0.9]

# 8. 计算预测误差
mae = np.mean(
    np.abs(pred_df["actual"] - pred_df["q50"])
)

rmse = np.sqrt(
    np.mean((pred_df["actual"] - pred_df["q50"]) ** 2)
)

print(f"\n整体 MAE:  {mae:.6f}")
print(f"整体 RMSE: {rmse:.6f}")

# 9. 每只股票单独绘图
stocks = pred_df["stock"].unique()

fig, axes = plt.subplots(
    len(stocks),
    1,
    figsize=(14, 5 * len(stocks)),
    squeeze=False,
)

for i, stock in enumerate(stocks):
    ax = axes[i, 0]

    sub = pred_df[
        pred_df["stock"] == stock
    ].sort_values("time_idx")

    x = sub["time_idx"].to_numpy()

    # 真实收益率
    ax.plot(
        x,
        sub["actual"],
        label="Actual",
        linewidth=1.5,
    )

    # TFT 中位数预测
    ax.plot(
        x,
        sub["q50"],
        label="TFT Prediction (q=0.5)",
        linestyle="--",
        linewidth=1.5,
    )

    # 10%～90% 分位数区间
    ax.fill_between(
        x,
        sub["q10"].to_numpy(),
        sub["q90"].to_numpy(),
        alpha=0.2,
        label="q10–q90 interval",
    )

    # 训练与预测的分界线
    ax.axvline(
        cutoff,
        linestyle=":",
        label="Training cutoff",
    )

    ax.set_title(
        f"Stock {stock}: Actual vs Rolling Forecast"
    )
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()