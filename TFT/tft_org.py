"""
基于 Transformer 的时间融合模型 TFT（Temporal Fusion Transformer）示例脚本。

整理自 tft.md，沿用 pytorch_forecasting 官方建模流程。
原文中引用了 time_df 却未给出数据构造，本脚本补充了一份合成数据，
使代码无需额外数据即可运行（多品类异构时间序列的回归预测）。

依赖：
    pip install numpy pandas matplotlib seaborn lightning pytorch-forecasting tensorboard
"""

import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无图形界面时使用，避免报错
import matplotlib.pyplot as plt
import seaborn as sns

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import logging

logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 1. 构造合成数据（原文缺失，这里补齐，使脚本可直接运行）
# ----------------------------------------------------------------------------
def build_synthetic_time_df(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成多品类（l1_name）的日度时间序列，覆盖 TimeSeriesDataSet 所需字段。"""
    rng = np.random.default_rng(seed)
    categories = ["采购/贸易", "房地产/建筑", "服务业", "供应链/物流",
                  "生产制造", "金融", "运营", "产品"]
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    for cat_idx, cat in enumerate(categories):
        for d in range(n_days):
            date = base_date + pd.Timedelta(days=d)
            # 周期性 + 品类差异 + 噪声，作为目标变量（未归一化）
            trend = 100 + 5 * cat_idx + 0.2 * d
            seasonal = 10 * np.sin(2 * np.pi * d / 30.0)
            noise = rng.normal(0, 3)
            act = max(1.0, trend + seasonal + noise)
            rows.append(
                {
                    "l1_name": cat,
                    "days_from_start": d,
                    "day": date.day,
                    "day_of_week": date.dayofweek,
                    "month": date.month,
                    "year": date.year,
                    "act": act,
                    "act_normalized": act,  # 原数据已做归一化，这里直接用原值占位
                }
            )
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# 2. 建立训练集 / 验证集
# ----------------------------------------------------------------------------
def build_datasets(time_df: pd.DataFrame):
    max_prediction_length = 60
    max_encoder_length = 2 * 60
    training_cutoff = time_df["days_from_start"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        time_df[lambda x: x.days_from_start <= training_cutoff],
        time_idx="days_from_start",
        target="act_normalized",
        group_ids=["l1_name"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["l1_name"],
        time_varying_known_reals=["days_from_start", "day", "day_of_week", "month", "year"],
        time_varying_unknown_reals=["act_normalized"],
        target_normalizer=GroupNormalizer(
            groups=["l1_name"], transformation="softplus"
        ),  # we normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, time_df, predict=True, stop_randomization=True
    )

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    return training, validation, train_dataloader, val_dataloader


# ----------------------------------------------------------------------------
# 3. 模型训练
# ----------------------------------------------------------------------------
def train_model(training, train_dataloader, val_dataloader):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=45,  # 原文配置；演示时可调小（如 3）以加快运行
        accelerator="auto",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 模型训练时间较长，及时保存模型
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # 原文使用 Jupyter 魔法命令 !zip，这里改为可运行的压缩逻辑
    import shutil

    shutil.make_archive("modeltft", "zip", "lightning_logs")
    print("已打包模型到 modeltft.zip")

    return best_tft, trainer


# ----------------------------------------------------------------------------
# 4. 模型预估
# ----------------------------------------------------------------------------
def predict_and_plot(best_tft, val_dataloader):
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="auto"))
    print("验证集 MAE:", MAE()(predictions.output, predictions.y))

    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    for idx in range(8):
        fig, ax = plt.subplots(figsize=(10, 4))
        best_tft.plot_prediction(
            raw_predictions.x, raw_predictions.output,
            idx=idx, add_loss_to_title=QuantileLoss(), ax=ax,
        )
        fig.savefig(f"prediction_idx_{idx}.png")
        plt.close(fig)


# ----------------------------------------------------------------------------
# 5. 可解释性分析
# ----------------------------------------------------------------------------
def interpret(best_tft, val_dataloader, train_dataloader):
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    fig = best_tft.plot_interpretation(interpretation)
    fig.savefig("interpretation.png")
    plt.close(fig)

    # 训练集上的预测 vs 实际（可用于初步异常检测）
    predictions = best_tft.predict(train_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
        predictions.x, predictions.output
    )
    fig = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    fig.savefig("prediction_actual_by_variable.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. 调参模块（计算耗时，默认关闭）
# ----------------------------------------------------------------------------
def tune(train_dataloader, val_dataloader, enabled: bool = False):
    if not enabled:
        print("跳过超参搜索（将 tune_enabled=True 以启用，耗时较长）")
        return

    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=1,
        max_epochs=1,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(30, 128),
        hidden_continuous_size_range=(30, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )

    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)
    print(study.best_trial.params)


def main():
    time_df = build_synthetic_time_df(n_days=200)
    print("合成数据形状:", time_df.shape)

    training, validation, train_dataloader, val_dataloader = build_datasets(time_df)

    # 基线对照
    baseline = Baseline()
    print("Baseline SMAPE:", SMAPE()(baseline.predict(val_dataloader, return_y=True)))

    best_tft, trainer = train_model(training, train_dataloader, val_dataloader)
    predict_and_plot(best_tft, val_dataloader)
    interpret(best_tft, val_dataloader, train_dataloader)
    tune(train_dataloader, val_dataloader, enabled=False)


if __name__ == "__main__":
    main()
