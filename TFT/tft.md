
时间序列大模型篇--基于Transformer的时间融合TFT（九）
时间序列大模型篇--基于Transformer的时间融合TFT（九）
叫我阿哲就好
叫我阿哲就好​
北京理工大学 应用统计硕士
收录于 · 时间序列分析
43 人赞同了该文章
​
目录
PRE
上篇介绍了基于LSTM框架的时序大模型DeepAR，本篇则挑选基于Transformer架构的时序模型TFT（Temporal Fusion Transformer）进行讲解，正如上篇所说，虽然现在Transformer框架确实在NLP 领域占据主导地位，但它们在时间序列相关任务中的表现并没有明显优于 LSTM。还是需要具体数据具体分析。

前置知识：


叫我阿哲就好：时间序列分析--长短期记忆模型LSTM（七）
30 赞同 · 10 评论 文章

叫我阿哲就好：transformer(学习笔记):Attention Is All You Need（2017）
6 赞同 · 0 评论 文章
模型简介
Temporal Fusion Transformer (TFT) 是 Google 2020年发布的基于 Transformer 的时间序列预测模型。

论文：

https://arxiv.org/abs/1912.09363
arxiv.org/abs/1912.09363
其凭借以下几大优点以及新颖之处，为其在时序预估领域博得了一席之地：

1、可输入丰富的特征：

i） known future inputs：如未来高校毕业生人数、未来一周天气、GDP等

ii） exogenous time series that are only observed in the past：如至今平台新增流量、月活等数据

iii）static (i.e. time-invariant) covariates：时不变特征，如：产品标签、国家等。


图来自原文
2、异构时间序列：支持来自不同分布的多个时间序列的训练。为了实现这一目标，TFT 架构将处理分为两部分：专注于特定事件特征的局部处理和捕获所有时间序列的集体特征的全局处理。

3、多水平预测：支持多步预测。除了实际预测之外，TFT 还通过使用分位数回归输出预测区间。

4、可解释性： TFT 的核心是基于transformer的架构。通过利用自注意力（ Multi Head 注意力机制），在分析时可以提供对特征重要性的额外见解。

5、高性能：在基准测试中，TFT 的性能优于传统统计模型 (ARIMA) 以及基于 DNN 的模型。

可以看出，其除了支持多序列预估的能力外，提供的额外可解释性都能够较好的满足现实业务场景，

这也是深度学习发展的一大主调。

模型框架
第一次看TFT的模型框架还是挺吓人的，但其实仔细品一下，还是较容易理解的：


原文模型框架
整个模型由多个核心组件构成：门控残差网络（GRN）、变量选择网络（VSN），LSTM编码器、LSTM解码器以及multi head attention机制最后拼接标准神经网络的输出结构，多个组件进行协同工作以实现模型的预估工作。

对于给定的时间步t、回溯窗口k和已知未来变量窗口 
 ，其中⋹ 
 。

模型的主要输入包括：

1) 已观察到过去的时间段 
 的输入x；

2)未来时间段 
 已知的辅助外生变量x；

3）静态时不变变量 S；

受变量窗口 
 的影响，最终目标变量y的预估窗口应该也是处于 
 时间段。接下来我们针对框架中的核心组件进行逐个解剖。

门控残差网络（GRN）
GRN作为模型的一大基本模块，多次出现在模型框架中：


精确衡量外部输入与目标变量间的相关关系是一件较为复杂的事情，因此模型选择使用门控单元，仅在需要时灵活地应用非线性处理，它结合了门控机制和残差连接的思想，其核心组成为两个dense layer以及两种类型的激活函数，称为ELU（指数线性单元）和GLU（门控线性单元）。

GRN输入包括一个主要输入a和一个上下文向量c：


layerNorm表示常规的归一化处理，ELU为激活函数


ELU激活函数
ELU 激活函数的主要作用包括对负值的抑制作用、对梯度消失问题的缓解以及对网络训练的稳定性有所提高。

GLU（门控线性单元）：


门控线性单元
 函数为sigmod型激活函数，W，b分别表示权重与偏差，计算两项的哈马达积以得到GLU的输出值。GLU 主要用于选择最重要的特征（赋权）来进行后续的预估（获取相关性）。

最终输出通过标准层归一化。GRN 还包含残差连接，这意味着网络可以在必要时完全跳过输入，因为 GLU 输出可以全部接近于 0 以抑制非线性贡献。如果没有上下文向量，GRN 只会将上下文输入视为零——即等式中 c = 0 。在某些情况下，根据 GRN 所在的位置，网络还可以使用静态变量。

变量选择网络（VSN）
顾名思义，VSN充当特征选择机制，并非所有时间序列都是复杂的。因此模型应该能够区分有用的特征和嘈杂的特征。此外，由于TFT框架包含 3 种类型的输入，因此TFT 使用VSN的 3 个实例，并且每个实例具有不同的权重（每个 VSN 单元的颜色不同）。



变量选择网络（VSN）
可以看到VSN在底层利用 GRN 来实现其过滤的功能。

此时，在t时刻的所有过去输入经过flattened之后通过 GRN 单元（蓝色），然后通过 softmax分类函数，生成归一化的权重向量u。

此外，每个特征都会通过自己的 GRN，这会导致创建一个名为 
 的处理向量，每个变量对应一个向量。

最后，输出就可以表示为 
 和u的线性组合。

每个特征都有自己的 GRN，但每个特征的 GRN 在同一回溯周期内的所有时间步上都是相同的。
静态变量的 VSN 不考虑上下文向量c
LSTM Encoder Decoder
LSTM 编码器解码器在 NLP模型中较为常见 中。该组件主要包含两个用途：

1、到目前为止，输入已通过VSN，并对特征进行了相应的编码和加权。然而，由于我们的输入是时间序列数据，因此模型还应该理解时间/顺序顺序。因此，LSTM 编码器解码器模块的首要目标是生成识别上下文的embedding，称为 
 。类似于经典 Transformer 中使用的位置编码。

又由于我们TFT框架的输入包含三种类型的数据：

1、已知的过去输入被馈送到编码器，

2、已知的未来输入被馈送到解码器。

3、那么静态信息呢？静态变量在处理中起着重要作用

根据外部数据正确调节输入：具体来说，不是将初始隐藏状态以及LSTM 的单元状态设置为 0，而是将它们初始化分别为两个向量 
 、 
 （由 TFT 的静态协变量编码器产生）。因此，最终的embedding 
 将以外生信息为条件。

Multi-Head Attention
这是 TFT 架构的最后一部分,也是具有亮点的一部分。在此步骤中，应用了近几年大火的的自注意力机制，这有助于解决模型学习不同时间步长的长期依赖问题。

所有基于 Transformer 的架构都利用注意力来学习输入数据之间的复杂依赖关系。

TFT模型提出了一种新颖的可解释多头注意力机制，与标准实现相反，它提供了特征可解释性。在原始架构中，有不同的“head”（查询/键/值权重矩阵），以便将输入投影到不同的表示子空间中。这种方法的缺点是权重矩阵没有共同点，因此无法解释。



而TFT 的多头注意力增加了一个新的矩阵/分组，


使得不同的头共享一些权重，然后可以用季节性分析来解释这些权重。


分位数回归
在模型框架的最后，应用了分位数回归的技巧，目前也成为了大多数预估模型的标配，简单的点预估已经难以满足复杂业务的需求，我们更需要的是一个置信区间。

在涉及时间序列预测的许多应用中，仅目标变量的预测是不够的。估计预测的不确定性同样重要。通常，这以预测区间的形式出现。如果我们决定在输出中包含预测区间，线性回归和均方误差将变得不适用。

分位数回归是标准线性回归的扩展，它估计目标变量的条件中位数，并且可以在不满足线性回归的假设时使用。除了中位数之外，分位数回归还可以计算 0.25 和 0.75 分位数（或与此相关的任何百分位数），这意味着模型能够输出围绕实际预测的预测区间。


给定 y 和 ŷ 分别为实际值和预测值，q 为 0 到 1 之间的分位数值，分位数损失函数定义为：


随着 q 值的增加，与低估相比，高估会受到更大的惩罚。例如，当 q 等于 0.75 时，高估将受到 0.75 倍的惩罚，而低估则将受到 0.25 倍的惩罚。这就是预测区间的创建方式。

TFT模型实现是通过最小化 q ⋹ [0.1, 0.5, 0.9] 上的分位数损失总和来训练的。分位数损失的使用并不是排他性的 - 可以使用其他类型的损失函数，例如 MSE、MAPE 等。

以上就是TFT模型的整个模型框架以及训练流程，接下来来看一下具体python建模应用以及其可解释性的实际应用。

模型应用
整体模型框架仍是基于pytorch，与上篇DeepAR建模流程类似。

导包，当时自己建模时所涉模块较多，懒得做区分了，因此import的较多

import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import logging
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
关于数据处理模块就直接跳过了，关于这一部分，前面相关文章介绍的比较详尽。

建立训练集

max_prediction_length = 60
max_encoder_length = 2*60
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
    time_varying_known_reals=["days_from_start","day","day_of_week", "month", 'year'],
    time_varying_unknown_reals=['act_normalized'],
    target_normalizer=GroupNormalizer(
        groups=["l1_name"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
     
模型训练

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=45,
    accelerator='auto',
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
     

模型训练时间较长，及时保存模型

best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
#zip and download the model
!zip  -r modeltft.zip lightning_logs/lightning_logs/version_14/*
模型预估

## using the new pytorch-forecasting method (1.0)
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
MAE()(predictions.output, predictions.y)
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

for idx in range(8):  
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=QuantileLoss(),ax=ax)

     
#list=['采购/贸易','房地产/建筑','服务业','供应链/物流','生产制造','金融','运营','产品']
我这里直接进行了多品类（多重异构时间序列）的预估过程，整体效果如下：



还是较为不错的。

可解释性
最后针对TFT模型可解释能力进行一个应用说明，毕竟这也是TFT的一大亮点之处。

首先针对于训练过程，图中灰色曲线表示的则是attention机制，可以理解为历史各时间点数据对于后续预估结果的贡献程度。


可以调用interpret_output函数进行进一步分析：

raw_predictions= best_tft.predict(val_dataloader, mode="raw", return_x=True)
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)

attention 机制

特征重要性
Temporal Fusion Transformer尝试通过考虑预测的稳健性来计算每个特征的影响，也可以根据attention曲线来观察预估时刻的重要性变化。

以及TFT模型支持针对历史数据在各种维度上进行初步的异常检测，帮助我们进行分析：

#Analysis on the training set
predictions = best_tft.predict(train_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)


最后附一个调参模块，由于计算时间实在久，本文模型均为手动调参，有条件的同学可以尝试

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
    use_learning_rate_finder=False 
)

# save study results
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# print best hyperparameters
print(study.best_trial.params)
end
以上就是本篇基于transfromer框架的时序大模型介绍，随着本篇结束，时序领域的模型的坑基本也都填完了，后续看是否存在有意思的点遗漏在进行补充。后续准备开一个新坑，做causal。