# 双流量化选股模型：TSMixer + 基本面 MLP + 门控融合

这是一个端到端研究基线，包含 CSV 数据读取、基本面 PIT（Point-in-Time）对齐、技术因子生成、按时间切分并 purge 重叠标签、仅用训练期拟合缺失值填充/稳健缩放、双流神经网络训练、验证集早停及测试期预测输出。

> 重要：这是预测模型研究代码，不是完整的实盘交易系统。报告里的 Top/Bottom 组未来收益是重叠的未来 H 日标签均值，不等于考虑了持仓重叠、成交约束、费用、税费、滑点和市场冲击后的真实策略收益。

## 1. 环境

建议 Python 3.10+，依赖：

```bash
pip install -r requirements.txt
```

## 2. 输入 CSV

### `bars.csv`

至少包含：

```text
date,symbol,open,high,low,close,volume,amount,turnover
2020-01-02,000001.SZ,16.20,16.50,16.10,16.40,123456,201234567,0.35
```

- `date`：交易日；`symbol`：股票代码，建议读入为字符串并保留交易所后缀。
- 必须有 `open/high/low/close/volume`；`amount` 和 `turnover` 可选。
- 同一只股票同一交易日只能有一行。
- 同一列价格应采用一致的复权口径。该代码把未来标签定义为信号日收盘后形成信号、下一交易日开盘买入、H 个交易日后对应开盘卖出。复权价格计算出的标签是研究目标，不等于实际可成交回报。
- 默认会在每只股票首末观测日期之间补齐市场日历缺失的行。若缺失是停牌/无成交，原始 open 仍为空，因此在相应入场/退出价格不可用时该标签会被剔除。输入最好包含可靠、完整的市场交易日记录及可识别的停牌状态。

### `fundamentals.csv`

至少包含：

```text
available_date,symbol,roe,roa,pe,pb,revenue_yoy,profit_yoy,debt_to_asset
2020-04-28,000001.SZ,0.12,0.01,8.5,1.1,0.08,0.10,0.72
2020-08-28,000001.SZ,0.13,0.01,8.2,1.0,0.09,0.11,0.71
```

- `available_date` **必须表示这条基本面信息最早可用于模型决策的日期**，不能简单使用报告期结束日。例如年报截止日期是 12 月 31 日，但公告在次年 4 月才公开，则不能在 12 月使用这份年报。
- 如果公告发生在收盘后，而模型信号在当天收盘生成，应将该记录的 `available_date` 设置为下一交易日，除非数据确实在决策时点之前可用。
- 推荐每个 `symbol + available_date` 是一条完整 PIT 快照。如果某个指标在新快照里缺失，代码会向前延续其之前已知值；请确认这符合数据语义。
- 基本面数据必须是历史 PIT/vintage 数据。若供应商把后来修订的财报值覆盖回历史记录，单靠模型代码无法修复这种泄漏。
- 尽可能明确指定真实特征列，避免把数值型证券代码、报告期编号等元数据误当特征。

## 3. 运行

例：

```bash
python two_stream_quant.py \
  --bars data/bars.csv \
  --fundamentals data/fundamentals.csv \
  --fund-date-col available_date \
  --fund-cols roe,roa,pe,pb,revenue_yoy,profit_yoy,debt_to_asset \
  --horizon 5 \
  --seq-len 60 \
  --epochs 50 \
  --batch-size 512 \
  --device auto \
  --out-dir runs/two_stream_tsmixer
```

参数解释：

- `--horizon 5`：信号日为 t，开盘 t+1 入场，以 t+6 的开盘价作为 H=5 的退出参考价。
- `--seq-len 60`：每只股票技术面分支使用过去 60 个市场交易日的特征。
- `--fund-cols`：基本面列清单。自动检测可用，但明确指定更安全。
- 默认按日期顺序切分为训练/验证/测试 70%/15%/15%，并剔除标签结束日期跨越下一分段的样本。
- `--device auto` 会在 CUDA 可用时使用 GPU；否则使用 CPU。

## 4. 输出文件

- `best_model.pt`：最佳验证 RankIC 对应模型、特征列、训练期填充/缩放参数及配置。
- `report.json`：数据切分、特征名、样本数、验证/测试指标和训练参数。
- `training_history.csv`：每轮训练与验证指标。
- `validation_predictions.csv` / `test_predictions.csv`：样本日期、股票、未来 H 日标签、模型分数。
- `validation_daily_metrics.csv` / `test_daily_metrics.csv`：按日 RankIC 和 Top/Bottom 组未来收益。

主要指标包括 RankIC 均值、未年化 ICIR、预测误差和 Top-Bottom 未来收益差。对重叠 H 日标签，不能简单把这些结果当作真实组合净值、年化 Sharpe 或可成交 PnL。

## 5. 数据和研究边界

1. 特征只做历史/当期滚动计算；基本面按 `available_date` 向后 as-of 对齐。
2. 缺失值填充与缩放统计量仅用训练期数据拟合。
3. 验证集用于早停，测试集只在训练完成后评估一次。
4. 双流融合必须与“仅技术面”“仅基本面”“简单拼接”基线做同区间消融对比。单模型 RankIC 高，并不能证明扣费后策略更好。
5. 本代码不模拟涨跌停无法成交、停牌期间持仓、印花税/佣金、冲击成本、仓位约束、指数基准、调仓交叠和真实撮合。因此不要直接将测试结果用于实盘决策。
6. 真实部署需要另行实现完全一致的在线特征工程、训练特征缩放、数据版本控制与订单执行模块。
