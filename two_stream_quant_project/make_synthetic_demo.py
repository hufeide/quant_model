#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate synthetic CSVs solely to smoke-test the training script; not investment data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='demo_data')
    parser.add_argument('--symbols', type=int, default=20)
    parser.add_argument('--days', type=int, default=360)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range('2022-01-03', periods=args.days)
    bars, funds = [], []
    for j in range(args.symbols):
        symbol = f'{j:06d}.SZ'
        start_price = 8.0 + j * 0.8
        rets = rng.normal(0.0001, 0.015, len(dates)) + (j - args.symbols / 2) * 0.00001
        close = start_price * np.exp(np.cumsum(rets))
        overnight = rng.normal(0.0, 0.003, len(dates))
        open_ = close * np.exp(-rng.normal(0.0, 0.006, len(dates)))
        if len(dates) > 1:
            open_[1:] = close[:-1] * np.exp(overnight[1:])
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.008, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.008, len(dates)))
        volume = rng.lognormal(12.0, 0.55, len(dates))
        amount = volume * close
        turnover = rng.uniform(0.1, 4.0, len(dates))
        bars.extend(zip(dates, [symbol] * len(dates), open_, high, low, close, volume, amount, turnover))

        for k in range(0, len(dates), 45):
            funds.append((dates[k], symbol,
                          rng.normal(0.10, 0.03), rng.normal(12.0, 3.0),
                          rng.normal(1.4, 0.25), rng.normal(0.08, 0.04)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bars, columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']).to_csv(out / 'bars.csv', index=False)
    pd.DataFrame(funds, columns=['available_date', 'symbol', 'roe', 'pe', 'pb', 'revenue_yoy']).to_csv(out / 'fundamentals.csv', index=False)
    print(f'Wrote synthetic (non-investment) test data to {out.resolve()}')
    print('Run: python two_stream_quant.py --bars demo_data/bars.csv --fundamentals demo_data/fundamentals.csv --fund-cols roe,pe,pb,revenue_yoy --horizon 5 --seq-len 60 --epochs 2 --d-model 16 --depth 1 --device cpu --out-dir demo_runs')


if __name__ == '__main__':
    main()
