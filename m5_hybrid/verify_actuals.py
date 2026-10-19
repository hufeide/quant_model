"""独立验证：两个脚本的 data['sales'] 是否一致（即图中 Actual 线是否同一批数据）。"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "m5"
SEED = 42
N_ITEMS = 50
SEQ_LEN = 56
PRED_LEN = 28
STRIDE = 7


def make_origins(n_days, seq_len, pred_len, stride):
    T_train = int(n_days * 0.70)
    T_val = int(n_days * 0.85)
    origins = np.arange(0, n_days - seq_len - pred_len + 1, stride)
    test = origins[(origins + seq_len >= T_val) & (origins + seq_len + pred_len <= n_days)]
    return test


# ---- 公共：选商品 + 日列排序 ----
sales = pd.read_csv(DATA / "sales_train_validation.csv")
day_cols = [c for c in sales.columns if c.startswith("d_")]
day_num = np.array([int(c[2:]) for c in day_cols])
order = np.argsort(day_num)
day_cols = [day_cols[i] for i in order]
day_num = day_num[order]

rng = np.random.default_rng(SEED)
available = np.array(sorted(sales["item_id"].unique()))
chosen = sorted(rng.choice(available, size=N_ITEMS, replace=False).tolist())
item_to_idx = {it: i for i, it in enumerate(chosen)}

# ===== 方法 A：transfomer2.py 的 numpy 求和 =====
sampled = sales.loc[sales["item_id"].isin(chosen),
                    ["item_id", "store_id", "state_id", *day_cols]].copy()
sampled = sampled.set_index(["item_id", "store_id", "state_id"])
raw = sampled[day_cols].to_numpy(dtype=np.float32)
raw = np.maximum(raw, 0.0)
row_items = sampled.index.get_level_values("item_id").to_numpy()
n_days = len(day_cols)
matA = np.zeros((n_days, len(chosen)), dtype=np.float32)
for r, it in enumerate(row_items):
    matA[:, item_to_idx[it]] += raw[r, :]

# ===== 方法 B：tranformer.py 的 pandas 聚合 =====
calendar = pd.read_csv(DATA / "calendar.csv")
cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
cal_cols = [c for c in cal_cols if c in calendar.columns]
cal_sub = calendar[cal_cols].copy()
cal_sub["d_num"] = cal_sub["d"].str.replace("d_", "", regex=False).astype(int)
cal_sub = cal_sub.sort_values("d_num").drop_duplicates("d_num")
cal_sub["date"] = pd.to_datetime(cal_sub["date"])

smp = sales.loc[sales["item_id"].isin(chosen),
                ["item_id", "store_id", "state_id", *day_cols]].copy()
long = smp.melt(id_vars=["item_id", "store_id", "state_id"], value_vars=day_cols,
                var_name="d", value_name="sales")
long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype(np.float32)
long = long.merge(cal_sub[["d", *[c for c in cal_cols if c != "d"]]],
                  on="d", how="left")
group_cols = ["item_id", "d", *[c for c in cal_cols if c not in ("d",)]]
panel = (long.groupby(group_cols, as_index=False, dropna=False)
         .agg(sales=("sales", "sum"))
         .sort_values(["item_id", "date" if "date" in long.columns else "d"])
         .reset_index(drop=True))
wide = panel.pivot(index="d", columns="item_id", values="sales")
day_to_date = cal_sub.set_index("d")["date"]
wide.index = pd.to_datetime(wide.index.map(day_to_date))
wide = wide.sort_index()
wide = wide.reindex(columns=chosen).fillna(0)
matB = wide.to_numpy(dtype=np.float32)

print("matA shape:", matA.shape, "matB shape:", matB.shape)
print("n_days A:", matA.shape[0], "n_days B:", matB.shape[0])
diff = np.abs(matA - matB)
print("max abs diff (全表):", float(diff.max()))
print("mean abs diff:", float(diff.mean()))
print("是否有任一位置差 > 1e-3:", bool((diff > 1e-3).any()))

# 测试窗口（两图都画 test_o[-1]）
test_o = make_origins(matA.shape[0], SEQ_LEN, PRED_LEN, STRIDE)
fo = int(test_o[-1])
w0, w1 = fo + SEQ_LEN, fo + SEQ_LEN + PRED_LEN
print("\nfinal_origin:", fo, "window rows:", w0, "->", w1)
print("\nchosen[:6]:", chosen[:6])
print("\n--- Actual (方法A / transfomer2) 前6商品 测试窗口 ---")
print(np.round(matA[w0:w1, :6], 2))
print("\n--- Actual (方法B / tranformer.py) 前6商品 测试窗口 ---")
print(np.round(matB[w0:w1, :6], 2))
print("\n窗口内逐元素 max abs diff:", float(np.abs(matA[w0:w1, :6] - matB[w0:w1, :6]).max()))
