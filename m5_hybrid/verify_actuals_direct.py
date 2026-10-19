"""直接调用两个脚本的 prepare_data() 对比 data['sales']（Actual 数据源）。"""
import importlib.util
import numpy as np

SEQ_LEN = 56
PRED_LEN = 28
STRIDE = 7


def make_origins(n_days, seq_len, pred_len, stride):
    T_train = int(n_days * 0.70)
    T_val = int(n_days * 0.85)
    origins = np.arange(0, n_days - seq_len - pred_len + 1, stride)
    test = origins[(origins + seq_len >= T_val) & (origins + seq_len + pred_len <= n_days)]
    return test


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Loading tranformer.py ...")
t1 = load_module("/home/administator/workspace/quant_model/m5_hybrid/tranformer.py", "t1_mod")
data1 = t1.prepare_data()

print("\nLoading transfomer2.py ...")
t2 = load_module("/home/administator/workspace/quant_model/m5_hybrid/transfomer2.py", "t2_mod")
data2 = t2.prepare_data()

sales1 = data1["sales"]
sales2 = data2["sales"]
print("\nsales1 shape:", sales1.shape, "dtype:", sales1.dtype)
print("sales2 shape:", sales2.shape, "dtype:", sales2.dtype)

chosen1 = data1["chosen"]
chosen2 = data2["chosen"]
print("chosen equal:", chosen1 == chosen2)
print("chosen[:6]:", chosen1[:6])

dates1 = data1["dates"]
dates2 = data2["dates"]
print("dates equal:", np.array_equal(np.asarray(dates1), np.asarray(dates2)))

n_days = sales1.shape[0]
test_o = make_origins(n_days, SEQ_LEN, PRED_LEN, STRIDE)
fo = int(test_o[-1])
w0, w1 = fo + SEQ_LEN, fo + SEQ_LEN + PRED_LEN
print("\nfinal_origin:", fo, "window:", w0, "->", w1)

print("\n--- sales1 (tranformer.py) actuals 测试窗口，前6商品 ---")
print(np.round(sales1[w0:w1, :6], 2))
print("\n--- sales2 (transfomer2.py) actuals 测试窗口，前6商品 ---")
print(np.round(sales2[w0:w1, :6], 2))

diff = np.abs(sales1 - sales2)
print("\n全表 max abs diff:", float(diff.max()))
print("窗口内 max abs diff (前6商品):", float(np.abs(sales1[w0:w1, :6] - sales2[w0:w1, :6]).max()))
