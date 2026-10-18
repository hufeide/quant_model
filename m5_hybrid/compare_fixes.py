#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B test of the model fixes against the previous optimized baseline.

Same optimized hyper-parameters (LR=3e-3, heads=8, ensemble) are used for every
run so the comparison isolates the *architecture* changes:
  v1_baseline : original architecture (mean-pool cross-item, no horizon self-attn,
                late covariate residual)
  v2_fixed    : fixed architecture (per-patch cross-item, horizon self-attention,
                future-conditioned query) + low-coverage price mask
  v2_mae      : v2_fixed trained with MAE instead of Huber (objective alignment)

Reports the best non-overlapping WAPE across sales_only / multi_feature / ensemble.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
EXP = BASE / "experiment_runs"
EXP.mkdir(parents=True, exist_ok=True)
SCRIPT = BASE / "m5_hybrid.py"

# Common optimized hyper-parameters discovered by the 25-round sweep.
COMMON = {
    "M5_LR": "3e-3",
    "M5_N_HEADS": "8",
    "M5_ENSEMBLE": "true",
}

EXPERIMENTS = [
    ("v1_baseline", {**COMMON, "M5_ARCH": "v1", "M5_LOSS": "huber", "M5_PRICE_MASK": "false"}),
    ("v2_fixed", {**COMMON, "M5_ARCH": "v2", "M5_LOSS": "huber", "M5_PRICE_MASK": "true"}),
    ("v2_mae", {**COMMON, "M5_ARCH": "v2", "M5_LOSS": "mae", "M5_PRICE_MASK": "true"}),
]


def run(label: str, config: dict) -> dict:
    log_path = EXP / f"fix_{label}.log"
    env = {**os.environ, **config}
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)], env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(BASE)
        )
    elapsed = time.time() - t0
    rec = {"label": label, "config": json.dumps(config, sort_keys=True),
           "exit_code": proc.returncode, "elapsed_sec": round(elapsed, 1)}
    if proc.returncode != 0:
        rec["best_wape"] = float("nan")
        rec["note"] = "ERROR"
        return rec
    df = pd.read_csv(OUT / "metrics.csv")
    sub = df[df["evaluation"] == "non_overlapping"]
    rec["best_wape"] = round(float(sub["WAPE"].min()), 6)
    rec["note"] = " | ".join(f"{r['model']}={r['WAPE']:.4f}" for _, r in sub.iterrows())
    df.to_csv(EXP / f"fix_{label}_metrics.csv", index=False)
    return rec


def main() -> None:
    results = [run(label, cfg) for label, cfg in EXPERIMENTS]
    out_df = pd.DataFrame(results)
    out_df.to_csv(EXP / "fix_comparison.csv", index=False)
    print("\n===== FIX COMPARISON (lower WAPE is better) =====")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
