#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Greedy 25-round optimization loop for m5_hybrid.py.

Each round proposes ONE change (an environment-variable delta) on top of the
current best configuration, runs m5_hybrid.py, and keeps the change only if it
improves the primary objective: the best non-overlapping WAPE among the reported
models (sales_only / multi_feature / ensemble). Improvements accumulate across
rounds, so the final best config is the union of all winning changes.

Usage:
    python3 experiment_loop.py
Outputs:
    experiment_runs/round_XX_<label>.log   (stdout/stderr of each run)
    experiment_runs/round_XX_metrics.csv   (metrics.csv snapshot)
    experiment_runs/results.csv            (per-round objective + decision)
    experiment_runs/best_config.json       (accumulated best env overrides)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
EXP = BASE / "experiment_runs"
EXP.mkdir(parents=True, exist_ok=True)

SCRIPT = BASE / "m5_hybrid.py"


def parse_best_wape() -> tuple[float, pd.DataFrame]:
    """Return (best non_overlapping WAPE, full metrics dataframe)."""
    df = pd.read_csv(OUT / "metrics.csv")
    sub = df[df["evaluation"] == "non_overlapping"]
    return float(sub["WAPE"].min()), df


# (label, env delta applied on top of current best). Mutually exclusive choices
# simply override the same key, so the greedy step tests the new value.
EXPERIMENTS = [
    ("baseline", {}),
    ("cov_gate", {"M5_COV_FUSION": "gate"}),
    ("cov_proj", {"M5_COV_FUSION": "proj"}),
    ("lr_5e-4", {"M5_LR": "5e-4"}),
    ("lr_3e-3", {"M5_LR": "3e-3"}),
    ("dmodel_96", {"M5_D_MODEL": "96"}),
    ("dmodel_128", {"M5_D_MODEL": "128"}),
    ("temporal_3", {"M5_TEMPORAL_LAYERS": "3"}),
    ("temporal_4", {"M5_TEMPORAL_LAYERS": "4"}),
    ("heads_8", {"M5_N_HEADS": "8"}),
    ("dropout_0.15", {"M5_DROPOUT": "0.15"}),
    ("dropout_0.05", {"M5_DROPOUT": "0.05"}),
    ("patch_8_8", {"M5_PATCH_LEN": "8", "M5_PATCH_STRIDE": "8"}),
    ("patch_24_12", {"M5_PATCH_LEN": "24", "M5_PATCH_STRIDE": "12"}),
    ("wd_1e-4", {"M5_WEIGHT_DECAY": "1e-4"}),
    ("wd_1e-6", {"M5_WEIGHT_DECAY": "1e-6"}),
    ("batch_64", {"M5_BATCH_SIZE": "64"}),
    ("loss_huber", {"M5_LOSS": "huber"}),
    ("lr_schedule", {"M5_LR_SCHEDULE": "true"}),
    ("gradclip_5", {"M5_GRAD_CLIP": "5.0"}),
    ("item_embed", {"M5_ITEM_EMBED": "true"}),
    ("history_168", {"M5_HISTORY": "168"}),
    ("epochs_80", {"M5_EPOCHS": "80", "M5_PATIENCE": "12"}),
    ("ensemble", {"M5_ENSEMBLE": "true"}),
    ("combo_gate_lr", {"M5_COV_FUSION": "gate", "M5_LR": "5e-4"}),
]


def run_round(idx: int, label: str, config: dict) -> dict:
    log_path = EXP / f"round_{idx:02d}_{label}.log"
    env = {**os.environ, **config}
    t0 = __import__("time").time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(BASE),
        )
    elapsed = __import__("time").time() - t0
    record = {
        "round": idx,
        "label": label,
        "config": json.dumps(config, sort_keys=True),
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
    }
    if proc.returncode != 0:
        record["best_wape"] = float("nan")
        record["decision"] = "ERROR"
        record["note"] = "subprocess failed; see log"
        return record

    try:
        best_wape, df = parse_best_wape()
    except Exception as exc:  # noqa: BLE001
        record["best_wape"] = float("nan")
        record["decision"] = "ERROR"
        record["note"] = f"metrics parse failed: {exc}"
        return record

    # Snapshot metrics for this round.
    df.to_csv(EXP / f"round_{idx:02d}_{label}_metrics.csv", index=False)
    # Capture per-model non-overlapping WAPE for transparency.
    note_parts = []
    for _, r in df[df["evaluation"] == "non_overlapping"].iterrows():
        note_parts.append(f"{r['model']}={r['WAPE']:.4f}")
    record["best_wape"] = round(best_wape, 6)
    record["note"] = " | ".join(note_parts)
    record["decision"] = "pending"
    return record


def main() -> None:
    results: list[dict] = []
    best_config: dict = {}
    best_obj = float("inf")
    best_label = "baseline"

    for idx, (label, delta) in enumerate(EXPERIMENTS):
        config = {**best_config, **delta}
        print(f"\n=== Round {idx:02d}/{len(EXPERIMENTS)-1}: {label} ===", flush=True)
        print(f"config={config}", flush=True)
        rec = run_round(idx, label, config)

        if rec["exit_code"] == 0 and not np.isnan(rec["best_wape"]):
            # Keep the change only if it strictly improves the objective.
            if rec["best_wape"] < best_obj - 1e-4:
                best_config = config
                best_obj = rec["best_wape"]
                best_label = label
                rec["decision"] = "KEPT"
            else:
                rec["decision"] = "REJECTED"
        rec["best_config_label"] = best_label
        rec["best_obj"] = round(best_obj, 6)
        results.append(rec)
        pd.DataFrame(results).to_csv(EXP / "results.csv", index=False)
        with (EXP / "best_config.json").open("w", encoding="utf-8") as f:
            json.dump({"label": best_label, "config": best_config, "best_wape": best_obj}, f, indent=2)
        print(
            f"  -> best_wape={rec['best_wape']:.5f} decision={rec['decision']} "
            f"(running best={best_obj:.5f} via '{best_label}')",
            flush=True,
        )

    print("\n===== FINAL =====", flush=True)
    print(f"Best objective (min non-overlapping WAPE): {best_obj:.5f}", flush=True)
    print(f"Best config label: {best_label}", flush=True)
    print(f"Best config: {json.dumps(best_config, sort_keys=True)}", flush=True)
    pd.DataFrame(results).to_csv(EXP / "results.csv", index=False)


if __name__ == "__main__":
    main()
