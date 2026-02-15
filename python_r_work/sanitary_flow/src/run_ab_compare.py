#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RunResult:
    backend: str
    run_dir: str
    run_ts: str
    elapsed_sec: Optional[float]
    metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run A/B comparison for sanitary decomposition backends (xgb vs cat)."
    )
    p.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to run Extran_decomp_02082026.py",
    )
    p.add_argument(
        "--model_script",
        default=os.path.join(
            os.path.dirname(__file__), "Extran_decomp_02082026.py"
        ),
        help="Path to Extran_decomp_02082026.py",
    )
    p.add_argument("--data_dir", required=True, help="Input data folder")
    p.add_argument("--results_parent_dir", required=True, help="Output parent folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--usage_units", default="AUTO")
    p.add_argument("--fast", type=int, default=1, help="1 fast mode, 0 full mode")
    p.add_argument("--plots", type=int, default=0, help="1 plots on, 0 off")
    p.add_argument("--em_iters", type=int, default=None)
    p.add_argument("--early_stopping_rounds", type=int, default=None)
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--xgb_nthread", type=int, default=None)
    p.add_argument("--wee_hours_start", type=int, default=None)
    p.add_argument("--wee_hours_end", type=int, default=None)
    p.add_argument("--night_weight_ex", type=float, default=None)
    p.add_argument("--night_weight_sh", type=float, default=None)
    p.add_argument("--night_anchor_alpha", type=float, default=None)
    p.add_argument(
        "--backends",
        nargs="+",
        default=["xgb", "cat"],
        help="Backends to compare. Default: xgb cat",
    )
    return p.parse_args()


def list_run_dirs(results_parent_dir: str) -> set[str]:
    if not os.path.isdir(results_parent_dir):
        return set()
    out: set[str] = set()
    for name in os.listdir(results_parent_dir):
        path = os.path.join(results_parent_dir, name)
        if os.path.isdir(path):
            out.add(os.path.abspath(path))
    return out


def newest_dir(paths: List[str]) -> str:
    return sorted(paths, key=lambda p: os.path.getmtime(p))[-1]


def append_optional(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def run_one_backend(args: argparse.Namespace, backend: str) -> RunResult:
    before = list_run_dirs(args.results_parent_dir)
    cmd = [
        args.python_bin,
        args.model_script,
        "--data_dir",
        args.data_dir,
        "--results_parent_dir",
        args.results_parent_dir,
        "--model_backend",
        backend,
        "--seed",
        str(args.seed),
        "--usage_units",
        str(args.usage_units),
        "--fast",
        str(args.fast),
        "--plots",
        str(args.plots),
    ]
    append_optional(cmd, "--em_iters", args.em_iters)
    append_optional(cmd, "--early_stopping_rounds", args.early_stopping_rounds)
    append_optional(cmd, "--max_train_rows", args.max_train_rows)
    append_optional(cmd, "--xgb_nthread", args.xgb_nthread)
    append_optional(cmd, "--wee_hours_start", args.wee_hours_start)
    append_optional(cmd, "--wee_hours_end", args.wee_hours_end)
    append_optional(cmd, "--night_weight_ex", args.night_weight_ex)
    append_optional(cmd, "--night_weight_sh", args.night_weight_sh)
    append_optional(cmd, "--night_anchor_alpha", args.night_anchor_alpha)

    print(f"\n=== Running backend: {backend} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    after = list_run_dirs(args.results_parent_dir)
    created = sorted(after - before)
    if not created:
        raise RuntimeError(f"No new run directory found for backend {backend}")
    run_dir = newest_dir(created)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    run_ts = manifest["run_ts"]

    holdout_csv = os.path.join(run_dir, "tables", "dry_holdout_predictions.csv")
    df = pd.read_csv(holdout_csv)
    dt = pd.to_datetime(df["datetime"], utc=True)
    hours = dt.dt.hour.to_numpy(dtype=int)
    night_mask = (hours >= 0) & (hours <= 4)
    day_mask = ~night_mask

    def calc(sub: pd.DataFrame) -> Tuple[float, float, float]:
        y = sub["plant_flow"].to_numpy(dtype=float)
        yhat = sub["plant_recon"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y - yhat)))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        denom = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - (np.sum((y - yhat) ** 2) / denom)) if denom > 0 else float("nan")
        return mae, rmse, r2

    mae, rmse, r2 = calc(df)
    mae_n, rmse_n, r2_n = calc(df.loc[night_mask])
    mae_d, rmse_d, r2_d = calc(df.loc[day_mask])

    elapsed_sec = None
    log_path = os.path.join(args.results_parent_dir, "experiment_log.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        m = log["run_ts"].astype(str) == str(run_ts)
        if m.any():
            v = log.loc[m, "elapsed_sec"].tail(1).iloc[0]
            elapsed_sec = float(v) if pd.notna(v) else None

    metrics = {
        "overall_mae": mae,
        "overall_rmse": rmse,
        "overall_r2": r2,
        "night_mae": mae_n,
        "night_rmse": rmse_n,
        "night_r2": r2_n,
        "day_mae": mae_d,
        "day_rmse": rmse_d,
        "day_r2": r2_d,
    }
    return RunResult(
        backend=backend,
        run_dir=run_dir,
        run_ts=run_ts,
        elapsed_sec=elapsed_sec,
        metrics=metrics,
    )


def main() -> None:
    args = parse_args()
    backends = [b.strip().lower() for b in args.backends]
    allowed = {"xgb", "cat"}
    for b in backends:
        if b not in allowed:
            raise ValueError(f"Unsupported backend in --backends: {b}")

    results: List[RunResult] = []
    for b in backends:
        results.append(run_one_backend(args, b))

    rows: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "backend": r.backend,
            "run_ts": r.run_ts,
            "elapsed_sec": r.elapsed_sec,
            "run_dir": r.run_dir,
        }
        row.update(r.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.results_parent_dir, f"ab_compare_{ts}.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== A/B Summary ===")
    show_cols = [
        "backend",
        "elapsed_sec",
        "overall_mae",
        "overall_rmse",
        "overall_r2",
        "night_mae",
        "night_rmse",
        "night_r2",
        "day_mae",
        "day_rmse",
        "day_r2",
        "run_ts",
    ]
    print(df[show_cols].to_string(index=False))
    print(f"\nSaved comparison: {out_csv}")


if __name__ == "__main__":
    main()
