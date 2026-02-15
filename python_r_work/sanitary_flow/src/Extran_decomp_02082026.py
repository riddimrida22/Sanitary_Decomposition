# =============================================================================
# EXTRANEOUS FLOW DECOMPOSITION — DRY (LEARN SANITARY SHAPE) + WET BONUS (PYTHON)
# FAST / NON-HANGING — ROBUST IO + USAGE UNITS + RF AUTO-WIDEN + OPTIONAL PLOTS
#
# Inputs in --data_dir:
#   plant_flow_hourly.csv    : datetime, flow_mgd (MGD)
#   tidal_levels_hourly.csv  : datetime, tide_ft
#   water_usage_daily.csv    : date, usage column (MGD or MG daily volume)
#   rainfall_hourly.csv      : datetime, rain_in (incremental, hourly)
#
# Outputs:
#   <results_parent_dir>/<timestamp>/
#     tables/hourly_results_final.csv
#     tables/dry_holdout_predictions.csv
#     results.xlsx
#     models/model_bundle.pkl
#     plots/*.png   (if matplotlib installed)
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import math
import pickle
import argparse
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("xgboost is required. Install with: pip install xgboost") from e

try:
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
except Exception as e:
    raise ImportError("openpyxl is required. Install with: pip install openpyxl") from e

# Optional plots
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# =============================================================================
# Progress meter
# =============================================================================
class Progress:
    def __init__(self, total_steps: int):
        self.total = int(total_steps)
        self.done = 0

    def tick(self, label: str = ""):
        self.done += 1
        pct = 100.0 * self.done / max(1, self.total)
        bar_n = 30
        filled = int(bar_n * pct / 100.0)
        bar = "#" * filled + "-" * (bar_n - filled)
        sys.stdout.write(f"\r[{bar}] {pct:6.2f}% ({self.done}/{self.total}) {label}")
        sys.stdout.flush()
        if self.done >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    data_dir: str = r"/mnt/c/Users/bertr/Documents/Projects/Codex/python_r_work/sanitary_flow/data/Data"
    results_parent_dir: str = r"/mnt/c/Users/bertr/Documents/Projects/Codex/python_r_work/sanitary_flow/output"
    tz: str = "UTC"

    seed: int = 42

    # Usage units:
    #   "AUTO" -> infer MG vs MGD using plant/usage ratio
    #   "MGD"  -> usage column already MGD
    #   "MG"   -> usage column is MG daily volume
    usage_units: str = "AUTO"

    files: Dict[str, str] = None
    cols: Dict[str, str] = None

    # RF estimation with auto widen if pegging
    rf_bounds_primary: Tuple[float, float] = (0.85, 0.95)
    rf_bounds_secondary: Tuple[float, float] = (0.75, 1.05)
    rf_grid_step_primary: float = 0.005
    rf_grid_step_secondary: float = 0.01
    rf_min_dry_days_for_est: int = 60
    rf_pegging_threshold_seasons: int = 2  # if >= this many seasons pegged -> widen

    travel_time_same_day_fraction: float = 0.67

    dwf_filter: Dict[str, Any] = None

    outlier_sigma: float = 3.0
    gw_proxy_rolling_days: int = 14
    tidal_lag_hours: Optional[int] = None

    em_iters: int = 6
    train_fraction_days: float = 0.75

    early_stopping_rounds: int = 80
    valid_fraction_rows: float = 0.10
    min_valid_rows: int = 500
    max_train_rows: Optional[int] = 200000

    plots_enable: bool = True
    plots_dpi: int = 120
    plots_width: int = 1400
    plots_height: int = 800
    plots_sample_days_for_hourly_ts: int = 21
    plots_min_points: int = 200

    xgb_ex_params: Dict[str, Any] = None
    xgb_sh_params: Dict[str, Any] = None


def make_default_config() -> Config:
    cfg = Config()
    cfg.files = dict(
        plant_flow="plant_flow_hourly.csv",
        tide="tidal_levels_hourly.csv",
        usage="water_usage_daily.csv",
        rainfall="rainfall_hourly.csv",
    )
    cfg.cols = dict(
        plant_dt="datetime", plant_val="flow_mgd",
        tide_dt="datetime", tide_val="tide_ft",
        usage_dt="date", usage_val="usage_mgd",
        rain_dt="datetime", rain_val="rain_in",
    )
    cfg.dwf_filter = dict(
        trace_threshold_in=0.02,
        event_min_in=0.05,
        antecedent_hours_try=[24, 12, 6],
        min_dry_days_target=30,
        response_lag_lo_hr=12,
        response_lag_hi_hr=24,
        baseline_roll_hr=168,
        impact_threshold_mgd=8
    )
    cfg.xgb_ex_params = dict(
        max_depth=6,
        eta=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        seed=cfg.seed,
    )
    cfg.xgb_sh_params = dict(
        max_depth=5,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        seed=cfg.seed,
    )
    return cfg


# =============================================================================
# Helpers
# =============================================================================
def season_of_date(d: Date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def odd_k(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_to_datetime(
    s,
    *,
    utc: bool = False,
    as_date: bool = False,
    dayfirst: Optional[bool] = None,
    allow_excel_serial: bool = True,
    fmt: Optional[str] = None,
) -> pd.Series:
    if isinstance(s, (pd.Series, pd.Index)):
        ser = pd.Series(s)
    else:
        ser = pd.Series([s])

    txt = ser.astype(str).str.strip()
    txt = txt.replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan})

    out_num = pd.Series(pd.NaT, index=ser.index)
    if allow_excel_serial:
        num = pd.to_numeric(txt, errors="coerce")
        out_num = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce", utc=utc)

    if fmt is not None:
        out_txt = pd.to_datetime(txt, errors="coerce", utc=utc, format=fmt)
    else:
        try:
            out_txt = pd.to_datetime(txt, errors="coerce", utc=utc, format="mixed", dayfirst=dayfirst)
        except TypeError:
            out_txt = pd.to_datetime(txt, errors="coerce", utc=utc)

    out = out_num.where(out_num.notna(), out_txt)

    if out.isna().any():
        bad = int(out.isna().sum())
        examples = txt[out.isna()].head(10).tolist()
        raise ValueError(f"Datetime parsing failed for {bad} rows. Examples: {examples}")

    if as_date:
        return out.dt.date

    return out


def parse_dt_series(s: pd.Series) -> pd.Series:
    return safe_to_datetime(s, utc=True)


def fill_numeric_gaps(arr: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(pd.Series(arr), errors="coerce")
    if x.notna().sum() == 0:
        return np.zeros(len(x), dtype=float)
    x = x.interpolate(limit_direction="both")
    med = float(x.median()) if np.isfinite(x.median()) else 0.0
    x = x.fillna(med).fillna(0.0)
    return x.to_numpy(dtype=float)


def impute_feature_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = fill_numeric_gaps(df[c].to_numpy())
    return df


def scale_to_daily_mean(
    values: pd.Series,
    groups: pd.Series,
    target_daily_mean: pd.Series,
    *,
    min_denominator: float = 1e-9,
) -> pd.Series:
    """
    Scale hourly values so each day's mean matches target_daily_mean for that day.
    """
    values_num = pd.to_numeric(values, errors="coerce").fillna(0.0)
    target_num = pd.to_numeric(target_daily_mean, errors="coerce")

    day_mean = values_num.groupby(groups).transform("mean")
    ratio = target_num / day_mean.clip(lower=min_denominator)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return np.maximum(0.0, values_num.to_numpy(dtype=float) * ratio.to_numpy(dtype=float))


def detect_usage_value_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    candidates = ["usage_mgd", "usage_mg", "daily_usage", "usage", "water_usage", "flow", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: pick column with highest finite numeric fraction, excluding date-ish cols
    exclude = set(["date", "Date", "DATE", "datetime", "Datetime", "timestamp", "time"])
    best_col, best_score = None, -1.0
    for c in df.columns:
        if c in exclude:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        score = float(np.isfinite(x).mean())
        if score > best_score:
            best_score, best_col = score, c
    if best_col is None:
        raise KeyError(f"Could not detect usage value column. Columns: {list(df.columns)}")
    return best_col


def infer_usage_units(plant_daily_mgd: pd.Series, usage_raw: pd.Series) -> str:
    # Heuristic:
    # - If usage is MG daily volume, usage numbers are often ~ (MGD*24) scale,
    #   so plant/usage ratio is small (~0.02–0.2 typically).
    # - If usage is MGD, ratio is around ~0.2–5.
    p = pd.to_numeric(plant_daily_mgd, errors="coerce")
    u = pd.to_numeric(usage_raw, errors="coerce")
    ok = np.isfinite(p) & np.isfinite(u) & (u > 0)
    if ok.sum() < 30:
        return "MGD"
    ratio = float(np.nanmedian(p[ok] / u[ok]))
    # conservative thresholds
    if ratio < 0.2:
        return "MG"
    return "MGD"


# =============================================================================
# Output dirs
# =============================================================================
def init_output_dirs(cfg: Config) -> Dict[str, str]:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = os.path.join(os.path.abspath(cfg.results_parent_dir), run_ts)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))
    ensure_dir(os.path.join(out_dir, "tables"))
    ensure_dir(os.path.join(out_dir, "models"))

    manifest = {
        "run_ts": run_ts,
        "seed": cfg.seed,
        "data_dir": os.path.abspath(cfg.data_dir),
        "results_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"out_dir": out_dir, "run_ts": run_ts}


# =============================================================================
# 4) Load + align
# =============================================================================
def load_and_align(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    print("=" * 70)
    print("4) LOAD + ALIGN DATA")
    print("=" * 70)

    data_dir = cfg.data_dir
    cc = cfg.cols

    # ---- Plant
    plant = pd.read_csv(os.path.join(data_dir, cfg.files["plant_flow"]))
    plant = plant.rename(columns={cc["plant_dt"]: "datetime_raw", cc["plant_val"]: "plant_flow"})
    plant["datetime"] = parse_dt_series(plant["datetime_raw"])
    plant = plant.drop(columns=["datetime_raw"]).sort_values("datetime").drop_duplicates("datetime")

    # ---- Tide
    tide = pd.read_csv(os.path.join(data_dir, cfg.files["tide"]))
    tide = tide.rename(columns={cc["tide_dt"]: "datetime_raw", cc["tide_val"]: "tide"})
    tide["datetime"] = parse_dt_series(tide["datetime_raw"])
    tide = tide.drop(columns=["datetime_raw"]).sort_values("datetime").drop_duplicates("datetime")

    # Intersection coverage
    t0 = max(plant["datetime"].min(), tide["datetime"].min())
    t1 = min(plant["datetime"].max(), tide["datetime"].max())

    full_hours = pd.date_range(t0, t1, freq="h", tz="UTC")
    full_hours_df = pd.DataFrame({"datetime": full_hours})

    plant = full_hours_df.merge(plant, on="datetime", how="left")
    tide = full_hours_df.merge(tide, on="datetime", how="left")

    plant["plant_flow"] = fill_numeric_gaps(plant["plant_flow"].to_numpy())
    tide["tide"] = fill_numeric_gaps(tide["tide"].to_numpy())

    # ---- Rain
    rain = pd.read_csv(os.path.join(data_dir, cfg.files["rainfall"]))
    rain = rain.rename(columns={cc["rain_dt"]: "datetime_raw", cc["rain_val"]: "rain_in"})
    rain["datetime"] = parse_dt_series(rain["datetime_raw"])
    rain = rain.drop(columns=["datetime_raw"])
    rain["datetime"] = rain["datetime"].dt.floor("h")
    rain["rain_in"] = pd.to_numeric(rain["rain_in"], errors="coerce").fillna(0.0)
    rain_hr = rain.groupby("datetime", as_index=False)["rain_in"].sum().rename(columns={"rain_in": "rain_incr"})
    rain_hr = full_hours_df.merge(rain_hr, on="datetime", how="left")
    rain_hr["rain_incr"] = rain_hr["rain_incr"].fillna(0.0).clip(lower=0.0)

    # ---- Usage (robust)
    usage = pd.read_csv(os.path.join(data_dir, cfg.files["usage"]))
    # date col
    cand_date = [cc["usage_dt"], "date", "Date", "DATE", "day", "Day", "sample_date", "SampleDate"]
    date_col = next((c for c in cand_date if c in usage.columns), None)
    if date_col is None:
        raise KeyError(f"water_usage_daily.csv: could not find date column. Found: {list(usage.columns)}")

    val_col = detect_usage_value_col(usage, cc["usage_val"])
    usage = usage.rename(columns={date_col: "date", val_col: "usage_raw"})
    usage["date"] = safe_to_datetime(usage["date"], as_date=True, utc=False)
    usage["usage_raw"] = pd.to_numeric(usage["usage_raw"], errors="coerce")
    usage = usage.dropna(subset=["usage_raw"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    # Merge hourly master
    hourly = plant.merge(tide, on="datetime", how="inner").merge(rain_hr, on="datetime", how="left")
    hourly["rain_incr"] = hourly["rain_incr"].fillna(0.0)
    hourly["date"] = hourly["datetime"].dt.date
    hourly["season"] = hourly["date"].apply(season_of_date)

    # Infer/Apply usage units
    daily_plant = hourly.groupby("date", as_index=False)["plant_flow"].mean().rename(columns={"plant_flow": "plant_flow_daily"})
    usage_units = cfg.usage_units.upper().strip()
    if usage_units == "AUTO":
        usage_units = infer_usage_units(daily_plant["plant_flow_daily"], usage["usage_raw"])
    if usage_units not in ("MGD", "MG"):
        raise ValueError("usage_units must be AUTO, MGD, or MG")

    if usage_units == "MG":
        usage["usage_mg"] = usage["usage_raw"]
        usage["usage_mgd"] = usage["usage_raw"] / 1.0
    else:
        usage["usage_mgd"] = usage["usage_raw"]
        usage["usage_mg"] = usage["usage_raw"] * 1.0

    usage = usage.drop(columns=["usage_raw"])

    print(f"  Hourly rows: {len(hourly):,} ({hourly['datetime'].iloc[0]} to {hourly['datetime'].iloc[-1]})")
    print(f"  Daily usage rows: {len(usage):,} ({usage['date'].iloc[0]} to {usage['date'].iloc[-1]})")
    print(f"  Usage units detected/used: {cfg.usage_units} -> {usage_units} (stored as MGD in pipeline)")
    return hourly.reset_index(drop=True), usage, usage_units


# =============================================================================
# 5) Dry/Wet flagging
# =============================================================================
def flag_dry_wet(hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("5) FLAG DRY/WET HOURS")
    print("=" * 70)

    f = cfg.dwf_filter
    trace = float(f["trace_threshold_in"])
    event_min = float(f["event_min_in"])
    tries = list(f["antecedent_hours_try"])
    target = int(f["min_dry_days_target"])

    lag_lo = int(f["response_lag_lo_hr"])
    lag_hi = int(f["response_lag_hi_hr"])
    base_k = odd_k(int(f["baseline_roll_hr"]))
    impact_thr = float(f["impact_threshold_mgd"])

    dt = hourly.sort_values("datetime").reset_index(drop=True).copy()

    dt["rain_clean"] = np.where(dt["rain_incr"] < trace, 0.0, dt["rain_incr"])

    dt["plant_base"] = pd.Series(dt["plant_flow"]).rolling(window=base_k, center=True, min_periods=1).median()
    dt["plant_base"] = fill_numeric_gaps(dt["plant_base"].to_numpy())
    dt["plant_anom"] = dt["plant_flow"] - dt["plant_base"]

    anom = dt["plant_anom"].to_numpy(dtype=float)
    fwd_max = np.zeros(len(anom), dtype=float)
    for L in range(lag_lo, lag_hi + 1):
        shifted = np.roll(anom, -L)
        shifted[-L:] = np.nan
        fwd_max = np.maximum(fwd_max, np.nan_to_num(shifted, nan=0.0))
    dt["anom_fwd_max"] = fwd_max

    dt["is_effective_rain"] = (dt["rain_clean"] >= event_min) & (dt["anom_fwd_max"] >= impact_thr)
    dt["is_effective_rain"] = dt["is_effective_rain"].fillna(False)

    chosen_ante = tries[-1]
    chosen_dry_days = 0

    for ante in tries:
        eff = dt["is_effective_rain"].astype(int).to_numpy()
        eff_roll = pd.Series(eff).rolling(window=int(ante), min_periods=1).sum().to_numpy()
        dt["is_wet"] = eff_roll > 0
        dt["is_dry"] = ~dt["is_wet"]

        dry_days_n = int(dt.groupby("date")["is_dry"].all().sum())
        print(f"  antecedent_hours={ante} -> fully dry days={dry_days_n}")
        chosen_ante, chosen_dry_days = ante, dry_days_n
        if dry_days_n >= target:
            break

    print(f"\n  Using antecedent_hours = {chosen_ante}")
    print(f"  Fully dry days: {chosen_dry_days}")
    print(f"  Dry hours: {int(dt['is_dry'].sum()):,} ({100.0 * dt['is_dry'].mean():.1f}%)")
    print(f"  Wet hours: {int(dt['is_wet'].sum()):,} ({100.0 * dt['is_wet'].mean():.1f}%)")
    print(f"  Effective rain pulses: {int(dt['is_effective_rain'].sum()):,}")

    dt = dt.drop(columns=["rain_clean", "plant_base", "plant_anom", "anom_fwd_max"], errors="ignore")
    return dt


# =============================================================================
# 5B) Return factor estimation with pegging detection
# =============================================================================
def _fit_rf_for_bounds(hourly: pd.DataFrame, usage: pd.DataFrame, bounds: Tuple[float, float], step: float, cfg: Config) -> pd.DataFrame:
    rf_lo, rf_hi = bounds
    f_same = float(cfg.travel_time_same_day_fraction)
    f_prev = 1.0 - f_same

    daily_plant = hourly.groupby("date").agg(
        plant_flow_daily=("plant_flow", "mean"),
        is_fully_dry=("is_dry", "all"),
    ).reset_index()
    daily_plant["season"] = daily_plant["date"].apply(season_of_date)

    u = usage.sort_values("date").reset_index(drop=True).copy()
    u["usage_adj"] = f_same * u["usage_mgd"] + f_prev * u["usage_mgd"].shift(1)

    daily = daily_plant.merge(u[["date", "usage_mg", "usage_mgd", "usage_adj"]], on="date", how="left")
    daily = daily[(daily["is_fully_dry"]) & np.isfinite(daily["usage_adj"]) & np.isfinite(daily["plant_flow_daily"])]

    rf_grid = np.arange(rf_lo, rf_hi + 1e-12, float(step))

    def best_rf_for_season(seas: str) -> Tuple[float, float, int]:
        d = daily[daily["season"] == seas].copy()
        n_days = int(len(d))
        if n_days < 30:
            return (float((rf_lo + rf_hi) / 2.0), float("inf"), n_days)

        usage_adj = d["usage_adj"].to_numpy(dtype=float)
        plant_daily = d["plant_flow_daily"].to_numpy(dtype=float)

        best_rf = float((rf_lo + rf_hi) / 2.0)
        best_score = float("inf")

        for rf in rf_grid:
            ex_raw = plant_daily - rf * usage_adj
            neg_pen = np.mean(np.maximum(0.0, -ex_raw))
            cor_pen = abs(np.corrcoef(ex_raw, usage_adj)[0, 1]) if np.std(ex_raw) > 0 and np.std(usage_adj) > 0 else 0.0
            sd_pen = np.std(ex_raw) / (np.mean(plant_daily) + 1e-9)
            score = cor_pen + 5.0 * neg_pen / (np.std(plant_daily) + 1e-9) + 0.25 * sd_pen
            if np.isfinite(score) and score < best_score:
                best_score, best_rf = float(score), float(rf)

        return best_rf, best_score, n_days

    rows = []
    for seas in ["DJF", "MAM", "JJA", "SON"]:
        rf, score, n_days = best_rf_for_season(seas)
        rows.append((seas, rf, score, n_days))

    res = pd.DataFrame(rows, columns=["season", "rf", "score", "n_days"])

    # DJF >= max(other seasons) like your earlier constraint
    other_max = float(res.loc[res["season"] != "DJF", "rf"].max())
    res.loc[res["season"] == "DJF", "rf"] = np.maximum(res.loc[res["season"] == "DJF", "rf"], other_max)
    res["rf"] = res["rf"].clip(rf_lo, rf_hi)

    return res


def estimate_return_factor_by_season(hourly: pd.DataFrame, usage: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("5B) ESTIMATE RETURN FACTOR BY SEASON")
    print("=" * 70)

    # Sanity check ratio (like your console)
    f_same = float(cfg.travel_time_same_day_fraction)
    f_prev = 1.0 - f_same
    daily_plant = hourly.groupby("date", as_index=False)["plant_flow"].mean().rename(columns={"plant_flow": "plant_flow_daily"})
    u = usage.sort_values("date").copy()
    u["usage_adj"] = f_same * u["usage_mgd"] + f_prev * u["usage_mgd"].shift(1)
    chk = daily_plant.merge(u[["date", "usage_adj"]], on="date", how="left")
    ok = np.isfinite(chk["plant_flow_daily"]) & np.isfinite(chk["usage_adj"]) & (chk["usage_adj"] > 0)
    ratio = float(np.nanmedian(chk.loc[ok, "plant_flow_daily"] / chk.loc[ok, "usage_adj"])) if ok.sum() else float("nan")
    print(f"  Usage sanity check: usage_units {cfg.usage_units} ok; median plant/usage_adj ratio={ratio:.2f}")

    # Need enough dry days to estimate
    dry_days = hourly.groupby("date")["is_dry"].all()
    if int(dry_days.sum()) < cfg.rf_min_dry_days_for_est:
        mid = float(sum(cfg.rf_bounds_primary) / 2.0)
        print("  WARNING: Few dry days available; using mid primary bound.")
        return pd.DataFrame({"season": ["DJF", "MAM", "JJA", "SON"], "rf": [mid, mid, mid, mid]})

    print(f"  RF bounds try 1/2: [{cfg.rf_bounds_primary[0]:.3f}, {cfg.rf_bounds_primary[1]:.3f}]")
    res1 = _fit_rf_for_bounds(hourly, usage, cfg.rf_bounds_primary, cfg.rf_grid_step_primary, cfg)
    print(res1[["season", "rf", "score", "n_days"]].to_string(index=False))

    tol = cfg.rf_grid_step_primary + 1e-12
    pegged = int(((abs(res1["rf"] - cfg.rf_bounds_primary[0]) <= tol) | (abs(res1["rf"] - cfg.rf_bounds_primary[1]) <= tol)).sum())

    if pegged >= int(cfg.rf_pegging_threshold_seasons):
        print("  RF pegging detected -> widening bounds + refitting...")
        print(f"  RF bounds try 2/2: [{cfg.rf_bounds_secondary[0]:.3f}, {cfg.rf_bounds_secondary[1]:.3f}]")
        res2 = _fit_rf_for_bounds(hourly, usage, cfg.rf_bounds_secondary, cfg.rf_grid_step_secondary, cfg)
        print(res2[["season", "rf", "score", "n_days"]].to_string(index=False))
        final = res2[["season", "rf"]].copy()
    else:
        final = res1[["season", "rf"]].copy()

    print("\n  Final seasonal return factors:")
    print(final.to_string(index=False))
    return final


# =============================================================================
# 6) Daily mass balance
# =============================================================================
def compute_daily_mass_balance(hourly: pd.DataFrame, usage: pd.DataFrame, rf_table: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 70)
    print("6) STAGE 1 — DAILY MASS BALANCE")
    print("=" * 70)

    f_same = float(cfg.travel_time_same_day_fraction)
    f_prev = 1.0 - f_same

    dt_daily_plant = hourly.groupby("date").agg(plant_flow_daily=("plant_flow", "mean")).reset_index()
    dt_daily_flag = hourly.groupby("date").agg(is_fully_dry=("is_dry", "all")).reset_index()
    dt_daily_flag["season"] = dt_daily_flag["date"].apply(season_of_date)

    u = usage.copy().sort_values("date").reset_index(drop=True)
    u["usage_adj"] = f_same * u["usage_mgd"] + f_prev * u["usage_mgd"].shift(1)

    rf_map = rf_table.set_index("season")["rf"].to_dict()
    dt_daily_flag["rf"] = dt_daily_flag["season"].map(rf_map)
    dt_daily_flag["rf"] = dt_daily_flag["rf"].fillna(float(sum(cfg.rf_bounds_primary) / 2.0))

    daily = dt_daily_plant.merge(u[["date", "usage_mg", "usage_mgd", "usage_adj"]], on="date", how="inner")
    daily = daily.merge(dt_daily_flag[["date", "is_fully_dry", "season", "rf"]], on="date", how="left")
    daily["is_fully_dry"] = daily["is_fully_dry"].fillna(False)

    daily = daily[np.isfinite(daily["usage_adj"]) & np.isfinite(daily["rf"])].copy()
    daily["sanitary_daily"] = daily["rf"] * daily["usage_adj"]
    daily["extraneous_daily"] = daily["plant_flow_daily"] - daily["sanitary_daily"]
    daily["extraneous_daily_nonneg"] = np.maximum(0.0, daily["extraneous_daily"])

    dry = daily[daily["is_fully_dry"]].copy()

    if len(dry) >= 20:
        dry["month"] = pd.to_datetime(dry["date"].astype(str)).dt.month
        stats = dry.groupby("month")["extraneous_daily_nonneg"].agg(["mean", "std"]).reset_index()
        dry = dry.merge(stats, on="month", how="left")
        keep = (
            np.abs(dry["extraneous_daily_nonneg"] - dry["mean"])
            <= cfg.outlier_sigma * dry["std"].replace(0, np.nan)
        ).fillna(True)
        dry = dry[keep].copy().drop(columns=["month", "mean", "std"], errors="ignore")

    print(f"  Fully dry days used: {len(dry)}")
    return daily, dry


# =============================================================================
# 7) GW proxy
# =============================================================================
def compute_gw_proxy(hourly: pd.DataFrame, daily_all: pd.DataFrame, daily_dry: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 70)
    print("7) GW PROXY")
    print("=" * 70)

    k = odd_k(int(cfg.gw_proxy_rolling_days))
    dt = hourly.copy()

    if len(daily_dry) >= 2:
        print("  Using dry-day GW proxy.")
        src = daily_dry[["date", "extraneous_daily_nonneg"]].rename(columns={"extraneous_daily_nonneg": "extraneous_daily"}).copy()
    else:
        print("  WARNING: Too few dry days; using all-days GW proxy fallback.")
        src = daily_all[["date", "extraneous_daily"]].copy()
        src["extraneous_daily"] = np.maximum(0.0, src["extraneous_daily"])

    src = src.sort_values("date").reset_index(drop=True)
    src["gw_proxy_daily"] = pd.Series(src["extraneous_daily"]).rolling(window=k, center=True, min_periods=1).median()
    src["gw_proxy_daily"] = fill_numeric_gaps(src["gw_proxy_daily"].to_numpy())

    full_dates = pd.DataFrame({
        "date": pd.date_range(pd.to_datetime(dt["date"].min()), pd.to_datetime(dt["date"].max()), freq="D").date
    })
    gw_full = full_dates.merge(src[["date", "gw_proxy_daily"]], on="date", how="left")
    gw_full["gw_proxy_daily"] = fill_numeric_gaps(gw_full["gw_proxy_daily"].to_numpy())

    noon_times = pd.to_datetime(gw_full["date"].astype(str) + " 12:00:00", utc=True)
    x = noon_times.astype("int64").to_numpy(dtype=float)
    y = gw_full["gw_proxy_daily"].to_numpy(dtype=float)
    xq = dt["datetime"].astype("int64").to_numpy(dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    dt["gw_proxy"] = np.interp(xq, x, y, left=y[0], right=y[-1])
    dt["gw_proxy"] = np.where(np.isfinite(dt["gw_proxy"]), dt["gw_proxy"], 0.0)
    dt["gw_rate"] = np.concatenate([[0.0], np.diff(dt["gw_proxy"].to_numpy(dtype=float))])
    dt["gw_rate"] = np.where(np.isfinite(dt["gw_rate"]), dt["gw_rate"], 0.0)

    return dt, gw_full


# =============================================================================
# 8) Tidal lag estimate
# =============================================================================
def estimate_tidal_lag(hourly: pd.DataFrame, max_lag: int = 24) -> int:
    dry = hourly[hourly["is_dry"]].copy()
    if len(dry) < 500:
        return 6

    x = dry["tide"].to_numpy(dtype=float)
    y = dry["plant_flow"].to_numpy(dtype=float)

    x = x - pd.Series(x).rolling(168, center=True, min_periods=1).mean().to_numpy()
    y = y - pd.Series(y).rolling(168, center=True, min_periods=1).mean().to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 200:
        return 6

    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for L in lags:
        if L >= 0:
            xv = x[:-L or None]
            yv = y[L:]
        else:
            L2 = abs(L)
            xv = x[L2:]
            yv = y[:-L2]

        if len(xv) < 3 or np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(xv, yv)[0, 1])

    corr_arr = np.asarray(corrs, dtype=float)
    if not np.isfinite(corr_arr).any():
        return 6

    best = int(lags[int(np.nanargmax(np.abs(corr_arr)))])
    return abs(best)


# =============================================================================
# 9) Feature builders
# =============================================================================
def build_ex_features(df: pd.DataFrame, lag_opt: int) -> List[str]:
    d = df

    def shift(col: str, n: int) -> np.ndarray:
        arr = d[col].to_numpy(dtype=float)
        out = np.roll(arr, n)
        out[:n] = np.nan
        return out

    d["tide_lag_opt"] = shift("tide", lag_opt)
    d["tide_lag_1"] = shift("tide", 1)
    d["tide_lag_2"] = shift("tide", 2)
    d["tide_lag_3"] = shift("tide", 3)
    d["tide_lag_6"] = shift("tide", 6)
    d["tide_lag_12"] = shift("tide", 12)

    tide = d["tide"].to_numpy(dtype=float)
    tide_prev = np.roll(tide, 1); tide_prev[0] = np.nan
    d["tide_rate"] = tide - tide_prev

    d["tide_range_12h"] = pd.Series(tide).rolling(12, min_periods=1).apply(lambda z: float(np.nanmax(z) - np.nanmin(z))).to_numpy()
    d["tide_range_24h"] = pd.Series(tide).rolling(24, min_periods=1).apply(lambda z: float(np.nanmax(z) - np.nanmin(z))).to_numpy()
    d["tide_mean_24h"] = pd.Series(tide).rolling(24, min_periods=1).mean().to_numpy()
    d["tide_mean_48h"] = pd.Series(tide).rolling(48, min_periods=1).mean().to_numpy()

    ts = d["datetime"]
    month = ts.dt.month.astype(float)
    dow = ((ts.dt.dayofweek + 1) / 7.0).astype(float)

    d["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    d["dow"] = dow

    return [
        "tide", "tide_lag_opt", "tide_lag_1", "tide_lag_2", "tide_lag_3", "tide_lag_6", "tide_lag_12",
        "tide_rate", "tide_range_12h", "tide_range_24h", "tide_mean_24h", "tide_mean_48h",
        "gw_proxy", "gw_rate", "month_sin", "month_cos", "dow"
    ]


def build_shape_features(df: pd.DataFrame) -> List[str]:
    ts = df["datetime"]
    hr = ts.dt.hour.astype(float)
    month = ts.dt.month.astype(float)
    dow = ((ts.dt.dayofweek + 1) / 7.0).astype(float)

    df["hour_sin"] = np.sin(2.0 * np.pi * hr / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hr / 24.0)
    df["month_sin2"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos2"] = np.cos(2.0 * np.pi * month / 12.0)
    df["dow2"] = dow

    return ["hour_sin", "hour_cos", "month_sin2", "month_cos2", "dow2"]


# =============================================================================
# 10) Split dry days
# =============================================================================
def split_days(dates: List[Date], train_frac: float) -> Dict[str, List[Date]]:
    d = sorted(set(dates))
    n = len(d)
    n_train = max(1, int(math.floor(n * train_frac)))
    return {"train": d[:n_train], "test": d[n_train:]}


# =============================================================================
# 11) XGB train helper
# =============================================================================
def xgb_train_es(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    nrounds_max: int,
    early_stopping_rounds: int,
    valid_fraction_rows: float,
    min_valid_rows: int,
    max_train_rows: Optional[int],
) -> xgb.Booster:
    n = X.shape[0]
    if max_train_rows is not None and n > max_train_rows:
        start = n - max_train_rows
        X = X[start:, :]
        y = y[start:]
        n = X.shape[0]

    nva = max(min_valid_rows, int(math.floor(n * valid_fraction_rows)))
    nva = min(nva, max(50, n - 50))

    tr_idx = np.arange(0, n - nva)
    va_idx = np.arange(n - nva, n)

    dtr = xgb.DMatrix(X[tr_idx, :], label=y[tr_idx])
    dva = xgb.DMatrix(X[va_idx, :], label=y[va_idx])

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(nrounds_max),
        evals=[(dtr, "train"), (dva, "eval")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False
    )
    return booster


# =============================================================================
# 12) Fit dry decomposition
# =============================================================================
def fit_dry_decomposition_models(hourly: pd.DataFrame, daily_all: pd.DataFrame, daily_dry: pd.DataFrame, lag_opt: int, cfg: Config) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("12) FIT DRY DECOMPOSITION (ALTERNATING)")
    print("=" * 70)

    if len(daily_dry) < 10:
        raise RuntimeError("Not enough fully dry days to train (need ~10+).")

    dry_days = daily_dry["date"].tolist()
    dt = hourly[(hourly["is_dry"]) & (hourly["date"].isin(dry_days))].copy()

    dt = dt.merge(daily_all[["date", "sanitary_daily", "usage_mg", "usage_mgd", "rf"]], on="date", how="left")
    dt = dt.merge(
        daily_dry[["date", "extraneous_daily_nonneg"]].rename(columns={"extraneous_daily_nonneg": "extraneous_daily"}),
        on="date", how="left"
    )
    dt = dt.sort_values("datetime").reset_index(drop=True)

    ex_cols = build_ex_features(dt, lag_opt)
    sh_cols = build_shape_features(dt)
    feat_cols = list(dict.fromkeys(ex_cols + sh_cols))
    dt = impute_feature_cols(dt, feat_cols)

    split = split_days(dry_days, cfg.train_fraction_days)
    dt_train = dt[dt["date"].isin(split["train"])].copy()
    dt_test = dt[dt["date"].isin(split["test"])].copy()

    dt_train["s_shape"] = 1.0
    dt_test["s_shape"] = 1.0

    ex_model = None
    sh_model = None

    def predict_shape_factor(model: xgb.Booster, dframe: pd.DataFrame) -> np.ndarray:
        raw = model.predict(xgb.DMatrix(dframe[sh_cols].to_numpy(dtype=float)))
        raw = np.maximum(1e-9, raw)
        tmp = pd.DataFrame({"date": dframe["date"].to_numpy(), "raw": raw})
        day_mean = tmp.groupby("date")["raw"].transform("mean").to_numpy()
        return raw / np.maximum(1e-9, day_mean)

    for it in range(1, cfg.em_iters + 1):
        print(f"  Iteration {it} / {cfg.em_iters} ({100.0 * it / cfg.em_iters:.0f}%)")

        dt_train["sanitary_hourly"] = np.maximum(0.0, dt_train["sanitary_daily"].to_numpy(dtype=float) * dt_train["s_shape"].to_numpy(dtype=float))
        dt_train["ex_target_raw"] = np.maximum(0.0, dt_train["plant_flow"].to_numpy(dtype=float) - dt_train["sanitary_hourly"].to_numpy(dtype=float))

        ex_raw_mean = dt_train.groupby("date")["ex_target_raw"].transform("mean").to_numpy(dtype=float)
        scale = np.where(ex_raw_mean > 0, dt_train["extraneous_daily"].to_numpy(dtype=float) / ex_raw_mean, 0.0)
        dt_train["ex_target"] = np.maximum(0.0, dt_train["ex_target_raw"].to_numpy(dtype=float) * scale)

        Xex = dt_train[ex_cols].to_numpy(dtype=float)
        yex = dt_train["ex_target"].to_numpy(dtype=float)

        ex_model = xgb_train_es(
            X=Xex, y=yex,
            params=cfg.xgb_ex_params,
            nrounds_max=4000,
            early_stopping_rounds=cfg.early_stopping_rounds,
            valid_fraction_rows=cfg.valid_fraction_rows,
            min_valid_rows=cfg.min_valid_rows,
            max_train_rows=cfg.max_train_rows
        )

        ex_pred_raw = np.maximum(0.0, ex_model.predict(xgb.DMatrix(Xex)))
        dt_train["ex_pred_raw"] = ex_pred_raw
        ex_pred_mean = dt_train.groupby("date")["ex_pred_raw"].transform("mean").to_numpy(dtype=float)
        ex_pred = np.where(ex_pred_mean > 0, ex_pred_raw * (dt_train["extraneous_daily"].to_numpy(dtype=float) / ex_pred_mean), 0.0)
        dt_train["ex_pred"] = np.maximum(0.0, ex_pred)

        san_resid = np.maximum(0.0, dt_train["plant_flow"].to_numpy(dtype=float) - dt_train["ex_pred"].to_numpy(dtype=float))
        dt_train["san_resid"] = san_resid
        san_mean = dt_train.groupby("date")["san_resid"].transform("mean").to_numpy(dtype=float)
        dt_train["san_shape_obs"] = np.where(san_mean > 0, san_resid / san_mean, 1.0)

        Xsh = dt_train[sh_cols].to_numpy(dtype=float)
        ysh = dt_train["san_shape_obs"].to_numpy(dtype=float)

        sh_model = xgb_train_es(
            X=Xsh, y=ysh,
            params=cfg.xgb_sh_params,
            nrounds_max=3000,
            early_stopping_rounds=cfg.early_stopping_rounds,
            valid_fraction_rows=cfg.valid_fraction_rows,
            min_valid_rows=cfg.min_valid_rows,
            max_train_rows=cfg.max_train_rows
        )

        dt_train["s_shape"] = predict_shape_factor(sh_model, dt_train)

        sanitary_pred = np.maximum(0.0, dt_train["sanitary_daily"].to_numpy(dtype=float) * dt_train["s_shape"].to_numpy(dtype=float))
        plant_recon = sanitary_pred + dt_train["ex_pred"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((dt_train["plant_flow"].to_numpy(dtype=float) - plant_recon) ** 2)))
        print(f"    Train recon RMSE (plant): {rmse:.4f} MGD")

    # Holdout test
    dt_test["s_shape"] = predict_shape_factor(sh_model, dt_test)
    dt_test["sanitary_pred"] = np.maximum(0.0, dt_test["sanitary_daily"].to_numpy(dtype=float) * dt_test["s_shape"].to_numpy(dtype=float))

    Xex_t = dt_test[ex_cols].to_numpy(dtype=float)
    ex_raw_t = np.maximum(0.0, ex_model.predict(xgb.DMatrix(Xex_t)))
    dt_test["ex_pred_raw"] = ex_raw_t
    ex_pred_mean_t = dt_test.groupby("date")["ex_pred_raw"].transform("mean").to_numpy(dtype=float)
    dt_test["extraneous_pred"] = np.maximum(0.0, np.where(ex_pred_mean_t > 0, ex_raw_t * (dt_test["extraneous_daily"].to_numpy(dtype=float) / ex_pred_mean_t), 0.0))
    dt_test["plant_recon"] = dt_test["sanitary_pred"] + dt_test["extraneous_pred"]

    mae = float(np.mean(np.abs(dt_test["plant_flow"] - dt_test["plant_recon"])))
    rmse = float(np.sqrt(np.mean((dt_test["plant_flow"] - dt_test["plant_recon"]) ** 2)))
    denom = float(np.sum((dt_test["plant_flow"] - dt_test["plant_flow"].mean()) ** 2))
    r2 = float(1.0 - (np.sum((dt_test["plant_flow"] - dt_test["plant_recon"]) ** 2) / denom)) if denom > 0 else float("nan")

    print("\n  DRY TEST PERFORMANCE (holdout dry days):")
    print(f"    Plant recon MAE:  {mae:.4f} MGD")
    print(f"    Plant recon RMSE: {rmse:.4f} MGD")
    print(f"    Plant recon R2:   {r2:.4f}")

    return {
        "ex_model": ex_model,
        "sh_model": sh_model,
        "ex_feature_cols": ex_cols,
        "sh_feature_cols": sh_cols,
        "dry_test_metrics": {"plant_MAE": mae, "plant_RMSE": rmse, "plant_R2": r2},
        "split": split,
        "dt_test": dt_test,
    }


# =============================================================================
# 13) Apply models to all hours
# =============================================================================
def apply_models_all_hours(hourly: pd.DataFrame, daily_all: pd.DataFrame, daily_dry: pd.DataFrame, models: Dict[str, Any], lag_opt: int) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("13) APPLY MODELS TO ALL HOURS")
    print("=" * 70)

    dt = hourly.copy()
    dt = dt.merge(daily_all[["date", "usage_mg", "usage_mgd", "sanitary_daily", "rf"]], on="date", how="left")
    dt = dt.merge(
        daily_dry[["date", "extraneous_daily_nonneg"]].rename(columns={"extraneous_daily_nonneg": "extraneous_daily"}),
        on="date", how="left"
    )
    dt = dt.sort_values("datetime").reset_index(drop=True)

    ex_cols = build_ex_features(dt, lag_opt)
    sh_cols = build_shape_features(dt)
    feat_cols = list(dict.fromkeys(ex_cols + sh_cols))
    dt = impute_feature_cols(dt, feat_cols)

    ex_model: xgb.Booster = models["ex_model"]
    sh_model: xgb.Booster = models["sh_model"]

    sh_raw = np.maximum(1e-9, sh_model.predict(xgb.DMatrix(dt[sh_cols].to_numpy(dtype=float))))
    tmp = pd.DataFrame({"date": dt["date"].to_numpy(), "raw": sh_raw})
    day_mean = tmp.groupby("date")["raw"].transform("mean").to_numpy(dtype=float)
    dt["sanitary_shape"] = sh_raw / np.maximum(1e-9, day_mean)
    dt["sanitary_hourly_pred"] = np.maximum(0.0, dt["sanitary_daily"].to_numpy(dtype=float) * dt["sanitary_shape"].to_numpy(dtype=float))

    ex_raw = np.maximum(0.0, ex_model.predict(xgb.DMatrix(dt[ex_cols].to_numpy(dtype=float))))
    dt["extraneous_baseline_raw"] = ex_raw
    dt["extraneous_hourly_pred"] = ex_raw.copy()

    # Scale to match daily extraneous on dry days.
    mask = (dt["is_dry"]) & dt["extraneous_daily"].notna()
    if mask.any():
        dt.loc[mask, "extraneous_hourly_pred"] = scale_to_daily_mean(
            values=dt.loc[mask, "extraneous_baseline_raw"],
            groups=dt.loc[mask, "date"],
            target_daily_mean=dt.loc[mask, "extraneous_daily"],
        )

    dt["plant_recon"] = dt["sanitary_hourly_pred"] + dt["extraneous_hourly_pred"]
    dt["residual"] = dt["plant_flow"] - dt["plant_recon"]
    dt["wet_residual"] = np.where(~dt["is_dry"], dt["residual"], np.nan)

    return dt


# =============================================================================
# Diagnostics
# =============================================================================
def test_tide_effect(dt_final: pd.DataFrame) -> pd.DataFrame:
    d = dt_final[(dt_final["is_dry"]) & np.isfinite(dt_final["extraneous_hourly_pred"]) & np.isfinite(dt_final["tide"])].copy()
    if len(d) < 1000:
        return pd.DataFrame({"note": ["Too few dry rows for tide test"]})
    try:
        d["tide_q"] = pd.qcut(d["tide"], q=4, duplicates="drop")
    except Exception:
        return pd.DataFrame({"note": ["Could not compute tide quantiles (insufficient unique tide values)"]})
    return d.groupby("tide_q").agg(
        n=("tide", "size"),
        extr_mean=("extraneous_hourly_pred", "mean"),
        extr_med=("extraneous_hourly_pred", "median"),
    ).reset_index()


def test_suppression_effect(dt_final: pd.DataFrame) -> pd.DataFrame:
    d = dt_final[(dt_final["is_dry"]) & np.isfinite(dt_final["extraneous_hourly_pred"]) & np.isfinite(dt_final["sanitary_hourly_pred"])].copy()
    if len(d) < 1000:
        return pd.DataFrame({"note": ["Too few dry rows for suppression test"]})
    try:
        d["san_q"] = pd.qcut(d["sanitary_hourly_pred"], q=4, duplicates="drop")
    except Exception:
        return pd.DataFrame({"note": ["Could not compute sanitary quantiles (insufficient unique values)"]})
    return d.groupby("san_q").agg(
        n=("sanitary_hourly_pred", "size"),
        extr_mean=("extraneous_hourly_pred", "mean"),
        extr_med=("extraneous_hourly_pred", "median"),
    ).reset_index()


def test_lag_stability_by_season(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seas in ["DJF", "MAM", "JJA", "SON"]:
        d = hourly[hourly["season"] == seas].copy()
        lag = estimate_tidal_lag(d, max_lag=24)
        rows.append((seas, lag))
    return pd.DataFrame(rows, columns=["season", "tidal_lag_hours"])


# =============================================================================
# Plots (optional, matplotlib)
# =============================================================================
def _choose_plot_window(dt: pd.DataFrame, days: int, min_pts: int) -> Optional[pd.DataFrame]:
    if len(dt) < min_pts:
        return None
    d = dt.sort_values("datetime").copy()
    dates = sorted(d["date"].unique())
    if len(dates) < 5:
        return None
    for d0 in dates:
        d1 = d0 + pd.Timedelta(days=days)
        sub = d[(pd.to_datetime(d["date"]) >= pd.to_datetime(d0)) & (pd.to_datetime(d["date"]) <= pd.to_datetime(d1))]
        if len(sub) < min_pts:
            continue
        ok1 = np.isfinite(sub["plant_flow"]).mean()
        ok2 = np.isfinite(sub["plant_recon"]).mean()
        if ok1 < 0.80 or ok2 < 0.50:
            continue
        if float(np.nanstd(sub["plant_flow"])) <= 1e-6:
            continue
        return sub
    return None


def save_plots(dt_final: pd.DataFrame, daily_all: pd.DataFrame, rf_table: pd.DataFrame, out_dir: str, cfg: Config):
    if not cfg.plots_enable:
        return
    if not HAVE_MPL:
        print("  NOTE: matplotlib not installed; skipping plots.")
        return

    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    def savefig(path: str):
        plt.tight_layout()
        plt.savefig(path, dpi=cfg.plots_dpi)
        plt.close()

    dt = dt_final.sort_values("datetime").copy()

    # 1) timeseries plant vs recon sample window
    sub = _choose_plot_window(dt, cfg.plots_sample_days_for_hourly_ts, cfg.plots_min_points)
    plt.figure(figsize=(cfg.plots_width/100, cfg.plots_height/100))
    if sub is None:
        plt.axis("off")
        plt.text(0.5, 0.5, "No suitable window found (too many NA / too flat).", ha="center", va="center")
    else:
        plt.plot(sub["datetime"], sub["plant_flow"], label="Plant (obs)")
        plt.plot(sub["datetime"], sub["plant_recon"], label="Recon (san+ex)")
        plt.title("Plant flow vs Reconstructed (san+ex) — sample window")
        plt.xlabel("Datetime"); plt.ylabel("MGD"); plt.legend()
    savefig(os.path.join(plots_dir, "01_timeseries_plant_vs_recon.png"))

    # 2) dry scatter tide vs extraneous
    d = dt[(dt["is_dry"]) & np.isfinite(dt["tide"]) & np.isfinite(dt["extraneous_hourly_pred"])]
    if len(d) > cfg.plots_min_points:
        plt.figure(figsize=(cfg.plots_width/100, cfg.plots_height/100))
        plt.scatter(d["tide"], d["extraneous_hourly_pred"], s=6, alpha=0.3)
        plt.title("Dry hours: Tide vs predicted extraneous")
        plt.xlabel("Tide (ft)"); plt.ylabel("Predicted extraneous (MGD)")
        savefig(os.path.join(plots_dir, "02_dry_scatter_tide_vs_extraneous.png"))

    # 3) wet scatter rain vs wet residual
    w = dt[(~dt["is_dry"]) & np.isfinite(dt["rain_incr"]) & np.isfinite(dt["wet_residual"])]
    if len(w) > cfg.plots_min_points:
        plt.figure(figsize=(cfg.plots_width/100, cfg.plots_height/100))
        plt.scatter(w["rain_incr"], w["wet_residual"], s=6, alpha=0.3)
        plt.axhline(0)
        plt.title("Wet hours: rain vs wet residual (storm excess proxy)")
        plt.xlabel("Rain (in/hr)"); plt.ylabel("Wet residual (MGD)")
        savefig(os.path.join(plots_dir, "03_wet_scatter_rain_vs_wet_residual.png"))

    # 4) daily mass balance
    da = daily_all.sort_values("date").copy()
    if len(da) > 30:
        plt.figure(figsize=(cfg.plots_width/100, cfg.plots_height/100))
        plt.plot(da["date"], da["plant_flow_daily"], label="Plant")
        plt.plot(da["date"], da["sanitary_daily"], label="Sanitary")
        plt.plot(da["date"], np.maximum(0, da["extraneous_daily"]), label="Extraneous (>=0)")
        plt.title("Daily mean flow: plant vs sanitary vs extraneous")
        plt.xlabel("Date"); plt.ylabel("MGD"); plt.legend()
        savefig(os.path.join(plots_dir, "04_daily_mass_balance.png"))

    # 5) seasonal RF
    rf = rf_table.copy()
    if len(rf) >= 4:
        plt.figure(figsize=(cfg.plots_width/100, cfg.plots_height/100))
        plt.bar(rf["season"], rf["rf"])
        plt.title("Seasonal Return Factor")
        plt.ylabel("RF")
        savefig(os.path.join(plots_dir, "05_seasonal_return_factor.png"))


# =============================================================================
# Excel export helpers
# =============================================================================
def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        dtype = out[c].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_convert("UTC").dt.tz_localize(None)
        if isinstance(dtype, pd.IntervalDtype):
            out[c] = out[c].astype(str)
        if isinstance(dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
        if isinstance(dtype, pd.PeriodDtype):
            out[c] = out[c].astype(str)

    def _safe_obj(x):
        if isinstance(x, pd.Interval):
            return str(x)
        return x

    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in obj_cols:
        out[c] = out[c].map(_safe_obj)

    return out


def write_df_sheet(wb: Workbook, name: str, df: pd.DataFrame):
    ws = wb.create_sheet(title=name)
    df2 = _excel_safe_df(df)
    for r in dataframe_to_rows(df2, index=False, header=True):
        ws.append(r)


def export_excel(out_dir: str, dt_final: pd.DataFrame, daily_all: pd.DataFrame, daily_dry: pd.DataFrame, rf_table: pd.DataFrame, models: Dict[str, Any], gw_daily: pd.DataFrame, lag_opt: int, tests: Dict[str, pd.DataFrame], cfg: Config):
    print("\n" + "=" * 70)
    print("14) EXPORT EXCEL")
    print("=" * 70)

    wb = Workbook()
    wb.remove(wb.active)

    dry_cols = [
        "datetime", "date", "plant_flow", "tide", "gw_proxy", "rain_incr",
        "usage_mg", "usage_mgd", "rf", "sanitary_daily",
        "sanitary_hourly_pred", "extraneous_hourly_pred",
        "plant_recon", "residual"
    ]
    wet_cols = [
        "datetime", "date", "plant_flow", "tide", "gw_proxy", "rain_incr",
        "usage_mg", "usage_mgd", "rf", "sanitary_daily",
        "sanitary_hourly_pred", "extraneous_baseline_raw", "extraneous_hourly_pred",
        "plant_recon", "wet_residual"
    ]

    dry = dt_final[dt_final["is_dry"]].copy()
    wet = dt_final[~dt_final["is_dry"]].copy()

    write_df_sheet(wb, "Dry_Decomposition", dry[[c for c in dry_cols if c in dry.columns]])
    write_df_sheet(wb, "Wet_Bonus", wet[[c for c in wet_cols if c in wet.columns]])

    write_df_sheet(wb, "Daily_All_MassBalance", daily_all)
    write_df_sheet(wb, "Daily_Dry_MassBalance", daily_dry)
    write_df_sheet(wb, "ReturnFactor_Seasonal", rf_table)
    write_df_sheet(wb, "GW_Proxy_Daily", gw_daily)

    perf = pd.DataFrame(list(models["dry_test_metrics"].items()), columns=["metric", "value"])
    write_df_sheet(wb, "Performance_Dry", perf)

    write_df_sheet(wb, "Test_TideEffect", tests.get("tide_effect", pd.DataFrame({"note": ["None"]})))
    write_df_sheet(wb, "Test_SanSuppression", tests.get("suppression", pd.DataFrame({"note": ["None"]})))
    write_df_sheet(wb, "Test_LagBySeason", tests.get("lag_stability", pd.DataFrame({"note": ["None"]})))

    counts = dt_final.groupby("date").size()
    incomplete_days = int((counts < 24).sum())

    summary = pd.DataFrame([
        ("run_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("seed", cfg.seed),
        ("tz", cfg.tz),
        ("usage_units", cfg.usage_units),
        ("tidal_lag_hours", lag_opt),
        ("rf_bounds_primary", f"{cfg.rf_bounds_primary[0]}–{cfg.rf_bounds_primary[1]}"),
        ("rf_bounds_secondary", f"{cfg.rf_bounds_secondary[0]}–{cfg.rf_bounds_secondary[1]}"),
        ("travel_time_same_day_fraction", cfg.travel_time_same_day_fraction),
        ("gw_proxy_rolling_days", cfg.gw_proxy_rolling_days),
        ("em_iterations", cfg.em_iters),
        ("early_stopping_rounds", cfg.early_stopping_rounds),
        ("valid_fraction_rows", cfg.valid_fraction_rows),
        ("max_train_rows", "NULL" if cfg.max_train_rows is None else str(cfg.max_train_rows)),
        ("dry_days_train", len(models["split"]["train"])),
        ("dry_days_test", len(models["split"]["test"])),
        ("incomplete_days_in_output", incomplete_days),
        ("effective_rain_lag_window_hr", f"{cfg.dwf_filter['response_lag_lo_hr']}-{cfg.dwf_filter['response_lag_hi_hr']}"),
        ("impact_threshold_mgd", cfg.dwf_filter["impact_threshold_mgd"]),
    ], columns=["item", "value"])
    write_df_sheet(wb, "Summary", summary)

    xlsx_path = os.path.join(out_dir, "results.xlsx")
    wb.save(xlsx_path)
    print(f"  Saved: {xlsx_path}")


def save_model_bundle(out_dir: str, cfg: Config, lag_opt: int, rf_table: pd.DataFrame, models: Dict[str, Any], gw_daily: pd.DataFrame):
    bundle = {
        "config": {
            "seed": cfg.seed,
            "tz": cfg.tz,
            "usage_units": cfg.usage_units,
            "rf_bounds_primary": cfg.rf_bounds_primary,
            "rf_bounds_secondary": cfg.rf_bounds_secondary,
            "travel_time_same_day_fraction": cfg.travel_time_same_day_fraction,
            "gw_proxy_rolling_days": cfg.gw_proxy_rolling_days,
            "dwf_filter": cfg.dwf_filter,
            "em_iters": cfg.em_iters,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "valid_fraction_rows": cfg.valid_fraction_rows,
            "min_valid_rows": cfg.min_valid_rows,
            "max_train_rows": cfg.max_train_rows,
            "xgb_ex_params": cfg.xgb_ex_params,
            "xgb_sh_params": cfg.xgb_sh_params,
        },
        "tidal_lag_hours": int(lag_opt),
        "return_factor_seasonal": rf_table.copy(),
        "models": {
            "extraneous_xgb": models.get("ex_model"),
            "sanitary_shape_xgb": models.get("sh_model"),
            "ex_feature_cols": models.get("ex_feature_cols"),
            "sh_feature_cols": models.get("sh_feature_cols"),
            "dry_test_metrics": models.get("dry_test_metrics"),
            "split": models.get("split"),
        },
        "gw_daily_history": gw_daily.copy(),
    }

    models_dir = os.path.join(out_dir, "models")
    ensure_dir(models_dir)

    path = os.path.join(models_dir, "model_bundle.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"  Saved model bundle: {path}")


# =============================================================================
# Main
# =============================================================================
def main(cfg: Config):
    np.random.seed(cfg.seed)

    prog = Progress(total_steps=12)

    out = init_output_dirs(cfg)
    out_dir = out["out_dir"]
    prog.tick("Initialized output dirs")

    print("=" * 70)
    print("EXTRANEOUS FLOW DECOMPOSITION RUN (Python)")
    print("=" * 70)
    print("Inputs folder:\n ", os.path.abspath(cfg.data_dir))
    print("Results folder:\n ", out_dir)
    print(f"Seed: {cfg.seed}")

    hourly, usage, usage_units_used = load_and_align(cfg)
    prog.tick("Loaded + aligned inputs")

    hourly = flag_dry_wet(hourly, cfg)
    prog.tick("Flagged dry vs wet hours")

    rf_table = estimate_return_factor_by_season(hourly, usage, cfg)
    prog.tick("Estimated seasonal return factor")

    daily_all, daily_dry = compute_daily_mass_balance(hourly, usage, rf_table, cfg)
    prog.tick("Computed daily mass balance")

    hourly, gw_daily = compute_gw_proxy(hourly, daily_all, daily_dry, cfg)
    prog.tick("Computed GW proxy")

    lag_opt = cfg.tidal_lag_hours
    if lag_opt is None:
        lag_opt = estimate_tidal_lag(hourly, max_lag=24)
        print(f"\nEstimated tidal_lag_hours = {lag_opt}")
    else:
        print(f"\nUsing provided tidal_lag_hours = {lag_opt}")
    lag_opt = int(abs(lag_opt))
    prog.tick("Estimated/confirmed tidal lag")

    models = fit_dry_decomposition_models(hourly, daily_all, daily_dry, lag_opt, cfg)
    prog.tick("Trained dry decomposition models")

    dt_final = apply_models_all_hours(hourly, daily_all, daily_dry, models, lag_opt)
    prog.tick("Applied models to all hours")

    tests = {
        "tide_effect": test_tide_effect(dt_final),
        "suppression": test_suppression_effect(dt_final),
        "lag_stability": test_lag_stability_by_season(hourly),
    }
    prog.tick("Ran diagnostic tests")

    ensure_dir(os.path.join(out_dir, "tables"))
    dt_final.to_csv(os.path.join(out_dir, "tables", "hourly_results_final.csv"), index=False)
    models["dt_test"].to_csv(os.path.join(out_dir, "tables", "dry_holdout_predictions.csv"), index=False)
    prog.tick("Saved CSV tables")

    save_plots(dt_final, daily_all, rf_table, out_dir, cfg)
    prog.tick("Saved plots (if enabled)")

    export_excel(out_dir, dt_final, daily_all, daily_dry, rf_table, models, gw_daily, lag_opt, tests, cfg)
    save_model_bundle(out_dir, cfg, lag_opt, rf_table, models, gw_daily)
    prog.tick("Exported Excel + saved model bundle")

    print("\nCOMPLETE\nOutputs in:\n ", out_dir)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="Folder containing input CSVs")
    p.add_argument("--results_parent_dir", default=None, help="Parent output folder")
    p.add_argument("--tz", default="UTC")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--usage_units", default="AUTO", help="AUTO | MGD | MG")
    p.add_argument("--plots", type=int, default=1, help="1 enable plots, 0 disable")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = make_default_config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.results_parent_dir:
        cfg.results_parent_dir = args.results_parent_dir
    if args.tz:
        cfg.tz = args.tz
    cfg.seed = int(args.seed)
    cfg.usage_units = str(args.usage_units).upper().strip()
    if cfg.usage_units not in {"AUTO", "MGD", "MG"}:
        raise ValueError("usage_units must be one of: AUTO, MGD, MG")

    if int(args.plots) not in (0, 1):
        raise ValueError("--plots must be 0 or 1")
    cfg.plots_enable = bool(int(args.plots))

    # ensure xgb seeds match chosen seed
    cfg.xgb_ex_params["seed"] = cfg.seed
    cfg.xgb_sh_params["seed"] = cfg.seed

    main(cfg)
