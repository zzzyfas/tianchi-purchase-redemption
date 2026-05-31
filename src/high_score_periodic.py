from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["purchase", "redeem"]
RAW_COLUMNS = ["report_date", "total_purchase_amt", "total_redeem_amt"]


@dataclass(frozen=True)
class PeriodicConfig:
    factor_days: int = 84
    base_days: int = 14
    factor_stat: str = "median"
    segment_large_transactions: bool = True
    holiday_shrinkage: float = 0.5


def find_raw_file(raw_root: Path, filename: str) -> Path:
    matches = sorted(raw_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {raw_root}")
    return matches[0]


def load_daily_segments(raw_root: Path, large_threshold: int = 100_000_000) -> dict[str, pd.DataFrame]:
    balance_path = find_raw_file(raw_root, "user_balance_table.csv")
    raw = pd.read_csv(balance_path, usecols=RAW_COLUMNS, engine="python")
    raw["report_date"] = pd.to_datetime(raw["report_date"].astype(str), format="%Y%m%d")

    frames = {
        "all": raw,
        "regular": raw[
            (raw["total_purchase_amt"] < large_threshold)
            & (raw["total_redeem_amt"] < large_threshold)
        ],
        "large": raw[
            (raw["total_purchase_amt"] >= large_threshold)
            | (raw["total_redeem_amt"] >= large_threshold)
        ],
    }

    return {name: aggregate_daily(frame) for name, frame in frames.items()}


def aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (
        frame.groupby("report_date", as_index=False)[["total_purchase_amt", "total_redeem_amt"]]
        .sum()
        .rename(
            columns={
                "total_purchase_amt": "purchase",
                "total_redeem_amt": "redeem",
            }
        )
    )
    complete_dates = pd.DataFrame(
        {"report_date": pd.date_range(daily["report_date"].min(), daily["report_date"].max())}
    )
    daily = complete_dates.merge(daily, on="report_date", how="left").fillna(0)
    return daily.sort_values("report_date").reset_index(drop=True)


HOLIDAY_PERIODS = {
    "2013_mid_autumn": ("2013-09-19", "2013-09-21"),
    "2013_national": ("2013-10-01", "2013-10-07"),
    "2014_qingming": ("2014-04-05", "2014-04-07"),
    "2014_labour": ("2014-05-01", "2014-05-03"),
    "2014_dragon_boat": ("2014-05-31", "2014-06-02"),
    "2014_mid_autumn": ("2014-09-06", "2014-09-08"),
    "2014_national": ("2014-10-01", "2014-10-07"),
}


def holiday_category(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date).normalize()
    for start, end in HOLIDAY_PERIODS.values():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts <= date <= end_ts:
            return "holiday"
        if start_ts - pd.Timedelta(days=2) <= date < start_ts:
            return "pre_holiday"
        if end_ts < date <= end_ts + pd.Timedelta(days=2):
            return "post_holiday"
    return "normal"


def periodic_components(train: pd.DataFrame, target: str, config: PeriodicConfig) -> tuple[float, pd.Series]:
    recent = train.tail(config.factor_days).copy()
    recent["weekday"] = recent["report_date"].dt.dayofweek
    overall = recent[target].mean()
    if config.factor_stat == "median":
        weekday_level = recent.groupby("weekday")[target].median()
    elif config.factor_stat == "mean":
        weekday_level = recent.groupby("weekday")[target].mean()
    else:
        raise ValueError(config.factor_stat)

    weekday_factor = (weekday_level / weekday_level.mean()).reindex(range(7)).fillna(1.0)
    base = train[target].tail(config.base_days).mean()
    if not np.isfinite(base) or base <= 0:
        base = overall
    return float(base), weekday_factor


def learn_holiday_factors(train: pd.DataFrame, target: str, config: PeriodicConfig) -> dict[str, float]:
    if config.holiday_shrinkage <= 0:
        return {}

    base, weekday_factor = periodic_components(train, target, config)
    history = train.copy()
    history["category"] = history["report_date"].map(holiday_category)
    history = history[history["category"] != "normal"].copy()
    history["expected"] = history["report_date"].dt.dayofweek.map(weekday_factor) * base
    history["ratio"] = history[target] / history["expected"].replace(0, np.nan)
    factors = history.groupby("category")["ratio"].median().clip(0.7, 1.3).to_dict()
    return {
        category: 1 + config.holiday_shrinkage * (factor - 1)
        for category, factor in factors.items()
    }


def forecast_periodic(train: pd.DataFrame, start_date: str | pd.Timestamp, config: PeriodicConfig, periods: int = 30) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    train = train[train["report_date"] < start_ts].copy()
    dates = pd.date_range(start_ts, periods=periods)
    out = pd.DataFrame({"report_date": dates})
    for target in TARGETS:
        base, weekday_factor = periodic_components(train, target, config)
        holiday_factors = learn_holiday_factors(train, target, config)
        out[target] = out["report_date"].dt.dayofweek.map(weekday_factor) * base
        out[target] = out[target] * out["report_date"].map(
            lambda date: holiday_factors.get(holiday_category(date), 1.0)
        )
    return out


def forecast_segmented(segments: dict[str, pd.DataFrame], start_date: str | pd.Timestamp, config: PeriodicConfig, periods: int = 30) -> pd.DataFrame:
    if not config.segment_large_transactions:
        return forecast_periodic(segments["all"], start_date, config, periods)

    regular = forecast_periodic(segments["regular"], start_date, config, periods)
    large = forecast_periodic(segments["large"], start_date, config, periods)
    out = regular[["report_date"]].copy()
    for target in TARGETS:
        out[target] = regular[target] + large[target]
    return out


def forecast_public_cycle_rule(
    train: pd.DataFrame,
    start_date: str | pd.Timestamp,
    periods: int = 30,
    history_start: str = "2014-03-01",
) -> pd.DataFrame:
    """Reproduce the public Tianchi baseline: day-of-month base times weekday factor."""
    start_ts = pd.Timestamp(start_date)
    history = train[
        (train["report_date"] >= pd.Timestamp(history_start))
        & (train["report_date"] < start_ts)
    ].copy()
    history["weekday"] = history["report_date"].dt.dayofweek
    history["day"] = history["report_date"].dt.day
    history["month_period"] = history["report_date"].dt.to_period("M")
    month_count = history["month_period"].nunique()

    weekday_level = history.groupby("weekday")[TARGETS].mean()
    weekday_factor = weekday_level / history[TARGETS].mean()

    weekday_count = (
        history.groupby(["day", "weekday"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    weekday_count = weekday_count.merge(
        weekday_factor.reset_index(),
        on="weekday",
        how="left",
    )
    for target in TARGETS:
        weekday_count[target] = weekday_count[target] * weekday_count["count"] / month_count

    day_rate = weekday_count.groupby("day", as_index=False)[TARGETS].sum()
    day_mean = history.groupby("day", as_index=False)[TARGETS].mean()
    day_base = day_mean.merge(day_rate, on="day", suffixes=("_mean", "_rate"))
    for target in TARGETS:
        day_base[target] = day_base[f"{target}_mean"] / day_base[f"{target}_rate"]

    out = pd.DataFrame({"report_date": pd.date_range(start_ts, periods=periods)})
    out["day"] = out["report_date"].dt.day
    out["weekday"] = out["report_date"].dt.dayofweek
    out = out.merge(day_base[["day"] + TARGETS], on="day", how="left")
    out = out.merge(
        weekday_factor.reset_index().rename(
            columns={target: f"{target}_weekday_factor" for target in TARGETS}
        ),
        on="weekday",
        how="left",
    )
    for target in TARGETS:
        out[target] = out[target] * out[f"{target}_weekday_factor"]
    return out[["report_date"] + TARGETS]


def apply_target_holiday_template(
    pred: pd.DataFrame,
    mid_autumn_shrinkage: float = 0.0,
    national_pre_shrinkage: float = 0.0,
) -> pd.DataFrame:
    """Apply same-holiday prior-year residual patterns to September 2014."""
    out = pred.copy()
    templates = {
        "2014-09-06": (0.378, 0.268),
        "2014-09-07": (0.527, 0.336),
        "2014-09-08": (0.934, 1.043),
        "2014-09-09": (1.222, 1.837),
        "2014-09-10": (0.507, 1.659),
        "2014-09-29": (0.656, 1.051),
        "2014-09-30": (0.298, 0.677),
    }
    for date_text, (purchase_factor, redeem_factor) in templates.items():
        date = pd.Timestamp(date_text)
        mask = out["report_date"] == date
        shrinkage = mid_autumn_shrinkage if date.day <= 10 else national_pre_shrinkage
        out.loc[mask, "purchase"] *= 1 + shrinkage * (purchase_factor - 1)
        out.loc[mask, "redeem"] *= 1 + shrinkage * (redeem_factor - 1)
    return out


def actual_for_window(daily: pd.DataFrame, start_date: str | pd.Timestamp, periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=periods)
    return pd.DataFrame({"report_date": dates}).merge(daily, on="report_date", how="left")


def relative_abs_error(actual: pd.Series, pred: pd.Series) -> np.ndarray:
    actual_values = actual.to_numpy(dtype=float)
    pred_values = pred.to_numpy(dtype=float)
    return np.abs(actual_values - pred_values) / np.maximum(np.abs(actual_values), 1.0)


def official_proxy_metrics(actual: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    merged = actual.merge(pred, on="report_date", suffixes=("_actual", "_pred"))
    purchase_error = relative_abs_error(merged["purchase_actual"], merged["purchase_pred"])
    redeem_error = relative_abs_error(merged["redeem_actual"], merged["redeem_pred"])
    purchase_points = 10 * np.exp(-purchase_error / 0.3)
    redeem_points = 10 * np.exp(-redeem_error / 0.3)
    return {
        "weighted_mape": float(0.45 * purchase_error.mean() + 0.55 * redeem_error.mean()),
        "proxy_score": float(0.45 * purchase_points.sum() + 0.55 * redeem_points.sum()),
        "purchase_zero_days": int((purchase_error > 0.3).sum()),
        "redeem_zero_days": int((redeem_error > 0.3).sum()),
        "max_weighted_error": float(np.max(0.45 * purchase_error + 0.55 * redeem_error)),
    }


def backtest_config(
    segments: dict[str, pd.DataFrame],
    config: PeriodicConfig,
    validation_starts: list[str],
) -> dict[str, float]:
    records = []
    for start in validation_starts:
        start_ts = pd.Timestamp(start)
        train_segments = {
            name: frame[frame["report_date"] < start_ts].copy()
            for name, frame in segments.items()
        }
        actual = actual_for_window(segments["all"], start_ts)
        pred = forecast_segmented(train_segments, start_ts, config)
        records.append(official_proxy_metrics(actual, pred))

    return {
        "mean_weighted_mape": float(np.mean([r["weighted_mape"] for r in records])),
        "mean_proxy_score": float(np.mean([r["proxy_score"] for r in records])),
        "total_purchase_zero_days": int(np.sum([r["purchase_zero_days"] for r in records])),
        "total_redeem_zero_days": int(np.sum([r["redeem_zero_days"] for r in records])),
        "max_weighted_error": float(np.max([r["max_weighted_error"] for r in records])),
    }


def search_configs(segments: dict[str, pd.DataFrame], validation_starts: list[str]) -> pd.DataFrame:
    rows = []
    for factor_days in [56, 84, 112]:
        for base_days in [7, 14, 21, 28]:
            for factor_stat in ["mean", "median"]:
                for segmented in [False, True]:
                    for holiday_shrinkage in [0.0, 0.5, 1.0]:
                        config = PeriodicConfig(
                            factor_days=factor_days,
                            base_days=base_days,
                            factor_stat=factor_stat,
                            segment_large_transactions=segmented,
                            holiday_shrinkage=holiday_shrinkage,
                        )
                        metrics = backtest_config(segments, config, validation_starts)
                        rows.append({**config.__dict__, **metrics})
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_proxy_score", "total_redeem_zero_days", "mean_weighted_mape"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def config_from_row(row: pd.Series) -> PeriodicConfig:
    return PeriodicConfig(
        factor_days=int(row["factor_days"]),
        base_days=int(row["base_days"]),
        factor_stat=str(row["factor_stat"]),
        segment_large_transactions=bool(row["segment_large_transactions"]),
        holiday_shrinkage=float(row["holiday_shrinkage"]),
    )


def write_submission(pred: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pred.copy()
    out["report_date"] = out["report_date"].dt.strftime("%Y%m%d").astype(int)
    out["purchase"] = np.maximum(out["purchase"], 0).round().astype("int64")
    out["redeem"] = np.maximum(out["redeem"], 0).round().astype("int64")
    out[["report_date", "purchase", "redeem"]].to_csv(path, index=False, header=False)
