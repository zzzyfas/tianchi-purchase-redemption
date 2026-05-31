from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.high_score_periodic import (
    PeriodicConfig,
    TARGETS,
    actual_for_window,
    forecast_segmented,
    forecast_public_cycle_rule,
    official_proxy_metrics,
)


@dataclass(frozen=True)
class CalendarLinearConfig:
    train_months: int = 4
    alpha: float = 10.0
    fit_residual_ratio: bool = True
    include_day_onehot: bool = True
    include_distance_features: bool = False
    holiday_shrinkage: float = 1.0


@dataclass(frozen=True)
class StaticCalendarConfig:
    train_months: int = 3
    alpha: float = 100.0
    include_day_onehot: bool = True
    include_distance_features: bool = False


HOLIDAY_PERIODS = [
    ("2013-09-19", "2013-09-21"),
    ("2013-10-01", "2013-10-07"),
    ("2014-01-01", "2014-01-01"),
    ("2014-01-31", "2014-02-06"),
    ("2014-04-05", "2014-04-07"),
    ("2014-05-01", "2014-05-03"),
    ("2014-05-31", "2014-06-02"),
    ("2014-09-06", "2014-09-08"),
    ("2014-10-01", "2014-10-07"),
]


def _calendar_state(dates: pd.Series) -> pd.DataFrame:
    normalized = pd.to_datetime(dates).dt.normalize()
    flags = holiday_flags(pd.Series(normalized, index=dates.index))
    weekend = normalized.dt.dayofweek >= 5
    special_work_days = {pd.Timestamp("2014-05-04"), pd.Timestamp("2014-09-28")}
    return pd.DataFrame(
        {
            "is_holiday": flags["is_holiday"].to_numpy(),
            "is_holiday_end": flags["is_lastday_of_holiday"].to_numpy(),
            "is_work": (((flags["is_holiday"] == 0) & ~weekend) | normalized.isin(special_work_days)).astype(int),
        },
        index=dates.index,
    )


def _distance_feature(dates: pd.Series, state_column: str, value: int, direction: int) -> np.ndarray:
    normalized = pd.to_datetime(dates).dt.normalize()
    expanded = pd.Series(pd.date_range(normalized.min() - pd.Timedelta(days=15), normalized.max() + pd.Timedelta(days=15)))
    state = _calendar_state(expanded)
    lookup = dict(zip(expanded, state[state_column]))
    distances = []
    for date in normalized:
        distance = 0
        for step in range(1, 11):
            if lookup[date + pd.Timedelta(days=direction * step)] == value:
                distance = step
                break
        distances.append(10 if distance > 5 else distance)
    return np.asarray(distances)


def holiday_flags(dates: pd.Series) -> pd.DataFrame:
    normalized = pd.to_datetime(dates).dt.normalize()
    out = pd.DataFrame(index=dates.index)
    out["is_holiday"] = 0
    out["is_pre_holiday_1"] = 0
    out["is_pre_holiday_2"] = 0
    out["is_post_holiday_1"] = 0
    out["is_post_holiday_2"] = 0
    out["is_firstday_of_holiday"] = 0
    out["is_lastday_of_holiday"] = 0
    for start, end in HOLIDAY_PERIODS:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        out.loc[normalized.between(start_ts, end_ts), "is_holiday"] = 1
        out.loc[normalized == start_ts, "is_firstday_of_holiday"] = 1
        out.loc[normalized == end_ts, "is_lastday_of_holiday"] = 1
        out.loc[normalized == start_ts - pd.Timedelta(days=1), "is_pre_holiday_1"] = 1
        out.loc[normalized == start_ts - pd.Timedelta(days=2), "is_pre_holiday_2"] = 1
        out.loc[normalized == end_ts + pd.Timedelta(days=1), "is_post_holiday_1"] = 1
        out.loc[normalized == end_ts + pd.Timedelta(days=2), "is_post_holiday_2"] = 1
    return out


def make_calendar_features(
    dates: pd.Series,
    include_day_onehot: bool = True,
    include_distance_features: bool = False,
) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    normalized = dates.dt.normalize()
    features = pd.DataFrame(index=dates.index)
    features["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    features["is_month_start_5"] = (dates.dt.day <= 5).astype(int)
    features["is_month_end_7"] = (dates.dt.day >= 25).astype(int)
    features["is_month_end_3"] = (dates.dt.day >= 28).astype(int)
    features["is_monday"] = (dates.dt.dayofweek == 0).astype(int)
    features["is_friday"] = (dates.dt.dayofweek == 4).astype(int)
    features = pd.concat([features, holiday_flags(dates)], axis=1)
    special_work_days = {pd.Timestamp("2014-05-04"), pd.Timestamp("2014-09-28")}
    features["is_work"] = (
        ((features["is_holiday"] == 0) & (features["is_weekend"] == 0))
        | normalized.isin(special_work_days)
    ).astype(int)

    adjacent = holiday_flags(pd.Series(normalized + pd.Timedelta(days=1)))
    previous = holiday_flags(pd.Series(normalized - pd.Timedelta(days=1)))
    tomorrow_weekend = (dates.dt.dayofweek.add(1).mod(7) >= 5).astype(int)
    yesterday_weekend = (dates.dt.dayofweek.sub(1).mod(7) >= 5).astype(int)
    tomorrow_work = (
        ((adjacent["is_holiday"].to_numpy() == 0) & (tomorrow_weekend.to_numpy() == 0))
        | (normalized + pd.Timedelta(days=1)).isin(special_work_days)
    ).astype(int)
    yesterday_work = (
        ((previous["is_holiday"].to_numpy() == 0) & (yesterday_weekend.to_numpy() == 0))
        | (normalized - pd.Timedelta(days=1)).isin(special_work_days)
    ).astype(int)
    features["is_gonna_work_tomorrow"] = ((features["is_work"] == 0) & (tomorrow_work == 1)).astype(int)
    features["is_worked_yesterday"] = yesterday_work
    features["is_lastday_of_workday"] = ((features["is_holiday"] == 0) & (adjacent["is_holiday"].to_numpy() == 1)).astype(int)
    features["is_work_on_sunday"] = ((dates.dt.dayofweek == 6) & (features["is_work"] == 1)).astype(int)
    features["is_firstday_of_month"] = (dates.dt.day == 1).astype(int)
    features["is_secday_of_month"] = (dates.dt.day == 2).astype(int)
    features["is_premonth"] = (dates.dt.day <= 10).astype(int)
    features["is_midmonth"] = dates.dt.day.between(11, 20).astype(int)
    features["is_tailmonth"] = (dates.dt.day >= 21).astype(int)
    week_mod = dates.dt.isocalendar().week.astype(int).mod(4)
    for value, name in [(1, "first"), (2, "second"), (3, "third"), (0, "fourth")]:
        features[f"is_{name}_week"] = (week_mod == value).astype(int)
    features["dis_from_middleofweek"] = (dates.dt.dayofweek - 3).abs()
    features["dis_from_purchase_peak"] = (dates.dt.dayofweek - 1).abs()
    features["dis_from_purchase_valley"] = (dates.dt.dayofweek - 6).abs()
    if include_distance_features:
        features["dis_to_nowork"] = _distance_feature(dates, "is_work", 0, 1)
        features["dis_from_nowork"] = _distance_feature(dates, "is_work", 0, -1)
        features["dis_to_work"] = _distance_feature(dates, "is_work", 1, 1)
        features["dis_from_work"] = _distance_feature(dates, "is_work", 1, -1)
        features["dis_to_holiday"] = _distance_feature(dates, "is_holiday", 1, 1)
        features["dis_from_holiday"] = _distance_feature(dates, "is_holiday", 1, -1)
        features["dis_from_holiendday"] = _distance_feature(dates, "is_holiday_end", 1, -1)
        features["dis_from_startofmonth"] = dates.dt.day

    weekday = pd.get_dummies(dates.dt.dayofweek, prefix="weekday", dtype=int)
    features = pd.concat([features, weekday], axis=1)
    if include_day_onehot:
        day = pd.get_dummies(dates.dt.day, prefix="day", dtype=int)
        features = pd.concat([features, day], axis=1)
    return features


def monthly_cycle_features(history: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    rows = []
    for month in sorted(pd.to_datetime(dates).dt.to_period("M").unique()):
        month_start = month.to_timestamp()
        month_dates = pd.to_datetime(dates)[pd.to_datetime(dates).dt.to_period("M") == month]
        train = history[history["report_date"] < month_start]
        pred = forecast_public_cycle_rule(train, month_start, periods=month.days_in_month)
        rows.append(pred[pred["report_date"].isin(month_dates)])
    return pd.concat(rows, ignore_index=True)


def align_features(train_features: pd.DataFrame, test_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train_features.columns) | set(test_features.columns))
    return (
        train_features.reindex(columns=columns, fill_value=0),
        test_features.reindex(columns=columns, fill_value=0),
    )


def forecast_calendar_linear(
    history: pd.DataFrame,
    start_date: str | pd.Timestamp,
    config: CalendarLinearConfig,
    periods: int = 30,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    train_start = start_ts - pd.DateOffset(months=config.train_months)
    train = history[(history["report_date"] >= train_start) & (history["report_date"] < start_ts)].copy()
    test = pd.DataFrame({"report_date": pd.date_range(start_ts, periods=periods)})

    train_cycle = monthly_cycle_features(history, train["report_date"])
    test_cycle = forecast_public_cycle_rule(history[history["report_date"] < start_ts], start_ts, periods=periods)
    for target in TARGETS:
        fallback = train_cycle[target].median()
        train_cycle[target] = train_cycle[target].fillna(fallback)
        test_cycle[target] = test_cycle[target].fillna(fallback)

    x_train = make_calendar_features(
        train["report_date"],
        config.include_day_onehot,
        config.include_distance_features,
    )
    x_test = make_calendar_features(
        test["report_date"],
        config.include_day_onehot,
        config.include_distance_features,
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    for target in TARGETS:
        x_train[f"{target}_cycle"] = train_cycle[target].to_numpy()
        x_test[f"{target}_cycle"] = test_cycle[target].to_numpy()
    x_train, x_test = align_features(x_train, x_test)

    out = test.copy()
    for target in TARGETS:
        if config.fit_residual_ratio:
            y = train[target].to_numpy() / np.maximum(train_cycle[target].to_numpy(), 1.0)
            model = make_pipeline(StandardScaler(), Ridge(alpha=config.alpha))
            model.fit(x_train, y)
            pred = model.predict(x_test) * test_cycle[target].to_numpy()
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=config.alpha))
            model.fit(x_train, train[target].to_numpy())
            pred = model.predict(x_test)
        out[target] = np.maximum(pred, 0)
    return out


def forecast_static_calendar(
    history: pd.DataFrame,
    start_date: str | pd.Timestamp,
    config: StaticCalendarConfig = StaticCalendarConfig(),
    periods: int = 30,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    train_start = start_ts - pd.DateOffset(months=config.train_months)
    train = history[(history["report_date"] >= train_start) & (history["report_date"] < start_ts)].copy()
    test = pd.DataFrame({"report_date": pd.date_range(start_ts, periods=periods)})
    x_train = make_calendar_features(
        train["report_date"],
        config.include_day_onehot,
        config.include_distance_features,
    ).reset_index(drop=True)
    x_test = make_calendar_features(
        test["report_date"],
        config.include_day_onehot,
        config.include_distance_features,
    ).reset_index(drop=True)
    x_train, x_test = align_features(x_train, x_test)

    out = test.copy()
    for target in TARGETS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=config.alpha))
        model.fit(x_train, train[target].to_numpy())
        out[target] = np.maximum(model.predict(x_test), 0)
    return out


def forecast_robust_ensemble(
    history: pd.DataFrame,
    start_date: str | pd.Timestamp,
    periods: int = 30,
) -> pd.DataFrame:
    """Use the locally robust target-specific blend selected by rolling validation."""
    rich = forecast_calendar_linear(
        history,
        start_date,
        CalendarLinearConfig(
            train_months=3,
            alpha=300.0,
            fit_residual_ratio=True,
            include_day_onehot=False,
            include_distance_features=False,
        ),
        periods,
    )
    static = forecast_static_calendar(history, start_date, periods=periods)
    out = rich.copy()
    out["redeem"] = static["redeem"]
    return out


def forecast_final_ensemble(
    segments: dict[str, pd.DataFrame],
    start_date: str | pd.Timestamp,
    periods: int = 30,
) -> pd.DataFrame:
    """Combine the leakage-safe purchase cycle with the stable redeem model."""
    purchase_cycle = forecast_segmented(
        segments,
        start_date,
        PeriodicConfig(
            factor_days=84,
            base_days=14,
            factor_stat="mean",
            segment_large_transactions=True,
            holiday_shrinkage=0.5,
        ),
        periods,
    )
    redeem_model = forecast_robust_ensemble(segments["all"], start_date, periods)
    out = purchase_cycle.copy()
    out["redeem"] = redeem_model["redeem"]
    return out


def backtest_calendar_config(
    daily: pd.DataFrame,
    config: CalendarLinearConfig,
    validation_starts: list[str],
) -> dict[str, float]:
    records = []
    for start in validation_starts:
        actual = actual_for_window(daily, start)
        pred = forecast_calendar_linear(daily, start, config)
        records.append(official_proxy_metrics(actual, pred))
    return {
        "mean_weighted_mape": float(np.mean([r["weighted_mape"] for r in records])),
        "mean_proxy_score": float(np.mean([r["proxy_score"] for r in records])),
        "total_purchase_zero_days": int(np.sum([r["purchase_zero_days"] for r in records])),
        "total_redeem_zero_days": int(np.sum([r["redeem_zero_days"] for r in records])),
        "max_weighted_error": float(np.max([r["max_weighted_error"] for r in records])),
    }


def search_calendar_configs(daily: pd.DataFrame, validation_starts: list[str]) -> pd.DataFrame:
    rows = []
    for train_months in [3, 4, 5]:
        for alpha in [1.0, 10.0, 100.0]:
            for fit_residual_ratio in [True, False]:
                for include_day_onehot in [True, False]:
                    config = CalendarLinearConfig(
                        train_months=train_months,
                        alpha=alpha,
                        fit_residual_ratio=fit_residual_ratio,
                        include_day_onehot=include_day_onehot,
                    )
                    metrics = backtest_calendar_config(daily, config, validation_starts)
                    rows.append({**config.__dict__, **metrics})
    return (
        pd.DataFrame(rows)
        .sort_values(["mean_proxy_score", "mean_weighted_mape"], ascending=[False, True])
        .reset_index(drop=True)
    )


def config_from_row(row: pd.Series) -> CalendarLinearConfig:
    return CalendarLinearConfig(
        train_months=int(row["train_months"]),
        alpha=float(row["alpha"]),
        fit_residual_ratio=bool(row["fit_residual_ratio"]),
        include_day_onehot=bool(row["include_day_onehot"]),
        include_distance_features=bool(row.get("include_distance_features", False)),
        holiday_shrinkage=float(row.get("holiday_shrinkage", 1.0)),
    )
