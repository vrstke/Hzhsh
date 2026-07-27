
"""
Online Forecasting App
======================

A Streamlit application for:
- entering monthly online data for a website
- importing CSV data
- cleaning and validating time series
- exploring patterns (trend, seasonality, autocorrelation, decomposition)
- comparing several forecasting models
- forecasting online for the next 12 months
- exporting results

Run:
    pip install -r requirements.txt
    streamlit run online_forecast_app.py

Input CSV format (recommended):
    date,online
    2021-01-01,120
    2021-02-01,135
    ...

The app also accepts daily data; it will aggregate to monthly averages.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# -----------------------------
# Optional dependencies
# -----------------------------
HAS_PROPHET = False
HAS_XGBOOST = False

try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_SEASONAL_PERIOD = 12
DEFAULT_FORECAST_HORIZON = 12
DEFAULT_TEST_MONTHS = 12
MIN_HISTORY_FOR_SEASONAL_MODELS = 24


# -----------------------------
# Utility functions
# -----------------------------
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    out = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.nanmean(out))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_monthly_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.Series:
    """Convert user data into a clean monthly series with a monthly DatetimeIndex."""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Ожидались колонки '{date_col}' и '{value_col}'.")

    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    work = work.dropna(subset=[date_col, value_col]).copy()
    if work.empty:
        raise ValueError("Нет валидных строк после очистки.")

    # If dates look daily or irregular, aggregate to monthly mean.
    work["month"] = work[date_col].dt.to_period("M").dt.to_timestamp("MS")
    monthly = work.groupby("month")[value_col].mean().sort_index()

    # Reindex to complete monthly range.
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index)

    # Fill missing months using interpolation + edge filling.
    monthly = monthly.interpolate(method="time")
    monthly = monthly.ffill().bfill()

    monthly.name = "online"
    monthly.index.name = "month"
    return monthly


def monthly_summary_table(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame("online").copy()
    df["year"] = df.index.year
    df["month_num"] = df.index.month
    df["month_name"] = df.index.strftime("%B")
    pivot = (
        df.pivot_table(index="month_num", columns="year", values="online", aggfunc="mean")
        .sort_index()
    )
    pivot["5y_avg"] = df.groupby("month_num")["online"].mean()
    pivot["5y_median"] = df.groupby("month_num")["online"].median()
    pivot.index.name = "month_num"
    return pivot.reset_index()


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def make_future_index(last_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    start = (last_index.max() + pd.offsets.MonthBegin(1)).normalize()
    return pd.date_range(start=start, periods=horizon, freq="MS")


def series_to_frame(series: pd.Series) -> pd.DataFrame:
    return series.reset_index().rename(columns={"month": "ds", "online": "y"})


def build_feature_frame(series: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    df = series.to_frame("y").copy()
    idx = df.index

    # Calendar features
    df["month_num"] = idx.month
    df["quarter"] = idx.quarter
    df["year"] = idx.year
    df["trend_idx"] = np.arange(len(df), dtype=float)

    # Cyclical annual seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        if lag <= max_lag:
            df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling features
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["roll_std_3"] = df["y"].shift(1).rolling(3).std()
    df["roll_std_6"] = df["y"].shift(1).rolling(6).std()
    df["yoy_diff"] = df["y"] - df["y"].shift(12)
    return df


def train_test_split_last_n(series: pd.Series, n_test: int) -> Tuple[pd.Series, pd.Series]:
    if len(series) <= n_test + 6:
        raise ValueError(
            f"Слишком мало данных ({len(series)} месяцев). Нужно хотя бы {n_test + 6}."
        )
    train = series.iloc[:-n_test].copy()
    test = series.iloc[-n_test:].copy()
    return train, test


# -----------------------------
# Model implementations
# -----------------------------
@dataclass
class ForecastResult:
    model_name: str
    prediction: pd.Series
    lower_80: Optional[pd.Series]
    upper_80: Optional[pd.Series]
    lower_95: Optional[pd.Series]
    upper_95: Optional[pd.Series]
    residual_std: float
    metrics: Dict[str, float]


def _ci_from_std(pred: pd.Series, residual_std: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simple growing uncertainty bands."""
    steps = np.arange(1, len(pred) + 1, dtype=float)
    sigma = residual_std * np.sqrt(steps)
    lower_80 = pred - 1.2816 * sigma
    upper_80 = pred + 1.2816 * sigma
    lower_95 = pred - 1.96 * sigma
    upper_95 = pred + 1.96 * sigma
    return lower_80, upper_80, lower_95, upper_95


def forecast_naive(train: pd.Series, horizon: int) -> pd.Series:
    last_val = float(train.iloc[-1])
    idx = make_future_index(train.index, horizon)
    return pd.Series([last_val] * horizon, index=idx, name="naive")


def forecast_seasonal_naive(train: pd.Series, horizon: int, seasonal_period: int = 12) -> pd.Series:
    idx = make_future_index(train.index, horizon)
    values = []
    for i in range(horizon):
        pos = len(train) - seasonal_period + i
        if pos >= 0:
            values.append(float(train.iloc[pos]))
        else:
            values.append(float(train.iloc[-1]))
    return pd.Series(values, index=idx, name="seasonal_naive")


def forecast_drift(train: pd.Series, horizon: int) -> pd.Series:
    idx = make_future_index(train.index, horizon)
    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / max(len(train) - 1, 1)
    values = [float(train.iloc[-1]) + slope * (i + 1) for i in range(horizon)]
    return pd.Series(values, index=idx, name="drift")


def fit_ets(train: pd.Series, horizon: int) -> Tuple[pd.Series, float]:
    seasonal_periods = 12 if len(train) >= 24 else None
    use_seasonality = seasonal_periods is not None and len(train) >= 2 * seasonal_periods

    model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal="add" if use_seasonality else None,
        seasonal_periods=seasonal_periods if use_seasonality else None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, use_brute=True)
    pred = fit.forecast(horizon)
    resid = np.asarray(train - fit.fittedvalues, dtype=float)
    resid_std = float(np.nanstd(resid, ddof=1))
    return pred.rename("ets"), resid_std


def fit_sarimax(train: pd.Series, horizon: int) -> Tuple[pd.Series, float]:
    seasonal_ok = len(train) >= 24
    candidates = [
        ((1, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 0), (1, 1, 1, 12)),
        ((0, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 1), (0, 1, 1, 12)),
        ((1, 0, 1), (1, 0, 1, 12)) if seasonal_ok else None,
        ((0, 1, 1), (0, 1, 1, 12)) if seasonal_ok else None,
    ]
    candidates = [c for c in candidates if c is not None]

    best_fit = None
    best_aic = np.inf
    best_order = None
    best_seasonal = None

    for order, seasonal_order in candidates:
        try:
            mod = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mod.fit(disp=False)
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_order = order
                best_seasonal = seasonal_order
        except Exception:
            continue

    if best_fit is None:
        # Fallback to a very simple model
        mod = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        best_fit = mod.fit(disp=False)

    pred_res = best_fit.get_forecast(steps=horizon)
    pred = pred_res.predicted_mean.rename("sarimax")
    resid = np.asarray(best_fit.resid, dtype=float)
    resid_std = float(np.nanstd(resid, ddof=1))
    return pred, resid_std


def fit_prophet(train: pd.Series, horizon: int) -> Tuple[pd.Series, float]:
    if not HAS_PROPHET:
        raise RuntimeError("Prophet is not installed.")

    dfp = train.reset_index().rename(columns={"month": "ds", "online": "y"})
    # Prophet prefers daily timestamps; month-start works fine.
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(future)
    tail = fc.tail(horizon).copy()
    pred = pd.Series(tail["yhat"].values, index=make_future_index(train.index, horizon), name="prophet")
    resid = np.asarray(train.values - m.predict(dfp)["yhat"].values, dtype=float)
    resid_std = float(np.nanstd(resid, ddof=1))
    return pred, resid_std


def _iterative_ml_forecast(
    series: pd.Series,
    horizon: int,
    model_type: str = "ridge",
) -> Tuple[pd.Series, float]:
    """Recursive forecasting for ML models using lag/calendar features."""
    hist = series.copy()
    preds = []
    future_index = make_future_index(series.index, horizon)

    for dt in future_index:
        temp = hist.copy()
        temp.index.name = "month"
        feat = build_feature_frame(temp)
        row = feat.iloc[[-1]].copy()

        # Build one-step-ahead feature row.
        row["month_num"] = dt.month
        row["quarter"] = dt.quarter
        row["year"] = dt.year
        row["trend_idx"] = float(len(hist))
        row["month_sin"] = math.sin(2 * math.pi * dt.month / 12.0)
        row["month_cos"] = math.cos(2 * math.pi * dt.month / 12.0)

        # Fill lags / rolling features from history.
        for lag in [1, 2, 3, 6, 12]:
            if f"lag_{lag}" in row.columns:
                row[f"lag_{lag}"] = float(hist.iloc[-lag]) if len(hist) >= lag else float(hist.iloc[-1])
        row["roll_mean_3"] = float(hist.iloc[-3:].mean()) if len(hist) >= 3 else float(hist.mean())
        row["roll_mean_6"] = float(hist.iloc[-6:].mean()) if len(hist) >= 6 else float(hist.mean())
        row["roll_std_3"] = float(hist.iloc[-3:].std(ddof=1)) if len(hist) >= 3 else 0.0
        row["roll_std_6"] = float(hist.iloc[-6:].std(ddof=1)) if len(hist) >= 6 else 0.0
        row["yoy_diff"] = float(hist.iloc[-1] - hist.iloc[-13]) if len(hist) >= 13 else 0.0

        model_data = build_feature_frame(series).dropna()
        feature_cols = [
            c for c in model_data.columns
            if c != "y"
        ]

        X_train = model_data[feature_cols]
        y_train = model_data["y"].astype(float)

        if model_type == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=2.0, random_state=42)),
            ])
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=42,
            )
        elif model_type == "xgb" and HAS_XGBOOST:
            model = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=2.0, random_state=42)),
            ])

        # For the one-step row, align columns
        row = row[feature_cols]
        model.fit(X_train, y_train)
        pred_val = float(model.predict(row)[0])
        preds.append(pred_val)
        hist.loc[dt] = pred_val

    preds_series = pd.Series(preds, index=future_index, name=model_type)
    # Estimate residual std using in-sample fit
    model_data = build_feature_frame(series).dropna()
    feature_cols = [c for c in model_data.columns if c != "y"]
    X_train = model_data[feature_cols]
    y_train = model_data["y"].astype(float)

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
        )
    elif model_type == "xgb" and HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=2.0, random_state=42)),
        ])

    model.fit(X_train, y_train)
    fitted = pd.Series(model.predict(X_train), index=X_train.index)
    resid_std = float(np.nanstd(y_train.values - fitted.values, ddof=1))
    return preds_series, resid_std


def model_predictions_on_test(
    series: pd.Series,
    n_test: int,
    seasonal_period: int = 12,
    use_prophet: bool = True,
) -> List[ForecastResult]:
    train, test = train_test_split_last_n(series, n_test=n_test)
    results: List[ForecastResult] = []

    def eval_pred(model_name: str, pred: pd.Series, resid_std: float) -> ForecastResult:
        pred = pred.reindex(test.index)
        metrics = {
            "MAE": float(mean_absolute_error(test.values, pred.values)),
            "RMSE": rmse(test.values, pred.values),
            "MAPE": safe_mape(test.values, pred.values),
        }
        l80, u80, l95, u95 = _ci_from_std(pred, resid_std)
        return ForecastResult(
            model_name=model_name,
            prediction=pred,
            lower_80=l80,
            upper_80=u80,
            lower_95=l95,
            upper_95=u95,
            residual_std=resid_std,
            metrics=metrics,
        )

    # Baselines
    results.append(eval_pred("Naive", forecast_naive(train, n_test), residual_std=float(np.std(train.diff().dropna(), ddof=1))))
    if len(train) >= seasonal_period:
        results.append(
            eval_pred(
                "Seasonal Naive",
                forecast_seasonal_naive(train, n_test, seasonal_period=seasonal_period),
                residual_std=float(np.std((train - train.shift(seasonal_period)).dropna(), ddof=1)) if len(train) > seasonal_period else float(np.std(train.diff().dropna(), ddof=1)),
            )
        )
    results.append(eval_pred("Drift", forecast_drift(train, n_test), residual_std=float(np.std(train.diff().dropna(), ddof=1))))

    # ETS
    try:
        pred, resid_std = fit_ets(train, n_test)
        results.append(eval_pred("ETS", pred, resid_std))
    except Exception:
        pass

    # SARIMAX
    try:
        pred, resid_std = fit_sarimax(train, n_test)
        results.append(eval_pred("SARIMAX", pred, resid_std))
    except Exception:
        pass

    # Ridge
    try:
        pred, resid_std = _iterative_ml_forecast(train, n_test, model_type="ridge")
        results.append(eval_pred("Ridge", pred, resid_std))
    except Exception:
        pass

    # Random Forest
    try:
        pred, resid_std = _iterative_ml_forecast(train, n_test, model_type="rf")
        results.append(eval_pred("Random Forest", pred, resid_std))
    except Exception:
        pass

    # XGBoost (optional)
    if HAS_XGBOOST:
        try:
            pred, resid_std = _iterative_ml_forecast(train, n_test, model_type="xgb")
            results.append(eval_pred("XGBoost", pred, resid_std))
        except Exception:
            pass

    # Prophet (optional)
    if use_prophet and HAS_PROPHET:
        try:
            pred, resid_std = fit_prophet(train, n_test)
            results.append(eval_pred("Prophet", pred, resid_std))
        except Exception:
            pass

    return results


def select_best_model(results: List[ForecastResult]) -> ForecastResult:
    if not results:
        raise ValueError("Нет доступных моделей.")
    return sorted(results, key=lambda r: (r.metrics["RMSE"], r.metrics["MAE"]))[0]


def forecast_future(
    series: pd.Series,
    horizon: int,
    best_model_name: str,
    residual_std: float,
) -> ForecastResult:
    """Fit the chosen approach on full data and forecast the future."""
    if best_model_name == "Naive":
        pred = forecast_naive(series, horizon)
        resid_std = float(np.std(series.diff().dropna(), ddof=1))
    elif best_model_name == "Seasonal Naive":
        pred = forecast_seasonal_naive(series, horizon)
        resid_std = float(np.std((series - series.shift(12)).dropna(), ddof=1)) if len(series) > 12 else float(np.std(series.diff().dropna(), ddof=1))
    elif best_model_name == "Drift":
        pred = forecast_drift(series, horizon)
        resid_std = float(np.std(series.diff().dropna(), ddof=1))
    elif best_model_name == "ETS":
        pred, resid_std = fit_ets(series, horizon)
    elif best_model_name == "SARIMAX":
        pred, resid_std = fit_sarimax(series, horizon)
    elif best_model_name == "Ridge":
        pred, resid_std = _iterative_ml_forecast(series, horizon, model_type="ridge")
    elif best_model_name == "Random Forest":
        pred, resid_std = _iterative_ml_forecast(series, horizon, model_type="rf")
    elif best_model_name == "XGBoost" and HAS_XGBOOST:
        pred, resid_std = _iterative_ml_forecast(series, horizon, model_type="xgb")
    elif best_model_name == "Prophet" and HAS_PROPHET:
        pred, resid_std = fit_prophet(series, horizon)
    else:
        # Safe fallback
        pred = forecast_seasonal_naive(series, horizon) if len(series) >= 12 else forecast_naive(series, horizon)
        resid_std = residual_std

    l80, u80, l95, u95 = _ci_from_std(pred, resid_std)
    return ForecastResult(
        model_name=best_model_name,
        prediction=pred,
        lower_80=l80,
        upper_80=u80,
        lower_95=l95,
        upper_95=u95,
        residual_std=resid_std,
        metrics={},
    )


# -----------------------------
# Visualizations
# -----------------------------
def make_line_figure(history: pd.Series, forecast: Optional[ForecastResult] = None, title: str = "Онлайн по месяцам") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history.values,
        mode="lines+markers",
        name="История",
    ))
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast.prediction.index,
            y=forecast.prediction.values,
            mode="lines+markers",
            name=f"Прогноз ({forecast.model_name})",
        ))
        if forecast.lower_95 is not None and forecast.upper_95 is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast.prediction.index) + list(forecast.prediction.index[::-1]),
                y=list(forecast.upper_95.values) + list(forecast.lower_95.values[::-1]),
                fill="toself",
                fillcolor="rgba(0, 123, 255, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% интервал",
                showlegend=True,
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Месяц",
        yaxis_title="Онлайн",
        template="plotly_white",
        height=460,
        legend=dict(orientation="h"),
    )
    return fig


def make_monthly_avg_plot(series: pd.Series) -> go.Figure:
    m = series.groupby(series.index.month).mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=[m.get(i, np.nan) for i in range(1, 13)], name="Средний онлайн"))
    fig.update_layout(
        title="Средний онлайн по месяцам (5 лет)",
        xaxis_title="Месяц",
        yaxis_title="Среднее значение",
        template="plotly_white",
        height=420,
    )
    return fig


def make_stl_figure(series: pd.Series, seasonal_period: int = 12) -> go.Figure:
    res = STL(series, period=seasonal_period, robust=True).fit()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=res.observed, name="Observed"))
    fig.add_trace(go.Scatter(x=series.index, y=res.trend, name="Trend"))
    fig.add_trace(go.Scatter(x=series.index, y=res.seasonal, name="Seasonal"))
    fig.add_trace(go.Scatter(x=series.index, y=res.resid, name="Residual"))
    fig.update_layout(
        title="STL-разложение",
        template="plotly_white",
        height=560,
        legend=dict(orientation="h"),
    )
    return fig


def make_acf_pacf_figures(series: pd.Series, lags: int = 24):
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(10, 3))
    plot_acf(series.dropna(), lags=min(lags, max(1, len(series) - 1)), ax=ax1)
    ax1.set_title("ACF")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    pacf_lags = min(lags, max(1, len(series) // 2 - 1))
    plot_pacf(series.dropna(), lags=max(1, pacf_lags), ax=ax2, method="ywm")
    ax2.set_title("PACF")
    fig2.tight_layout()
    return fig1, fig2


# -----------------------------
# Sample / template data
# -----------------------------
def template_months(n_months: int = 60) -> pd.DataFrame:
    end = pd.Timestamp.today().to_period("M").to_timestamp("MS")
    start = end - pd.DateOffset(months=n_months - 1)
    idx = pd.date_range(start=start, periods=n_months, freq="MS")
    return pd.DataFrame({"date": idx, "online": [np.nan] * n_months})


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=60, freq="MS")
    base = 120
    trend = np.linspace(0, 25, len(idx))
    seasonal = 12 * np.sin(2 * np.pi * (idx.month.values - 1) / 12.0)
    noise = np.array([np.random.default_rng(42).normal(0, 5) for _ in idx])
    vals = base + trend + seasonal + noise
    return pd.DataFrame({"date": idx, "online": np.round(vals, 2)})


# -----------------------------
# Streamlit UI
# -----------------------------
def render_sidebar() -> Tuple[str, int, int, int, bool]:
    st.sidebar.header("Настройки")
    site_name = st.sidebar.text_input("Название сайта", value="Мой сайт")
    horizon = st.sidebar.slider("Горизонт прогноза (месяцев)", 3, 24, DEFAULT_FORECAST_HORIZON)
    seasonal_period = st.sidebar.slider("Сезонный период", 6, 24, DEFAULT_SEASONAL_PERIOD)
    test_months = st.sidebar.slider("Месяцев для валидации", 6, 18, DEFAULT_TEST_MONTHS)
    use_prophet = st.sidebar.checkbox("Включать Prophet, если установлен", value=True)
    return site_name, horizon, seasonal_period, test_months, use_prophet


def render_data_input() -> pd.DataFrame:
    st.subheader("Ввод данных")

    tab_upload, tab_manual, tab_example = st.tabs(["Загрузить CSV", "Ввести вручную", "Пример"])

    with tab_upload:
        uploaded = st.file_uploader("CSV файл", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            st.write("Предпросмотр:")
            st.dataframe(raw.head(10), use_container_width=True)

            cols = list(raw.columns)
            date_col = st.selectbox("Колонка с датой", cols, index=0 if cols else None)
            value_col = st.selectbox("Колонка с онлайном", cols, index=1 if len(cols) > 1 else 0)

            if st.button("Использовать этот CSV", key="use_csv"):
                try:
                    series = ensure_monthly_series(raw, date_col, value_col)
                    return series.reset_index().rename(columns={"month": "date", "online": "online"})
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    with tab_manual:
        st.caption("Можно добавлять строки вручную. Формат: месяц + средний онлайн за месяц.")
        manual_df = template_months(60)
        edited = st.data_editor(
            manual_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "date": st.column_config.DateColumn("Месяц", format="YYYY-MM"),
                "online": st.column_config.NumberColumn("Онлайн", help="Средний онлайн за месяц", min_value=0.0, step=1.0),
            },
            hide_index=True,
            key="manual_editor",
        )
        if st.button("Использовать введённые данные", key="use_manual"):
            return edited.copy()

    with tab_example:
        st.write("Синтетический пример на 5 лет.")
        ex = sample_data()
        st.dataframe(ex.head(12), use_container_width=True)
        if st.button("Загрузить пример", key="use_example"):
            return ex.copy()

    return pd.DataFrame(columns=["date", "online"])


def summarize_series(series: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Месяцев", f"{len(series)}")
    c2.metric("Среднее", f"{series.mean():.2f}")
    c3.metric("Мин", f"{series.min():.2f}")
    c4.metric("Макс", f"{series.max():.2f}")

    st.markdown("### Аномалии")
    outliers = detect_outliers_iqr(series)
    if outliers.any():
        st.warning(f"Найдено аномалий: {int(outliers.sum())}")
        st.dataframe(series[outliers].to_frame("online"), use_container_width=True)
    else:
        st.success("Явных выбросов по IQR не найдено.")


def render_monthly_table(series: pd.Series) -> None:
    st.markdown("### Исторические значения")
    hist_df = series.reset_index().rename(columns={"month": "Месяц", "online": "Онлайн"})
    st.dataframe(hist_df, use_container_width=True, height=320)

    st.markdown("### Средний онлайн по месяцам")
    pivot = monthly_summary_table(series)
    st.dataframe(pivot, use_container_width=True, height=340)
    st.plotly_chart(make_monthly_avg_plot(series), use_container_width=True)


def render_model_comparison(results: List[ForecastResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Model": r.model_name, **r.metrics}
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    st.markdown("### Сравнение моделей")
    st.dataframe(comp, use_container_width=True, height=320)
    return comp


def render_forecast_table(forecast: ForecastResult) -> pd.DataFrame:
    out = pd.DataFrame({
        "month": forecast.prediction.index,
        "forecast": forecast.prediction.values,
        "lower_80": forecast.lower_80.values if forecast.lower_80 is not None else np.nan,
        "upper_80": forecast.upper_80.values if forecast.upper_80 is not None else np.nan,
        "lower_95": forecast.lower_95.values if forecast.lower_95 is not None else np.nan,
        "upper_95": forecast.upper_95.values if forecast.upper_95 is not None else np.nan,
    })
    out["month"] = out["month"].dt.strftime("%Y-%m")
    st.markdown("### Прогноз на следующий год")
    st.dataframe(out, use_container_width=True, height=360)
    return out


def main() -> None:
    st.set_page_config(page_title="Forecast Online App", layout="wide")
    st.title("Прогноз онлайн по месяцам")
    st.caption("Вводи данные за прошлые месяцы, анализируй сезонность и получай прогноз на следующий год.")

    site_name, horizon, seasonal_period, test_months, use_prophet = render_sidebar()

    if "loaded_df" not in st.session_state:
        st.session_state.loaded_df = pd.DataFrame(columns=["date", "online"])

    new_df = render_data_input()
    if isinstance(new_df, pd.DataFrame) and not new_df.empty:
        st.session_state.loaded_df = new_df.copy()

    df = st.session_state.loaded_df.copy()
    if df.empty:
        st.info("Загрузи CSV, введи данные вручную или открой пример.")
        return

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", list(df.columns)[0])
    value_col = cols.get("online", list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0])

    try:
        monthly = ensure_monthly_series(df, date_col, value_col)
    except Exception as e:
        st.error(f"Не удалось обработать данные: {e}")
        return

    st.subheader(f"Сайт: {site_name}")

    summarize_series(monthly)
    render_monthly_table(monthly)

    st.markdown("### Графики анализа")
    fig = make_line_figure(monthly, title="Исторический онлайн")
    st.plotly_chart(fig, use_container_width=True)

    try:
        st.plotly_chart(make_stl_figure(monthly, seasonal_period=seasonal_period), use_container_width=True)
    except Exception as e:
        st.warning(f"STL-разложение недоступно для этих данных: {e}")

    try:
        acf_fig, pacf_fig = make_acf_pacf_figures(monthly, lags=min(24, len(monthly) - 1))
        st.pyplot(acf_fig, clear_figure=True, use_container_width=True)
        st.pyplot(pacf_fig, clear_figure=True, use_container_width=True)
    except Exception as e:
        st.warning(f"ACF/PACF не построены: {e}")

    st.markdown("### Модельный анализ")
    try:
        results = model_predictions_on_test(
            monthly,
            n_test=min(test_months, max(6, len(monthly) // 4)),
            seasonal_period=seasonal_period,
            use_prophet=use_prophet,
        )
    except Exception as e:
        st.error(f"Не удалось обучить модели: {e}")
        return

    if not results:
        st.error("Ни одна модель не смогла обучиться на ваших данных.")
        return

    comp = render_model_comparison(results)
    best_name = comp.iloc[0]["Model"]
    st.success(f"Лучшая модель на валидации: {best_name}")

    best_result = select_best_model(results)
    future = forecast_future(monthly, horizon=horizon, best_model_name=best_name, residual_std=best_result.residual_std)

    # Narrative insight block
    st.markdown("### Интерпретация")
    slope = (monthly.iloc[-1] - monthly.iloc[0]) / max(len(monthly) - 1, 1)
    if slope > 0:
        st.info("В ряде виден общий рост.")
    elif slope < 0:
        st.info("В ряде виден общий спад.")
    else:
        st.info("Общий тренд близок к нулю.")
    if len(monthly) >= 24:
        seasonal_strength = monthly.groupby(monthly.index.month).mean().std()
        st.write(f"Сезонность по средним месяцам: стандартное отклонение месячных средних = {seasonal_strength:.2f}")

    forecast_df = render_forecast_table(future)
    st.plotly_chart(make_line_figure(monthly, forecast=future, title="История + прогноз"), use_container_width=True)

    # Download buttons
    st.markdown("### Скачать результаты")
    csv_hist = monthly.reset_index().rename(columns={"month": "date", "online": "online"})
    csv_forecast = forecast_df.copy()
    buffer1 = csv_hist.to_csv(index=False).encode("utf-8")
    buffer2 = csv_forecast.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    c1.download_button("Скачать историю CSV", data=buffer1, file_name="online_history.csv", mime="text/csv")
    c2.download_button("Скачать прогноз CSV", data=buffer2, file_name="online_forecast.csv", mime="text/csv")

    st.markdown("### Как использовать")
    st.write(
        "1) Вставь месячные значения онлайн или загрузи CSV. "
        "2) Проверь очистку и аналитику. "
        "3) Смотри, какая модель лучше по последним месяцам. "
        "4) Забирай прогноз на следующий год."
    )


if __name__ == "__main__":
    main()
