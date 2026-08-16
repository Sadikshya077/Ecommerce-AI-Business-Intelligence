"""models/forecasting/prophet_forecast.py"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DAILY_SALES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "daily_sales.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Trailing days held out to evaluate the forecast against real, already-known values
HOLDOUT_DAYS = 60
# Days forecast beyond the end of the (trimmed) historical data
FORECAST_HORIZON_DAYS = 30
# A day counts as "active" if order_count is at least this fraction of the dataset's median
MIN_ACTIVITY_FRACTION = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prophet_forecast")


# Olist's collection window trails off sharply at the end -- auto-detect and
# trim that collapsing tail rather than let Prophet learn a fake demand crash
def trim_collection_tail(daily: pd.DataFrame) -> pd.DataFrame:
    median_count = daily["order_count"].median()
    threshold = max(median_count * MIN_ACTIVITY_FRACTION, 1)
    active_dates = daily.loc[daily["order_count"] >= threshold, "order_date"]
    cutoff = active_dates.max()
    trimmed = daily[daily["order_date"] <= cutoff]

    n_trimmed = len(daily) - len(trimmed)
    if n_trimmed:
        logger.warning(
            "Trimmed %d trailing days after %s where order_count fell below "
            "%.1f%% of the dataset median -- consistent with Olist's known "
            "collection tail rather than genuine demand collapse.",
            n_trimmed, cutoff.date(), MIN_ACTIVITY_FRACTION * 100,
        )
    return trimmed


# Prophet requires exactly two columns named ds (date) and y (target)
def to_prophet_format(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[["order_date", "revenue"]].rename(columns={"order_date": "ds", "revenue": "y"})


# Trains on everything except the last HOLDOUT_DAYS, forecasts that period,
# and scores against the real values -- this is what MAPE/RMSE in the report
# actually validate, not the final future-looking forecast itself
def evaluate_holdout(df: pd.DataFrame) -> dict:
    train = df.iloc[:-HOLDOUT_DAYS]
    test = df.iloc[-HOLDOUT_DAYS:]

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(train)

    future = model.make_future_dataframe(periods=HOLDOUT_DAYS)
    forecast = model.predict(future)

    predicted = forecast.set_index("ds").loc[test["ds"], "yhat"].clip(lower=0)
    actual = test.set_index("ds")["y"]

    mape = mean_absolute_percentage_error(actual, predicted.reindex(actual.index))
    rmse = mean_squared_error(actual, predicted.reindex(actual.index)) ** 0.5

    logger.info("Holdout evaluation (last %d days): MAPE=%.1f%%  RMSE=%.2f", HOLDOUT_DAYS, mape * 100, rmse)
    return {"holdout_days": HOLDOUT_DAYS, "mape": mape, "rmse": rmse}


# Refits on the full trimmed series and forecasts beyond the end of the data
def fit_final_model(df: pd.DataFrame):
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=FORECAST_HORIZON_DAYS)
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    return model, forecast


def plot_forecast(model, forecast, out_path: Path):
    fig = model.plot(forecast)
    plt.title("Daily revenue: historical and forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue (R$)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_components(model, forecast, out_path: Path):
    fig = model.plot_components(forecast)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# Condenses the forecast into a short trend note rather than a full time
# series -- this is what the API's /insights endpoint embeds for the LLM
def build_forecast_summary(df: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    last_actual_date = df["ds"].max()
    prior_30d = df[df["ds"] > last_actual_date - pd.Timedelta(days=30)]["y"].sum()

    future_only = forecast[forecast["ds"] > last_actual_date]
    next_30d = future_only.head(30)["yhat"].sum()

    pct_change = ((next_30d - prior_30d) / prior_30d * 100) if prior_30d > 0 else 0.0
    if pct_change > 5:
        direction = "up"
    elif pct_change < -5:
        direction = "down"
    else:
        direction = "flat"

    return {
        "last_actual_date": str(last_actual_date.date()),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "prior_30d_actual_revenue": round(float(prior_30d), 2),
        "next_30d_predicted_revenue": round(float(next_30d), 2),
        "pct_change": round(float(pct_change), 1),
        "trend_direction": direction,
    }


def run():
    daily = pd.read_parquet(DAILY_SALES_PATH)
    logger.info(
        "Loaded %d days of sales data (%s to %s)",
        len(daily), daily["order_date"].min().date(), daily["order_date"].max().date(),
    )

    trimmed = trim_collection_tail(daily)
    df = to_prophet_format(trimmed)

    if len(df) < HOLDOUT_DAYS * 2:
        raise ValueError(
            f"Only {len(df)} usable days after trimming -- not enough for a "
            f"{HOLDOUT_DAYS}-day holdout evaluation. Lower HOLDOUT_DAYS or "
            "check trim_collection_tail's threshold."
        )

    eval_metrics = evaluate_holdout(df)
    model, forecast = fit_final_model(df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_forecast(model, forecast, FIGURES_DIR / "sales_forecast.png")
    plot_components(model, forecast, FIGURES_DIR / "sales_forecast_components.png")

    future_forecast = forecast[forecast["ds"] > df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    future_forecast.to_parquet(OUTPUT_DIR / "sales_forecast.parquet", index=False)

    pd.DataFrame([eval_metrics]).to_csv(OUTPUT_DIR / "sales_forecast_evaluation.csv", index=False)

    summary = build_forecast_summary(df, forecast)
    with open(OUTPUT_DIR / "sales_forecast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Forecast summary: %s", summary)

    logger.info("Sales forecasting complete.")


if __name__ == "__main__":
    run()
