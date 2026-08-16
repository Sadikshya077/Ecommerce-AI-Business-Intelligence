"""api/data_store.py"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"


# Sales forecast is global (same for every customer), not customer-indexed --
# loaded separately from CustomerStore rather than merged into it
def load_forecast_summary() -> Optional[dict]:
    path = DATA_DIR / "sales_forecast_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# Segment profile (Phase 3's k=4 result) is also global, one row per
# segment, not customer-indexed -- soft-loaded like forecast_summary rather
# than added to the hard required-files check, since it's supplementary
def load_segment_profile() -> list:
    path = DATA_DIR / "segment_profile.parquet"
    if not path.exists():
        return []
    profile = pd.read_parquet(path)
    profile["segment_label"] = profile["segment_id"].map(SEGMENT_LABELS)
    return profile.to_dict(orient="records")

# Fixed mapping from k=4 segmentation result
SEGMENT_LABELS = {
    0: "Lapsed one-time buyers",
    1: "Recent one-time buyers",
    2: "Loyal repeat customers",
    3: "High-value one-time buyers",
}


# In-memory store: loads every model output once and merges into one
# customer_unique_id-indexed table, so requests never hit disk
class CustomerStore:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self.forecast_summary: Optional[dict] = None
        self.segment_profile: list = []

    # Merges churn, CLV, and SHAP outputs into a single lookup table
    def load(self):
        required = {
            "churn_predictions.parquet": "python -m models.churn.evaluate_churn",
            "clv_predictions.parquet": "python -m models.clv.evaluate_clv",
            "churn_shap.parquet": "python -m explainability.shap_explainer",
            "clv_shap.parquet": "python -m explainability.shap_explainer",
        }
        missing = {f: cmd for f, cmd in required.items() if not (DATA_DIR / f).exists()}
        if missing:
            lines = "\n".join(f"  {f} -- run: {cmd}" for f, cmd in missing.items())
            raise RuntimeError(f"Cannot start API: required model outputs are missing.\n{lines}")

        churn = pd.read_parquet(DATA_DIR / "churn_predictions.parquet")
        clv = pd.read_parquet(DATA_DIR / "clv_predictions.parquet")
        churn_shap = pd.read_parquet(DATA_DIR / "churn_shap.parquet")
        clv_shap = pd.read_parquet(DATA_DIR / "clv_shap.parquet")

        df = churn.merge(
            clv[["customer_unique_id", "clv_ml", "clv_formula", "monetary"]],
            on="customer_unique_id", how="left",
        )
        df = df.merge(
            churn_shap[["customer_unique_id", "shap_top_features"]]
            .rename(columns={"shap_top_features": "churn_shap_top_features"}),
            on="customer_unique_id", how="left",
        )
        df = df.merge(
            clv_shap[["customer_unique_id", "shap_top_features"]]
            .rename(columns={"shap_top_features": "clv_shap_top_features"}),
            on="customer_unique_id", how="left",
        )

        df["segment_label"] = df["segment_id"].map(SEGMENT_LABELS)
        self._df = df.set_index("customer_unique_id")
        self.forecast_summary = load_forecast_summary()
        self.segment_profile = load_segment_profile()
        return self

    # Returns one customer's merged record as a dict, or None if not found
    def get(self, customer_unique_id: str) -> Optional[dict]:
        if self._df is None:
            raise RuntimeError("CustomerStore.load() must be called before use")
        if customer_unique_id not in self._df.index:
            return None
        return self._df.loc[customer_unique_id].to_dict()

    # Random sample of customers -- useful for grabbing real IDs to test with
    def sample(self, n: int) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("CustomerStore.load() must be called before use")
        return self._df.sample(min(n, len(self._df))).reset_index()

    def __len__(self):
        return 0 if self._df is None else len(self._df)


store = CustomerStore()