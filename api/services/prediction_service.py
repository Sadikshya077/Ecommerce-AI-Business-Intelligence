"""api/services/prediction_service.py"""

import json

from api.exceptions import PredictionServiceError
from api.schemas.common import SHAPFeature


# Parses the JSON-string SHAP features stored per customer into typed
# models. Wrapped explicitly: malformed SHAP data would be a data-pipeline
# bug, not a bad client request -- it should surface as a specific, logged
# service error, not an unhandled JSONDecodeError falling through to the
# generic catch-all.
def parse_shap_features(raw_json: str) -> list:
    try:
        items = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PredictionServiceError(f"Malformed SHAP feature data: {exc}") from exc
    return [SHAPFeature(**f) for f in items]


def build_churn_result(record: dict) -> dict:
    return {
        "churn_probability": record["churn_probability"],
        "segment_id": int(record["segment_id"]),
        "top_features": parse_shap_features(record["churn_shap_top_features"]),
    }


def build_clv_result(record: dict) -> dict:
    return {
        "clv_ml": record["clv_ml"],
        "clv_formula": record["clv_formula"],
        "historical_spend": record["monetary"],
        "top_features": parse_shap_features(record["clv_shap_top_features"]),
    }