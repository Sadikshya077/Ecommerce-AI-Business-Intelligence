"""api/services/customer_service.py"""

from api.data_store import store
from api.exceptions import CustomerNotFoundError, ModelArtifactMissingError


# Thin wrapper around CustomerStore -- routers call this, never the store
# directly, so the store's internal shape can change without every router
# needing to change too. Converts "not found" from a None return into the
# typed exception the centralized handler in exceptions.py expects.
def get_customer_record(customer_unique_id: str) -> dict:
    record = store.get(customer_unique_id)
    if record is None:
        raise CustomerNotFoundError(customer_unique_id)
    return record


def sample_customers(n: int) -> list:
    df = store.sample(n)
    return df[["customer_unique_id", "segment_label", "churn_probability", "clv_ml"]].to_dict(orient="records")


def get_segment_profiles() -> list:
    if not store.segment_profile:
        raise ModelArtifactMissingError("segment_profile.parquet not loaded")
    return store.segment_profile
