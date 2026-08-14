"""
models/segmentation/kmeans_segmentation.py

Customer segmentation via K-Means on standardized RFM features
(recency, frequency, monetary). k is chosen by silhouette score
across a range rather than picked arbitrarily -- the comparison
table and plot are saved so the choice is documented, not asserted.

Output:
    data/processed/models/customer_segments.parquet
        customer_unique_id, recency_days, frequency, monetary, segment_id
    data/processed/models/segment_profile.parquet
        one row per segment: size and average RFM values
    reports/figures/segmentation_silhouette.png
    reports/figures/segmentation_k_selection.csv
    models/segmentation/artifacts/kmeans_model.joblib
    models/segmentation/artifacts/scaler.joblib

Run from project root:
    python -m models.segmentation.kmeans_segmentation
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
K_RANGE = range(2, 9)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 5000

# A k that scores well but produces a cluster smaller than this fraction of
# the dataset is excluded from automatic selection -- a cluster that small
# usually isn't a usable business segment even if it's statistically clean.
MIN_CLUSTER_FRACTION = 0.03

# Set to an int (e.g. 4) to override automatic k selection after reviewing
# the diagnostics table below. Leave as None to auto-select. Either way,
# the full comparison table is always logged and saved, so the choice is
# documented rather than asserted -- see the k-selection writeup this
# should feed into your methodology chapter.
FORCE_K = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kmeans_segmentation")


def load_rfm() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    missing = [c for c in RFM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"customer_features.parquet is missing expected columns: {missing}")

    before = len(df)
    df = df.dropna(subset=RFM_COLUMNS)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d customers with null RFM values before clustering", dropped)
    return df


def select_k(X_scaled, n_samples: int) -> tuple:
    scores = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=RANDOM_STATE
        )
        # Davies-Bouldin is cheap (no O(n^2) pairwise step) so it runs on
        # the full dataset -- lower is better, unlike silhouette.
        db_index = davies_bouldin_score(X_scaled, labels)

        cluster_sizes = pd.Series(labels).value_counts()
        min_cluster_pct = cluster_sizes.min() / n_samples

        scores.append({
            "k": k,
            "silhouette_score": silhouette,
            "davies_bouldin_index": db_index,
            "inertia": km.inertia_,
            "min_cluster_size": int(cluster_sizes.min()),
            "min_cluster_pct": min_cluster_pct,
        })
        logger.info(
            "k=%d  silhouette=%.4f  davies_bouldin=%.4f  inertia=%.1f  smallest_cluster=%.1f%%",
            k, silhouette, db_index, km.inertia_, min_cluster_pct * 100,
        )

    scores_df = pd.DataFrame(scores)

    if FORCE_K is not None:
        best_k = FORCE_K
        logger.info(
            "FORCE_K=%d set -- overriding automatic selection. "
            "Review the table above to confirm this is still the right call.", best_k
        )
        return best_k, scores_df

    eligible = scores_df[scores_df["min_cluster_pct"] >= MIN_CLUSTER_FRACTION]
    if eligible.empty:
        logger.warning(
            "No k in K_RANGE keeps every cluster above %.0f%% of the dataset -- "
            "falling back to the full candidate set. Segments may be unbalanced; "
            "consider lowering MIN_CLUSTER_FRACTION or reviewing K_RANGE.",
            MIN_CLUSTER_FRACTION * 100,
        )
        eligible = scores_df

    best_k = int(eligible.loc[eligible["silhouette_score"].idxmax(), "k"])
    logger.info(
        "Selected k=%d (highest silhouette score among candidates with no "
        "cluster below %.0f%% of the dataset)", best_k, MIN_CLUSTER_FRACTION * 100,
    )
    return best_k, scores_df


def plot_k_selection(scores_df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(scores_df["k"], scores_df["silhouette_score"], marker="o", color="tab:blue", label="Silhouette score")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Silhouette score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(scores_df["k"], scores_df["inertia"], marker="s", color="tab:orange", label="Inertia (elbow)")
    ax2.set_ylabel("Inertia", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Silhouette score and inertia")

    ax3.plot(scores_df["k"], scores_df["davies_bouldin_index"], marker="^", color="tab:green")
    ax3.set_xlabel("Number of clusters (k)")
    ax3.set_ylabel("Davies-Bouldin index (lower is better)")
    ax3.set_title("Davies-Bouldin index")

    fig.suptitle("K-Means k selection diagnostics")
    fig.tight_layout()
    out_path = FIGURES_DIR / "segmentation_silhouette.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def build_segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby("segment_id")
        .agg(
            n_customers=("customer_unique_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
    )
    profile["pct_of_customers"] = (profile["n_customers"] / profile["n_customers"].sum() * 100).round(1)
    return profile


def run():
    df = load_rfm()
    logger.info("Clustering on %d customers", len(df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[RFM_COLUMNS])

    best_k, scores_df = select_k(X_scaled, n_samples=len(df))
    plot_k_selection(scores_df)

    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["segment_id"] = final_model.fit_predict(X_scaled)

    profile = build_segment_profile(df)
    logger.info("Segment profile:\n%s", profile.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df[["customer_unique_id", *RFM_COLUMNS, "segment_id"]].to_parquet(
        OUTPUT_DIR / "customer_segments.parquet", index=False
    )
    profile.to_parquet(OUTPUT_DIR / "segment_profile.parquet", index=False)
    scores_df.to_csv(FIGURES_DIR / "segmentation_k_selection.csv", index=False)

    joblib.dump(final_model, ARTIFACTS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    logger.info("Segmentation complete: k=%d, %d customers segmented.", best_k, len(df))


if __name__ == "__main__":
    run()