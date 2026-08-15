# Mine product category co-purchase patterns using FP-Growth
# Generate frequent itemsets and association rules from order baskets

import logging
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASKETS_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "order_baskets.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "models"

# Retry with lower support when no multi-category itemsets are found
MIN_SUPPORT = 0.005
MIN_LIFT = 1.2

# Retry with lower lift when no association rules are found
MIN_SUPPORT_FLOOR = 0.00005
SUPPORT_RETRY_FACTOR = 0.5

# If rules are found but none clear MIN_LIFT, retry down to this floor.
# 1.0 is the natural floor: lift <= 1 means no positive association, so
# there's nothing meaningful below it to search for.
MIN_LIFT_FLOOR = 1.0
LIFT_RETRY_STEP = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fp_growth")


# Load order baskets and group categories into transactions
def load_transactions() -> list:
    baskets = pd.read_parquet(BASKETS_PATH)
    transactions = baskets.groupby("order_id")["category_name_english"].apply(list).tolist()
    return transactions

# Run FP-Growth and association rule mining on the order transactions
def run():
    transactions = load_transactions()
    logger.info("Loaded %d order baskets", len(transactions))

    # Analyze basket sizes to measure the amount of cross-category signal
    basket_sizes = pd.Series([len(t) for t in transactions])
    logger.info("Basket size distribution:\n%s", basket_sizes.value_counts().sort_index().to_string())
    multi_item = int((basket_sizes > 1).sum())
    logger.info(
        "%d of %d baskets (%.1f%%) contain more than one category",
        multi_item, len(transactions), 100 * multi_item / len(transactions),
    )

    # Convert transactions into a one-hot encoded category matrix
    encoder = TransactionEncoder()
    encoded = encoder.fit_transform(transactions)
    encoded_df = pd.DataFrame(encoded, columns=encoder.columns_)

    # Find frequent itemsets, lowering support when necessary
    support_used = MIN_SUPPORT
    frequent_itemsets = pd.DataFrame()
    while True:
        logger.info("Mining frequent itemsets (min_support=%.5f)...", support_used)
        frequent_itemsets = fpgrowth(encoded_df, min_support=support_used, use_colnames=True)

        has_multi_item = (
            not frequent_itemsets.empty
            and (frequent_itemsets["itemsets"].apply(len) >= 2).any()
        )
        if has_multi_item or support_used <= MIN_SUPPORT_FLOOR:
            break
        support_used = max(support_used * SUPPORT_RETRY_FACTOR, MIN_SUPPORT_FLOOR)
        logger.warning(
            "No itemsets of 2+ categories found -- retrying at a lower min_support=%.5f",
            support_used,
        )

    # Sort discovered itemsets by their support
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    n_multi = int((frequent_itemsets["itemsets"].apply(len) >= 2).sum()) if not frequent_itemsets.empty else 0
    logger.info(
        "Found %d frequent itemsets (%d with 2+ categories) at min_support=%.5f",
        len(frequent_itemsets), n_multi, support_used,
    )

    # Stop when no meaningful multi-category patterns can be found
    if frequent_itemsets.empty or n_multi == 0:
        logger.warning(
            "No multi-category itemsets found even at the floor min_support=%.5f. "
            "With only %.1f%% of baskets spanning more than one category, this "
            "dataset likely offers too little co-purchase signal for FP-Growth "
            "to surface anything meaningful -- document this as a finding "
            "(limited cross-category signal in the Olist marketplace), not a bug.",
            MIN_SUPPORT_FLOOR, 100 * multi_item / len(transactions),
        )
        return
    if support_used != MIN_SUPPORT:
        logger.warning(
            "NOTE: MIN_SUPPORT=%.4f found no usable itemsets; results below use "
            "the lower threshold %.5f instead. Update MIN_SUPPORT at the top of "
            "this file to match, so the constant reflects what was actually used.",
            MIN_SUPPORT, support_used,
        )

    # Generate association rules and lower the lift threshold if necessary
    lift_used = MIN_LIFT
    rules = pd.DataFrame()
    while True:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_used)
        if not rules.empty or lift_used <= MIN_LIFT_FLOOR:
            break
        lift_used = round(max(lift_used - LIFT_RETRY_STEP, MIN_LIFT_FLOOR), 2)
        logger.warning("No rules cleared the lift threshold -- retrying at lift > %.2f", lift_used)

    # Sort rules by lift so the strongest associations appear first
    rules = rules.sort_values("lift", ascending=False)
    logger.info("Found %d rules with lift > %.2f", len(rules), lift_used)
    if rules.empty:
        logger.warning(
            "No rules found even at the lift floor of %.2f. The %d multi-category "
            "itemsets found aren't co-occurring more than chance would predict -- "
            "document this as a finding, not a bug.", MIN_LIFT_FLOOR, n_multi,
        )
        return
    if lift_used != MIN_LIFT:
        logger.warning(
            "NOTE: MIN_LIFT=%.2f found no rules; results below use the lower "
            "threshold %.2f instead. Update MIN_LIFT at the top of this file "
            "to match, so the constant reflects what was actually used.",
            MIN_LIFT, lift_used,
        )

    # Convert frozensets into strings so the results can be stored in Parquet
    frequent_itemsets_out = frequent_itemsets.copy()
    frequent_itemsets_out["itemsets"] = frequent_itemsets_out["itemsets"].apply(
        lambda s: ", ".join(sorted(s))
    )

    # Convert rule antecedents and consequents into readable strings
    rules_out = rules.copy()
    for col in ["antecedents", "consequents"]:
        rules_out[col] = rules_out[col].apply(lambda s: ", ".join(sorted(s)))
    rules_out = rules_out[["antecedents", "consequents", "support", "confidence", "lift"]]

    # Save frequent itemsets and association rules to the models directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frequent_itemsets_out.to_parquet(OUTPUT_DIR / "frequent_itemsets.parquet", index=False)
    rules_out.to_parquet(OUTPUT_DIR / "association_rules.parquet", index=False)

    # Log the strongest rules for quick inspection
    if not rules_out.empty:
        logger.info("Top rules by lift:\n%s", rules_out.head(10).to_string(index=False))
    logger.info("Wrote frequent_itemsets.parquet and association_rules.parquet to %s", OUTPUT_DIR)


if __name__ == "__main__":
    run()