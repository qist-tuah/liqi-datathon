"""Step 4 - feature engineering (features that need no free-text parsing).

Reads data/processed/transactions_clean.parquet in batches and writes
data/processed/transactions_features.parquet = every clean column, untouched,
plus the new feature columns listed in FEATURES below.

Groups built here:
  time      - calendar parts of epoch_seconds, all in UTC
  parties   - ordering/beneficiary type pair
  amount    - log scale, order of magnitude, whole-dollar flag
  geography - sender/receiver bank country (from the BIC8), domestic flag, corridor
  routing   - parsed bic8_path: entries, gaps, endpoint mismatches

Groups that depend on the free-text columns (accounts, tokens, anomaly
signatures) are added later, once their formats are confirmed by EDA.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import settings as S

# The paths live in settings.py; these names are kept so existing notebooks keep working.
FEATURES_PARQUET = S.FEATURES_PARQUET
FEATURE_DICTIONARY_CSV = S.FEATURE_DICTIONARY_CSV

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PAIRS = ["B2B", "B2C", "C2B", "C2C"]
DICT = pa.dictionary(pa.int32(), pa.string())
# Powers of ten parsed from decimal text, exactly as the CSV amounts were parsed.
EXPONENTS = np.arange(-12, 25)
POWERS = np.array([float(f"1e{k}") for k in EXPONENTS])

# name, arrow type, group, source column(s), description
FEATURES = [
    ("utc_date", pa.date32(), "time", "epoch_seconds", "Calendar date of the payment in UTC"),
    ("utc_month", pa.int8(), "time", "epoch_seconds", "Month number 1-12 in UTC"),
    ("utc_iso_week", pa.int8(), "time", "epoch_seconds", "ISO 8601 week number in UTC (weeks start on Monday)"),
    ("utc_weekday", pa.int8(), "time", "epoch_seconds", "Day of week in UTC: 0 = Monday ... 6 = Sunday"),
    ("utc_weekday_name", DICT, "time", "epoch_seconds", "Day of week in UTC as Mon ... Sun"),
    ("utc_is_weekend", pa.bool_(), "time", "epoch_seconds", "True when the UTC day is Saturday or Sunday"),
    ("utc_hour", pa.int8(), "time", "epoch_seconds", "Hour of day 0-23 in UTC"),
    ("party_pair", DICT, "parties", "ordering_type + beneficiary_type",
     "Sender type to receiver type: B2B, B2C, C2B or C2C (B = business, C = consumer)"),
    ("usd_amount_log10", pa.float64(), "amount", "usd_amount", "log10 of usd_amount (missing if the amount is not above 0)"),
    ("usd_amount_magnitude", pa.int8(), "amount", "usd_amount",
     "Order of magnitude: largest k with 10^k <= usd_amount (0.01 -> -2, 999.99 -> 2, 1000 -> 3)"),
    ("usd_amount_is_whole", pa.bool_(), "amount", "usd_amount", "True when usd_amount has no cents (e.g. 500.00)"),
    ("sender_country", DICT, "geography", "ordering_bic8",
     "Country of the sending bank: characters 5-6 of ordering_bic8 (0621US30 -> US)"),
    ("receiver_country", DICT, "geography", "beneficiary_bic8",
     "Country of the receiving bank: characters 5-6 of beneficiary_bic8"),
    ("is_domestic", pa.bool_(), "geography", "sender_country + receiver_country",
     "True when the sending and receiving banks are in the same country"),
    ("country_corridor", DICT, "geography", "sender_country + receiver_country",
     "Sender country -> receiver country, e.g. US->GB"),
    ("path_entries", pa.int8(), "routing", "bic8_path",
     "Number of entries in bic8_path; each '...' gap counts as one entry"),
    ("path_gaps", pa.int8(), "routing", "bic8_path",
     "Number of '...' gap entries in bic8_path (0 when continuity_qflag is true)"),
    ("path_first_differs", pa.bool_(), "routing", "bic8_path + ordering_bic8",
     "True when the first bank in bic8_path is not ordering_bic8"),
    ("path_last_differs", pa.bool_(), "routing", "bic8_path + beneficiary_bic8",
     "True when the last bank in bic8_path is not beneficiary_bic8"),
]
REQUIRED = ["epoch_seconds", "ordering_type", "beneficiary_type", "usd_amount",
            "ordering_bic8", "beneficiary_bic8", "bic8_path"]


def as_text(column: pa.ChunkedArray) -> pa.Array:
    """One column as plain strings, whether the clean file stores it as text or as a category.

    The clean step turns a text column into a category when it has few distinct
    values (clean.category_max_distinct), and the string functions used below
    do not accept category input.
    """
    array = column.combine_chunks()
    return array.cast(pa.string()) if pa.types.is_dictionary(array.type) else array


def build(table: pa.Table) -> dict[str, pa.Array]:
    """Compute every feature for one batch. Input columns are not modified."""
    out = {}

    ts = table["epoch_seconds"].combine_chunks()
    weekday = pc.day_of_week(ts)  # 0 = Monday
    out["utc_date"] = pc.cast(pc.floor_temporal(ts, unit="day"), pa.timestamp("us")).cast(pa.date32())
    out["utc_month"] = pc.month(ts).cast(pa.int8())
    out["utc_iso_week"] = pc.iso_week(ts).cast(pa.int8())
    out["utc_weekday"] = weekday.cast(pa.int8())
    out["utc_weekday_name"] = pa.DictionaryArray.from_arrays(weekday.cast(pa.int32()), pa.array(WEEKDAYS))
    out["utc_is_weekend"] = pc.greater_equal(weekday, 5)
    out["utc_hour"] = pc.hour(ts).cast(pa.int8())

    o = table["ordering_type"].to_pandas().astype("string")
    b = table["beneficiary_type"].to_pandas().astype("string")
    pair = pd.Categorical(o + "2" + b, categories=PAIRS)
    unexpected = int(pair.isna().sum() - (o.isna() | b.isna()).sum())
    if unexpected:
        seen = sorted((o + "2" + b).dropna().unique())
        raise SystemExit(f"\nordering_type/beneficiary_type contain values other than B and C: {seen}")
    out["party_pair"] = pa.DictionaryArray.from_pandas(pair, type=DICT)

    amount = table["usd_amount"].to_pandas().to_numpy(dtype="float64", na_value=np.nan)
    positive = amount > 0  # False for missing values too
    log10 = np.full(len(amount), np.nan)
    log10[positive] = np.log10(amount[positive])
    magnitude = EXPONENTS[np.clip(np.searchsorted(POWERS, amount, side="right") - 1, 0, len(POWERS) - 1)]
    out["usd_amount_log10"] = pa.array(log10, mask=~positive)
    out["usd_amount_magnitude"] = pa.array(magnitude.astype("int8"), mask=~positive)
    out["usd_amount_is_whole"] = pa.array(np.round(amount * 100) % 100 == 0, mask=np.isnan(amount))

    # geography: the country is characters 5-6 of the BIC8 (0-based slice 4:6), e.g. 0621US30 -> US
    obic = as_text(table["ordering_bic8"])
    bbic = as_text(table["beneficiary_bic8"])
    sender = pc.utf8_slice_codeunits(obic, 4, 6)
    receiver = pc.utf8_slice_codeunits(bbic, 4, 6)
    out["sender_country"] = sender.dictionary_encode()
    out["receiver_country"] = receiver.dictionary_encode()
    out["is_domestic"] = pc.equal(sender, receiver)
    out["country_corridor"] = pc.binary_join_element_wise(sender, receiver, "->").dictionary_encode()

    # routing: bic8_path is "BIC;BIC;...;BIC" where "..." marks a gap in the observed route
    path = as_text(table["bic8_path"])
    if path.null_count:
        raise SystemExit(f"\nbic8_path has {path.null_count} missing values - routing features expect none")
    parts = pc.split_pattern(path, ";")
    flat = pc.list_flatten(parts)
    offsets = parts.offsets.to_numpy()
    offsets = offsets - offsets[0]
    is_gap = pc.equal(flat, "...").to_numpy(zero_copy_only=False)
    gaps = np.add.reduceat(is_gap.astype(np.int64), offsets[:-1]) if len(path) else np.array([], dtype=np.int64)
    first = flat.take(pa.array(offsets[:-1]))
    last = flat.take(pa.array(offsets[1:] - 1))
    out["path_entries"] = pc.list_value_length(parts).cast(pa.int8())
    out["path_gaps"] = pa.array(gaps.astype("int8"))
    out["path_first_differs"] = pc.not_equal(first, obic)
    out["path_last_differs"] = pc.not_equal(last, bbic)
    return out


def run() -> None:
    cfg = S.load_config()
    if not S.CLEAN_PARQUET.exists():
        raise SystemExit("Run the clean step first (notebooks/01_preprocessing.ipynb).")
    pf = pq.ParquetFile(S.CLEAN_PARQUET)
    clean_schema = pf.schema_arrow

    missing = [c for c in REQUIRED if c not in clean_schema.names]
    if missing:
        raise SystemExit(f"Clean file is missing columns needed for features: {missing}. Columns found: {clean_schema.names}")
    if not pa.types.is_timestamp(clean_schema.field("epoch_seconds").type):
        raise SystemExit(f"epoch_seconds must be a datetime in the clean file, found {clean_schema.field('epoch_seconds').type}")
    if not pa.types.is_floating(clean_schema.field("usd_amount").type) and not pa.types.is_integer(clean_schema.field("usd_amount").type):
        raise SystemExit(f"usd_amount must be numeric in the clean file, found {clean_schema.field('usd_amount').type}")
    names = [f[0] for f in FEATURES]
    clash = [n for n in names if n in clean_schema.names]
    if clash:
        raise SystemExit(f"Feature names already exist in the clean file: {clash}")

    schema = clean_schema.remove_metadata()
    for name, dtype, *_ in FEATURES:
        schema = schema.append(pa.field(name, dtype))

    S.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"feature": n, "group": g, "type": str(t), "source": s, "description": d} for n, t, g, s, d in FEATURES]
    ).to_csv(FEATURE_DICTIONARY_CSV, index=False)

    total = pf.metadata.num_rows
    rows = 0
    nulls = dict.fromkeys(names, 0)
    tmp = FEATURES_PARQUET.with_suffix(".parquet.tmp")
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for batch in pf.iter_batches(batch_size=cfg["raw"]["chunk_rows"]):
                table = pa.Table.from_batches([batch])
                for name, array in build(table).items():
                    nulls[name] += array.null_count
                    table = table.append_column(name, array)
                writer.write_table(table.cast(schema))
                rows += table.num_rows
                print(f"\r  rows processed: {rows:,} / {total:,}", end="", flush=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print()
    if rows != total:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Integrity check FAILED, nothing saved: row count changed {total:,} -> {rows:,}")
    tmp.replace(FEATURES_PARQUET)

    print(f"Added {len(names)} features to {len(clean_schema.names)} clean columns = {len(schema.names)} columns, {rows:,} rows")
    with_nulls = {n: v for n, v in nulls.items() if v}
    print("Missing values in new features:", with_nulls if with_nulls else "none")
    print(f"Saved -> {FEATURES_PARQUET.relative_to(S.ROOT)}")
    print(f"Feature dictionary -> {FEATURE_DICTIONARY_CSV.relative_to(S.ROOT)}")