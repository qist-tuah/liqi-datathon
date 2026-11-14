"""Step 3 - typed, analysis-ready Parquet.

Types are inferred from the full-file profile (not from a sample), written to
reports/profiling/type_plan.csv for review, and can be overridden in
config/config.yaml (clean.force_types). Empty strings become missing values.
No rows are dropped and no values are edited.
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import settings as S

EPOCH_S = (946_684_800, 4_102_444_800)  # 2000-01-01 .. 2100-01-01 in seconds
EPOCH_MS = (EPOCH_S[0] * 1000, EPOCH_S[1] * 1000)
TRUE, FALSE = {"Y", "YES", "TRUE", "T", "1"}, {"N", "NO", "FALSE", "F", "0"}
ARROW = {
    "string": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "category": pa.dictionary(pa.int32(), pa.string()),
    "datetime_s": pa.timestamp("us", tz="UTC"),
    "datetime_ms": pa.timestamp("us", tz="UTC"),
}


def infer_type(c: dict, category_max: int) -> tuple[str, str]:
    """Return (type, reason) for one profiled column."""
    n = c["non_empty"]
    if n == 0:
        return "string", "column is entirely empty"
    values = c["all_values"]
    if values is not None and {v.upper() for v in values} <= {"Y", "N"}:
        return "bool", "only Y/N values"
    if c["numeric"] == n:
        if c["leading_zero"] > 0:
            return "string", f"{c['leading_zero']} numeric-looking values have leading zeros (identifier)"
        lo, hi = c["numeric_min"], c["numeric_max"]
        if EPOCH_S[0] <= lo and hi <= EPOCH_S[1]:
            return "datetime_s", "all values fall in the Unix-seconds range 2000-2100"
        if EPOCH_MS[0] <= lo and hi <= EPOCH_MS[1]:
            return "datetime_ms", "all values fall in the Unix-milliseconds range 2000-2100"
        if c["integer"] == n:
            if c["max_len"] > 18:
                return "string", "integers too long for int64"
            return "int", "all values are integers"
        return "float", "all values are numeric"
    if isinstance(c["distinct"], int) and c["distinct"] <= category_max:
        return "category", f"{c['distinct']} distinct text values"
    return "string", "free text / identifier"


def cast(s: pd.Series, kind: str, categories: list[str] | None) -> pd.Series:
    s = s.astype("string")
    s = s.mask(s == "", pd.NA)
    if kind == "string":
        return s
    if kind == "category":
        return pd.Series(pd.Categorical(s, categories=categories), index=s.index)
    if kind == "bool":
        up = s.str.upper()
        bad = up.notna() & ~up.isin(TRUE | FALSE)
        if bad.any():
            raise ValueError(f"values that are not boolean: {sorted(s[bad].unique())[:10]}")
        return up.map(lambda x: pd.NA if pd.isna(x) else x in TRUE).astype("boolean")
    num = pd.to_numeric(s, errors="raise")
    if kind == "int":
        return num.astype("Int64")
    as_float = pd.Series(num.astype("Float64").to_numpy(dtype="float64", na_value=np.nan), index=s.index)
    if kind == "float":
        return as_float
    if kind in ("datetime_s", "datetime_ms"):
        return pd.to_datetime(as_float, unit="s" if kind == "datetime_s" else "ms", utc=True)
    raise ValueError(f"unknown type {kind!r}")


def run() -> None:
    cfg = S.load_config()
    c = cfg["clean"]
    if not S.PROFILE_JSON.exists():
        raise SystemExit("Run the profile step first: python run_pipeline.py profile")
    profile = json.loads(S.PROFILE_JSON.read_text(encoding="utf-8"))
    prof = {col["column"]: col for col in profile["columns"]}

    drop, force, rename = set(c["drop_columns"] or []), c["force_types"] or {}, c["rename"] or {}
    for name in list(drop) + list(force) + list(rename):
        if name not in prof:
            raise SystemExit(f"config.yaml refers to unknown column {name!r}. Valid names: {list(prof)}")
    for name, kind in force.items():
        if kind not in ARROW:
            raise SystemExit(f"force_types.{name}: {kind!r} is not one of {list(ARROW)}")

    plan = []
    for name, col in prof.items():
        if name in drop:
            plan.append({"column": name, "output_name": "", "type": "DROPPED", "reason": "clean.drop_columns"})
            continue
        kind, reason = (force[name], "clean.force_types") if name in force else infer_type(col, c["category_max_distinct"])
        if kind == "category" and col["all_values"] is None:
            raise SystemExit(f"{name}: too many distinct values to be a category")
        plan.append({"column": name, "output_name": rename.get(name, name), "type": kind, "reason": reason})
    pd.DataFrame(plan).to_csv(S.TYPE_PLAN_CSV, index=False)
    kept = [p for p in plan if p["type"] != "DROPPED"]

    label = c["label_column"]
    add_label = bool(label) and label in prof and label not in drop
    if label and not add_label:
        print(f"Note: label column {label!r} not found - is_anomaly not created.")

    fields = [(p["output_name"], ARROW[p["type"]]) for p in kept]
    if add_label:
        fields.append(("is_anomaly", pa.int8()))
    out_names = [f[0] for f in fields]
    if len(set(out_names)) != len(out_names):
        raise SystemExit(f"Duplicate output column names after rename: {out_names}")
    schema = pa.schema(fields)

    S.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(S.RAW_PARQUET)
    if pf.metadata.num_rows != profile["rows"]:
        raise SystemExit("profile.json is out of date for the raw Parquet file - run the profile step again.")
    nulls = {p["output_name"]: 0 for p in kept}
    rows = 0
    tmp = S.CLEAN_PARQUET.with_suffix(".parquet.tmp")
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for batch in pf.iter_batches(batch_size=cfg["raw"]["chunk_rows"], columns=[p["column"] for p in kept]):
                raw = batch.to_pandas()
                out = {}
                for p in kept:
                    try:
                        col = cast(raw[p["column"]], p["type"], prof[p["column"]]["all_values"])
                    except Exception as e:
                        raise SystemExit(f"\nColumn {p['column']!r} could not be cast to {p['type']}: {e}")
                    out[p["output_name"]] = col
                    nulls[p["output_name"]] += int(col.isna().sum())
                df = pd.DataFrame(out)
                if add_label:
                    df["is_anomaly"] = df[rename.get(label, label)].notna().astype("int8")
                writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
                rows += len(df)
                print(f"\r  rows cleaned: {rows:,} / {profile['rows']:,}", end="", flush=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print()

    # Integrity checks: same row count, and casting created no new missing values.
    problems = []
    if rows != profile["rows"]:
        problems.append(f"row count changed: {profile['rows']:,} -> {rows:,}")
    for p in kept:
        expected = prof[p["column"]]["empty"]
        if nulls[p["output_name"]] != expected:
            problems.append(f"{p['column']}: {expected:,} empty in raw but {nulls[p['output_name']]:,} missing after cast")
    if problems:
        tmp.unlink(missing_ok=True)
        raise SystemExit("Integrity check FAILED, nothing saved:\n  " + "\n  ".join(problems))
    tmp.replace(S.CLEAN_PARQUET)

    summary = pd.DataFrame(
        [{"column": p["output_name"], "type": p["type"], "missing": nulls[p["output_name"]],
          "missing_pct": round(100 * nulls[p["output_name"]] / rows, 4) if rows else 0.0, "reason": p["reason"]} for p in kept]
    )
    summary.to_csv(S.CLEAN_SUMMARY_CSV, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_colwidth", 70):
        print(summary.to_string(index=False))
    print(f"\nIntegrity checks passed. Saved {rows:,} rows -> {S.CLEAN_PARQUET.relative_to(S.ROOT)}")
    print(f"Type plan -> {S.TYPE_PLAN_CSV.relative_to(S.ROOT)}")
