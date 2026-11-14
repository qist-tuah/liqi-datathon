"""Step 2 - profile every column of the raw Parquet file (streamed, low memory).

Writes reports/profiling/profile.json (full detail, used by the clean step)
and profile_summary.csv (one row per column, easy to read or share).
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import settings as S

INT_RE = r"[+-]?\d+"
NUM_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
LEADING_ZERO_RE = r"[+-]?0\d.*"  # "007", "00.5" - but not "0" or "0.5"


class ColumnStats:
    def __init__(self, name: str, max_distinct: int):
        self.name = name
        self.max_distinct = max_distinct
        self.n = self.n_empty = self.n_whitespace_only = self.n_padded = 0
        self.n_numeric = self.n_integer = self.n_leading_zero = 0
        self.min_len = self.max_len = None
        self.num_min = self.num_max = None
        self.counter: Counter | None = Counter()

    def update(self, s: pd.Series) -> None:
        self.n += len(s)
        empty = s == ""
        self.n_empty += int(empty.sum())
        v = s[~empty]
        if v.empty:
            return
        stripped = v.str.strip()
        self.n_whitespace_only += int((stripped == "").sum())
        self.n_padded += int(((stripped != v) & (stripped != "")).sum())

        lens = v.str.len()
        self.min_len = int(lens.min()) if self.min_len is None else min(self.min_len, int(lens.min()))
        self.max_len = int(lens.max()) if self.max_len is None else max(self.max_len, int(lens.max()))

        is_num = v.str.fullmatch(NUM_RE)
        self.n_numeric += int(is_num.sum())
        self.n_integer += int(v.str.fullmatch(INT_RE).sum())
        self.n_leading_zero += int(v.str.fullmatch(LEADING_ZERO_RE).sum())
        if is_num.any():
            nums = pd.to_numeric(v[is_num])
            lo, hi = float(nums.min()), float(nums.max())
            self.num_min = lo if self.num_min is None else min(self.num_min, lo)
            self.num_max = hi if self.num_max is None else max(self.num_max, hi)

        if self.counter is not None:
            self.counter.update(v.value_counts().to_dict())
            if len(self.counter) > self.max_distinct:
                self.counter = None  # high cardinality: stop tracking values

    def result(self, top_n: int) -> dict:
        non_empty = self.n - self.n_empty
        tracked = self.counter is not None
        return {
            "column": self.name,
            "rows": self.n,
            "empty": self.n_empty,
            "empty_pct": round(100 * self.n_empty / self.n, 4) if self.n else 0.0,
            "non_empty": non_empty,
            "whitespace_only": self.n_whitespace_only,
            "padded_with_whitespace": self.n_padded,
            "distinct": len(self.counter) if tracked else f">{self.max_distinct}",
            "min_len": self.min_len,
            "max_len": self.max_len,
            "numeric": self.n_numeric,
            "integer": self.n_integer,
            "leading_zero": self.n_leading_zero,
            "numeric_min": self.num_min,
            "numeric_max": self.num_max,
            "top_values": dict(self.counter.most_common(top_n)) if tracked else None,
            "all_values": sorted(self.counter) if tracked and len(self.counter) <= 1000 else None,
        }


def run() -> None:
    cfg = S.load_config()
    p = cfg["profile"]
    if not S.RAW_PARQUET.exists():
        raise SystemExit("Run the convert step first: python run_pipeline.py convert")

    pf = pq.ParquetFile(S.RAW_PARQUET)
    stats = {n: ColumnStats(n, p["max_distinct_tracked"]) for n in pf.schema_arrow.names}
    hashes, rows = [], 0
    for batch in pf.iter_batches(batch_size=cfg["raw"]["chunk_rows"]):
        df = batch.to_pandas()
        for name, st in stats.items():
            st.update(df[name])
        if p["count_duplicates"]:
            hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy())
        rows += len(df)
        print(f"\r  rows profiled: {rows:,} / {pf.metadata.num_rows:,}", end="", flush=True)
    print()

    duplicates = None
    if p["count_duplicates"] and hashes:
        duplicates = int(rows - len(np.unique(np.concatenate(hashes))))

    columns = [st.result(p["top_n"]) for st in stats.values()]
    profile = {"rows": rows, "n_columns": len(columns), "exact_duplicate_rows": duplicates, "columns": columns}
    S.PROFILE_JSON.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = pd.DataFrame(columns).drop(columns=["all_values"])
    summary["top_values"] = summary["top_values"].map(
        lambda d: "" if d is None else " | ".join(f"{k!r}: {v}" for k, v in list(d.items())[:5])
    )
    summary.to_csv(S.PROFILE_CSV, index=False, encoding="utf-8-sig")

    print(f"Rows: {rows:,}   Columns: {len(columns)}   Exact duplicate rows: {duplicates}")
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_columns", None):
        print(summary[["column", "empty_pct", "distinct", "min_len", "max_len", "numeric", "leading_zero"]].to_string(index=False))
    print(f"\nFull profile -> {S.PROFILE_JSON.relative_to(S.ROOT)}")
    print(f"Summary      -> {S.PROFILE_CSV.relative_to(S.ROOT)}")
