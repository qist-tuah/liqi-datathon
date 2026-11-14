"""Step 1 - raw CSV -> Parquet, lossless.

Every column is read as text so nothing is altered: leading zeros survive
("0009559"), and strings such as "NA" or "NULL" are not turned into missing
values. Types are decided later, in the clean step.
"""
import csv
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import settings as S


def detect_delimiter(header_line: str) -> str:
    counts = {d: header_line.count(d) for d in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get)


def normalise_names(original: list[str]) -> list[str]:
    """lower snake_case; blank headers become col_<position>; duplicates get _2, _3..."""
    out, seen = [], {}
    for i, name in enumerate(original):
        s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
        if not s:
            s = f"col_{i:02d}"
        seen[s] = seen.get(s, 0) + 1
        out.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    return out


def run() -> None:
    cfg = S.load_config()
    raw = cfg["raw"]
    path = S.find_raw_file(cfg)
    encoding = raw["encoding"]

    with open(path, encoding=encoding, newline="") as f:
        first_line = f.readline()
        delimiter = raw["delimiter"] or detect_delimiter(first_line)
        f.seek(0)
        original = next(csv.reader(f, delimiter=delimiter))
    names = normalise_names(original)

    S.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    S.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"position": range(len(names)), "original_name": original, "normalised_name": names}
    ).to_csv(S.COLUMN_MAPPING_CSV, index=False)

    print(f"File      : {path.name} ({path.stat().st_size / 1e6:,.1f} MB)")
    print(f"Delimiter : {delimiter!r}   Encoding: {encoding}   Columns: {len(names)}")

    schema = pa.schema([(n, pa.string()) for n in names])
    reader = pd.read_csv(
        path,
        sep=delimiter,
        header=0,
        names=names,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding=encoding,
        chunksize=raw["chunk_rows"],
    )
    rows = 0
    tmp = S.RAW_PARQUET.with_suffix(".parquet.tmp")
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for chunk in reader:
                chunk = chunk.fillna("")  # rows shorter than the header
                writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
                rows += len(chunk)
                print(f"\r  rows written: {rows:,}", end="", flush=True)
    except UnicodeDecodeError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"\nCould not decode the file as {encoding}: {e}\n"
            "Set raw.encoding in config/config.yaml (try cp1252) and run again."
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(S.RAW_PARQUET)
    print(f"\nSaved {rows:,} rows -> {S.RAW_PARQUET.relative_to(S.ROOT)}")
    print(f"Column name mapping -> {S.COLUMN_MAPPING_CSV.relative_to(S.ROOT)}")
