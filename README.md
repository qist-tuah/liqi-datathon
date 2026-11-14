# SWIFT UMDAC Datathon 2026

## Folder structure

```
swift-umdac-datathon-2026/
├── run_pipeline.py          entry point: python run_pipeline.py [convert|profile|clean|all]
├── requirements.txt
├── config/config.yaml       every setting (file name, chunk size, type overrides, label column)
├── data/
│   ├── raw/                 the original CSV. Never edit or re-save it (especially not from Excel)
│   ├── interim/             transactions_raw.parquet   - lossless copy, every column as text
│   └── processed/           transactions_clean.parquet - typed, analysis-ready
├── src/
│   ├── settings.py          paths + config loading
│   ├── convert.py           step 1: CSV -> Parquet in chunks
│   ├── profiling.py         step 2: per-column profile of the whole file
│   └── clean.py             step 3: type casting + integrity checks
├── reports/
│   ├── profiling/           column_mapping.csv, profile.json, profile_summary.csv,
│   │                        type_plan.csv, clean_summary.csv
│   └── figures/             charts for the final presentation
├── notebooks/               EDA and modelling notebooks (01_eda.ipynb, 02_features.ipynb, ...)
├── models/                  saved models
├── submissions/             files handed to the organisers
└── tests/make_synthetic_sample.py   fake CSV for a smoke test
```

## Setup (Windows)

```
cd swift-umdac-datathon-2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

1. Copy the dataset CSV into `data/raw/` (it must be the only `.csv` there, or set `raw.file` in `config/config.yaml`).
2. `python run_pipeline.py all`

## What each step does

| Step | Reads | Writes | Notes |
|---|---|---|---|
| convert | `data/raw/*.csv` | `data/interim/transactions_raw.parquet`, `column_mapping.csv` | Chunked, so file size is not limited by RAM. Everything stays text: leading zeros are kept and strings like `NA` are not treated as missing. Headers become lower snake_case; a blank header becomes `col_<position>`. |
| profile | raw Parquet | `profile.json`, `profile_summary.csv` | Per column: empty count, distinct count, length range, numeric/integer counts, leading-zero count, whitespace padding, top values. Also counts exact duplicate rows. |
| clean | raw Parquet + `profile.json` | `data/processed/transactions_clean.parquet`, `type_plan.csv`, `clean_summary.csv` | Infers a type per column from the full-file profile, casts, converts empty strings to missing, adds `is_anomaly`. Fails without saving if the row count changes or a cast creates a missing value. |

## Type inference rules (clean step), in order

1. Only `Y`/`N` values -> `bool`
2. Every value numeric, none with a leading zero:
   - all within Unix seconds for 2000-2100 -> `datetime_s` (UTC); milliseconds -> `datetime_ms`
   - all integers -> `int`, otherwise `float`
3. Numeric-looking with leading zeros -> `string` (identifier)
4. At most `category_max_distinct` (50) distinct values -> `category`
5. Anything else -> `string`

Override any column in `config/config.yaml` under `clean.force_types`, then re-run `python run_pipeline.py clean`.

## Deliberately not done

No rows are removed (duplicates are counted only, since repeated transactions can be a signal),
no whitespace is stripped, no outliers are touched, nothing is imputed.

## Load the result

```python
import pandas as pd
df = pd.read_parquet("data/processed/transactions_clean.parquet")
```
