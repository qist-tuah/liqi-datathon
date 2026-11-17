"""Dashboard 1 - Anomaly types.

Run from the project root:   python dashboards/d01_anomalies.py
Output:                      reports/dashboards/D01_anomalies.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/02_anomaly_types.ipynb).
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash_style as ds  # noqa: E402
from src import eda  # noqa: E402

NAMES = {
    "ROMANCE_SCAM": "Romance scam",
    "FAT_FINGER_MISTAKE": "Fat-finger mistake",
    "CEO_SCAM": "CEO scam",
    "DUPLICATE_PAYMENT": "Duplicate payment",
    "ACCOUNT_REPLACEMENT_SCAM": "Account-replacement scam",
    "CUSTOMER_RECALL": "Customer recall",
    "NORMAL": "Normal payments",
}
RULES = {
    "ROMANCE_SCAM": "2-8 payments, rising",
    "FAT_FINGER_MISTAKE": "Amount 30x+ off, extra zeros",
    "CEO_SCAM": "Business, new payee, 3x usual",
    "DUPLICATE_PAYMENT": "Full copy within 2 hours",
    "ACCOUNT_REPLACEMENT_SCAM": "Usual amount, new payee",
    "CUSTOMER_RECALL": "No rule found",
    "NORMAL": "",
}
SCAMS = ["ROMANCE_SCAM", "CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM"]
FAR = 1.5  # |log10(amount / usual)| >= 1.5, i.e. about 30x larger or smaller


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    eda.connect(memory_limit=os.environ.get("DASH_MEMORY_LIMIT"))
    eda.require_columns("feat", ["path_last_differs"])
    q, con = eda.q, eda.con
    # Let DuckDB spill to disk instead of running out of memory (data/interim is gitignored)
    spill = ROOT / "data" / "interim" / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    d = {}

    # per type: rows, sender type, last-hop mismatch, duration on Direct routes
    d["types"] = q("""
        SELECT coalesce(anomaly_type, 'NORMAL')                                        AS type,
               count(*)                                                               AS n,
               100 * avg((ordering_type = 'C')::INT)                                  AS pct_sender_consumer,
               100 * avg(path_last_differs::INT)                                      AS pct_last_hop_differs,
               median(total_duration) FILTER (WHERE transaction_type_code = 'Direct') AS direct_median_s
        FROM feat GROUP BY 1
    """).set_index("type")

    # known payee: did the same sender also pay the same receiver in a normal payment?
    known_normal = q("""
        SELECT 100 * sum(c) FILTER (WHERE c > 1) / sum(c) AS pct
        FROM (SELECT count(*) AS c FROM feat WHERE is_anomaly = 0 GROUP BY ordering_account, beneficiary_account)
    """).pct[0]
    known = q("""
        WITH a AS (SELECT anomaly_type, ordering_account, beneficiary_account FROM feat WHERE is_anomaly = 1),
        p AS (   -- only the normal pairs that also appear among the anomalies: small, so it needs little memory
            SELECT DISTINCT n.ordering_account, n.beneficiary_account
            FROM feat n SEMI JOIN a USING (ordering_account, beneficiary_account)
            WHERE n.is_anomaly = 0
        )
        SELECT a.anomaly_type AS type, 100 * avg((p.ordering_account IS NOT NULL)::INT) AS pct
        FROM a LEFT JOIN p USING (ordering_account, beneficiary_account)
        GROUP BY 1
    """).set_index("type").pct
    known["NORMAL"] = known_normal
    d["types"]["pct_known_payee"] = known

    # amount compared with the sender's usual amount (median of the sender's normal payments)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE sender_usual AS
        SELECT ordering_account, median(usd_amount) AS usual FROM feat WHERE is_anomaly = 0 GROUP BY 1
    """)
    ratio = q("""
        SELECT f.anomaly_type AS type, log10(f.usd_amount / s.usual) AS log_ratio
        FROM feat f JOIN sender_usual s USING (ordering_account) WHERE f.is_anomaly = 1
    """)
    d["types"]["median_ratio"] = 10 ** ratio.groupby("type").log_ratio.median()
    normal_ratio = q(f"""
        SELECT median(f.usd_amount / s.usual)                                   AS median_ratio,
               100 * avg((abs(log10(f.usd_amount / s.usual)) >= {FAR})::INT)    AS pct_far
        FROM feat f JOIN sender_usual s USING (ordering_account) WHERE f.is_anomaly = 0
    """)
    d["types"].loc["NORMAL", "median_ratio"] = normal_ratio.median_ratio[0]
    d["normal_pct_far"] = normal_ratio.pct_far[0]
    d["normal_hist"] = q("""
        SELECT floor(log10(f.usd_amount / s.usual) / 0.25)::INT AS bin, count(*) AS n
        FROM feat f JOIN sender_usual s USING (ordering_account) WHERE f.is_anomaly = 0
        GROUP BY 1 ORDER BY 1
    """)
    d["fat_finger"] = ratio.loc[ratio.type == "FAT_FINGER_MISTAKE", "log_ratio"].to_numpy()

    # romance scams: the sequence of payments from each victim to the beneficiary
    d["romance"] = q("""
        SELECT ordering_account, beneficiary_account, usd_amount,
               row_number() OVER (PARTITION BY ordering_account, beneficiary_account ORDER BY epoch_seconds, usd_amount) AS payment_no
        FROM feat WHERE anomaly_type = 'ROMANCE_SCAM'
    """)

    # duplicates: groups identical in sender, receiver, amount, currency, bank path and duration
    d["dup"] = q("""
        WITH cand AS (   -- sender, receiver, amount, currency combinations that occur more than once
            SELECT ordering_account, beneficiary_account, usd_amount, currency_code
            FROM feat GROUP BY ALL HAVING count(*) > 1
        ),
        rep AS (         -- the few rows in those groups
            SELECT f.ordering_account, f.beneficiary_account, f.usd_amount, f.currency_code, f.bic8_path,
                   f.total_duration, f.anomaly_type, f.epoch_seconds
            FROM feat f SEMI JOIN cand USING (ordering_account, beneficiary_account, usd_amount, currency_code)
        ),
        g AS (
            SELECT count(*)                                                   AS copies,
                   count(*) FILTER (WHERE anomaly_type = 'DUPLICATE_PAYMENT') AS n_dup,
                   count(*) FILTER (WHERE anomaly_type = 'ROMANCE_SCAM')      AS n_romance,
                   epoch(max(epoch_seconds)) - epoch(min(epoch_seconds))      AS spread_s
            FROM rep
            GROUP BY ordering_account, beneficiary_account, usd_amount, currency_code, bic8_path, total_duration
            HAVING count(*) > 1
        )
        SELECT CASE WHEN n_dup = copies THEN 'duplicate' WHEN n_dup > 0 THEN 'partly'
                    WHEN n_romance > 0 THEN 'romance' ELSE 'unlabelled' END AS label,
               spread_s <= 7200 AS within_2h, count(*) AS n_groups, sum(copies) AS n_rows
        FROM g GROUP BY 1, 2 ORDER BY 1, 2
    """)
    return d


def pct(v: float) -> str:
    return "0%" if v == 0 else "100%" if v == 100 else f"{v:.1f}%"


def times(v: float) -> str:
    return f"{v:.0f}x" if v >= 10 else f"{v:.1f}x"


def draw(d: dict):
    t = d["types"]
    order = [k for k in t.sort_values("n", ascending=False).index if k != "NORMAL"]
    normal = t.loc["NORMAL"]
    n_anom, n_all = int(t.loc[order, "n"].sum()), int(t.n.sum())

    rom = d["romance"].sort_values(["ordering_account", "beneficiary_account", "payment_no"])
    victims = rom.groupby(["ordering_account", "beneficiary_account"])
    n_victims = victims.ngroups

    dup = d["dup"]
    assert not (dup.label == "partly").any(), "a duplicate group is only partly labelled"
    groups = lambda label, within: int(dup.loc[(dup.label == label) & (dup.within_2h == within), "n_groups"].sum())  # noqa: E731
    dup_in, dup_out = groups("duplicate", True), groups("duplicate", False)
    un_in, un_out, rom_in = groups("unlabelled", True), groups("unlabelled", False), groups("romance", True)
    precision = 100 * dup_in / (dup_in + un_in)
    assert dup_out == 0, "a labelled duplicate group is more than 2 hours apart"

    fig = ds.new_dashboard(
        1, "Anomalies: each type leaves its own fingerprint",
        f"{n_anom:,} anomalies in {n_all:,} payments   |   red = differs from normal payments")

    ds.kpi_row(fig, [
        (f"{n_anom:,}", f"anomalies ({100 * n_anom / n_all:.2f}%)"),
        ("5 of 6", "types decoded"),
        (f"{precision:.1f}%", "duplicate-rule precision"),
        (f"{n_victims}", "romance victims"),
    ])

    # ---------------------------------------------------------------- fingerprint table
    # layout: 3 equal columns; the table spans columns 1-2, so its edges line up with the two cards below it
    (x1, w1), (x2, w2), (x3, w3) = ds.columns(3)
    TOP_Y, TOP_H, BOT_Y, BOT_H = 0.40, 0.36, 0.085, 0.30
    ds.card(fig, [x1, TOP_Y, x2 + w2 - x1, TOP_H])    # fingerprint table
    ds.card(fig, [x3, TOP_Y, w3, TOP_H])              # romance
    ds.card(fig, [x1, BOT_Y, w1, BOT_H])              # fat finger
    ds.card(fig, [x2, BOT_Y, w2, BOT_H])              # slower scams
    ds.card(fig, [x3, BOT_Y, w3, BOT_H])              # duplicates
    PAD = 0.012                                       # inner margin of every card
    ax = ds.table_panel(fig, [x1 + PAD, 0.412, x2 + w2 - x1 - 2 * PAD, 0.265],
                        "Fingerprint of each type",
                        "medians and shares; red = differs from normal")
    cols = [  # (header, x of the right edge, formatter, highlight test)
        ("payments", 0.285, lambda r: f"{int(r.n):,}", lambda r: False),
        ("consumer\nsender", 0.375, lambda r: pct(r.pct_sender_consumer),
         lambda r: abs(r.pct_sender_consumer - normal.pct_sender_consumer) > 15),
        ("known\npayee", 0.470, lambda r: pct(r.pct_known_payee), lambda r: r.pct_known_payee < 1),
        ("amount vs\nusual", 0.565, lambda r: times(r.median_ratio),
         lambda r: r.median_ratio >= 2 or r.median_ratio <= 0.5),
        ("Direct-route\nseconds", 0.665, lambda r: f"{r.direct_median_s:,.0f} s",
         lambda r: r.direct_median_s >= 5 * normal.direct_median_s),
        ("last-hop\nmismatch", 0.765, lambda r: pct(r.pct_last_hop_differs),
         lambda r: r.pct_last_hop_differs == 0),
    ]
    rows = order + ["NORMAL"]
    row_h = 0.82 / len(rows)
    y_of = lambda i: 0.82 - (i + 0.5) * row_h  # noqa: E731
    ax.text(0.0, 0.91, "type", fontsize=9, color=ds.MUTED, va="center")
    ax.text(0.79, 0.91, "rule", fontsize=9, color=ds.MUTED, va="center")
    for header, x, _, _ in cols:
        ax.text(x, 0.91, header, fontsize=9, color=ds.MUTED, va="center", ha="right", linespacing=1.15)
    ax.plot([0, 1], [0.82, 0.82], color=ds.AXIS, lw=1)
    ax.plot([0, 1], [row_h, row_h], color=ds.AXIS, lw=1)
    for i, key in enumerate(rows):
        r, y = t.loc[key], y_of(i)
        is_normal = key == "NORMAL"
        ax.text(0.0, y, ds_name(key), fontsize=10.5, va="center", color=ds.INK2 if is_normal else ds.INK,
                fontweight="normal" if is_normal else "bold")
        for _, x, fmt, test in cols:
            hot = (not is_normal) and bool(test(r))
            if hot:
                ax.add_patch(Rectangle((x - 0.066, y - row_h * 0.36), 0.074, row_h * 0.72, facecolor=ds.TINT, edgecolor="none"))
            ax.text(x, y, fmt(r), fontsize=10.5, va="center", ha="right", color=ds.INK2 if is_normal else ds.INK,
                    fontweight="bold" if hot else "normal")
        ax.text(0.79, y, RULES[key], fontsize=9.5, va="center", color=ds.INK2)
        if 0 < i < len(rows) - 1:
            ax.plot([0, 1], [y + row_h / 2, y + row_h / 2], color=ds.GRID, lw=0.8)

    # ---------------------------------------------------------------- romance escalation
    by_no = rom.groupby("payment_no").usd_amount.quantile([0.25, 0.5, 0.75]).unstack()
    ax = ds.panel(fig, [x3 + 0.052, 0.44, w3 - 0.052 - PAD, 0.237],
                  "Romance: amounts escalate",
                  "median USD per payment", title_x=x3 + PAD)
    ax.fill_between(by_no.index, by_no[0.25], by_no[0.75], color=ds.ACCENT, alpha=0.12, linewidth=0)
    ax.plot(by_no.index, by_no[0.5], color=ds.ACCENT, marker="o", markersize=6, markeredgecolor=ds.SURFACE, markeredgewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(5, 20000)
    ax.set_yticks([10, 100, 1000, 10000], ["\\$10", "\\$100", "\\$1,000", "\\$10,000"])
    ax.minorticks_off()
    ax.set_xticks(list(by_no.index), [f"{k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}" for k in by_no.index])
    ax.set_xlim(0.6, by_no.index.max() + 0.6)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    first, last = by_no.index.min(), by_no.index.max()
    ax.annotate(f"\\${by_no.loc[first, 0.5]:,.0f}", (first, by_no.loc[first, 0.5]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold")
    ax.annotate(f"\\${by_no.loc[last, 0.5]:,.0f}", (last, by_no.loc[last, 0.5]), xytext=(-8, 6), textcoords="offset points",
                ha="right", fontsize=10, fontweight="bold")

    # ---------------------------------------------------------------- fat finger
    ff = d["fat_finger"]
    pct_far = 100 * (np.abs(ff) >= FAR).mean()
    edges = np.arange(-6, 8.01, 0.25)
    nh = d["normal_hist"]
    normal_density = (nh.set_index("bin").n.reindex(range(-24, 32), fill_value=0) / nh.n.sum() / 0.25).to_numpy()
    ff_density, _ = np.histogram(ff, bins=edges, density=True)
    ax = ds.panel(fig, [x1 + PAD, 0.12, w1 - 2 * PAD, 0.18],
                  f"Fat finger: {pct_far:.1f}% are 30x+ off",
                  "amount vs the sender's usual")
    ax.stairs(normal_density, edges, color=ds.CONTEXT, fill=True, alpha=0.45, linewidth=0)
    ax.stairs(normal_density, edges, color=ds.CONTEXT, linewidth=1.5, label="normal")
    ax.stairs(ff_density, edges, color=ds.ACCENT, linewidth=2, label="fat finger")
    ax.axvspan(-5, -FAR, color=ds.TINT, alpha=0.45, linewidth=0, zorder=0)
    ax.axvspan(FAR, 7, color=ds.TINT, alpha=0.45, linewidth=0, zorder=0)
    ax.set_xlim(-5, 7)
    ax.set_xticks([-4, -2, 0, 2, 4, 6], ["÷10,000", "÷100", "usual", "×100", "×10,000", "×1,000,000"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=1.4)
    ax.text(FAR + 0.15, ax.get_ylim()[1] * 0.50, "too large", fontsize=8.5, color=ds.INK2, va="center")
    ax.text(-FAR - 0.15, ax.get_ylim()[1] * 0.50, "too small", fontsize=8.5, color=ds.INK2, va="center", ha="right")

    # ---------------------------------------------------------------- duration on Direct routes
    dur = t.direct_median_s.sort_values()
    slow = dur[SCAMS] / normal.direct_median_s
    ax = ds.panel(fig, [x2 + 0.105, 0.12, w2 - 0.105 - 0.025, 0.18],
                  f"Scams are {slow.min():.0f}-{slow.max():.0f}x slower",
                  "median seconds, Direct route", title_x=x2 + PAD)
    colors = [ds.ACCENT if k in SCAMS else ds.CONTEXT for k in dur.index]
    ax.barh([ds_name(k, short=True) for k in dur.index], dur.values, color=colors, height=0.62)
    for y, v in enumerate(dur.values):
        ax.text(v + dur.max() * 0.02, y, f"{v:,.0f} s", va="center", fontsize=9.5)
    ax.set_xlim(0, dur.max() * 1.22)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)

    # ---------------------------------------------------------------- duplicate rule
    ax = ds.panel(fig, [x3 + 0.105, 0.12, w3 - 0.105 - 0.03, 0.18],
                  "Duplicate = copy within 2 h",
                  "groups of identical payments", title_x=x3 + PAD)
    labels = ["Unlabelled, later", "Romance, within 2 h", "Unlabelled, within 2 h", "Labelled, within 2 h"]
    values = [un_out, rom_in, un_in, dup_in]
    ax.barh(labels, values, color=[ds.CONTEXT, ds.CONTEXT, ds.CONTEXT, ds.ACCENT], height=0.62)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.03, y, f"{v:,}", va="center", fontsize=9.5)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    ds.footer(fig, "Not decoded: customer recall looks like a normal payment in all 12 attributes tested.")
    return fig


def ds_name(key: str, short: bool = False) -> str:
    name = NAMES[key]
    if short:
        name = name.replace(" (baseline)", "").replace("Account-replacement scam", "Account replacement")
    return name


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D01_anomalies")
