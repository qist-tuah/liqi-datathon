"""Dashboard 7 - Settlement duration.

Run from the project root:   python dashboards/d07_duration.py
Output:                      reports/dashboards/D07_duration.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/10_duration.ipynb).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

SLOW_SECONDS = 75           # boundary between the two humps of the duration histogram
NAMES = {"NORMAL": "Normal payments", "ROMANCE_SCAM": "Romance scam", "CEO_SCAM": "CEO scam",
         "ACCOUNT_REPLACEMENT_SCAM": "Account replacement", "DUPLICATE_PAYMENT": "Duplicate payment",
         "FAT_FINGER_MISTAKE": "Fat-finger mistake", "CUSTOMER_RECALL": "Customer recall"}


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}
    con.execute("""
        CREATE OR REPLACE TEMP TABLE direct AS
        SELECT split_part(bic8_path, ';', 1) AS a, string_split(bic8_path, ';')[-1] AS b, total_duration AS s, anomaly_type
        FROM feat WHERE participant_count = 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT a, b, count(*) AS n, min(s) AS base, avg(s) AS mean,
               count(*) FILTER (WHERE anomaly_type IS NULL) AS n_normal,
               min(s) FILTER (WHERE anomaly_type IS NULL)   AS base_normal
        FROM direct GROUP BY 1, 2
    """)
    d["hist"] = q("SELECT floor(log10(s) * 10) / 10 AS bin, count(*) AS n FROM direct WHERE anomaly_type IS NULL GROUP BY 1 ORDER BY 1")
    d["fast"] = float(q(f"SELECT 100 * avg((s <= {SLOW_SECONDS})::INT) AS p FROM direct WHERE anomaly_type IS NULL").p[0])
    d["var"] = q("""
        SELECT var_pop(x.s) AS total, avg((x.s - p.mean) * (x.s - p.mean)) AS within
        FROM direct x JOIN base p USING (a, b)
    """).iloc[0]
    # random delay = seconds above the pair's fastest payment (pairs with 1,000+ payments, so the fastest is a reliable base)
    d["delay"] = q("SELECT x.s - p.base AS delay, count(*) AS n FROM direct x JOIN base p USING (a, b) WHERE p.n >= 1000 GROUP BY 1 ORDER BY 1")
    d["busy"] = q("""
        SELECT a, b, count(*) AS n, min(s) AS fastest, median(s) AS median_s, max(s) AS slowest
        FROM direct WHERE anomaly_type IS NULL GROUP BY 1, 2 ORDER BY n DESC LIMIT 8
    """)
    d["slow_pairs"] = float(q("SELECT 100 * avg((base >= 650)::INT) AS p FROM base WHERE n >= 30").p[0])
    # routes with intermediaries: duration vs the sum of the base times of their links
    con.execute("""
        CREATE OR REPLACE TEMP TABLE multi AS
        SELECT string_split(bic8_path, ';') AS h, participant_count AS k, total_duration AS s, transaction_type_code AS t
        FROM feat WHERE participant_count IN (3, 4) AND path_gaps = 0 AND anomaly_type IS NULL
    """)
    d["sum"] = q("""
        SELECT k, t, count(*) AS n, 100 * avg((abs(s - pred) <= 30)::INT) AS pct_within_30s FROM (
            SELECT 3 AS k, m.t, m.s, l1.base + l2.base AS pred FROM multi m
            JOIN base l1 ON m.h[1] = l1.a AND m.h[2] = l1.b JOIN base l2 ON m.h[2] = l2.a AND m.h[3] = l2.b
            WHERE m.k = 3 AND l1.n >= 30 AND l2.n >= 30
            UNION ALL
            SELECT 4, m.t, m.s, l1.base + l2.base + l3.base FROM multi m
            JOIN base l1 ON m.h[1] = l1.a AND m.h[2] = l1.b JOIN base l2 ON m.h[2] = l2.a AND m.h[3] = l2.b
            JOIN base l3 ON m.h[3] = l3.a AND m.h[4] = l3.b
            WHERE m.k = 4 AND l1.n >= 30 AND l2.n >= 30 AND l3.n >= 30
        ) GROUP BY 1, 2 ORDER BY 1, 2 DESC
    """)
    # anomalies: duration vs the normal time of the same pair of banks (pairs that normal payments also use)
    d["anom"] = q("""
        SELECT coalesce(x.anomaly_type, 'NORMAL') AS type, count(*) AS n, median(x.s - p.base_normal) AS above_s
        FROM direct x JOIN base p USING (a, b) WHERE p.n_normal >= 5 GROUP BY 1
    """).set_index("type")
    d["anom_all"] = q("""
        SELECT coalesce(anomaly_type, 'NORMAL') AS type, count(*) AS n, median(s) AS median_s
        FROM direct GROUP BY 1
    """).set_index("type")
    return d


def draw(d: dict):
    h = d["hist"]
    n_direct = int(h.n.sum())
    fast = d["fast"]
    sd_all, sd_pair = float(np.sqrt(d["var"].total)), float(np.sqrt(d["var"].within))
    dl = d["delay"]
    n_delay = int(dl.n.sum())
    mean_delay = float((dl.delay * dl.n).sum() / n_delay)
    sm = d["sum"]
    within = float((sm.pct_within_30s * sm.n).sum() / sm.n.sum())

    fig = ds.new_dashboard(7, "Duration: a fixed time per bank pair, plus a few seconds",
                           f"{n_direct:,} normal 2-bank payments for the shapes   |   red = differs from normal")
    ds.kpi_row(fig, [
        (f"{sd_pair:.1f} s", f"spread within a bank pair ({sd_all:,.0f} s overall)"),
        (f"{mean_delay:.0f} s", "random delay on top (mean)"),
        (f"{within:.1f}%", "of 3-4 bank routes = sum of links ±30 s"),
        (f"{d['slow_pairs']:.1f}%", "of bank pairs take 650 s or more"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- two speeds (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.03, right=0.02), "Two speeds: seconds, or minutes to hours",
                  "normal 2-bank payments by total_duration, log scale", title_x=R["wide"][0] + ds.PAD)
    left = 10 ** h.bin
    ax.bar(left, h.n / n_direct * 100, width=10 ** (h.bin + 0.1) - left, align="edge",
           color=[ds.ACCENT if v < SLOW_SECONDS else ds.CONTEXT for v in left], edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(SLOW_SECONDS, color=ds.INK2, ls="--", lw=1)
    peak = (h.n / n_direct * 100).max()
    ax.set_ylim(0, peak * 1.25)
    ax.text(SLOW_SECONDS / 1.4, peak * 1.12, f"{fast:.1f}% under {SLOW_SECONDS} s", ha="right", fontsize=10.5, fontweight="bold", color=ds.ACCENT)
    ax.text(SLOW_SECONDS * 1.4, peak * 1.12, f"{100 - fast:.1f}% slower", ha="left", fontsize=10.5, fontweight="bold", color=ds.INK2)
    ax.set_xticks([1, 10, 60, 600, 3600, 10_000], ["1 s", "10 s", "1 min", "10 min", "1 h", "10,000 s"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- random delay (side)
    shown = dl[dl.delay <= 45]
    ax = ds.panel(fig, ds.inner(R["side"], left=0.02, right=0.02), "Base + a random delay",
                  "seconds above the pair's base; line = Gamma(2, 5 s)", title_x=R["side"][0] + ds.PAD)
    ax.bar(shown.delay, shown.n / n_delay * 100, width=0.85, color=ds.ACCENT)
    xs = np.linspace(0, 45, 300)
    ax.plot(xs + 0.5, 100 * xs / 25 * np.exp(-xs / 5), color=ds.INK, lw=1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.98, 0.9, f"mean {mean_delay:.1f} s\n(Gamma: 10 s)", transform=ax.transAxes, ha="right", va="top", fontsize=9.5)

    # ---------------------------------------------------------------- busy pairs (b1)
    b = d["busy"].sort_values("median_s")
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.09, right=0.02), "Each pair has its own time",
                  "8 busiest pairs: fastest to slowest payment", title_x=R["b1"][0] + ds.PAD)
    for y, r in enumerate(b.itertuples()):
        ax.plot([r.fastest, r.slowest], [y, y], color=ds.ACCENT, lw=5, solid_capstyle="round")
        ax.text(r.slowest * 1.15, y, f"{r.fastest:,}-{r.slowest:,} s", va="center", fontsize=8.5, color=ds.INK2)
    ax.set_yticks(range(len(b)), [f"{r.a} → {r.b}" for r in b.itertuples()], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(3, 20_000)
    ax.set_xticks([10, 100, 1000, 10_000], ["10 s", "100 s", "1,000 s", "10,000 s"])
    ax.minorticks_off()
    ax.set_ylim(-0.6, len(b) - 0.4)

    # ---------------------------------------------------------------- additivity (b2)
    labels = [f"{k} banks, {t}" for k, t in zip(sm.k, sm.t)]
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.075, right=0.02, bottom=0.06), "Routes add up their links",
                  "% within 30 s of the sum of the links' base times", title_x=R["b2"][0] + ds.PAD)
    ds.hbars(ax, labels, sm.pct_within_30s, ds.ACCENT, lambda v: f"{v:.1f}%", xmax=100, fontsize=9.5)
    fig.text(R["b2"][0] + ds.PAD, R["b2"][1] + 0.025, f"{int(sm.n.sum()):,} gap-free normal routes whose links also occur as 2-bank payments",
             fontsize=8.5, color=ds.INK2, va="center")

    # ---------------------------------------------------------------- anomalies (b3)
    an, al = d["anom"], d["anom_all"]
    order = ["ROMANCE_SCAM", "CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "NORMAL"]
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.075, right=0.02, bottom=0.075), "Scams are slow only because of their banks",
                  "median seconds above the pair's normal time", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, [NAMES[k] for k in order], [an.loc[k, "above_s"] for k in order],
             [ds.CONTEXT if k == "NORMAL" else ds.ACCENT for k in order], lambda v: f"+{v:.0f} s", xmax=30, fontsize=9.5)
    shared = 100 * sum(an.loc[k, "n"] for k in order[:3]) / sum(al.loc[k, "n"] for k in order[:3])
    fig.text(R["b3"][0] + ds.PAD, R["b3"][1] + 0.033, f"Overall median: {al.loc['ROMANCE_SCAM', 'median_s']:,.0f} s romance vs {al.loc['NORMAL', 'median_s']:,.0f} s normal.\n"
                      f"Only {shared:.0f}% of scam rows use a pair normal payments also use.",
             fontsize=8.5, color=ds.INK2, va="center")

    ds.footer(fig, "Hour, weekend, gaps, currency conversions and the Covered label add no time. "
                   "Not explained: what makes a bank pair fast or slow.")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D07_duration")
