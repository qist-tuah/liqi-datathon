"""Dashboard 6 - Routing and payment path.

Run from the project root:   python dashboards/d06_routing.py
Output:                      reports/dashboards/D06_routing.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/09_routing.ipynb).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

SEGMENTS = ["domestic", "cross-border", "B2B", "C2C", "weekday", "weekend", "under $100", "$100K and more"]


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}
    con.execute("""
        CREATE OR REPLACE TEMP TABLE route AS
        SELECT participant_count                                    AS banks,
               transaction_type_code                                AS route_type,
               path_gaps                                            AS gaps,
               path_first_differs, path_last_differs,
               ordering_bic8, beneficiary_bic8,
               split_part(bic8_path, ';', 1)                        AS first_bank,
               string_split(bic8_path, ';')[-1]                     AS last_bank,
               sender_country, receiver_country, is_domestic, party_pair, utc_is_weekend, usd_amount_magnitude
        FROM feat
    """)
    d["banks"] = q("SELECT banks, count(*) AS n FROM route GROUP BY 1 ORDER BY 1")

    # share of 2-bank routes in segments that could plausibly change the route
    d["segments"] = q("""
        SELECT segment, count(*) AS n, 100 * avg((banks = 2)::INT) AS pct_two_banks FROM (
            SELECT CASE WHEN is_domestic THEN 'domestic' ELSE 'cross-border' END AS segment, banks FROM route
            UNION ALL SELECT party_pair, banks FROM route WHERE party_pair IN ('B2B', 'C2C')
            UNION ALL SELECT CASE WHEN utc_is_weekend THEN 'weekend' ELSE 'weekday' END, banks FROM route
            UNION ALL SELECT CASE WHEN usd_amount_magnitude <= 1 THEN 'under $100' ELSE '$100K and more' END, banks
                      FROM route WHERE usd_amount_magnitude <= 1 OR usd_amount_magnitude >= 5
        ) GROUP BY 1
    """).set_index("segment")

    # Covered or Indirect: one fixed answer per pair (first bank, last bank) of the path
    d["cover"] = q("""
        WITH p AS (SELECT first_bank, last_bank, count(*) AS n, sum((route_type = 'Covered')::INT) AS c
                   FROM route WHERE banks >= 3 GROUP BY 1, 2)
        SELECT count(*) AS pairs, sum((c = 0)::INT) AS never, sum((c = n)::INT) AS always,
               sum((c > 0 AND c < n)::INT) AS mixed, sum((n >= 2)::INT) AS pairs_2_plus
        FROM p
    """).iloc[0]
    d["mixed_bic"] = int(q("""
        SELECT count(*) AS n FROM (
            SELECT ordering_bic8, beneficiary_bic8 FROM route WHERE banks >= 3 GROUP BY 1, 2
            HAVING sum((route_type = 'Covered')::INT) > 0 AND sum((route_type = 'Covered')::INT) < count(*))
    """).n[0])

    # which of the two disagreeing banks is the odd one out: the pair that loses the same-country preference
    d["deviate"] = q("""
        SELECT CASE WHEN NOT path_first_differs AND NOT path_last_differs THEN 'match'
                    WHEN path_first_differs AND NOT path_last_differs     THEN 'first'
                    WHEN NOT path_first_differs AND path_last_differs     THEN 'last'
                    ELSE 'both' END                                               AS kind,
               count(*)                                                           AS n,
               100 * avg((sender_country = receiver_country)::INT)                AS bic_pair,
               100 * avg((substr(first_bank, 5, 2) = substr(last_bank, 5, 2))::INT) AS path_pair
        FROM route GROUP BY 1
    """).set_index("kind")
    d["chance"] = float(q("""
        WITH s AS (SELECT sender_country AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM route GROUP BY 1),
             r AS (SELECT receiver_country AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM route GROUP BY 1)
        SELECT 100 * sum(s.p * r.p) AS chance FROM s JOIN r USING (c)
    """).chance[0])
    d["ends"] = q("SELECT 100 * avg(path_first_differs::INT) AS first_pct, 100 * avg(path_last_differs::INT) AS last_pct FROM route").iloc[0]

    # gaps: one probability per hand-off between two banks
    d["gaps"] = q("SELECT banks, count(*) AS n, 100 * avg((gaps > 0)::INT) AS pct_with_gap FROM route WHERE banks <= 7 GROUP BY 1 ORDER BY 1")
    d["gap_total"] = q("SELECT sum(gaps) AS gaps, sum(banks - 1) AS links FROM route").iloc[0]
    return d


def draw(d: dict):
    dist = d["banks"]
    n = int(dist.n.sum())
    mean_extra = float(((dist.banks - 2) * dist.n).sum() / n)
    r_hat = mean_extra / (1 + mean_extra)                  # each extra bank is r_hat times as likely as one fewer
    R_RULE = math.exp(-1)
    dist["predicted"] = n * (1 - R_RULE) * R_RULE ** (dist.banks - 2)
    fit = dist[dist.predicted >= 5]
    z = (fit.n - fit.predicted) / np.sqrt(fit.predicted)
    cov = d["cover"]
    per_link = d["gap_total"].gaps / d["gap_total"].links
    g = per_link / (1 + per_link)                           # gap probability per hand-off
    seg = d["segments"].reindex(SEGMENTS)
    base = 100 * float(dist.loc[dist.banks == 2, "n"].sum()) / n
    spread = float((seg.pct_two_banks - base).abs().max())

    fig = ds.new_dashboard(6, "Routing: four random draws build every route",
                           f"{n:,} payments   |   red = differs from the rest")
    ds.kpi_row(fig, [
        (f"{r_hat:.3f}", "odds of each extra bank (1/e)"),
        (f"{int(cov.mixed)}", f"of {int(cov.pairs):,} bank pairs mix Covered and Indirect"),
        (f"{d['ends'].first_pct:.1f}%", "routes not starting at the sender's bank"),
        (f"{100 * g:.1f}%", "gap chance at each hand-off"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- number of banks (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.05, right=0.02),
                  "Number of banks: 2 + a random number that falls by 1/e each step",
                  f"payments by number of banks, log scale; all {len(fit)} lengths up to {int(fit.banks.max())} banks within "
                  f"{z.abs().max():.1f} standard deviations of the rule",
                  title_x=R["wide"][0] + ds.PAD)
    ax.bar(dist.banks, dist.n, color=ds.ACCENT, width=0.7, label="observed")
    ax.plot(fit.banks, fit.predicted, color=ds.INK, lw=1.4, marker="o", ms=4,
            label="rule: 2 + exponential(mean 1), rounded down")
    for row in dist[dist.banks <= 6].itertuples():
        ax.text(row.banks + 0.12, row.n * 2.2, f"{row.n / n * 100:.1f}%", ha="left", fontsize=9, color=ds.INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.5, n * 6)
    ax.set_xticks(dist.banks)
    ax.set_yticks([1, 100, 10_000, 1_000_000], ["1", "100", "10K", "1M"])
    ax.legend(loc="upper right")

    # ---------------------------------------------------------------- Covered (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.085, right=0.02, bottom=0.075), "Covered is fixed per bank pair",
                  "pairs of first and last bank in the path, 3+ banks", title_x=R["side"][0] + ds.PAD)
    ds.hbars(ax, ["always Covered", "never Covered", "both labels"], [cov.always, cov.never, cov.mixed],
             [ds.ACCENT, ds.CONTEXT, ds.HOT], lambda v: f"{int(v):,}", fontsize=10)
    fig.text(R["side"][0] + ds.PAD, R["side"][1] + 0.035,
             f"{int(cov.pairs_2_plus):,} pairs had 2+ payments, so a mix was possible.\n"
             f"Keyed by the BIC columns instead of the path: {d['mixed_bic']:,} mixed pairs.",
             fontsize=8.5, color=ds.INK2, va="center")

    # ---------------------------------------------------------------- route length vs payment (b1)
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.085, right=0.02), "The route ignores the payment",
                  f"% with 2 banks; every segment shown within {spread:.2f} points", title_x=R["b1"][0] + ds.PAD)
    ds.hbars(ax, SEGMENTS, seg.pct_two_banks, ds.ACCENT, lambda v: f"{v:.1f}%", xmax=100, fontsize=8.5)
    ax.axvline(base, color=ds.INK2, ls="--", lw=1)

    # ---------------------------------------------------------------- which bank deviates (b2)
    dv = d["deviate"]
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.035, right=0.02), "The party's bank is the odd one out",
                  "% of pairs in the same country, rows where path end ≠ BIC column", title_x=R["b2"][0] + ds.PAD)
    x = np.arange(3)
    ax.bar(0, dv.loc["match", "path_pair"], width=0.5, color=ds.CONTEXT)
    ax.text(0, dv.loc["match", "path_pair"] + 0.2, f"{dv.loc['match', 'path_pair']:.1f}%", ha="center", fontsize=9)
    b1 = ax.bar(x[1:] - 0.19, [dv.loc[k, "path_pair"] for k in ("first", "last")], width=0.38, color=ds.ACCENT, label="path ends")
    b2 = ax.bar(x[1:] + 0.19, [dv.loc[k, "bic_pair"] for k in ("first", "last")], width=0.38, color=ds.HOT, label="BIC columns")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.axhline(d["chance"], color=ds.INK2, ls=":", lw=1.3, label=f"two unrelated banks: {d['chance']:.1f}%")
    ax.set_xticks(x, ["both ends\nmatch", "first bank\ndiffers", "last bank\ndiffers"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    ax.set_ylim(0, 12.5)

    # ---------------------------------------------------------------- gaps (b3)
    gp = d["gaps"]
    pred = 100 * (1 - (1 - g) ** (gp.banks - 1))
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.03, right=0.02), f"One {100 * g:.0f}% gap chance per hand-off",
                  "% of routes with a gap; dots = predicted", title_x=R["b3"][0] + ds.PAD)
    bars = ax.bar(gp.banks, gp.pct_with_gap, color=ds.ACCENT, width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8.5)
    ax.plot(gp.banks, pred, "o", color=ds.INK, ms=4)
    ax.set_xticks(gp.banks, [f"{b} banks" if b == 2 else str(b) for b in gp.banks])
    ax.set_ylim(0, gp.pct_with_gap.max() * 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    ds.footer(fig, "Not explained: which bank pairs are cover pairs (one-way, far more often when the first bank is one of the largest).")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D06_routing")
