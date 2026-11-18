"""Dashboard 8 - Bank network and country corridors.

Run from the project root:   python dashboards/d08_network.py
Output:                      reports/dashboards/D08_network.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/11_network.ipynb).
Countries are those of the first and last bank in the path (the route), not of the parties' own banks.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

EXTRA_NAMES = {"UG": "Uganda"}
TIERS = ["200,000+", "50,000 to\n199,999", "10,000 to\n49,999", "2,000 to\n9,999", "200 to\n1,999", "under\n200"]


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}
    con.execute("""
        CREATE OR REPLACE TEMP TABLE net AS
        SELECT split_part(bic8_path, ';', 1) AS fb, string_split(bic8_path, ';')[-1] AS lb,
               substr(split_part(bic8_path, ';', 1), 5, 2) AS fc, substr(string_split(bic8_path, ';')[-1], 5, 2) AS lc,
               participant_count AS k, string_split(bic8_path, ';') AS h
        FROM feat
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE mid AS
        SELECT bank, substr(bank, 5, 2) AS c, fc, lc
        FROM (SELECT unnest(list_slice(h, 2, len(h) - 1)) AS bank, fc, lc FROM net WHERE k >= 3) WHERE bank <> '...'
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE bank AS
        SELECT bic, sum(f) AS as_first, sum(l) AS as_last, sum(m) AS as_mid FROM (
            SELECT fb AS bic, count(*) AS f, 0 AS l, 0 AS m FROM net GROUP BY 1
            UNION ALL SELECT lb, 0, count(*), 0 FROM net GROUP BY 1
            UNION ALL SELECT bank, 0, 0, count(*) FROM mid GROUP BY 1) GROUP BY 1
    """)
    d["n"] = int(q("SELECT count(*) AS n FROM net").n[0])
    d["n_mid"] = int(q("SELECT count(*) AS n FROM mid").n[0])
    d["banks"] = q("SELECT as_first, as_last, as_mid FROM bank")
    d["pairs"] = q("SELECT count(*) AS n FROM net GROUP BY fb, lb")
    d["possible"] = float(q("SELECT count(*) FILTER (WHERE as_first > 0) * 1.0 * count(*) FILTER (WHERE as_last > 0) AS p FROM bank").p[0])

    # intermediaries: same country as an end of the route, against chance
    d["geo"] = q("SELECT 100 * avg((c = fc)::INT) AS first_obs, 100 * avg((c = lc)::INT) AS last_obs FROM mid").iloc[0]
    d["geo_chance"] = q("""
        WITH m AS (SELECT c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM mid GROUP BY 1),
             f AS (SELECT fc AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM mid GROUP BY 1),
             l AS (SELECT lc AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM mid GROUP BY 1)
        SELECT (SELECT 100 * sum(m.p * f.p) FROM m JOIN f USING (c)) AS first_chance,
               (SELECT 100 * sum(m.p * l.p) FROM m JOIN l USING (c)) AS last_chance
    """).iloc[0]

    # corridors: observed vs expected from country sizes
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE corridor AS
        WITH f AS (SELECT fc AS c, count(*) * 1.0 / {d['n']} AS p FROM net GROUP BY 1),
             l AS (SELECT lc AS c, count(*) * 1.0 / {d['n']} AS p FROM net GROUP BY 1),
             o AS (SELECT fc, lc, count(*) AS observed FROM net GROUP BY 1, 2)
        SELECT o.fc, o.lc, o.observed, {d['n']} * f.p * l.p AS expected
        FROM o JOIN f ON o.fc = f.c JOIN l ON o.lc = l.c
    """)
    d["domestic"] = q("SELECT sum(observed) AS observed, sum(expected) AS expected FROM corridor WHERE fc = lc").iloc[0]
    d["home"] = q("""
        SELECT fc AS country, observed / expected AS ratio FROM corridor
        WHERE fc = lc AND fc IN (SELECT fc FROM net GROUP BY 1 ORDER BY count(*) DESC LIMIT 8)
    """)
    d["corridors"] = q("""
        WITH c AS (SELECT * FROM corridor WHERE fc <> lc AND expected >= 2000 ORDER BY observed / expected DESC LIMIT 6),
             p AS (SELECT substr(fb, 5, 2) AS fc, substr(lb, 5, 2) AS lc, max(n) AS busiest FROM
                   (SELECT fb, lb, count(*) AS n FROM net GROUP BY 1, 2) GROUP BY 1, 2)
        SELECT c.fc, c.lc, c.observed / c.expected AS ratio, 100.0 * p.busiest / c.observed AS pct_busiest_pair
        FROM c JOIN p USING (fc, lc) ORDER BY ratio DESC
    """)
    return d


def tier(v: float) -> int:
    for i, limit in enumerate([200_000, 50_000, 10_000, 2_000, 200]):
        if v >= limit:
            return i
    return 5


def draw(d: dict):
    n, n_mid = d["n"], d["n_mid"]
    pairs = d["pairs"].n.sort_values(ascending=False).to_numpy()
    cum_pairs = np.cumsum(pairs) / n
    top10 = 100 * cum_pairs[len(pairs) // 10 - 1]
    used = 100 * len(pairs) / d["possible"]
    dom = d["domestic"].observed / d["domestic"].expected
    geo, chance = d["geo"], d["geo_chance"]

    b = d["banks"].copy()
    b["ends"] = b.as_first + b.as_last
    b["tier"] = b.ends.map(tier)
    t = b.groupby("tier").agg(ends=("ends", "sum"), mid=("as_mid", "sum"), banks=("ends", "size"))
    t["ratio"] = (t.mid / n_mid) / (t.ends / (2 * n))

    fig = ds.new_dashboard(8, "Bank network: corridors come from bank pairs, intermediaries from size",
                           f"{n:,} payments, {n_mid:,} intermediary slots   |   red = differs from expected")
    ds.kpi_row(fig, [
        (f"{used:.2f}%", "of possible bank pairs are ever used"),
        (f"{top10:.1f}%", "of payments on the busiest 10% of pairs"),
        (f"{dom:.2f}x", "domestic payments vs country sizes"),
        (f"{geo.first_obs:.2f}%", f"in the first bank's country (chance {chance.first_chance:.2f}%)"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- concentration (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.035, right=0.02), "Intermediary traffic sits on far fewer banks",
                  "cumulative share of appearances, banks from the busiest down (log scale)", title_x=R["wide"][0] + ds.PAD)
    halves = {}
    for col, label, color in [("as_first", "first bank", ds.CONTEXT), ("as_last", "last bank", ds.SERIES[1]),
                              ("as_mid", "intermediary", ds.ACCENT)]:
        v = np.sort(b[col].to_numpy())[::-1]
        v = v[v > 0]
        cum = np.cumsum(v) / v.sum() * 100
        halves[label] = int(np.argmax(cum >= 50)) + 1
        ax.plot(np.arange(1, len(v) + 1), cum, color=color, lw=2.4 if label == "intermediary" else 1.8,
                label=f"{label}: {halves[label]} banks carry half")
    ax.axhline(50, color=ds.AXIS, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, 10_000)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100, 1000, 10_000], ["1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.minorticks_off()
    ax.legend(loc="lower right")

    # ---------------------------------------------------------------- bank size (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.02, right=0.02), "Big banks sit in the middle",
                  "intermediary share / end share; banks by first/last appearances", title_x=R["side"][0] + ds.PAD)
    t = t.reindex(range(6))
    bars = ax.bar(range(6), t.ratio, color=[ds.HOT if r > 1.2 else (ds.ACCENT if r > 0.8 else ds.CONTEXT) for r in t.ratio], width=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.axhline(1, color=ds.INK2, ls="--", lw=1)
    ax.set_xticks(range(6), TIERS, fontsize=7.5)
    ax.set_ylim(0, t.ratio.max() * 1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- geography (b1)
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.03, right=0.02), "Intermediaries ignore geography",
                  "% of intermediaries in the same country as ...", title_x=R["b1"][0] + ds.PAD)
    x = np.arange(2)
    b1 = ax.bar(x - 0.19, [geo.first_obs, geo.last_obs], width=0.38, color=ds.ACCENT, label="observed")
    b2 = ax.bar(x + 0.19, [chance.first_chance, chance.last_chance], width=0.38, color=ds.CONTEXT, label="by chance")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f%%", padding=2, fontsize=9)
    ax.set_xticks(x, ["the first bank", "the last bank"])
    ax.set_ylim(0, max(geo.last_obs, chance.last_chance) * 1.45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", ncols=2)

    # ---------------------------------------------------------------- corridors (b2)
    c = d["corridors"]
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.075, right=0.02), "One bank pair can make a corridor",
                  "most over-used corridors: % of volume from their busiest bank pair", title_x=R["b2"][0] + ds.PAD)
    ds.hbars(ax, [f"{EXTRA_NAMES.get(r.fc, ds.country(r.fc))} → {EXTRA_NAMES.get(r.lc, ds.country(r.lc))}\n{r.ratio:.1f}x expected" for r in c.itertuples()],
             c.pct_busiest_pair, ds.ACCENT, lambda v: f"{v:.0f}%", xmax=100, fontsize=8)

    # ---------------------------------------------------------------- domestic (b3)
    h = d["home"].sort_values("ratio", ascending=False)
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.075, right=0.02), f"Payments stay home {dom:.2f}x as often",
                  "domestic payments / expected from country sizes", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, [ds.country(cc) for cc in h.country], h.ratio,
             [ds.ACCENT if r > 1.05 else ds.CONTEXT for r in h.ratio], lambda v: f"{v:.2f}x", fontsize=8.5)
    ax.axvline(1, color=ds.INK2, ls="--", lw=1)

    ds.footer(fig, "Cross-border corridors follow which bank pairs are connected, not geography. "
                   "The currency follows the first bank in the path (Dashboard 3).")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D08_network")
