"""Dashboard 2 - Amounts.

Run from the project root:   python dashboards/d02_amounts.py
Output:                      reports/dashboards/D02_amounts.pdf and .png

Every number on the page is computed here from the normal payments (same logic as notebooks/04_amounts.ipynb).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

PAIRS = ["C2C", "B2C", "C2B", "B2B"]
BIN = 0.1  # histogram bin width on the log10 scale


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}

    # amount distribution per party pair (log10 scale)
    d["pairs"] = q("""
        SELECT party_pair, count(*) AS n, median(usd_amount) AS median_usd,
               avg(usd_amount_log10) AS mean_log10, stddev(usd_amount_log10) AS sd_log10
        FROM normal GROUP BY 1 ORDER BY 1
    """).set_index("party_pair")
    d["hist"] = q(f"""
        SELECT party_pair, floor(usd_amount_log10 / {BIN})::INT AS bin, count(*) AS n
        FROM normal GROUP BY 1, 2 ORDER BY 1, 2
    """)

    # share of the variance of log10(amount) explained by the sender account (senders with 5+ payments)
    d["sender_share"] = q("""
        WITH s AS (SELECT ordering_account, count(*) AS n, avg(usd_amount_log10) AS m FROM normal GROUP BY 1),
        x AS (SELECT usd_amount_log10 AS l, s.m FROM normal JOIN s USING (ordering_account) WHERE s.n >= 5)
        SELECT 1 - avg((l - m) * (l - m)) / var_pop(l) AS share FROM x
    """).share[0]

    # amount level of each sender country and each currency, after removing the party-pair level
    con.execute("""
        CREATE OR REPLACE TEMP TABLE res AS
        WITH pm AS (SELECT party_pair, avg(usd_amount_log10) AS pm FROM normal GROUP BY 1)
        SELECT sender_country, currency_code, usd_amount_log10 - pm.pm AS res
        FROM normal JOIN pm USING (party_pair)
    """)
    d["countries"] = q("""
        SELECT sender_country AS country, count(*) AS n, avg(res) AS level
        FROM res GROUP BY 1 HAVING count(*) >= 20000 ORDER BY 1
    """)
    d["currencies"] = q("""
        WITH c AS (SELECT sender_country, avg(res) AS country_level FROM res GROUP BY 1)
        SELECT currency_code, count(*) AS n, avg(res) AS level_before, avg(res - c.country_level) AS level_after
        FROM res JOIN c USING (sender_country)
        GROUP BY 1 HAVING count(*) >= 20000 ORDER BY 1
    """)
    con.execute("DROP TABLE res")

    # whole-dollar amounts (no cents) in USD vs other currencies
    d["whole"] = q("""
        SELECT currency_code = 'USD' AS is_usd, count(*) AS n, 100 * avg(usd_amount_is_whole::INT) AS pct
        FROM normal GROUP BY 1 ORDER BY 1
    """).set_index("is_usd")
    return d


def money(v: float) -> str:
    return f"\\${v:,.0f}"


def draw(d: dict):
    p = d["pairs"]
    lv = p["mean_log10"]
    sender_x = 10 ** (lv["B2C"] - lv["C2C"])     # business sender, receiver fixed
    receiver_x = 10 ** (lv["C2B"] - lv["C2C"])   # business receiver, sender fixed
    both_x = 10 ** (lv["B2B"] - lv["C2C"])
    cur = d["currencies"]
    usd, other = d["whole"].loc[True, "pct"], d["whole"].loc[False, "pct"]

    fig = ds.new_dashboard(2, "Amounts: one bell curve, scaled by who pays whom",
                           f"{int(p.n.sum()):,} normal payments   |   red = differs from normal")
    ds.kpi_row(fig, [
        (f"{sender_x:.1f}x", "business sender"),
        (f"{receiver_x:.1f}x", "business receiver"),
        (f"{100 * d['sender_share']:.0f}%", "set by the sender"),
        (f"{usd:.1f}%", "USD whole dollars"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- distributions (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.03, right=0.03), "Same shape, shifted by party type",
                  "share of payments by amount, log scale", title_x=R["wide"][0] + ds.PAD)
    h = d["hist"]
    bins = np.arange(h.bin.min(), h.bin.max() + 1)
    curves = {}
    for color, pair in zip(ds.SERIES, PAIRS):
        hp = h[h.party_pair == pair].set_index("bin").n.reindex(bins, fill_value=0)
        y = hp.to_numpy() / hp.sum() * 100 / BIN                     # % of payments per 1 log10 unit
        y = np.convolve(y, np.ones(5) / 5, mode="same")              # smooth over half a decade (hides round-number spikes)
        x = (bins + 0.5) * BIN
        curves[pair] = (x, y)
        ax.plot(x, y, color=color, lw=2.2, label=pair)
    # name each curve: C2C on its left slope, B2B on its right slope, B2C / C2B (on top of each other) at its peak
    for keys, side in ((["C2C"], -1), (["B2C", "C2B"], 0), (["B2B"], 1)):
        x, y = curves[keys[-1]]
        if side == 0:
            i = int(np.argmax(y))
            xy, off, ha = (x[i], y[i]), (0, 8), "center"
        else:
            xs = np.log10(p.loc[keys[0], "median_usd"]) + 1.1 * side   # one decade away from the median
            xy, off, ha = (xs, np.interp(xs, x, y)), (8 * side, 4), "left" if side > 0 else "right"
        ax.annotate(" / ".join(keys), xy, xytext=off, textcoords="offset points", ha=ha,
                    fontsize=10.5, fontweight="bold", color=ds.INK)
    ax.set_xlim(-1, 8)
    ax.set_ylim(0, max(c[1].max() for c in curves.values()) * 1.25)
    ax.set_xticks(range(0, 8), ["\\$1", "\\$10", "\\$100", "\\$1K", "\\$10K", "\\$100K", "\\$1M", "\\$10M"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", ncols=4, handlelength=1.4, columnspacing=1.2)

    # ---------------------------------------------------------------- medians (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.045, right=0.02), "Types multiply the amount",
                  f"median USD; B2B = {both_x:.0f}x C2C", title_x=R["side"][0] + ds.PAD)
    order = ["B2B", "C2B", "B2C", "C2C"]
    ds.hbars(ax, order, [p.loc[k, "median_usd"] for k in order],
             [ds.ACCENT if k == "B2B" else ds.CONTEXT for k in order], money, fontsize=10.5)

    # ---------------------------------------------------------------- sender country
    c = d["countries"].sort_values("level")
    pick = list(c.head(4).itertuples()) + list(c.tail(4).itertuples())
    pick = sorted(pick, key=lambda r: -r.level)
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.09, right=0.02), "Sender country sets a level",
                  "amount vs the average, highest and lowest", title_x=R["b1"][0] + ds.PAD)
    mult = [10 ** r.level for r in pick]
    ax.barh([ds.country(r.country) for r in pick], [np.log10(m) for m in mult],
            color=[ds.ACCENT if m >= 1 else ds.CONTEXT for m in mult], height=0.62)
    for yy, m in enumerate(mult):
        ax.text(np.log10(m) + (0.03 if m >= 1 else -0.03), yy, f"{m:.1f}x" if m >= 0.995 else f"{m:.2f}x", va="center",
                ha="left" if m >= 1 else "right", fontsize=9.5)
    lim = max(abs(np.log10(m)) for m in mult) * 1.45
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color=ds.AXIS, lw=1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.invert_yaxis()

    # ---------------------------------------------------------------- currency
    before, after = cur.level_before.std(), cur.level_after.std()
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.075, right=0.02), "Currency adds nothing",
                  f"one dot per currency ({len(cur)})", title_x=R["b2"][0] + ds.PAD)
    rng = np.random.default_rng(0)   # small vertical jitter so the dots do not hide each other
    for yy, col, label in [(1, "level_before", "as paid"), (0, "level_after", "same country")]:
        ax.scatter(10 ** cur[col], yy + rng.uniform(-0.12, 0.12, len(cur)), s=36,
                   color=ds.CONTEXT if yy == 1 else ds.ACCENT, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xscale("log")
    lo = min(10 ** cur[["level_before", "level_after"]].min().min() / 1.6, 0.05)   # axis always covers every dot
    hi = max(10 ** cur[["level_before", "level_after"]].max().max() * 1.6, 20)
    ax.set_xlim(lo, hi)
    ax.set_xticks([0.1, 1, 10], ["0.1x", "1x", "10x"])
    ax.minorticks_off()
    ax.set_yticks([0, 1], ["same\ncountry", "as paid"])
    ax.set_ylim(-0.6, 1.6)
    ax.axvline(1, color=ds.AXIS, lw=1, zorder=1)
    ax.text(hi, 1.4, f"spread {10 ** before:.1f}x", ha="right", fontsize=9.5, color=ds.INK2)
    ax.text(hi, -0.5, f"spread {10 ** after:.2f}x", ha="right", fontsize=9.5, color=ds.INK, fontweight="bold")

    # ---------------------------------------------------------------- whole dollars
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.075, right=0.02), "USD amounts end in .00",
                  "% of payments with no cents", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, ["USD", "other\ncurrencies"], [usd, other], [ds.HOT, ds.CONTEXT], lambda v: f"{v:.1f}%", xmax=100,
             fontsize=10.5)

    ds.footer(fig, "Route type and number of banks do not change the amount.")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D02_amounts")
