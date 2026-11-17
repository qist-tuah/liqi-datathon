"""Dashboard 3 - Currency and FX.

Run from the project root:   python dashboards/d03_currency_fx.py
Output:                      reports/dashboards/D03_currency_fx.pdf and .png

Every number on the page is computed here from the normal payments (same logic as notebooks/05_currency_fx.ipynb).
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

# Official currency (ISO 4217) of the 20 largest sending countries, to check the derived home currencies
ISO_4217 = {"US": "USD", "CH": "CHF", "GB": "GBP", "FR": "EUR", "DE": "EUR", "SG": "SGD", "CN": "CNY", "IT": "EUR",
            "BE": "EUR", "LU": "EUR", "NL": "EUR", "HK": "HKD", "JP": "JPY", "AU": "AUD", "IM": "GBP", "FI": "EUR",
            "ES": "EUR", "CA": "CAD", "KR": "KRW", "AT": "EUR"}
NO_SENDER = {"ER": "ERN"}  # Eritrea receives but never sends, so its home currency comes from ISO 4217


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}

    # home currency of each country = the currency its senders use most
    con.execute("""
        CREATE OR REPLACE TEMP TABLE home AS
        SELECT sender_country AS country, arg_max(currency_code, n) AS home_currency,
               max(n) / sum(n) AS home_share, sum(n) AS payments_sent
        FROM (SELECT sender_country, currency_code, count(*) AS n FROM normal GROUP BY 1, 2)
        GROUP BY 1
    """)
    d["home"] = q("SELECT * FROM home ORDER BY payments_sent DESC, country")

    # currency mix by party pair, when sender and receiver countries have different home currencies
    d["pairs"] = q("""
        SELECT n.party_pair, count(*) AS n, 100 * avg((n.currency_code = hs.home_currency)::INT) AS pct_sender_home
        FROM normal n
        JOIN home hs ON hs.country = n.sender_country
        JOIN home hr ON hr.country = n.receiver_country
        WHERE hs.home_currency <> hr.home_currency
        GROUP BY 1 ORDER BY 1
    """).set_index("party_pair")

    # where along the route the currency comes from, and the conversion-count rule (one streaming pass)
    d["route"] = q("""
        WITH m AS (SELECT map_from_entries(list((country, home_currency))) AS hm FROM home),
        p AS (
            SELECT n.currency_code AS cur, n.currency_conversion_count AS conversions, hs.home_currency AS ordering_bic_home,
                   list_transform(list_filter(string_split(n.bic8_path, ';'), lambda x: x <> '...'),
                                  lambda b: m.hm[substr(b, 5, 2)]) AS homes
            FROM normal n CROSS JOIN m JOIN home hs ON hs.country = n.sender_country
        ),
        c AS (SELECT *, list_transform(homes, lambda h: coalesce(h, 'ERN')) AS h2 FROM p)
        SELECT count(*)                                                                             AS n,
               100 * avg((cur = homes[1])::INT)                                                     AS first_bank,
               100 * avg((cur <> homes[1] AND cur = homes[-1])::INT)                               AS last_bank,
               100 * avg((cur <> homes[1] AND cur <> homes[-1] AND list_contains(homes, cur))::INT) AS middle_bank,
               100 * avg((NOT list_contains(homes, cur))::INT)                                      AS no_bank,
               sum((conversions = len(list_filter(list_zip(h2[1:-2], h2[2:]), lambda z: z[1] <> z[2])))::INT) AS exact,
               count(*) FILTER (WHERE ordering_bic_home <> homes[1])                                AS disagree_n,
               100 * avg((cur = ordering_bic_home)::INT) FILTER (WHERE ordering_bic_home <> homes[1]) AS disagree_ordering,
               100 * avg((cur = homes[1])::INT) FILTER (WHERE ordering_bic_home <> homes[1])        AS disagree_first
        FROM c
    """).iloc[0]

    # one real payment to illustrate the rule: 4 banks, no gaps, 2 conversions
    d["examples"] = q("""
        SELECT bic8_path, currency_code, currency_conversion_count
        FROM normal
        WHERE participant_count = 4 AND path_gaps = 0 AND currency_conversion_count = 2
        ORDER BY epoch_seconds, bic8_path LIMIT 2000
    """)
    return d


def pick_example(d: dict) -> tuple:
    """First example whose 4 banks have home currencies A, A, B, C and that is paid in A."""
    hm = dict(zip(d["home"].country, d["home"].home_currency)) | NO_SENDER
    for r in d["examples"].itertuples():
        banks = r.bic8_path.split(";")
        homes = [hm.get(b[4:6]) for b in banks]
        if None in homes or len(banks) != 4:
            continue
        a, b, c, e = homes
        if a == b and b != c and c != e and len({a, c, e}) == 3 and r.currency_code == a:
            return banks, homes, r.currency_code
    raise RuntimeError("no example route found")


def draw(d: dict):
    h, rt, pairs = d["home"], d["route"], d["pairs"]
    total = h.payments_sent.sum()
    pct_home = (h.home_share * h.payments_sent).sum() / total * 100
    top20 = h.head(20)
    iso_ok = int((top20.home_currency == top20.country.map(ISO_4217)).sum())
    pct_exact = rt.exact / rt.n * 100

    fig = ds.new_dashboard(3, "Currency and FX: the route decides the currency",
                           f"{int(rt.n):,} normal payments   |   red = differs from normal")
    ds.kpi_row(fig, [
        (f"{pct_home:.1f}%", "in the home currency"),
        (f"{iso_ok} of {len(top20)}", "official currencies"),
        (f"{rt.first_bank:.1f}%", "first bank's currency"),
        (f"{pct_exact:.0f}%", "conversions predicted"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- conversion rule (wide)
    banks, homes, cur = pick_example(d)
    ax = ds.table_panel(fig, ds.inner(R["wide"], left=ds.PAD, right=ds.PAD, bottom=ds.PAD),
                        "Conversions = currency changes along the route",
                        "a real payment through 4 banks", title_x=R["wide"][0] + ds.PAD)
    xs = [0.04 + i * 0.25 for i in range(4)]
    bw, by, bh = 0.17, 0.42, 0.34
    for i, (x, bank, home) in enumerate(zip(xs, banks, homes)):
        ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=ds.TINT_BLUE if i == 0 else "#f3f6fb", edgecolor=ds.AXIS, linewidth=1))
        ax.text(x + bw / 2, by + bh * 0.70, home, ha="center", va="center", fontsize=17, fontweight="bold", color=ds.NAVY)
        ax.text(x + bw / 2, by + bh * 0.30, f"bank in {ds.country(bank[4:6])}", ha="center", va="center",
                fontsize=9.5, color=ds.INK2)
        if i < 3:
            change = homes[i] != homes[i + 1]
            ax.add_patch(FancyArrowPatch((x + bw + 0.01, by + bh / 2), (xs[i + 1] - 0.01, by + bh / 2),
                                         arrowstyle="-|>", mutation_scale=14, linewidth=2,
                                         color=ds.HOT if change else ds.CONTEXT))
            ax.text((x + bw + xs[i + 1]) / 2, by + bh + 0.06, "+1 conversion" if change else "no change",
                    ha="center", fontsize=10, fontweight="bold" if change else "normal",
                    color=ds.HOT if change else ds.INK2)
    ax.text(xs[0] + bw / 2, by - 0.10, f"paid in {cur}", ha="center", fontsize=10, color=ds.INK, fontweight="bold")
    n_changes = sum(homes[i] != homes[i + 1] for i in range(3))
    ax.text(0.5, 0.08, f"{n_changes} changes  =  {n_changes} conversions   |   holds for {int(rt.exact):,} of {int(rt.n):,} payments*",
            ha="center", fontsize=11.5, color=ds.INK, fontweight="bold")

    # ---------------------------------------------------------------- source of the currency (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.075, right=0.02), "The first bank sets it",
                  "currency = home currency of ...", title_x=R["side"][0] + ds.PAD)
    labels = ["first bank", "last bank", "middle bank", "no bank"]
    vals = [rt.first_bank, rt.last_bank, rt.middle_bank, rt.no_bank]
    ds.hbars(ax, labels, vals, [ds.ACCENT] + [ds.CONTEXT] * 3, lambda v: f"{v:.1f}%", xmax=100, fontsize=10.5)

    # ---------------------------------------------------------------- home currency per country
    top = h.head(7)
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.10, right=0.02), "Each country has a home currency",
                  "share of its payments in that currency", title_x=R["b1"][0] + ds.PAD)
    ds.hbars(ax, [f"{ds.country(c)}  {k}" for c, k in zip(top.country, top.home_currency)], top.home_share * 100,
             ds.ACCENT, lambda v: f"{v:.0f}%", xmax=100)

    # ---------------------------------------------------------------- party pair
    order = ["B2B", "B2C", "C2B", "C2C"]
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.045, right=0.02), "Party type does not matter",
                  "% paid in the sender's home currency", title_x=R["b2"][0] + ds.PAD)
    ds.hbars(ax, order, [pairs.loc[k, "pct_sender_home"] for k in order], ds.ACCENT, lambda v: f"{v:.1f}%",
             xmax=100, fontsize=10.5)

    # ---------------------------------------------------------------- ordering BIC vs first bank
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.085, right=0.02), "Not the ordering BIC",
                  f"{int(rt.disagree_n):,} payments where the two differ", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, ["first bank\nin path", "ordering\nBIC"], [rt.disagree_first, rt.disagree_ordering],
             [ds.ACCENT, ds.HOT], lambda v: f"{v:.1f}%", xmax=100, fontsize=10.5)

    ds.footer(fig, "* Eritrea never sends payments, so its home currency (ERN) is taken from ISO 4217.")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D03_currency_fx")
