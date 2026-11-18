"""Dashboard 10 - Data structure and consistency rules.

Run from the project root:   python dashboards/d10_consistency.py
Output:                      reports/dashboards/D10_consistency.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/08_consistency.ipynb).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

sys.path.insert(0, str(ds.ROOT))
from src import settings as S  # noqa: E402

BIC = r"\d{4}[A-Z]{2}[A-Z0-9]{2}"
IBAN = r"_FAKE_[A-Z]{2}\d{20}"


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}
    d["path"] = q(f"""
        SELECT count(*) AS n,
               sum((continuity_qflag <> (path_gaps = 0))::INT)                                  AS flag,
               sum((participant_count <> path_entries - path_gaps)::INT)                        AS count_,
               sum(((transaction_type_code = 'Direct') <> (participant_count = 2))::INT)        AS direct,
               sum((bic8_path LIKE '...%' OR bic8_path LIKE '%...')::INT)                        AS gap_end,
               sum((NOT regexp_full_match(ordering_bic8, '{BIC}') OR NOT regexp_full_match(beneficiary_bic8, '{BIC}')
                    OR NOT regexp_full_match(bic8_path, '({BIC}|\\.\\.\\.)(;({BIC}|\\.\\.\\.))*'))::INT) AS bic_format,
               sum((currency_conversion_count > participant_count - 1)::INT)                    AS conversions,
               100 * avg((currency_conversion_count = participant_count - 1)::INT)              AS pct_at_limit
        FROM feat
    """).iloc[0]
    d["flag_grid"] = q("SELECT continuity_qflag AS flag, path_gaps > 0 AS gap, count(*) AS n FROM feat GROUP BY 1, 2")
    d["conv"] = q("SELECT participant_count AS banks, max(currency_conversion_count) AS max_conv, count(*) AS n FROM feat GROUP BY 1 ORDER BY 1")
    d["sender_acc"] = q("""
        SELECT count(*) AS accounts, sum((nb > 1 OR nt > 1)::INT) AS bad FROM
        (SELECT ordering_account, count(DISTINCT ordering_bic8) AS nb, count(DISTINCT ordering_type) AS nt FROM feat GROUP BY 1)
    """).iloc[0]
    d["receiver_acc"] = q("""
        SELECT count(*) AS accounts, sum((nb > 1)::INT) AS bad FROM
        (SELECT beneficiary_account, count(DISTINCT beneficiary_bic8) AS nb FROM feat GROUP BY 1)
    """).iloc[0]
    d["iban"] = q(f"""
        SELECT side, count(*) AS n, sum((substr(a, 7, 2) <> substr(bic, 5, 2) OR substr(a, 11, 4) <> substr(bic, 1, 4))::INT) AS vs_bic,
               sum((substr(a, 7, 2) <> substr(hop, 5, 2) OR substr(a, 11, 4) <> substr(hop, 1, 4))::INT) AS vs_path
        FROM (SELECT 'sender' AS side, ordering_account AS a, ordering_bic8 AS bic, split_part(bic8_path, ';', 1) AS hop FROM feat
              WHERE regexp_full_match(ordering_account, '{IBAN}')
              UNION ALL
              SELECT 'receiver', beneficiary_account, beneficiary_bic8, string_split(bic8_path, ';')[-1] FROM feat
              WHERE regexp_full_match(beneficiary_account, '{IBAN}'))
        GROUP BY 1
    """).set_index("side")
    d["checksum"] = q(f"""
        WITH a AS (SELECT DISTINCT ordering_account AS a FROM feat WHERE regexp_full_match(ordering_account, '{IBAN}')
                   UNION SELECT DISTINCT beneficiary_account FROM feat WHERE regexp_full_match(beneficiary_account, '{IBAN}'))
        SELECT count(*) AS n, sum((CAST(substr(a, 11, 18) || CAST(ascii(substr(a, 7, 1)) - 55 AS VARCHAR)
                                       || CAST(ascii(substr(a, 8, 1)) - 55 AS VARCHAR) || substr(a, 9, 2) AS HUGEINT) % 97 = 1)::INT) AS valid
        FROM a
    """).iloc[0]
    d["roles"] = q("""
        WITH o AS (SELECT DISTINCT ordering_account AS a, ordering_bic8 AS b, ordering_type AS t FROM feat),
             r AS (SELECT DISTINCT beneficiary_account AS a, beneficiary_bic8 AS b, beneficiary_type AS t FROM feat)
        SELECT count(*) AS pairs, sum((o.t = r.t)::INT) AS same_type FROM o JOIN r USING (a)
    """).iloc[0]
    d["text"] = q("""
        SELECT count(*) AS n,
               sum((starts_with(l1o, '/') AND l1o NOT IN ('/' || ao, '//' || ao))::INT)  AS line1_sender,
               sum((starts_with(l1b, '/') AND l1b NOT IN ('/' || ab, '//' || ab))::INT)  AS line1_receiver,
               sum(regexp_matches(od, '[^\\n]{36,}')::INT)                                AS long_sender,
               sum(regexp_matches(bd, '[^\\n]{36,}')::INT)                                AS long_receiver
        FROM (SELECT split_part(ordering_details, chr(10), 1) AS l1o, substr(ordering_account, 7) AS ao,
                     split_part(beneficiary_details, chr(10), 1) AS l1b, substr(beneficiary_account, 7) AS ab,
                     ordering_details AS od, beneficiary_details AS bd FROM feat)
    """).iloc[0]
    if S.RAW_PARQUET.exists():   # the amount exactly as written in the CSV
        d["cents"] = q(f"""
            SELECT count(*) AS n, sum((NOT regexp_full_match(usd_amount, '[0-9]+(\\.[0-9]{{1,2}})?')
                                       OR CAST(usd_amount AS DOUBLE) < 0.01)::INT) AS bad
            FROM read_parquet('{S.RAW_PARQUET.as_posix()}')
        """).iloc[0]
    d["dupes"] = int(q("""
        WITH k AS (SELECT epoch_seconds, ordering_account, beneficiary_account, usd_amount FROM tx GROUP BY ALL HAVING count(*) > 1)
        SELECT count(*) - count(DISTINCT hash(tx)) AS n FROM tx JOIN k USING (epoch_seconds, ordering_account, beneficiary_account, usd_amount)
    """).n[0])
    return d


def draw(d: dict):
    p, n = d["path"], int(d["path"].n)
    ib, tx = d["iban"], d["text"]
    rules = [  # (rule, tested, unit, exceptions)
        ("Flag N  ⇔  the path has a gap (...)", n, "payments", p.flag),
        ("Bank count = path entries − gaps", n, "payments", p.count_),
        ("Direct  ⇔  exactly 2 banks", n, "payments", p.direct),
        ("A gap is never the first or last entry", n, "payments", p.gap_end),
        ("Every BIC: 4 digits + country + 2 characters", n, "payments", p.bic_format),
        ("Conversions ≤ banks − 1", n, "payments", p.conversions),
        ("A sender account has one bank and one type", d["sender_acc"].accounts, "accounts", d["sender_acc"].bad),
        ("An IBAN holds its bank's country and code", int(ib.n.sum()), "IBAN rows", int(ib.vs_bic.sum())),
        ("Line 1 is /account, //account or absent", 2 * int(tx.n), "party texts", int(tx.line1_sender + tx.line1_receiver)),
    ]
    if "cents" in d:
        rules.append(("Amounts in whole cents, never below 0.01", int(d["cents"].n), "payments", int(d["cents"].bad)))
    held = sum(int(e) == 0 for *_, e in rules)

    ra, ch, ro = d["receiver_acc"], d["checksum"], d["roles"]
    partial = [  # (rule, % following, verdict colour)
        ("Detail lines\n≤ 35 characters", 100 - 100 * (tx.long_sender + tx.long_receiver) / (2 * tx.n), ds.ACCENT),
        ("Receiver account\nat one bank", 100 - 100 * ra.bad / ra.accounts, ds.ACCENT),
        ("Same type as\nsender and receiver", 100 * ro.same_type / ro.pairs, ds.HOT),
        ("IBAN check\ndigits valid", 100 * ch.valid / ch.n, ds.HOT),
    ]

    fig = ds.new_dashboard(10, f"Consistency: {held} rules hold in all {n / 1e6:.0f} million rows",
                           f"{n:,} payments   |   red = a rule that fails")
    ds.kpi_row(fig, [
        (f"{len(rules) - held}", f"exceptions in {len(rules)} structural rules" if held == len(rules) else f"of {len(rules)} rules have exceptions"),
        (f"{p.pct_at_limit:.1f}%", "of payments convert at every hand-off"),
        (f"{100 - 100 * ib.vs_path.sum() / ib.n.sum():.1f}%", "of IBANs also match the path end"),
        (f"{d['dupes']}", "fully identical row (a labelled duplicate)"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- scoreboard (wide)
    ax = ds.table_panel(fig, ds.inner(R["wide"]), "Rules that never break",
                        "each rule checked on every row; exceptions found")
    row_h = 0.92 / (len(rules) + 1)
    for x, text, ha in [(0.0, "rule", "left"), (0.72, "tested", "right"), (0.84, "unit", "left"), (0.995, "exceptions", "right")]:
        ax.text(x, 0.97, text, fontsize=9, color=ds.INK2, ha=ha, va="top")
    ax.plot([0, 1], [0.97 - row_h * 0.75] * 2, color=ds.AXIS, lw=0.8)
    for i, (rule, tested, unit, exc) in enumerate(rules):
        y = 0.97 - row_h * (i + 1)
        ax.text(0.0, y, rule, fontsize=10, va="top")
        ax.text(0.72, y, f"{int(tested):,}", fontsize=10, ha="right", va="top")
        ax.text(0.84, y, unit, fontsize=9, color=ds.INK2, va="top")
        ax.text(0.995, y, f"{int(exc):,}", fontsize=10.5, fontweight="bold", ha="right", va="top",
                color=ds.ACCENT if int(exc) == 0 else ds.HOT)

    # ---------------------------------------------------------------- partial rules (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.09, right=0.03), "Where the rules break",
                  "% of rows or accounts following the rule", title_x=R["side"][0] + ds.PAD)
    ds.hbars(ax, [r for r, _, _ in partial], [v for _, v, _ in partial], [c for *_, c in partial],
             lambda v: f"{v:.2f}%" if v > 99 else f"{v:.1f}%", xmax=100, fontsize=9)

    # ---------------------------------------------------------------- conversions (b1)
    cv = d["conv"][d["conv"].n >= 10]
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.03, right=0.02), "Conversions never pass banks − 1",
                  "most conversions seen, by banks in the route (10+ payments)", title_x=R["b1"][0] + ds.PAD)
    ax.plot(cv.banks, cv.banks - 1, color=ds.INK2, ls="--", lw=1.2, label="limit: banks − 1")
    ax.plot(cv.banks, cv.max_conv, "o", color=ds.ACCENT, ms=6, label="most seen")
    ax.set_xticks(cv.banks[::2])
    ax.set_yticks(range(0, int(cv.banks.max()), 4))
    ax.legend(loc="upper left")

    # ---------------------------------------------------------------- IBAN vs BIC vs path (b2)
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.075, right=0.02), "The IBAN is built from the party's bank",
                  "% of IBAN rows whose country and bank code match ...", title_x=R["b2"][0] + ds.PAD)
    vals = [100 - 100 * ib.vs_bic.sum() / ib.n.sum(), 100 - 100 * ib.vs_path.sum() / ib.n.sum()]
    ds.hbars(ax, ["the BIC column", "the path end"], vals, [ds.ACCENT, ds.CONTEXT], lambda v: f"{v:.2f}%", xmax=100, fontsize=10)

    # ---------------------------------------------------------------- flag vs gap (b3)
    g = d["flag_grid"].set_index(["flag", "gap"]).n
    ax = ds.table_panel(fig, ds.inner(R["b3"], left=ds.PAD + 0.01), "The flag is the gap",
                        "payments by continuity_qflag and gap in the path")
    cells = [[(True, False), (True, True)], [(False, False), (False, True)]]
    ax.text(0.48, 0.92, "no gap", ha="center", fontsize=9.5, color=ds.INK2)
    ax.text(0.82, 0.92, "gap (...)", ha="center", fontsize=9.5, color=ds.INK2)
    for r, (label, row) in enumerate(zip(["flag Y", "flag N"], cells)):
        y = 0.62 - r * 0.40
        ax.text(0.0, y, label, fontsize=10.5, va="center", fontweight="bold")
        for c, key in enumerate(row):
            v = int(g.get(key, 0))
            x = 0.48 + c * 0.34
            ax.add_patch(plt_rect(x, y, v))
            ax.text(x, y, f"{v:,}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color=ds.INK if v else ds.MUTED)

    ds.footer(fig, f"The {d['dupes']} identical pair of rows is labelled DUPLICATE_PAYMENT on both copies, so nothing was removed. "
                   "Notebook 08 lists every check.")
    return fig


def plt_rect(x, y, v):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch((x - 0.15, y - 0.15), 0.30, 0.30, boxstyle="round,pad=0,rounding_size=0.02",
                          facecolor=ds.TINT_BLUE if v else "#f3f3f1", edgecolor="none")


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D10_consistency")
