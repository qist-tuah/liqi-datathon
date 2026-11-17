"""Dashboard 5 - Parties and accounts.

Run from the project root:   python dashboards/d05_parties_accounts.py
Output:                      reports/dashboards/D05_parties_accounts.pdf and .png

Every number on the page is computed here (same logic as notebooks/07_parties_accounts.ipynb).
Party types and activity use normal payments; account roles and formats use every row.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

BUCKETS = ["1-4", "5-19", "20-99", "100-999", "1000+"]
IBAN_LIKE = "'^[A-Z]{2}[0-9]{2}'"   # after the '_FAKE_' prefix: 2 letters + 2 digits


def mod97_valid(s: str) -> bool:
    """ISO 13616 check: move the first 4 characters to the end, letters -> numbers (A=10 ... Z=35), remainder mod 97 must be 1."""
    r = s[4:] + s[:4]
    return int("".join(str(int(ch, 36)) for ch in r)) % 97 == 1


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {}

    # party pairs (normal payments)
    d["pairs"] = q("""
        SELECT party_pair, ordering_type, beneficiary_type, count(*) AS n
        FROM normal GROUP BY 1, 2, 3 ORDER BY 1
    """)

    # payments sent per sender account (normal payments)
    d["senders"] = q("""
        SELECT ordering_type AS t, count(*) AS n FROM normal GROUP BY ordering_account, ordering_type
    """)

    # repeat payee = the sender already paid this receiver before; by sender activity and type (normal payments)
    d["repeat"] = q(f"""
        WITH pr AS (SELECT ordering_account, any_value(ordering_type) AS t, count(*) AS c
                    FROM normal GROUP BY ordering_account, beneficiary_account),
        sd AS (SELECT ordering_account, any_value(t) AS t, sum(c) AS sent, count(*) AS payees FROM pr GROUP BY 1)
        SELECT t, CASE WHEN sent < 5 THEN '{BUCKETS[0]}' WHEN sent < 20 THEN '{BUCKETS[1]}' WHEN sent < 100 THEN '{BUCKETS[2]}'
                       WHEN sent < 1000 THEN '{BUCKETS[3]}' ELSE '{BUCKETS[4]}' END AS bucket,
               sum(sent) AS n, sum(sent - payees) AS repeats
        FROM sd GROUP BY 1, 2 ORDER BY 1, 2
    """)

    # the same account as sender and as receiver (all rows)
    d["roles"] = q("""
        WITH o AS (SELECT ordering_account AS acc, count(DISTINCT ordering_type) AS types, any_value(ordering_type) AS t,
                          arg_max(ordering_bic8, epoch_seconds) AS bank FROM feat GROUP BY 1),
        b AS (SELECT beneficiary_account AS acc, count(DISTINCT beneficiary_type) AS types, any_value(beneficiary_type) AS t,
                     arg_max(beneficiary_bic8, epoch_seconds) AS bank FROM feat GROUP BY 1),
        x AS (SELECT o.t AS ts, b.t AS tr, o.bank = b.bank AS same_bank FROM o JOIN b USING (acc))
        SELECT (SELECT count(*) FROM o WHERE types > 1) + (SELECT count(*) FROM b WHERE types > 1) AS accounts_with_2_types,
               count(*) AS both_roles,
               100 * avg(same_bank::INT) AS same_bank,
               100 * avg((ts = tr)::INT) AS same_type,
               100 * (avg((ts = 'B')::INT) * avg((tr = 'B')::INT) + avg((ts = 'C')::INT) * avg((tr = 'C')::INT)) AS chance
        FROM x
    """).iloc[0]

    # account format by bank country (all rows, both roles)
    d["formats"] = q(f"""
        WITH a AS (SELECT ordering_account AS acc, ordering_bic8 AS bic FROM feat
                   UNION ALL SELECT beneficiary_account, beneficiary_bic8 FROM feat)
        SELECT substr(bic, 5, 2) AS country, bool_and(regexp_matches(substr(acc, 7), {IBAN_LIKE})) AS all_iban,
               bool_or(regexp_matches(substr(acc, 7), {IBAN_LIKE})) AS any_iban
        FROM a GROUP BY 1 ORDER BY 1
    """)
    d["ibans"] = q(f"""
        WITH a AS (SELECT ordering_account AS acc, ordering_bic8 AS bic FROM feat
                   WHERE regexp_matches(substr(ordering_account, 7), {IBAN_LIKE})
                   UNION ALL
                   SELECT beneficiary_account, beneficiary_bic8 FROM feat
                   WHERE regexp_matches(substr(beneficiary_account, 7), {IBAN_LIKE}))
        SELECT DISTINCT rtrim(substr(acc, 7)) AS iban, bic FROM a
    """)
    return d


def draw(d: dict):
    pr, rl, ib = d["pairs"], d["roles"], d["ibans"]
    total = pr.n.sum()
    p_send_b = pr.loc[pr.ordering_type == "B", "n"].sum() / total
    p_recv_b = pr.loc[pr.beneficiary_type == "B", "n"].sum() / total
    pr = pr.set_index("party_pair")
    expected = {k: 100 * (p_send_b if k[0] == "B" else 1 - p_send_b) * (p_recv_b if k[2] == "B" else 1 - p_recv_b)
                for k in pr.index}
    observed = (pr.n / total * 100).to_dict()

    s = d["senders"]
    counts = np.sort(s.n.to_numpy())[::-1]
    share_of_senders = np.arange(1, len(counts) + 1) / len(counts) * 100
    cum_payments = np.cumsum(counts) / counts.sum() * 100
    top = {k: cum_payments[int(len(counts) * k / 100) - 1] for k in (1, 10)}
    mean_b, mean_c = s.loc[s.t == "B", "n"].mean(), s.loc[s.t == "C", "n"].mean()

    r = d["repeat"].assign(pct=lambda x: x.repeats / x.n * 100).pivot(index="bucket", columns="t", values="pct").loc[BUCKETS]

    valid = ib.iban.map(mod97_valid).mean() * 100
    assert (ib.iban.str.len() == 22).all(), "an IBAN is not 22 characters long"
    assert (ib.iban.str[:2] == ib.bic.str[4:6]).all(), "an IBAN country differs from its bank's country"
    assert (ib.iban.str[4:8] == ib.bic.str[:4]).all(), "an IBAN bank code differs from its BIC"
    assert (ib.iban.str[8:16] == "00000000").all(), "an IBAN has non-zero characters 9-16"
    assert rl.accounts_with_2_types == 0, "an account has two party types in the same role"
    f = d["formats"]
    n_iban, n_never, n_mixed = int(f.all_iban.sum()), int((~f.any_iban).sum()), int((f.any_iban & ~f.all_iban).sum())
    assert n_mixed == 0, "a bank country mixes IBAN and non-IBAN accounts"

    fig = ds.new_dashboard(5, "Parties and accounts: random roles, a fixed IBAN template",
                           f"{int(total):,} normal payments   |   red = differs from normal")
    ds.kpi_row(fig, [
        (f"{top[1]:.1f}%", "sent by the top 1%"),
        (f"{mean_b / mean_c:.1f}x", "business vs consumer senders"),
        (f"{rl.same_type:.1f}%", "same type in both roles"),
        (f"{valid:.1f}%", "valid IBAN check digits"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- concentration (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.04, right=0.03, bottom=0.065), "A few senders make most payments",
                  "share of payments made by the most active senders", title_x=R["wide"][0] + ds.PAD)
    ax.plot([0, 100], [0, 100], color=ds.CONTEXT, lw=1.5)
    ax.text(62, 52, "if every sender paid equally", fontsize=9.5, color=ds.INK2, rotation=0)
    idx = np.unique(np.r_[0, np.geomspace(1, len(counts) - 1, 2000).astype(int)])   # 2,000 points draw the same curve
    ax.fill_between(share_of_senders[idx], cum_payments[idx], color=ds.ACCENT, alpha=0.10, linewidth=0)
    ax.plot(share_of_senders[idx], cum_payments[idx], color=ds.ACCENT, lw=2.2)
    for k in (1, 10):
        ax.plot([k], [top[k]], "o", color=ds.ACCENT, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"top {k}% of senders = {top[k]:.1f}% of payments", (k, top[k]), xytext=(14, -4),
                    textcoords="offset points", fontsize=10.5, fontweight="bold", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.set_xlabel("senders, most active first")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # ---------------------------------------------------------------- independence (side)
    order = sorted(observed, key=observed.get, reverse=True)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.045, right=0.02), "Types are drawn independently",
                  "share of payments; line = if independent", title_x=R["side"][0] + ds.PAD)
    ds.hbars(ax, order, [observed[k] for k in order], ds.ACCENT, lambda v: f"{v:.1f}%", fontsize=10.5)
    for y, k in enumerate(order):
        ax.plot([expected[k]] * 2, [y - 0.42, y + 0.42], color=ds.NAVY, lw=2.2)

    # ---------------------------------------------------------------- repeat payees
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.04, right=0.03, bottom=0.065), "Repeat payees follow activity",
                  "% of payments to a payee paid before", title_x=R["b1"][0] + ds.PAD)
    for color, t, name in [(ds.SERIES[0], "B", "business"), (ds.SERIES[1], "C", "consumer")]:
        ax.plot(BUCKETS, r[t].values, color=color, lw=2.2, marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.5, label=f"{name} sender")
    ax.set_xlabel("payments sent by the sender")
    ax.set_yticks([0, 10, 20, 30], ["0%", "10%", "20%", "30%"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.4)

    # ---------------------------------------------------------------- IBAN template
    de = sorted(i for i in ib.iban if i.startswith("DE"))
    ex = de[len(de) // 2]                    # a typical German IBAN (the middle one, so it is the same on every run)
    parts = [(ex[0:2], "country", ds.TINT_BLUE), (ex[2:4], "check\nrandom", ds.TINT), (ex[4:8], "bank\n= BIC", ds.TINT_BLUE),
             (ex[8:16], "always\nzeros", "#eceef2"), (ex[16:22], "account", ds.TINT_BLUE)]
    ax = ds.table_panel(fig, ds.inner(R["b2"], left=ds.PAD, right=ds.PAD, bottom=ds.PAD), "Every IBAN: one template",
                        f"22 characters, {len(ib):,} IBANs", title_x=R["b2"][0] + ds.PAD)
    widths = [max(len(t), 4) for t, _, _ in parts]   # short parts get a wider box so their labels fit
    unit = 0.92 / sum(widths)
    x = 0.04
    for (text, label, color), w in zip(parts, widths):
        ax.add_patch(FancyBboxPatch((x + 0.004, 0.50), w * unit - 0.008, 0.26, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + w * unit / 2, 0.63, text, ha="center", va="center", fontsize=13, fontweight="bold",
                family="monospace", color=ds.NAVY)
        ax.text(x + w * unit / 2, 0.36, label, ha="center", va="center", fontsize=9.5, linespacing=1.2,
                color=ds.HOT if "random" in label else ds.INK2, fontweight="bold" if "random" in label else "normal")
        x += w * unit
    ax.text(0.5, 0.08, "real IBANs: length varies by country", ha="center", fontsize=9.5, color=ds.INK2)

    # ---------------------------------------------------------------- roles
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.075, right=0.02), "One account, two roles",
                  f"{int(rl.both_roles):,} accounts that send and receive", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, ["same bank", "same type", "by chance"], [rl.same_bank, rl.same_type, rl.chance],
             [ds.ACCENT, ds.HOT, ds.CONTEXT], lambda v: f"{v:.1f}%", xmax=100, fontsize=10.5)

    ds.footer(fig, f"{n_iban} bank countries always use IBANs; the other {n_never} never do.")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D05_parties_accounts")
