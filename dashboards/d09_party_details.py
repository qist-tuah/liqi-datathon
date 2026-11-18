"""Dashboard 9 - Party details and tokens.

Run from the project root:   python dashboards/d09_party_details.py
Output:                      reports/dashboards/D09_party_details.pdf and .png

Every number on the page is computed here from the full data (same logic as notebooks/12_details_tokens.ipynb).
The text columns are large, so one working table per side is stored in data/interim/details_tokens.duckdb
(the same file notebook 12 uses). The first run builds them (a few minutes); later runs take seconds.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

sys.path.insert(0, str(ds.ROOT))
from src import eda  # noqa: E402
from src import settings as S  # noqa: E402

DB_FILE = Path(os.environ.get("DASH_DETAILS_DB", S.INTERIM_DIR / "details_tokens.duckdb"))   # override only for testing
SIDES = {"sender": ("ordering", "sender_country"), "receiver": ("beneficiary", "receiver_country")}
IBAN = r"_FAKE_[A-Z]{2}\d{20}"
SHOW_CHARS = 74


def party_table(con, q, side: str) -> str:
    """Name of the per-side working table. Uses notebook 12's table if it exists, otherwise builds a smaller one."""
    tables = q("SELECT table_name FROM information_schema.tables")["table_name"].tolist()
    if f"party_{side}" in tables:
        return f"party_{side}"
    name = f"dash_party_{side}"
    if name not in tables:
        prefix, country_col = SIDES[side]
        print(f"Building {name} (first run only) ...", flush=True)
        con.execute(f"""
            CREATE TABLE {name} AS
            SELECT {prefix}_bic8 AS bank, {country_col} AS bank_country, {prefix}_type AS party_type, {prefix}_account AS account,
                   regexp_full_match({prefix}_account, '{IBAN}')                                      AS is_iban,
                   regexp_matches({prefix}_details, '(^|\\n)[0-9]/')                                  AS numbered_layout,
                   regexp_full_match({prefix}_details, '(/[^\\n]*\\n)?[0-9]{{4}}[A-Z]{{2}}[A-Z0-9]{{2}}') AS bic_only,
                   regexp_extract({prefix}_tokens, '<country_(?:code|name):([A-Z]{{2}})', 1)          AS address_country,
                   contains({prefix}_tokens, '<business:')                                            AS has_business,
                   contains({prefix}_tokens, '<company_keyword:')                                     AS has_company_keyword,
                   contains({prefix}_tokens, '<first_name:')                                          AS has_first_name,
                   contains({prefix}_tokens, '<last_name:')                                           AS has_last_name,
                   regexp_matches({prefix}_tokens, '<(finance_word|stop_word):')                      AS has_extra_words
            FROM feat
        """)
    return name


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    eda.connect(db_file=DB_FILE, memory_limit=os.environ.get("DASH_MEMORY_LIMIT"))
    q, con = eda.q, eda.con
    spill = ds.ROOT / "data" / "interim" / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    t = {side: party_table(con, q, side) for side in SIDES}
    d = {"n": int(q(f"SELECT count(*) AS n FROM {t['sender']}").n[0])}

    d["names"] = q(" UNION ALL ".join(f"""
        SELECT '{side}' AS side, party_type, count(*) AS n,
               sum(has_business::INT) AS business, sum(has_company_keyword::INT) AS company,
               sum(has_first_name::INT) AS first_name, sum(has_last_name::INT) AS last_name,
               sum(((party_type = 'B' AND (has_first_name OR has_last_name)) OR
                    (party_type = 'C' AND (has_business OR has_company_keyword)))::INT) AS wrong
        FROM {t[side]} GROUP BY 2""" for side in SIDES))
    d["layout"] = q(" UNION ALL ".join(f"""
        SELECT '{side}' AS side, count(*) AS accounts, sum((share = 0)::INT) AS never, sum((share = 1)::INT) AS always,
               sum((share > 0 AND share < 1)::INT) AS sometimes
        FROM (SELECT account, bank, avg(numbered_layout::INT) AS share FROM {t[side]} GROUP BY 1, 2 HAVING count(*) >= 200)"""
        for side in SIDES)).set_index("side")
    d["address"] = q(" UNION ALL ".join(f"""
        SELECT '{side}' AS side, 100 * avg((address_country = bank_country)::INT) AS pct_same,
               (SELECT 100 * sum(a.p * b.p) FROM
                   (SELECT address_country AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM {t[side]} WHERE address_country <> '' GROUP BY 1) a
                   JOIN (SELECT bank_country AS c, count(*) * 1.0 / sum(count(*)) OVER () AS p FROM {t[side]} WHERE address_country <> '' GROUP BY 1) b
                   USING (c)) AS pct_chance
        FROM {t[side]} WHERE address_country <> ''""" for side in SIDES)).set_index("side")
    d["bic_only"] = q(" UNION ALL ".join(f"SELECT '{side}' AS side, count(*) AS n, sum(bic_only::INT) AS k FROM {t[side]}" for side in SIDES))
    d["iban"] = q(" UNION ALL ".join(f"""
        SELECT '{side}' AS side, sum((p = 1)::INT) AS always, sum((p = 0)::INT) AS never, sum((p > 0 AND p < 1)::INT) AS mixed
        FROM (SELECT bank_country, avg(is_iban::INT) AS p FROM {t[side]} GROUP BY 1)""" for side in SIDES)).set_index("side")
    d["extra"] = q(f"SELECT numbered_layout, 100 * avg(has_extra_words::INT) AS pct FROM {t['sender']} GROUP BY 1").set_index("numbered_layout")

    # one busy receiver account and one busy sender account, as examples
    rec = q(f"SELECT account, bank FROM {t['receiver']} GROUP BY 1, 2 HAVING count(*) BETWEEN 300 AND 320 ORDER BY 1, 2 LIMIT 1").iloc[0]
    d["rec_texts"] = q(f"""
        SELECT beneficiary_details AS text, count(*) AS n FROM feat
        WHERE beneficiary_account = '{rec.account}' AND beneficiary_bic8 = '{rec.bank}' GROUP BY 1 ORDER BY 2 DESC, 1
    """)
    snd = q(f"SELECT account FROM {t['sender']} GROUP BY 1 HAVING count(*) BETWEEN 300 AND 320 ORDER BY 1 LIMIT 1").account[0]
    d["snd_texts"] = q(f"SELECT ordering_details AS text, count(*) AS n FROM feat WHERE ordering_account = '{snd}' GROUP BY 1 ORDER BY 2 DESC, 1")
    return d


def short(text: str) -> str:
    one = text.replace("\n", " | ")
    return one if len(one) <= SHOW_CHARS else one[: SHOW_CHARS - 1] + "…"


def draw(d: dict):
    nm = d["names"]
    wrong = int(nm.wrong.sum())
    lay = d["layout"]
    ad = d["address"]
    bo = d["bic_only"]
    bic_only = 100 * bo.k.sum() / bo.n.sum()

    fig = ds.new_dashboard(9, "Party details: senders are written once, receivers every time",
                           f"{2 * d['n']:,} party texts (sender and receiver of every payment)   |   red = differs from the rest")
    ds.kpi_row(fig, [
        (f"{wrong:,}", "names of the wrong party type"),
        (f"{int(lay.loc['sender', 'sometimes'])}", f"of {int(lay.loc['sender', 'accounts']):,} busy senders change layout"),
        (f"{ad.loc['sender', 'pct_same']:.0f}%", "of senders live in their bank's country"),
        (f"{bic_only:.1f}%", "of texts are only the party's bank code"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- examples (wide)
    ax = ds.table_panel(fig, ds.inner(R["wide"]), "One sender, one text; one receiver, many texts",
                        "real examples from the data (line breaks shown as | )")
    st, rt = d["snd_texts"], d["rec_texts"]
    split = 0.36
    ax.text(0, 0.97, f"A sender account: {int(st.n.sum())} payments", fontsize=10, fontweight="bold", va="top")
    ax.text(0, 0.86, "\n".join(st.text.iloc[0].split("\n")), fontsize=8.5, family="monospace", va="top", linespacing=1.5)
    ax.text(0, 0.30, f"{100 * st.n.iloc[0] / st.n.sum():.0f}% of its payments carry exactly this text;\n"
                     "the others add a line of extra words.", fontsize=9, color=ds.INK2, va="top")
    ax.plot([split - 0.02, split - 0.02], [0.02, 0.98], color=ds.GRID, lw=1)
    ax.text(split, 0.97, f"A receiver account: {int(rt.n.sum())} payments, {len(rt)} different texts", fontsize=10,
            fontweight="bold", va="top", color=ds.HOT)
    for i, r in enumerate(rt.head(9).itertuples()):
        yy = 0.86 - i * 0.092
        ax.text(split, yy, f"{r.n:>3}", fontsize=8.5, family="monospace", va="top", color=ds.INK2)
        ax.text(split + 0.035, yy, short(r.text), fontsize=8.5, family="monospace", va="top")

    # ---------------------------------------------------------------- names (side)
    tot = nm.groupby("party_type")[["n", "business", "company", "first_name", "last_name"]].sum()
    ax = ds.panel(fig, ds.inner(R["side"], left=0.03, right=0.02), "The name gives away the party type",
                  "% of texts with each name token, both sides", title_x=R["side"][0] + ds.PAD)
    cols = [("business", "business\nname"), ("company", "company\nkeyword"), ("first_name", "first\nname"), ("last_name", "last\nname")]
    xs = np.arange(len(cols))
    for off, typ, color, label in [(-0.19, "B", ds.ACCENT, "business (B)"), (0.19, "C", ds.SERIES[1], "consumer (C)")]:
        vals = [100 * tot.loc[typ, c] / tot.loc[typ, "n"] for c, _ in cols]
        bars = ax.bar(xs + off, vals, width=0.38, color=color, label=label)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8.5)
    ax.set_xticks(xs, [lab for _, lab in cols], fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", ncols=2, fontsize=8.5)

    # ---------------------------------------------------------------- layout per account (b1)
    ax = ds.panel(fig, ds.inner(R["b1"], left=0.03, right=0.02), "Senders keep their layout",
                  "accounts with 200+ payments: numbered layout ...", title_x=R["b1"][0] + ds.PAD)
    kinds = [("never", "never", ds.ACCENT), ("always", "always", ds.SERIES[1]), ("sometimes", "sometimes", ds.HOT)]
    xs = np.arange(2)
    for i, (col, label, color) in enumerate(kinds):
        vals = [lay.loc[s, col] for s in ("sender", "receiver")]
        bars = ax.bar(xs + (i - 1) * 0.26, vals, width=0.26, color=color, label=label)
        ax.bar_label(bars, labels=[f"{int(v):,}" for v in vals], padding=2, fontsize=8.5)
    ax.set_xticks(xs, ["sender accounts", "receiver accounts"])
    ax.set_ylim(0, lay[["never", "always", "sometimes"]].to_numpy().max() * 1.45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", ncols=3, fontsize=8.5)

    # ---------------------------------------------------------------- address (b2)
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.06, right=0.02, bottom=0.075), "Only senders live where they bank",
                  "% of texts whose address country = the bank's country", title_x=R["b2"][0] + ds.PAD)
    ds.hbars(ax, ["sender", "receiver"], [ad.loc["sender", "pct_same"], ad.loc["receiver", "pct_same"]],
             [ds.ACCENT, ds.HOT], lambda v: f"{v:.2f}%", xmax=100, fontsize=10)
    fig.text(R["b2"][0] + ds.PAD, R["b2"][1] + 0.033, f"By chance: {ad.loc['sender', 'pct_chance']:.1f}% (sender), {ad.loc['receiver', 'pct_chance']:.2f}% (receiver).\n"
                      "An account never changes its address country.", fontsize=8.5, color=ds.INK2, va="center")

    # ---------------------------------------------------------------- IBAN by country (b3)
    ib = d["iban"]
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.085, right=0.03, bottom=0.06), "The bank's country sets the account format",
                  "bank countries by share of IBAN-shaped accounts", title_x=R["b3"][0] + ds.PAD)
    ds.hbars(ax, ["always IBAN", "never IBAN", "mixed"], [ib.loc["sender", "always"], ib.loc["sender", "never"], ib.loc["sender", "mixed"]],
             [ds.ACCENT, ds.CONTEXT, ds.HOT], lambda v: f"{int(v)}", fontsize=10)
    fig.text(R["b3"][0] + ds.PAD, R["b3"][1] + 0.025, f"Sender side shown; receiver side: {int(ib.loc['receiver', 'always'])} always, "
             f"{int(ib.loc['receiver', 'never'])} never, {int(ib.loc['receiver', 'mixed'])} mixed.", fontsize=8.5, color=ds.INK2, va="center")

    ex = d["extra"]
    ds.footer(fig, f"Extra words are added to {ex.loc[False, 'pct']:.1f}% of free-text sender details and {ex.loc[True, 'pct']:.2f}% of numbered ones. "
                   "Not explained: why a receiver's address is in its bank's country less often than chance.")
    return fig


if __name__ == "__main__":
    data = load_data()
    if len(sys.argv) > 1 and sys.argv[1] == "--build-only":
        sys.exit(0)
    figure = draw(data)
    ds.save(figure, "D09_party_details")
