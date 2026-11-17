"""Dashboard 4 - Time.

Run from the project root:   python dashboards/d04_time.py
Output:                      reports/dashboards/D04_time.pdf and .png

Every number on the page is computed here from the normal payments (same logic as notebooks/06_time.ipynb).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_style as ds  # noqa: E402

# Official UTC offsets (IANA time-zone database) in January and July 2023; main financial centre for multi-zone countries
TZ = {
    "US": (-5, -4), "CA": (-5, -4), "PA": (-5, -5), "KY": (-5, -5), "AR": (-3, -3),
    "GB": (0, 1), "IM": (0, 1), "JE": (0, 1), "IE": (0, 1),
    "FR": (1, 2), "DE": (1, 2), "CH": (1, 2), "IT": (1, 2), "BE": (1, 2), "LU": (1, 2), "NL": (1, 2), "AT": (1, 2),
    "ES": (1, 2), "DK": (1, 2), "PL": (1, 2), "CZ": (1, 2), "MC": (1, 2), "LI": (1, 2), "MT": (1, 2),
    "FI": (2, 3), "CY": (2, 3), "UA": (2, 3), "IL": (2, 3), "ZA": (2, 2),
    "TR": (3, 3), "RU": (3, 3), "SA": (3, 3), "QA": (3, 3), "AE": (4, 4), "IN": (5.5, 5.5),
    "CN": (8, 8), "SG": (8, 8), "HK": (8, 8), "MY": (8, 8), "TW": (8, 8), "JP": (9, 9), "KR": (9, 9), "AU": (11, 10),
}
REGIONS = ["Americas", "Europe & Africa", "Middle East & S. Asia", "Asia-Pacific"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TABLE_COUNTRIES = ["US", "GB", "DE", "AU"]     # countries that change their clocks in real life
SUNDAY_COUNTRIES = ["AU", "JP", "GB", "US"]    # one per region, east to west


def estimate_offsets(prof: pd.DataFrame) -> pd.DataFrame:
    """Time-zone offset of each country vs the UK, from the shape of its UTC hour profile (countries with 5,000+ payments)."""
    P = prof.pivot(index="sender_country", columns="utc_hour", values="n").fillna(0)
    P = P[P.sum(axis=1) >= 5_000]
    Pn = P.div(P.sum(axis=1), axis=0)
    ref = Pn.loc["GB"].to_numpy()
    rows = []
    for country, v in Pn.iterrows():
        # k hours ahead of the UK: country_utc[h] = GB_utc[h + k] = np.roll(ref, -k)[h]
        corr = {k: np.corrcoef(v.to_numpy(), np.roll(ref, -k))[0, 1] for k in range(-12, 13)}
        k = max(corr, key=corr.get)
        rows.append({"country": country, "payments": int(P.loc[country].sum()), "offset_h": k})
    return pd.DataFrame(rows)


def load_data() -> dict:
    """Run every query the page needs. Returns small pandas objects only."""
    q, con = ds.connect()
    d = {"total": int(q("SELECT count(*) AS n FROM normal").n[0])}

    prof = q("SELECT sender_country, utc_hour, count(*) AS n FROM normal GROUP BY 1, 2")
    off = estimate_offsets(prof)
    d["offsets"] = off

    con.register("offsets_df", off[["country", "offset_h"]])
    con.execute("CREATE OR REPLACE TEMP TABLE offsets AS SELECT country AS sender_country, offset_h FROM offsets_df")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW local_time AS
        SELECT n.epoch_seconds, n.sender_country, n.utc_weekday,
               n.epoch_seconds + to_hours(o.offset_h) AS local_ts,
               CASE WHEN o.offset_h <= -3 THEN '{REGIONS[0]}' WHEN o.offset_h <= 2 THEN '{REGIONS[1]}'
                    WHEN o.offset_h <= 6 THEN '{REGIONS[2]}' ELSE '{REGIONS[3]}' END AS region
        FROM normal n JOIN offsets o USING (sender_country)
    """)
    d["hour"] = q("SELECT hour(local_ts) AS h, count(*) AS n FROM local_time GROUP BY 1 ORDER BY 1")
    d["weekday"] = q("SELECT region, isodow(local_ts) AS d, count(*) AS n FROM local_time GROUP BY 1, 2 ORDER BY 1, 2")
    d["weeks"] = q("""
        SELECT date_trunc('week', epoch_seconds)::DATE AS week, region, count(*) AS n
        FROM local_time WHERE epoch_seconds >= '2023-05-01' AND epoch_seconds < '2023-07-31'
        GROUP BY 1, 2 ORDER BY 1, 2
    """)
    d["sunday"] = q(f"""
        SELECT sender_country, count(*) AS n,
               100 * avg((utc_weekday = 6)::INT)       AS utc_sunday,
               100 * avg((isodow(local_ts) = 7)::INT)  AS local_sunday
        FROM local_time WHERE sender_country IN ({", ".join(f"'{c}'" for c in SUNDAY_COUNTRIES)})
        GROUP BY 1 ORDER BY 1
    """).set_index("sender_country")
    return d


def draw(d: dict):
    off = d["offsets"]
    chk = off[off.country.isin(TZ)].copy()
    jan = chk.country.map(lambda c: TZ[c][0] - TZ["GB"][0])
    jul = chk.country.map(lambda c: TZ[c][1] - TZ["GB"][1])
    n_jan = int(((chk.offset_h - jan).abs() <= 0.5).sum())   # India is +5.5: whole hours only
    covered = int(off.payments.sum())

    hr = d["hour"].set_index("h").n
    hr = hr / hr.sum() * 100
    office = hr.loc[7:18].sum()
    wd = d["weekday"].pivot(index="region", columns="d", values="n")
    wd_all = wd.sum() / wd.sum().sum() * 100
    sun_sat = wd_all[7] / wd_all[6]
    wk = d["weeks"].pivot(index="week", columns="region", values="n")
    wk = wk / wk.sum() * 100
    wk_all = d["weeks"].groupby("week").n.sum()
    wk_all = wk_all / wk_all.sum() * 100

    fig = ds.new_dashboard(4, "Time: one working week, in every sender's local time",
                           f"{covered:,} payments with a known time zone   |   red = differs from normal")
    ds.kpi_row(fig, [
        (f"{n_jan}/{len(chk)}", "countries on winter time"),
        (f"{office:.0f}%", "between 07:00 and 19:00"),
        (f"{sun_sat:.1f}x", "Sunday vs Saturday"),
        (f"{wk_all.max() / wk_all.min():.1f}x", "busiest vs quietest week"),
    ])
    R = ds.layout(fig)

    # ---------------------------------------------------------------- local hour (wide)
    ax = ds.panel(fig, ds.inner(R["wide"], left=0.03, right=0.02), "A working day in local time",
                  "% of payments by local hour of the sender", title_x=R["wide"][0] + ds.PAD)
    work = [7 <= h <= 18 for h in hr.index]
    ax.bar(hr.index, hr.values, width=0.72, color=[ds.ACCENT if w else ds.CONTEXT for w in work])
    for h in (7, int(hr.idxmax())):
        ax.text(h, hr[h] + hr.max() * 0.03, f"{hr[h]:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(2.5, hr.max() * 0.30, f"night: about {hr.loc[0:5].mean():.0f}% per hour", ha="center", fontsize=9.5,
            color=ds.INK2)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.7, 23.7)
    ax.set_ylim(0, hr.max() * 1.18)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- weekday (side)
    ax = ds.panel(fig, ds.inner(R["side"], left=0.02, right=0.02), "Quiet weekends",
                  "% of payments by local weekday", title_x=R["side"][0] + ds.PAD)
    ax.bar(DAYS, wd_all.values, width=0.62, color=[ds.ACCENT] * 5 + [ds.HOT] * 2)
    for i, v in enumerate(wd_all.values):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9.5, fontweight="bold" if i >= 5 else "normal")
    ax.set_ylim(0, wd_all.max() * 1.18)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- daylight saving table
    ax = ds.table_panel(fig, ds.inner(R["b1"], left=ds.PAD, right=ds.PAD, bottom=ds.PAD), "Clocks never change",
                        "UTC offset: found in the data vs official", title_x=R["b1"][0] + ds.PAD)
    found = off.set_index("country").offset_h
    cols = [("found", 0.47), ("January", 0.70), ("July", 0.93)]
    ax.text(0.0, 0.90, "country", fontsize=9, color=ds.MUTED, va="center")
    for name, x in cols:
        ax.text(x, 0.90, name, fontsize=9, color=ds.MUTED, va="center", ha="right")
    ax.plot([0, 1], [0.82, 0.82], color=ds.AXIS, lw=1)
    rh = 0.78 / len(TABLE_COUNTRIES)
    fmt = lambda v: f"{v:+g}" if v else "0"  # noqa: E731
    for i, c in enumerate(TABLE_COUNTRIES):
        y = 0.82 - (i + 0.5) * rh
        f, j, u = found[c], TZ[c][0], TZ[c][1]
        ax.text(0.0, y, ds.country(c), fontsize=10.5, va="center", fontweight="bold")
        for (name, x), v, bad in zip(cols, (f, j, u), (False, abs(f - j) > 0.5, abs(f - u) > 0.5)):
            if bad:
                ax.add_patch(Rectangle((x - 0.13, y - rh * 0.34), 0.15, rh * 0.68, facecolor=ds.TINT, edgecolor="none"))
            ax.text(x, y, fmt(v), fontsize=10.5, va="center", ha="right", fontweight="bold" if bad else "normal")
        if i < len(TABLE_COUNTRIES) - 1:
            ax.plot([0, 1], [y - rh / 2, y - rh / 2], color=ds.GRID, lw=0.8)

    # ---------------------------------------------------------------- weekly volume per region
    ax = ds.panel(fig, ds.inner(R["b2"], left=0.03, right=0.02), "Each week has its own level",
                  "% of each region's payments per week", title_x=R["b2"][0] + ds.PAD)
    x = np.arange(len(wk))
    for color, reg in zip(ds.SERIES, REGIONS):
        ax.plot(x, wk[reg].values, color=color, lw=2, label=reg)
    ax.set_xticks(x[::3], [w.strftime("%d %b") for w in wk.index[::3]])
    ax.set_yticks([5, 10], ["5%", "10%"])
    ax.set_ylim(0, wk.max().max() * 1.45)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, fontsize=8.5, handlelength=1.2, columnspacing=1.0)

    # ---------------------------------------------------------------- UTC vs local Sunday
    s = d["sunday"].loc[SUNDAY_COUNTRIES]
    ax = ds.panel(fig, ds.inner(R["b3"], left=0.075, right=0.02), "UTC distorts the weekend",
                  "Sunday share of payments", title_x=R["b3"][0] + ds.PAD)
    y = np.arange(len(s))
    ax.barh(y - 0.19, s.utc_sunday, height=0.36, color=ds.CONTEXT, label="UTC")
    ax.barh(y + 0.19, s.local_sunday, height=0.36, color=ds.ACCENT, label="local time")
    for yy, (u, lo) in enumerate(zip(s.utc_sunday, s.local_sunday)):
        ax.text(u + 0.2, yy - 0.19, f"{u:.1f}%", va="center", fontsize=9)
        ax.text(lo + 0.2, yy + 0.19, f"{lo:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y, [ds.country(c) for c in s.index])
    ax.set_xlim(0, max(s.utc_sunday.max(), s.local_sunday.max()) * 1.3)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8.5, handlelength=1.2)

    ds.footer(fig, "Amounts do not depend on the hour or the weekday.")
    return fig


if __name__ == "__main__":
    figure = draw(load_data())
    ds.save(figure, "D04_time")
