"""Shared look for the 10 dashboards (one 16:9 page each).

Every dashboard script imports this module, so all pages use the same colours, fonts and layout:

    import dash_style as ds
    fig = ds.new_dashboard(1, "Title = the main finding", "one line of context")
    ds.card(fig, [x, y, w, h])          # white card behind a panel (draw it before the panel)
    ds.kpi_row(fig, [("7,006", "what the number means"), ...])
    ax = ds.panel(fig, [x, y, w, h], "Panel title = takeaway", "what is plotted")
    ds.save(fig, "D01_anomalies")

Colour rule used on every page: ACCENT (blue) = the finding, CONTEXT (grey) = normal payments / baseline,
HOT (red) = a value that clearly differs from normal.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dashboards"

# colours
PAGE = "#e9eef6"      # page background
SURFACE = "#ffffff"   # cards and chart background
NAVY = "#0d2f5e"      # header band and headline numbers
BAND_TEXT = "#c9dcf5" # secondary text on the header band
INK = "#0b0b0b"       # titles and values
INK2 = "#52514e"      # subtitles and labels
MUTED = "#898781"     # axis text
GRID = "#e1e0d9"      # gridlines and tile borders
AXIS = "#c3c2b7"      # baselines
ACCENT = "#2a78d6"    # the finding
ACCENT2 = "#1baf7a"   # a second series, only when two findings share a chart
SERIES = ["#2a78d6", "#1baf7a", "#6250d6", "#e87ba4"]  # blue, green, violet, pink: 2-4 series (colour-blind safe, always labelled)
CONTEXT = "#b4b2a9"   # normal payments / baseline
HOT = "#e34948"       # differs from normal (stripes, badge, highlights)
TINT = "#fbd5d5"      # highlighted table cell (light HOT)
TINT_BLUE = "#dce9f9" # light ACCENT, for a highlighted box that is not "different from normal"
PROJECT = "UMDAC Datathon 2026  |  LIQI"

FIGSIZE = (16, 9)     # inches, 16:9
LEFT, RIGHT = 0.03, 0.97
GAP = 0.012           # space between cards (same everywhere, so all cards line up)
PAD = 0.012           # inner margin of every card
TOP_Y, TOP_H = 0.40, 0.36      # middle row of cards
BOT_Y, BOT_H = 0.085, 0.30     # bottom row of cards


def _font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Arial", "Helvetica", "DejaVu Sans"):
        if name in installed:
            return name
    return "sans-serif"


def _width(fig, text) -> float:
    """Width of a text object as a fraction of the figure width."""
    box = text.get_window_extent(fig.canvas.get_renderer())
    return box.width / (fig.get_figwidth() * fig.dpi)


def apply() -> None:
    """Set the matplotlib defaults shared by all dashboards."""
    matplotlib.rcParams.update({
        "font.family": _font(),
        "font.size": 10,
        "text.color": INK,
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,   # keep text as real text in the PDF
    })


def new_dashboard(number: int, title: str, subtitle: str, total: int = 10):
    """Create the page with its navy header band. Returns the figure."""
    apply()
    fig = plt.figure(figsize=FIGSIZE)
    fig.add_artist(Rectangle((0, 0.89), 1, 0.11, facecolor=NAVY, edgecolor="none", transform=fig.transFigure, zorder=-3))
    fig.add_artist(Rectangle((0, 0.884), 1, 0.006, facecolor=HOT, edgecolor="none", transform=fig.transFigure, zorder=-3))
    fig.text(LEFT, 0.957, title, fontsize=18, fontweight="bold", color="white", va="center")
    fig.text(LEFT, 0.917, subtitle, fontsize=11, color=BAND_TEXT, va="center")
    fig.text(RIGHT, 0.957, PROJECT, fontsize=10, color="white", va="center", ha="right", fontweight="bold")
    fig.text(RIGHT, 0.917, f"Dashboard {number} of {total}", fontsize=10, color=BAND_TEXT, va="center", ha="right")
    return fig


def columns(n: int, left: float = LEFT, right: float = RIGHT) -> list[tuple[float, float]]:
    """Split the page width into n equal columns separated by GAP. Returns [(x, width), ...].
    A card that spans columns i..j runs from cols[i][0] to cols[j][0] + cols[j][1], so its edges line up with the cards below."""
    width = (right - left - GAP * (n - 1)) / n
    return [(left + i * (width + GAP), width) for i in range(n)]


def layout(fig) -> dict:
    """Draw the standard 5 cards and return their rects:
    "wide" = middle row, columns 1-2 | "side" = middle row, column 3 | "b1", "b2", "b3" = bottom row.
    The wide card starts at b1's left edge and ends at b2's right edge; "side" and "b3" have the same x and width."""
    (x1, w1), (x2, w2), (x3, w3) = columns(3)
    rects = {"wide": [x1, TOP_Y, x2 + w2 - x1, TOP_H], "side": [x3, TOP_Y, w3, TOP_H],
             "b1": [x1, BOT_Y, w1, BOT_H], "b2": [x2, BOT_Y, w2, BOT_H], "b3": [x3, BOT_Y, w3, BOT_H]}
    for r in rects.values():
        card(fig, r)
    return rects


def inner(rect: list[float], left: float = PAD, right: float = PAD, top: float = 0.083, bottom: float = 0.035) -> list[float]:
    """The plot area inside a card: room for the panel title above and the axis labels below.
    left/right: extra space for tick labels (for example 0.05 for a y-axis with dollar labels)."""
    x, y, w, h = rect
    return [x + left, y + bottom, w - left - right, h - top - bottom]


def connect():
    """Open the data through src/eda.py with low-memory settings. Returns (q, con).
    Also creates the view `normal` (= feat without the labelled anomalies)."""
    import os
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import eda
    eda.connect(memory_limit=os.environ.get("DASH_MEMORY_LIMIT"))
    spill = ROOT / "data" / "interim" / "duckdb_tmp"   # gitignored: DuckDB writes here instead of running out of memory
    spill.mkdir(parents=True, exist_ok=True)
    eda.con.execute(f"SET temp_directory = '{spill.as_posix()}'")
    eda.con.execute("SET preserve_insertion_order = false")
    eda.con.execute("CREATE OR REPLACE TEMP VIEW normal AS SELECT * FROM feat WHERE is_anomaly = 0")
    return eda.q, eda.con


def hbars(ax, labels: list[str], values, colors, fmt, xmax: float | None = None, fontsize: float = 9.5) -> None:
    """Horizontal bars, first label on top, value printed at the end of each bar, no x-axis."""
    values = list(values)
    top = xmax if xmax is not None else max(values)
    ax.barh(labels, values, color=colors, height=0.62)
    for y, v in enumerate(values):
        ax.text(v + top * 0.02, y, fmt(v), va="center", fontsize=fontsize)
    ax.set_xlim(0, top * 1.22)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.invert_yaxis()


COUNTRY = {  # names for the country codes shown on the dashboards
    "US": "United States", "GB": "United Kingdom", "CH": "Switzerland", "FR": "France", "DE": "Germany",
    "SG": "Singapore", "CN": "China", "IT": "Italy", "BE": "Belgium", "LU": "Luxembourg", "NL": "Netherlands",
    "HK": "Hong Kong", "JP": "Japan", "AU": "Australia", "IM": "Isle of Man", "FI": "Finland", "ES": "Spain",
    "CA": "Canada", "KR": "South Korea", "AT": "Austria", "AE": "UAE", "SA": "Saudi Arabia", "IN": "India",
    "RU": "Russia", "TR": "Turkey", "IE": "Ireland", "DK": "Denmark", "UA": "Ukraine", "QA": "Qatar",
    "AR": "Argentina", "PA": "Panama", "KY": "Cayman Islands", "IL": "Israel", "MY": "Malaysia", "TW": "Taiwan",
    "ZA": "South Africa", "JE": "Jersey", "LI": "Liechtenstein", "MC": "Monaco", "PL": "Poland", "CZ": "Czechia",
    "MT": "Malta", "CY": "Cyprus", "SE": "Sweden", "NO": "Norway", "BR": "Brazil", "MX": "Mexico", "TH": "Thailand",
    "ID": "Indonesia", "PH": "Philippines", "VN": "Vietnam", "NZ": "New Zealand", "PT": "Portugal", "GR": "Greece",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia", "EG": "Egypt", "NG": "Nigeria",
    "KE": "Kenya", "MA": "Morocco", "CL": "Chile", "CO": "Colombia", "PE": "Peru", "PK": "Pakistan",
    "BD": "Bangladesh", "LK": "Sri Lanka", "KW": "Kuwait", "BH": "Bahrain", "OM": "Oman", "JO": "Jordan",
    "LB": "Lebanon", "BM": "Bermuda", "BS": "Bahamas", "VG": "British Virgin Islands", "GG": "Guernsey",
    "IS": "Iceland", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "SK": "Slovakia", "SI": "Slovenia",
}


def country(code: str) -> str:
    return COUNTRY.get(code, code)


def card(fig, rect: list[float]) -> None:
    """A white rounded card with a soft drop shadow, so it looks like it floats above the page.
    rect = [x, y, width, height] (figure fractions)."""
    x, y, w, h = rect
    aspect = FIGSIZE[0] / FIGSIZE[1]          # keeps the shadow the same thickness on all sides
    for i in range(1, 8):                     # several faint layers = a soft, blurred edge
        sx = 0.0008 * i
        sy = sx * aspect
        fig.add_artist(FancyBboxPatch((x - sx, y - 0.007 - sy), w + 2 * sx, h + 2 * sy,
                                      boxstyle="round,pad=0,rounding_size=0.010",
                                      facecolor=NAVY, edgecolor="none", alpha=0.022,
                                      transform=fig.transFigure, zorder=-2))
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                  facecolor=SURFACE, edgecolor="#dfe5ee", linewidth=0.8,
                                  transform=fig.transFigure, zorder=-1))


def kpi_row(fig, tiles: list[tuple[str, str]], y: float = 0.775, height: float = 0.09) -> None:
    """A row of headline numbers. tiles = [(value, description), ...]."""
    for value_label, (x, width) in zip(tiles, columns(len(tiles))):
        value, label = value_label
        card(fig, [x, y, width, height])
        fig.add_artist(Rectangle((x + 0.004, y + 0.014), 0.004, height - 0.028, facecolor=HOT, edgecolor="none",
                                 transform=fig.transFigure, zorder=0))
        big = fig.text(x + 0.018, y + height / 2, value, fontsize=26, fontweight="bold", color=NAVY, va="center")
        lx = x + 0.018 + _width(fig, big) + 0.010
        small = fig.text(lx, y + height / 2, label, fontsize=11, color=INK2, va="center", linespacing=1.35)
        while _width(fig, small) > x + width - 0.010 - lx and small.get_fontsize() > 7.5:   # never spill out of the box
            small.set_fontsize(small.get_fontsize() - 0.5)


def panel(fig, rect: list[float], title: str, subtitle: str = "", title_x: float | None = None):
    """Add a chart area. rect = [x, y, width, height] of the plot itself (figure fractions).
    The title (the takeaway) and subtitle (what is plotted) sit above it, starting at title_x
    (default: the left edge of the plot; give a smaller value when the chart has long labels on its left)."""
    x, y, w, h = rect
    tx = x if title_x is None else title_x
    ax = fig.add_axes(rect)
    fig.text(tx, y + h + 0.047, title, fontsize=12, fontweight="bold", va="bottom")
    if subtitle:
        fig.text(tx, y + h + 0.021, subtitle, fontsize=9.5, color=INK2, va="bottom")
    ax.tick_params(length=0)
    return ax


def table_panel(fig, rect: list[float], title: str, subtitle: str = "", title_x: float | None = None):
    """Like panel(), but with no axes: for tables drawn with ax.text (x and y run from 0 to 1)."""
    ax = panel(fig, rect, title, subtitle, title_x=title_x)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def footer(fig, text: str) -> None:
    """One short line at the bottom of the page (for example what stays unexplained)."""
    fig.text(LEFT, 0.04, text, fontsize=9.5, color=INK2, va="center")


def save(fig, name: str) -> None:
    """Write reports/dashboards/<name>.pdf (for the submission) and <name>.png (for a quick look)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    print(f"Saved {OUT_DIR / name}.pdf and .png")
