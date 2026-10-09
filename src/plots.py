"""Shared plotting: one colour-blind-safe palette (Okabe-Ito) used across all figures."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

C_TARGET = "#E69F00"  # amber  - targeted receiver
C_DECOY = "#0072B2"   # blue   - other eligible receivers (decoys)
C_DEF = "#D55E00"     # vermilion - coverage defenders
C_OTHER = "#A0A0A0"   # grey   - linemen, QB, pass rushers, secondary series
C_FIELD = "#F4F7F1"
C_YARD = "#CBD5C3"
POS_COLORS = {"WR": "#0072B2", "TE": "#009E73", "RB": "#CC79A7"}  # leaderboard bars

X_BINNED = "Total decoy metric on the play (defenders' worth of attention vs expected)"
Y_BINNED = "Target separation at the throw (yd)"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})


def titles(ax, title, subtitle=None, size=16):
    """Bold left-aligned title with a lighter subtitle underneath (same style on every chart)."""
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=size - 4, fontweight="normal", color="#444444", pad=8)
        extra = subtitle.count("\n") * (size - 4) * 1.2
        above = offset_copy(ax.transAxes, fig=ax.figure, y=size + 13 + extra, units="points")
        ax.text(0, 1, title, transform=above, fontsize=size, fontweight="bold", ha="left", va="bottom")
    else:
        ax.set_title(title, loc="left", fontsize=size, pad=10)


# ---------------------------------------------------------------- binned effect (Step 4)
def draw_binned(ax, series, xlabel=X_BINNED, ylabel=Y_BINNED):
    """series = {label: (binned_df, colour, linestyle)} from analysis.binned."""
    for label, (b, col, ls) in series.items():
        ax.errorbar(b.x_mean, b["mean"], yerr=b.ci95, color=col, ls=ls, marker="o", ms=8, lw=2.4, capsize=4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E5E5")
    ax.axvline(0, color="#BBBBBB", lw=1, ls=":")
    ax.legend(frameon=False, loc="upper left")


def binned_effect(series, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_binned(ax, series)
    titles(ax, title, subtitle)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- leaderboard
def draw_leaderboard(ax, lb, n=10, value="pull", second="goe_pre", fs=12):
    """Horizontal bars (value +/- 95% CI) coloured by position, with the second metric as a text column."""
    top = lb.head(n).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(top))
    lo, hi = top[value] - top.ci95, top[value] + top.ci95
    ax.barh(y, top[value], color=[POS_COLORS.get(g, C_OTHER) for g in top.pos_group], height=0.68, zorder=2)
    ax.errorbar(top[value], y, xerr=top.ci95, fmt="none", ecolor="#333333", elinewidth=1.3, capsize=3, zorder=3)
    for yi, v in zip(y, top[value]):
        ax.text(0.004, yi, f"+{v:.2f}", va="center", ha="left", color="white", fontsize=fs - 1, fontweight="bold", zorder=4)
    ax.set_yticks(y, [f"{r.player}  ({r.team} {r.pos})" for r in top.itertuples()], fontsize=fs)
    xcol = hi.max() * 1.13
    for yi, s in zip(y, top[second]):
        ax.text(xcol, yi, f"{s:+.2f}", va="center", ha="center", fontsize=fs - 1, color="#444444")
    ax.text(xcol, len(top) - 0.55, "Attention\nlevel", ha="center", va="bottom", fontsize=fs - 2, color="#444444")
    ax.set_xlim(min(0, lo.min()) - 0.01, hi.max() * 1.25)
    ax.set_ylim(-0.6, len(top) + 0.45)
    ax.axvline(0, color="#888888", lw=1)
    ax.grid(axis="x", color="#EEEEEE", zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=c, label=k) for k, c in POS_COLORS.items() if k in set(top.pos_group)]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=fs - 1)


def leaderboard_chart(lb, path, title, subtitle, xlabel, n=10):
    fig, ax = plt.subplots(figsize=(11, 6.8))
    draw_leaderboard(ax, lb, n=n)
    ax.set_xlabel(xlabel)
    titles(ax, title, subtitle)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- play diagrams
def draw_field(ax, x0, x1, los=None):
    """Field background between x0 and x1 (yards), yard lines every 5, numbers every 10."""
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 53.3, color=C_FIELD, zorder=0))
    for x in range(int(x0 // 5 * 5), int(x1) + 1, 5):
        if x0 <= x <= x1:
            ax.axvline(x, color=C_YARD, lw=1.2 if x % 10 == 0 else 0.6, zorder=1)
            if x % 10 == 0 and 10 < x < 110:
                ax.text(x, 2, str(x - 10 if x <= 60 else 110 - x), ha="center", color="#8A9A80", fontsize=10, zorder=1)
    for y in (0, 53.3):
        ax.axhline(y, color="#6F7F66", lw=1.5, zorder=1)
    if los is not None:
        ax.axvline(los, color=C_DECOY, ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-1, 54.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_play(ax, fr, att, grav, los=None, highlight=None, label_g=True, xlim=None, fs=10):
    """One moment of a play.

    fr    - tracking rows for this play & moment (all 22 players + ball)
    att   - attention pairs for this play (x_d, y_d, x_r, y_r, w, nflId_r)
    grav  - gravity rows for this play (nflId, displayName, G, is_target)
    highlight - nflId of a receiver whose incoming attention lines are emphasised (the decoy)
    """
    x0, x1 = xlim if xlim else (max(0, fr.x.min() - 4), min(120, fr.x.max() + 4))
    draw_field(ax, x0, x1, los)

    for r in att.itertuples():
        if highlight is None:
            col, lw, al = C_DEF, 0.3 + 6 * r.w, 0.15 + 0.6 * r.w
        elif r.nflId_r == highlight:
            col, lw, al = C_DEF, 0.8 + 7 * r.w, 0.35 + 0.6 * r.w
        else:
            col, lw, al = C_OTHER, 0.3 + 3 * r.w, 0.15 + 0.3 * r.w
        ax.plot([r.x_d, r.x_r], [r.y_d, r.y_r], color=col, lw=lw, alpha=al, zorder=2, solid_capstyle="round")

    other = fr[(fr.side != "ball") & ~fr.is_receiver & ~fr.is_defender]
    ax.scatter(other.x, other.y, s=55, color=C_OTHER, alpha=0.6, zorder=3, edgecolor="white", lw=0.5)
    d = fr[fr.is_defender]
    ax.scatter(d.x, d.y, s=100, marker="X", color=C_DEF, zorder=4, edgecolor="white", lw=0.6)
    rec = fr[fr.is_receiver].merge(grav[["nflId", "G"]], on="nflId")
    ax.scatter(rec.x, rec.y, s=150, color=[C_TARGET if t else C_DECOY for t in rec.is_target],
               zorder=5, edgecolor="black", lw=0.8)
    if highlight is not None:
        h = rec[rec.nflId == highlight]
        ax.scatter(h.x, h.y, s=520, facecolors="none", edgecolors=C_DEF, lw=2.2, zorder=5)
    b = fr[fr.side == "ball"]
    ax.scatter(b.x, b.y, s=45, marker="o", color="#7B3F00", zorder=6, edgecolor="white", lw=0.6)
    for r in rec.itertuples():
        strong = r.is_target or r.nflId == highlight
        show_g = label_g and (highlight is None or strong)  # with a decoy highlighted, only label the key two with G
        txt = r.displayName.split(" ", 1)[-1] + (f"\nG = {r.G:.2f}" if show_g else "")
        below = r.is_target and highlight is not None and r.y > 8  # keep target and decoy labels apart
        ax.annotate(txt, (r.x, r.y), xytext=(8, -10 if below else 8), textcoords="offset points",
                    va="top" if below else "bottom", fontsize=fs + (1 if strong else -1),
                    zorder=7, fontweight="bold" if strong else "normal", color="black" if strong else "#333333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def draw_separation(ax, sep, fs=11):
    """sep = dict(x_t, y_t, x_d, y_d, dist): dashed line from the target to his nearest defender."""
    ax.plot([sep["x_t"], sep["x_d"]], [sep["y_t"], sep["y_d"]], color=C_TARGET, lw=2.5, ls="--", zorder=6)
    ax.annotate(f"{sep['dist']:.1f} yd of\nseparation", (min(sep["x_t"], sep["x_d"]), (sep["y_t"] + sep["y_d"]) / 2),
                xytext=(-16, 0), textcoords="offset points", ha="right", va="center", fontsize=fs, fontweight="bold",
                color="#8A5A00", zorder=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_TARGET, lw=1.2))


PLAY_LEGEND = [
    Line2D([], [], marker="o", ls="", color=C_TARGET, mec="black", ms=11, label="Targeted receiver"),
    Line2D([], [], marker="o", ls="", color=C_DECOY, mec="black", ms=11, label="Other receivers"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=C_DEF, mew=2, ms=15, label="Decoy"),
    Line2D([], [], marker="X", ls="", color=C_DEF, ms=11, label="Coverage defender"),
    Line2D([], [], color=C_DEF, lw=3.5, label="Attention on the decoy (width = weight)"),
    Line2D([], [], color=C_OTHER, lw=2, label="Attention on others"),
]


def play_panels(path, panels, xlim, title, subtitle):
    """Side-by-side moments of one play. panels = list of dict(fr, att, grav, los, highlight, name, sep)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 8.2))
    for ax, pnl in zip(axes, panels):
        draw_play(ax, pnl["fr"], pnl["att"], pnl["grav"], los=pnl["los"], highlight=pnl["highlight"], xlim=xlim, fs=11)
        if pnl.get("sep"):
            draw_separation(ax, pnl["sep"])
        ax.set_title(pnl["name"], loc="left", fontsize=15, pad=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10, wspace=0.04)
    fig.legend(handles=PLAY_LEGEND, loc="lower center", ncol=6, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(0.02, 0.925, subtitle, ha="left", va="top", fontsize=13, color="#444444", linespacing=1.4)
    fig.savefig(path)
    plt.close(fig)


def play_gif(path, seq, xlim, los, highlight, title):
    """Animate a play. seq = list of dict(fr, att, grav, label) per frame (10 Hz)."""
    fig, ax = plt.subplots(figsize=(8.5, 8.2), dpi=72)

    def update(i):
        ax.clear()
        s = seq[i]
        draw_play(ax, s["fr"], s["att"], s["grav"], los=los, highlight=highlight, xlim=xlim, fs=11)
        if s.get("sep"):
            draw_separation(ax, s["sep"])
        ax.set_title(f"{title}\n{s['label']}", loc="left", fontsize=14)

    anim = FuncAnimation(fig, update, frames=len(seq), interval=100)
    anim.save(path, writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- one-page summary
def summary_page(path, title, definition, series, chart_title, chart_subtitle, lb, callouts, footnote):
    """16:9 infographic. callouts = list of (heading, body, colour)."""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1.02],
                          left=0.055, right=0.975, top=0.72, bottom=0.165, wspace=0.42, hspace=0.62)
    fig.text(0.055, 0.955, title, fontsize=27, fontweight="bold", va="top")
    fig.text(0.055, 0.885, definition, fontsize=14, color="#333333", va="top")

    ax1 = fig.add_subplot(gs[:, 0])
    draw_binned(ax1, series)
    titles(ax1, chart_title, chart_subtitle, size=15)

    ax2 = fig.add_subplot(gs[0, 1])
    draw_leaderboard(ax2, lb, n=5, fs=12)
    ax2.set_xlabel("Pull per decoy route (defenders' worth of attention vs expected)", fontsize=11)
    ax2.get_legend().remove()  # position is already in the tick labels
    titles(ax2, "Top 5 decoys", "Pull per non-targeted route, 95% CI (min 50 routes)", size=15)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    y = 1.12
    for head, body, col in callouts:
        text = textwrap.fill(body, 74)
        h = 0.11 + 0.085 * (text.count("\n") + 1)  # header + body lines, in axes fractions
        ax3.add_patch(plt.Rectangle((-0.03, y - h), 0.012, h, color=col, transform=ax3.transAxes, clip_on=False))
        ax3.text(0.0, y, head, fontsize=14.5, fontweight="bold", color=col, va="top", transform=ax3.transAxes)
        ax3.text(0.0, y - 0.11, text, fontsize=12, va="top", color="#222222", transform=ax3.transAxes)
        y -= h + 0.07

    fig.text(0.055, 0.02, textwrap.fill(footnote, 190), fontsize=10.5, color="#666666", va="bottom")
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)
