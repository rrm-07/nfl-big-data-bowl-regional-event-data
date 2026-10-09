"""Shared plotting: one colour-blind-safe palette (Okabe-Ito) used across all figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_TARGET = "#E69F00"  # amber  - targeted receiver
C_DECOY = "#0072B2"   # blue   - other eligible receivers (decoys)
C_DEF = "#D55E00"     # vermilion - coverage defenders
C_OTHER = "#A0A0A0"   # grey   - linemen, QB, pass rushers
C_FIELD = "#F4F7F1"
C_YARD = "#CBD5C3"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def draw_field(ax, x0, x1, los=None):
    """Field background between x0 and x1 (yards), yard lines every 5, numbers every 10."""
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 53.3, color=C_FIELD, zorder=0))
    for x in range(int(x0 // 5 * 5), int(x1) + 1, 5):
        if x0 <= x <= x1:
            ax.axvline(x, color=C_YARD, lw=1.2 if x % 10 == 0 else 0.6, zorder=1)
            if x % 10 == 0 and 10 < x < 110:
                ax.text(x, 2, str(x - 10 if x <= 60 else 110 - x), ha="center", color="#8A9A80", fontsize=9, zorder=1)
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


def draw_play(ax, fr, att, grav, los=None, highlight=None, label_g=True):
    """One moment of a play.

    fr    - tracking rows for this play & moment (all 22 players + ball)
    att   - attention pairs for this play (x_d, y_d, x_r, y_r, w, nflId_r)
    grav  - gravity rows for this play (nflId, displayName, G, is_target)
    highlight - nflId of a receiver whose incoming attention lines are emphasised
    """
    xs = fr.x
    draw_field(ax, max(0, xs.min() - 4), min(120, xs.max() + 4), los)

    for r in att.itertuples():
        if highlight is None:
            col, lw, al = C_DEF, 0.3 + 6 * r.w, 0.15 + 0.6 * r.w
        elif r.nflId_r == highlight:
            col, lw, al = C_DEF, 0.6 + 7 * r.w, 0.35 + 0.6 * r.w
        else:
            col, lw, al = C_OTHER, 0.3 + 3 * r.w, 0.15 + 0.3 * r.w
        ax.plot([r.x_d, r.x_r], [r.y_d, r.y_r], color=col, lw=lw, alpha=al, zorder=2, solid_capstyle="round")

    other = fr[(fr.side != "ball") & ~fr.is_receiver & ~fr.is_defender]
    ax.scatter(other.x, other.y, s=55, color=C_OTHER, alpha=0.6, zorder=3, edgecolor="white", lw=0.5)
    d = fr[fr.is_defender]
    ax.scatter(d.x, d.y, s=90, marker="X", color=C_DEF, zorder=4, edgecolor="white", lw=0.6)
    rec = fr[fr.is_receiver].merge(grav[["nflId", "G"]], on="nflId")
    ax.scatter(rec.x, rec.y, s=130, color=[C_TARGET if t else C_DECOY for t in rec.is_target],
               zorder=5, edgecolor="black", lw=0.8)
    b = fr[fr.side == "ball"]
    ax.scatter(b.x, b.y, s=40, marker="o", color="#7B3F00", zorder=6, edgecolor="white", lw=0.6)
    for r in rec.itertuples():
        txt = r.displayName.split(" ", 1)[-1] + (f"\nG = {r.G:.2f}" if label_g else "")
        ax.annotate(txt, (r.x, r.y), xytext=(7, 7), textcoords="offset points", fontsize=9, zorder=7,
                    fontweight="bold" if r.is_target else "normal",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
