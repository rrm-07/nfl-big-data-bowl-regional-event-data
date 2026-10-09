"""Defender attention -> receiver gravity, plus target separation.

Each coverage defender spreads one unit of attention over the eligible receivers
with a softmax on negative distance:
    w(d, r) = exp(-dist(d, r) / tau) / sum_r' exp(-dist(d, r') / tau)
Gravity G(r) = sum_d w(d, r) = "defenders' worth of attention" on receiver r.
"""
import numpy as np
import pandas as pd

KEY = ["gameId", "playId"]
TAU = 2.0       # yards
MAX_DIST = 15.0  # defenders farther than this from every receiver are ignored


def pairs_at(frames, moment="throw"):
    """All defender x receiver pairs on each play at one moment, with distance."""
    fr = frames[frames.moment == moment]
    rec = fr.loc[fr.is_receiver, KEY + ["nflId", "x", "y"]]
    dfn = fr.loc[fr.is_defender, KEY + ["nflId", "x", "y"]]
    p = dfn.merge(rec, on=KEY, suffixes=("_d", "_r"))
    p["dist"] = np.hypot(p.x_d - p.x_r, p.y_d - p.y_r)
    return p


def attention(pairs, tau=TAU, max_dist=MAX_DIST):
    """Softmax attention weight w(d, r) for every defender-receiver pair."""
    by_def = pairs.groupby(KEY + ["nflId_d"]).dist
    dmin = by_def.transform("min")
    p = pairs[dmin <= max_dist].copy()
    z = np.exp(-(p.dist - dmin[dmin <= max_dist]) / tau)  # shift by min for numerical stability
    p["w"] = z / z.groupby([p.gameId, p.playId, p.nflId_d]).transform("sum")
    return p


def gravity(frames, tau=TAU, max_dist=MAX_DIST, moment="throw"):
    """One row per eligible receiver per play: G = summed attention (0 if nobody is near)."""
    att = attention(pairs_at(frames, moment), tau, max_dist)
    g = att.groupby(KEY + ["nflId_r"]).w.sum().rename("G").reset_index().rename(columns={"nflId_r": "nflId"})
    fr = frames[(frames.moment == moment) & frames.is_receiver]
    out = fr[KEY + ["nflId", "displayName", "officialPosition", "pos_group", "is_target"]].merge(g, on=KEY + ["nflId"], how="left")
    out["G"] = out.G.fillna(0.0)
    return out, att


def separation(frames, moment="throw"):
    """Distance from the targeted receiver to the nearest defender (any defender, NGS-style)."""
    fr = frames[frames.moment == moment]
    tgt = fr.loc[fr.is_target & fr.is_receiver, KEY + ["x", "y"]]
    dfn = fr.loc[fr.side == "def", KEY + ["x", "y"]]
    m = tgt.merge(dfn, on=KEY, suffixes=("_t", "_d"))
    m["dist"] = np.hypot(m.x_t - m.x_d, m.y_t - m.y_d)
    return m.groupby(KEY).dist.min().rename(f"sep_{moment}")
