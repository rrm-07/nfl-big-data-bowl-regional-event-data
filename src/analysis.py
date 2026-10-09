"""Play-level tests (does decoy gravity create space?) and player-level summaries."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.metric import separation

KEY = ["gameId", "playId"]


def play_table(goe, frames, plays_df, goe_col="GOE"):
    """One row per play: total decoy GOE, the target's separation, depth and the play outcome."""
    nt = goe[~goe.is_target].groupby(KEY)[goe_col].agg(decoy_goe="sum", n_decoys="size")
    p = plays_df.set_index(KEY)[["week", "los_x", "passResult", "prePenaltyPlayResult", "pre_ok"]].join(nt)
    p[["decoy_goe", "n_decoys"]] = p[["decoy_goe", "n_decoys"]].fillna(0)
    p = p.join(goe[goe.is_target].set_index(KEY)[goe_col].rename("target_goe"))
    p = p.join(separation(frames, "throw")).join(separation(frames, "post_throw"))
    tgt = frames[(frames.moment == "throw") & frames.is_target & frames.is_receiver].set_index(KEY).x
    p["target_depth"] = tgt - p.los_x  # yards downfield of the LOS at the throw
    p["complete"] = (p.passResult == "C").astype(int)
    p["yards"] = p.prePenaltyPlayResult
    return p.reset_index()


def binned(p, x, y, q=5):
    """Quantile bins of x: mean y with a 95% CI."""
    b = pd.qcut(p[x], q)
    g = p.groupby(b, observed=True).agg(x_mean=(x, "mean"), mean=(y, "mean"), sd=(y, "std"), n=(y, "size"))
    g["ci95"] = 1.96 * g.sd / np.sqrt(g.n)
    return g.reset_index(drop=True)


def ols(p, y, x="decoy_goe", controls=("target_depth",)):
    """OLS with game-clustered SEs. Returns the coefficient on x, its 95% CI and p-value."""
    cols = [y, x, *controls, "gameId"]
    df = p[cols].dropna()
    fit = smf.ols(f"{y} ~ {x} + " + " + ".join(controls), df).fit(cov_type="cluster", cov_kwds={"groups": df.gameId})
    lo, hi = fit.conf_int().loc[x]
    return {"outcome": y, "coef": fit.params[x], "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[x],
            "per_sd": fit.params[x] * df[x].std(), "n": len(df)}


def split_half(goe, goe_col="GOE", min_n=10, split="weeks"):
    """Correlate each player's mean non-targeted GOE across two halves of the season.

    split="weeks": weeks 1-4 vs 5-8;  split="odd_even": odd vs even weeks.
    """
    nt = goe[~goe.is_target]
    first = nt.week <= 4 if split == "weeks" else nt.week % 2 == 1
    nt = nt.assign(half=np.where(first, "h1", "h2"))
    m = nt.groupby(["nflId", "half"])[goe_col].agg(["mean", "size"]).unstack()
    ok = (m[("size", "h1")] >= min_n) & (m[("size", "h2")] >= min_n)
    m = m[ok]
    return {"r": float(m[("mean", "h1")].corr(m[("mean", "h2")])), "players": int(ok.sum())}


def leaderboard(d_pre, d_throw, plays_df, min_n=20):
    """Mean decoy GOE per player on routes where he was NOT targeted.

    Ranked by GOE measured 1 s before the throw (the fair version); at-throw GOE shown alongside.
    """
    ok = plays_df.loc[plays_df.pre_ok, KEY + ["possessionTeam"]]  # drop the few throws < 1 s after the snap
    pre = d_pre[~d_pre.is_target].merge(ok, on=KEY)
    thr = d_throw[~d_throw.is_target].merge(ok[KEY], on=KEY)
    lb = pre.groupby("nflId").agg(player=("displayName", "first"), pos=("pos_group", "first"),
                                  team=("possessionTeam", lambda s: s.mode().iat[0]),
                                  decoy_routes=("GOE", "size"), goe_pre=("GOE", "mean"), sd=("GOE", "std"))
    lb["ci95"] = 1.96 * lb.sd / np.sqrt(lb.decoy_routes)
    lb = lb.join(thr.groupby("nflId").GOE.mean().rename("goe_throw"))
    lb = lb[lb.decoy_routes >= min_n].sort_values("goe_pre", ascending=False)
    return lb.drop(columns="sd").reset_index()
