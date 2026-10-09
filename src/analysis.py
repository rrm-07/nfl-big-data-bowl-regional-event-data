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


def split_half(goe, goe_col="GOE", min_n=10):
    """Correlate each player's mean non-targeted GOE in weeks 1-4 vs weeks 5-8."""
    nt = goe[~goe.is_target].assign(half=lambda d: np.where(d.week <= 4, "h1", "h2"))
    m = nt.groupby(["nflId", "half"])[goe_col].agg(["mean", "size"]).unstack()
    ok = (m[("size", "h1")] >= min_n) & (m[("size", "h2")] >= min_n)
    m = m[ok]
    return {"r": float(m[("mean", "h1")].corr(m[("mean", "h2")])), "players": int(ok.sum())}
