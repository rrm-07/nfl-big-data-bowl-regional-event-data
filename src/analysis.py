"""Play-level tests (does decoy gravity create space?) and player / team summaries.

A "GOE table" has one row per receiver per play with columns
gameId, playId, nflId, displayName, officialPosition, pos_group, is_target, week, GOE.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.metric import separation

KEY = ["gameId", "playId"]


# ---------------------------------------------------------------- play level
def play_table(goe, frames, plays_df, goe_col="GOE"):
    """One row per play: total decoy GOE, the target's separation, depth and the play outcome."""
    nt = goe[~goe.is_target].groupby(KEY)[goe_col].agg(decoy_goe="sum", n_decoys="size")
    p = plays_df.set_index(KEY)[["week", "los_x", "passResult", "prePenaltyPlayResult", "pre_ok",
                                 "pff_passCoverageType", "possessionTeam"]].join(nt)
    p[["decoy_goe", "n_decoys"]] = p[["decoy_goe", "n_decoys"]].fillna(0)
    p = p.join(goe[goe.is_target].set_index(KEY)[goe_col].rename("target_goe"))
    p = p.join(separation(frames, "throw")).join(separation(frames, "post_throw"))
    tgt = frames[(frames.moment == "throw") & frames.is_target & frames.is_receiver].set_index(KEY)
    p["target_id"] = tgt.nflId
    p["target_depth"] = tgt.x - p.los_x  # yards downfield of the LOS at the throw
    p["complete"] = (p.passResult == "C").astype(int)
    p["yards"] = p.prePenaltyPlayResult
    return p.reset_index()


def binned(p, x, y, q=5):
    """Quantile bins of x: mean y with a 95% CI."""
    b = pd.qcut(p[x], q)
    g = p.groupby(b, observed=True).agg(x_mean=(x, "mean"), mean=(y, "mean"), sd=(y, "std"), n=(y, "size"))
    g["ci95"] = 1.96 * g.sd / np.sqrt(g.n)
    return g.reset_index(drop=True)


def _fit(df, formula):
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df.gameId})


def ols(p, y, x="decoy_goe", controls=("target_depth",)):
    """OLS with game-clustered SEs. Returns the coefficient on x, its 95% CI and p-value."""
    df = p[[y, x, *controls, "gameId"]].dropna()
    fit = _fit(df, f"{y} ~ {x} + " + " + ".join(controls))
    lo, hi = fit.conf_int().loc[x]
    return {"outcome": y, "coef": fit.params[x], "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[x],
            "per_sd": fit.params[x] * df[x].std(), "n": len(df)}


def coverage_split(p, x="decoy_goe_pre", y="sep_throw"):
    """Effect of decoy GOE on target separation in man vs zone, plus a man x decoy interaction test."""
    df = p[p.pff_passCoverageType.isin(["Man", "Zone"])].copy()
    rows = [{"coverage": c, **ols(df[df.pff_passCoverageType == c], y, x)} for c in ["Man", "Zone"]]
    df["man"] = (df.pff_passCoverageType == "Man").astype(int)
    df = df[[y, x, "man", "target_depth", "gameId"]].dropna()
    fit = _fit(df, f"{y} ~ {x} * man + target_depth")
    term = f"{x}:man"
    lo, hi = fit.conf_int().loc[term]
    rows.append({"coverage": "Man minus Zone (interaction)", "outcome": y, "coef": fit.params[term],
                 "ci_lo": lo, "ci_hi": hi, "p": fit.pvalues[term], "n": len(df)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- player / team level
def nontarget(goe, plays_df):
    """Non-targeted routes on plays with a valid 1-s-before-throw frame, with the offense attached."""
    ok = plays_df.loc[plays_df.pre_ok, KEY + ["possessionTeam"]]
    return goe[~goe.is_target].merge(ok, on=KEY)


def split_half(goe, plays_df, by="nflId", min_n=10, split="weeks"):
    """Correlate mean non-targeted GOE per player (or team) across two halves of the season.

    split="weeks": weeks 1-4 vs 5-8;  split="odd_even": odd vs even weeks.
    """
    nt = nontarget(goe, plays_df)
    first = nt.week <= 4 if split == "weeks" else nt.week % 2 == 1
    m = nt.assign(half=np.where(first, "h1", "h2")).groupby([by, "half"]).GOE.agg(["mean", "size"]).unstack()
    ok = (m[("size", "h1")] >= min_n) & (m[("size", "h2")] >= min_n)
    m = m[ok]
    return {"r": float(m[("mean", "h1")].corr(m[("mean", "h2")])), "n": int(ok.sum())}


def leaderboard(main, plays_df, extra=None, min_n=50, name="goe_pre"):
    """Mean GOE per player on routes where he was NOT targeted, ranked by the `main` table.

    extra = {column_name: other GOE table} adds other versions of the metric alongside.
    """
    nt = nontarget(main, plays_df)
    lb = nt.groupby("nflId").agg(player=("displayName", "first"), pos=("officialPosition", "first"),
                                 pos_group=("pos_group", "first"),
                                 team=("possessionTeam", lambda s: s.mode().iat[0]),
                                 decoy_routes=("GOE", "size"), **{name: ("GOE", "mean")}, sd=("GOE", "std"))
    lb["ci95"] = 1.96 * lb.sd / np.sqrt(lb.decoy_routes)
    for col, t in (extra or {}).items():
        lb = lb.join(nontarget(t, plays_df).groupby("nflId").GOE.mean().rename(col))
    lb = lb[lb.decoy_routes >= min_n].sort_values(name, ascending=False).drop(columns="sd")
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    return lb.reset_index()


def team_board(goe, plays_df):
    """Mean non-targeted GOE per route for each offense."""
    nt = nontarget(goe, plays_df)
    t = nt.groupby("possessionTeam").GOE.agg(goe_pre="mean", sd="std", routes="size")
    t["ci95"] = 1.96 * t.sd / np.sqrt(t.routes)
    return t.drop(columns="sd").sort_values("goe_pre", ascending=False).reset_index()


def receiving_profile(p, goe_throw):
    """Per player: routes, targets, targets per route, and separation when targeted.

    target_sep_adj = separation minus what's typical for that target depth (5-yd bins),
    so deep and short targets are comparable.
    """
    routes = goe_throw.groupby("nflId").agg(routes=("is_target", "size"), targets=("is_target", "sum"))
    routes["tprr"] = routes.targets / routes.routes
    t = p[["sep_throw", "target_depth", "target_id"]].dropna().copy()
    t["depth_bin"] = (t.target_depth.clip(-10, 40) // 5).astype(int)
    t["sep_adj"] = t.sep_throw - t.groupby("depth_bin").sep_throw.transform("mean")
    s = t.groupby("target_id").agg(target_sep=("sep_throw", "mean"), target_sep_adj=("sep_adj", "mean"))
    return routes.join(s)
