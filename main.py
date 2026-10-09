"""Gravity: end-to-end pipeline. Run from the repo root:  python main.py

Writes every number used in the README to outputs/*.csv and every figure to outputs/*.png|gif.
"""
import os
import time

import pandas as pd

from src import plots as P
from src.analysis import (binned, coverage_split, leaderboard, ols, play_table, receiving_profile,
                          split_half, team_board)
from src.clean import build
from src.metric import TAU, gravity
from src.model import features, oof_gbm, score

pd.set_option("display.width", 250)
OUT = "outputs"
KEY = ["gameId", "playId"]
BASELINE = "v2"  # chosen over v1: higher out-of-fold R2 and (slightly) more stable player GOE
MIN_ROUTES = 50  # minimum non-targeted routes for the leaderboard (20 let small-sample players into the top 10)


def step(msg):
    print(f"\n{'=' * 8} {msg} {'=' * 8}")


def show(df, n=None, digits=3):
    print((df.head(n) if n else df).round(digits).to_string(index=False))


def goe_at(f, plays_df, moment, tau=TAU, keep=None):
    """GOE table at one moment: gravity -> context features -> out-of-fold expected G -> GOE."""
    g, _ = gravity(f, tau=tau, moment=moment)
    d = features(f, g, plays_df)
    if keep is not None:
        d = d.merge(keep, on=KEY)
    d = d.reset_index(drop=True)
    d["xG"] = oof_gbm(d, BASELINE)
    d["GOE"] = d.G - d.xG
    return d, g


def pull_goe(f, plays_df, keep, tau=TAU):
    """'Pull': change in attention from the snap to 1 s before the throw, adjusted the same way."""
    g_pre, _ = gravity(f, tau=tau, moment="pre_throw")
    g_snap, _ = gravity(f, tau=tau, moment="snap")
    dg = g_pre.merge(g_snap[KEY + ["nflId", "G"]], on=KEY + ["nflId"], suffixes=("", "_snap"))
    dg["G"] = dg.G - dg.G_snap  # > 0: defenders moved toward him after the snap
    d = features(f, dg.drop(columns="G_snap"), plays_df).merge(keep, on=KEY).reset_index(drop=True)
    d["GOE"] = d.G - oof_gbm(d, BASELINE)
    return d


def core():
    step("Step 1: clean + standardise")
    plays_df, f, info = build()
    pre_ok = plays_df.loc[plays_df.pre_ok, KEY]  # drops 6 throws made < 1 s after the snap
    print(f"{len(plays_df)} plays | target matched on {info['target_match_rate']:.1%} of {info['n_thrown']} thrown passes")

    step("Step 2-3: gravity, expected gravity, GOE")
    d, _ = goe_at(f, plays_df, "throw")
    d_v1 = d.assign(GOE=d.G - oof_gbm(d, "v1"))
    fit = pd.DataFrame([{"baseline": "v1", **score(d.G, d.G - d_v1.GOE)}, {"baseline": BASELINE, **score(d.G, d.xG)}])
    show(fit)
    d_pre, _ = goe_at(f, plays_df, "pre_throw", keep=pre_ok)  # the fair version: 1 s before the throw
    d_pull = pull_goe(f, plays_df, pre_ok)

    step("Step 4: does decoy gravity create space?")
    p = play_table(d, f, plays_df)
    for name, t in [("decoy_goe_pre", d_pre), ("decoy_pull", d_pull)]:
        p = p.merge(play_table(t, f, plays_df)[KEY + ["decoy_goe"]].rename(columns={"decoy_goe": name}), on=KEY)
    p_pre = p[p.pre_ok]
    rows = []
    for x, label, df in [("decoy_goe", "GOE at throw", p), ("decoy_goe_pre", "GOE 1 s before throw", p_pre),
                         ("decoy_pull", "Pull (snap -> 1 s before throw)", p_pre)]:
        for y in ["sep_throw", "sep_post_throw", "complete", "yards"]:
            rows.append({"decoy metric": label, **ols(df, y, x)})
    effects = pd.DataFrame(rows)
    effects.to_csv(f"{OUT}/effects.csv", index=False)
    print("outcome ~ decoy metric + target depth (game-clustered SEs); coef per +1 defender of decoy attention")
    show(effects, digits=4)

    b_throw, b_pre = binned(p, "decoy_goe", "sep_throw"), binned(p_pre, "decoy_goe_pre", "sep_throw")
    e = effects.set_index(["decoy metric", "outcome"])
    eff_thr, eff_pre = e.loc[("GOE at throw", "sep_throw")], e.loc[("GOE 1 s before throw", "sep_throw")]
    P.binned_effect(
        {"Decoy GOE 1.0 s before the throw": (b_pre, P.C_DECOY, "-"), "Decoy GOE at the throw": (b_throw, P.C_OTHER, "--")},
        f"{OUT}/decoy_goe_vs_separation.png",
        title="Decoys that hold defenders open space for the target",
        subtitle=(f"+{eff_pre.coef:.2f} yd target separation per extra defender pulled 1 s before the throw "
                  f"(+{eff_thr.coef:.2f} yd measured at the throw)"),
        xlabel="Total decoy GOE on the play (defenders' worth of attention above expected)",
        ylabel="Target separation at the throw (yd)")
    return dict(plays_df=plays_df, f=f, pre_ok=pre_ok, d=d, d_pre=d_pre, d_pull=d_pull, p=p, p_pre=p_pre,
                effects=effects, b_pre=b_pre, b_throw=b_throw)


def step5(c):
    step("Step 5: player leaderboard (non-targeted routes)")
    plays_df = c["plays_df"]
    lb = leaderboard(c["d_pre"], plays_df, extra={"goe_throw": c["d"], "pull": c["d_pull"]}, min_n=MIN_ROUTES)
    lb = lb.join(receiving_profile(c["p"], c["d"]), on="nflId")
    lb["pull_rank"] = lb.pull.rank(ascending=False).astype(int)
    lb.to_csv(f"{OUT}/leaderboard.csv", index=False)
    cols = ["rank", "player", "team", "pos", "decoy_routes", "goe_pre", "ci95", "goe_throw", "pull_rank"]
    print(f"{len(lb)} players with >= {MIN_ROUTES} non-targeted routes")
    show(lb[cols], 10)
    print("bottom 5:")
    show(lb[cols].tail(5))

    stab = []
    for who, by in [("player", "nflId"), ("team", "possessionTeam")]:
        for name, t in [("GOE 1 s before throw", c["d_pre"]), ("GOE at throw", c["d"]), ("Pull", c["d_pull"])]:
            for split in ["weeks", "odd_even"]:
                r = split_half(t, plays_df, by=by, split=split)
                stab.append({"level": who, "metric": name,
                             "split": "weeks 1-4 vs 5-8" if split == "weeks" else "odd vs even weeks", **r})
    stab = pd.DataFrame(stab)
    stab.to_csv(f"{OUT}/stability.csv", index=False)
    print("\nsplit-half stability (>= 10 non-targeted routes per half):")
    show(stab)
    c.update(lb=lb, stab=stab)


def block_a(c):
    plays_df, lb, stab, effects = c["plays_df"], c["lb"], c["stab"], c["effects"]

    step("A1: validity - 'drags defenders' or 'can't get open'?")
    v = lb[lb.targets >= 10]
    st = stab[stab.level == "player"].set_index(["metric", "split"]).r
    e = effects[effects.outcome == "sep_throw"].set_index("decoy metric")
    stars = lb[lb.pos_group == "WR"].nlargest(10, "targets")
    comp = pd.DataFrame([
        {"metric": "GOE 1 s before throw", "effect_yd": e.loc["GOE 1 s before throw", "coef"],
         "ci_lo": e.loc["GOE 1 s before throw", "ci_lo"], "ci_hi": e.loc["GOE 1 s before throw", "ci_hi"],
         "r_weeks": st[("GOE 1 s before throw", "weeks 1-4 vs 5-8")], "r_odd_even": st[("GOE 1 s before throw", "odd vs even weeks")],
         "r_with_own_sep": v.goe_pre.corr(v.target_sep_adj), "star_wr_median_rank": stars["rank"].median()},
        {"metric": "Pull (snap -> 1 s before throw)", "effect_yd": e.loc["Pull (snap -> 1 s before throw)", "coef"],
         "ci_lo": e.loc["Pull (snap -> 1 s before throw)", "ci_lo"], "ci_hi": e.loc["Pull (snap -> 1 s before throw)", "ci_hi"],
         "r_weeks": st[("Pull", "weeks 1-4 vs 5-8")], "r_odd_even": st[("Pull", "odd vs even weeks")],
         "r_with_own_sep": v.pull.corr(v.target_sep_adj), "star_wr_median_rank": stars.pull_rank.median()},
    ])
    comp.to_csv(f"{OUT}/metric_comparison.csv", index=False)
    print(f"players with >= {MIN_ROUTES} decoy routes and >= 10 targets: {len(v)}")
    show(comp)
    print("\n10 most-targeted WRs: rank by GOE vs rank by pull (of", len(lb), ")")
    show(stars[["player", "team", "targets", "target_sep_adj", "rank", "pull_rank"]])
    lb_pull = leaderboard(c["d_pull"], plays_df, extra={"goe_pre": c["d_pre"]}, min_n=MIN_ROUTES, name="pull")
    lb_pull.to_csv(f"{OUT}/leaderboard_pull.csv", index=False)
    print("\nTop 10 by pull:")
    show(lb_pull[["rank", "player", "team", "pos", "decoy_routes", "pull", "ci95", "goe_pre"]], 10)

    step("A2: man vs zone (decoy GOE 1 s before throw -> target separation at throw)")
    cov = pd.concat([coverage_split(c["p_pre"], "decoy_goe_pre").assign(metric="GOE 1 s before throw"),
                     coverage_split(c["p_pre"], "decoy_pull").assign(metric="Pull")])
    cov.to_csv(f"{OUT}/coverage_split.csv", index=False)
    show(cov[["metric", "coverage", "coef", "ci_lo", "ci_hi", "p", "n"]], digits=4)

    step("A3: team leaderboard (scheme vs skill)")
    tb = team_board(c["d_pre"], plays_df)
    tb.to_csv(f"{OUT}/team_leaderboard.csv", index=False)
    print("top 5:")
    show(tb, 5)
    print("bottom 5:")
    show(tb.tail(5))
    print("team vs player split-half r (GOE 1 s before throw):")
    show(stab[stab.metric == "GOE 1 s before throw"])

    step("A4: hidden heroes (top 15 decoy GOE, bottom half for targets per route)")
    med = lb.tprr.median()
    heroes = lb.head(15)[lambda t: t.tprr < med]
    heroes.to_csv(f"{OUT}/hidden_heroes.csv", index=False)
    print(f"median targets per route among the {len(lb)} players: {med:.3f}")
    show(heroes[["rank", "player", "team", "pos", "decoy_routes", "goe_pre", "tprr", "targets"]])

    step("A5: tau sensitivity (top 10 overlap with tau = 2)")
    rows = []
    base = {"GOE 1 s before throw": lb.set_index("nflId").goe_pre,
            "Pull": lb.set_index("nflId").pull}
    for tau in [1.5, 3.0]:
        d_tau, _ = goe_at(c["f"], plays_df, "pre_throw", tau=tau, keep=c["pre_ok"])
        for name, t in [("GOE 1 s before throw", d_tau), ("Pull", pull_goe(c["f"], plays_df, c["pre_ok"], tau=tau))]:
            alt = leaderboard(t, plays_df, min_n=MIN_ROUTES).set_index("nflId").goe_pre
            top_b, top_a = set(base[name].nlargest(10).index), set(alt.nlargest(10).index)
            both = base[name].index.intersection(alt.index)
            rows.append({"metric": name, "tau": tau, "top10_overlap": len(top_b & top_a),
                         "spearman_all_players": base[name][both].corr(alt[both], method="spearman")})
    tau_df = pd.DataFrame(rows)
    tau_df.to_csv(f"{OUT}/tau_sensitivity.csv", index=False)
    show(tau_df)
    c.update(comp=comp, lb_pull=lb_pull, cov=cov, tb=tb, heroes=heroes, tau_df=tau_df)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    c = core()
    step5(c)
    block_a(c)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return c


if __name__ == "__main__":
    main()
