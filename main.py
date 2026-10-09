"""Gravity: end-to-end pipeline. Run from the repo root:  python main.py

Writes every number used in the README to outputs/*.csv and every figure to outputs/*.png|gif.
"""
import os
import time

import numpy as np
import pandas as pd

from src import plots as P
from src.analysis import (binned, coverage_split, leaderboard, ols, play_table, receiving_profile,
                          split_half, team_board)
from src.clean import build, load_tracking, standardise
from src.metric import TAU, gravity
from src.model import features, oof_gbm, score

pd.set_option("display.width", 250)
OUT = "outputs"
KEY = ["gameId", "playId"]
BASELINE = "v2"  # chosen over v1: higher out-of-fold R2 and (slightly) more stable player GOE
MIN_ROUTES = 50  # minimum non-targeted routes for the leaderboard (20 let small-sample players into the top 10)
PULL = "Pull (snap -> 1 s before throw)"  # lead metric; GOE level 1 s before the throw is the second column
C_HERO = "#CC79A7"


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
    pd.DataFrame([{"thrown_passes": info["n_thrown"], "target_match_rate": info["target_match_rate"],
                   "plays_analysed": len(plays_df), "plays_with_pre_throw_frame": len(pre_ok),
                   "games": plays_df.gameId.nunique(), "weeks": plays_df.week.nunique()}]
                 ).to_csv(f"{OUT}/data_summary.csv", index=False)

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
                         ("decoy_pull", PULL, p_pre)]:
        for y in ["sep_throw", "sep_post_throw", "complete", "yards"]:
            rows.append({"decoy metric": label, **ols(df, y, x)})
    effects = pd.DataFrame(rows)
    effects.to_csv(f"{OUT}/effects.csv", index=False)
    print("outcome ~ decoy metric + target depth (game-clustered SEs); coef per +1 defender of decoy attention")
    show(effects, digits=4)

    b_pull, b_pre = binned(p_pre, "decoy_pull", "sep_throw"), binned(p_pre, "decoy_goe_pre", "sep_throw")
    bins = pd.concat([b_pull.assign(metric="Pull"), b_pre.assign(metric="GOE 1 s before throw")])
    bins.to_csv(f"{OUT}/binned_separation.csv", index=False)
    print("\ntarget separation at the throw by quintile of total decoy PULL (hockey-stick check):")
    show(b_pull, digits=2)
    e = effects.set_index(["decoy metric", "outcome"])
    eff_pull, eff_pre = e.loc[(PULL, "sep_throw")], e.loc[("GOE 1 s before throw", "sep_throw")]
    series = {"Pull: attention gained after the snap": (b_pull, P.C_DECOY, "-"),
              "Attention level 1 s before the throw": (b_pre, P.C_OTHER, "--")}
    subtitle = (f"+{eff_pull.coef:.2f} yd target separation per extra defender pulled toward the decoys after the snap "
                f"(attention level: +{eff_pre.coef:.2f} yd)")
    P.binned_effect(series, f"{OUT}/decoy_pull_vs_separation.png",
                    title="Decoys that drag defenders open space for the target", subtitle=subtitle)
    return dict(plays_df=plays_df, f=f, pre_ok=pre_ok, d=d, d_pre=d_pre, d_pull=d_pull, p=p, p_pre=p_pre,
                effects=effects, series=series, eff_pull=eff_pull, eff_pre=eff_pre)


def step5(c):
    step("Step 5: player leaderboard (non-targeted routes)")
    plays_df = c["plays_df"]
    lb = leaderboard(c["d_pull"], plays_df, extra={"goe_pre": c["d_pre"], "goe_throw": c["d"]},
                     min_n=MIN_ROUTES, name="pull")
    lb = lb.join(receiving_profile(c["p"], c["d"]), on="nflId")
    lb["goe_rank"] = lb.goe_pre.rank(ascending=False).astype(int)
    lb.to_csv(f"{OUT}/leaderboard.csv", index=False)
    cols = ["rank", "player", "team", "pos", "decoy_routes", "pull", "ci95", "goe_pre", "goe_rank"]
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
    def row(metric, stab_name, col, rank_col):
        return {"metric": metric, "effect_yd": e.loc[metric, "coef"], "ci_lo": e.loc[metric, "ci_lo"],
                "ci_hi": e.loc[metric, "ci_hi"], "r_weeks": st[(stab_name, "weeks 1-4 vs 5-8")],
                "r_odd_even": st[(stab_name, "odd vs even weeks")], "r_with_own_sep": v[col].corr(v.target_sep_adj),
                "star_wr_median_rank": stars[rank_col].median()}

    comp = pd.DataFrame([row(PULL, "Pull", "pull", "rank"),
                         row("GOE 1 s before throw", "GOE 1 s before throw", "goe_pre", "goe_rank")])
    comp.to_csv(f"{OUT}/metric_comparison.csv", index=False)
    print(f"players with >= {MIN_ROUTES} decoy routes and >= 10 targets: {len(v)}")
    show(comp)
    print("\n10 most-targeted WRs: rank by pull vs rank by GOE level (of", len(lb), ")")
    show(stars[["player", "team", "targets", "target_sep_adj", "rank", "goe_rank"]])

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

    step("A4: hidden heroes (top 15 by pull, bottom half for targets per route)")
    med = lb.tprr.median()
    heroes = lb.head(15)[lambda t: t.tprr < med]
    heroes.to_csv(f"{OUT}/hidden_heroes.csv", index=False)
    print(f"median targets per route among the {len(lb)} players: {med:.3f}")
    show(heroes[["rank", "player", "team", "pos", "decoy_routes", "pull", "goe_pre", "tprr", "targets"]])

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
    c.update(comp=comp, cov=cov, tb=tb, heroes=heroes, tau_df=tau_df)


def _nearest_defender(fr):
    """Target position and his nearest defender (any defender) in one frame."""
    t = fr[fr.is_target & fr.is_receiver].iloc[0]
    dd = fr[fr.side == "def"]
    dist = np.hypot(dd.x - t.x, dd.y - t.y)
    j = dist.idxmin()
    return {"x_t": t.x, "y_t": t.y, "x_d": dd.x[j], "y_d": dd.y[j], "dist": float(dist[j])}


def example_play(c):
    """A strong decoy play: a top decoy pulling well above expected on a completed downfield pass.

    Prefer the top-5 decoys (the named examples); fall back to the top 10.
    """
    lb = c["lb"]
    cols = KEY + ["complete", "sep_throw", "target_depth", "target_id", "decoy_pull"]
    cand = c["d_pull"][~c["d_pull"].is_target & c["d_pull"].nflId.isin(lb.head(10).nflId)].merge(c["p_pre"][cols], on=KEY)
    cand = cand[(cand.complete == 1) & cand.sep_throw.between(3, 10) & (cand.target_depth >= 5) & (cand.GOE >= 1.0)
                & (cand.decoy_pull > 0)]  # the decoys as a group also pulled coverage on this play
    cand = cand.assign(top5=cand.nflId.isin(lb.head(5).nflId)).sort_values(["top5", "GOE"], ascending=False)
    print("example play candidates (decoy pull GOE, target separation):")
    show(cand[KEY + ["displayName", "top5", "G", "GOE", "decoy_pull", "sep_throw", "target_depth"]], 6)
    return cand.iloc[0]


def visuals(c):
    step("Block B: visuals")
    f, plays_df, lb = c["f"], c["plays_df"], c["lb"]

    # 1. leaderboard
    a, b = lb.iloc[0], lb.iloc[1]
    who = f"{a.player} and {b.player} ({a.team})" if a.team == b.team else f"{a.player} and {b.player}"
    P.leaderboard_chart(
        lb, f"{OUT}/leaderboard.png", title=f"{who} drag the most defenders without the ball",
        subtitle=("Pull: defenders' worth of attention a receiver gains from the snap to 1 s before the throw, vs expected,\n"
                  "per non-targeted route (min 50 routes, 2021 weeks 1-8). Whiskers: 95% CI. Right column: attention level."),
        xlabel="Pull per decoy route (defenders' worth of attention vs expected)")

    # 2-3. example play: static panels + GIF
    row = example_play(c)
    gid, pid, decoy = row.gameId, row.playId, row.nflId
    fp = f[(f.gameId == gid) & (f.playId == pid)]
    pl = plays_df[(plays_df.gameId == gid) & (plays_df.playId == pid)].iloc[0]
    los = fp[(fp.moment == "snap") & (fp.side == "ball")].x.iloc[0]
    tname = fp[fp.is_target & fp.is_receiver].displayName.iloc[0]
    dname = row.displayName
    panels, gvals = [], {}
    for mom, name in [("snap", "At the snap"), ("pre_throw", "1 s before the throw"), ("throw", "At the throw")]:
        grav, att = gravity(fp, moment=mom)
        gvals[mom] = grav.loc[grav.nflId == decoy, "G"].iloc[0]
        fr = fp[fp.moment == mom]
        panels.append(dict(fr=fr, att=att, grav=grav, los=los, highlight=decoy, name=name,
                           sep=_nearest_defender(fr) if mom == "throw" else None))
    sep = panels[-1]["sep"]["dist"]

    t = load_tracking()
    tp = standardise(t[(t.gameId == gid) & (t.playId == pid)])
    flags = fp[fp.moment == "snap"][["nflId", "displayName", "officialPosition", "pos_group", "side",
                                     "is_target", "is_receiver", "is_defender"]]
    tp = tp.merge(flags, on="nflId", how="left")
    tp["side"] = tp.side.fillna("ball")
    for col in ["is_target", "is_receiver", "is_defender"]:
        tp[col] = tp[col].fillna(False).astype(bool)
    tp = tp[(tp.frameId >= pl.snap) & (tp.frameId <= pl.post_throw)]
    xlim = (max(0, tp.x.min() - 3), min(120, tp.x.max() + 3))

    desc = pl.playDescription.split(") ", 1)[-1] if pl.playDescription.startswith("(") else pl.playDescription
    title = f"{dname} drags the defense: {tname} gets {sep:.1f} yd of space"
    subtitle = (f"{pl.possessionTeam} vs {pl.defensiveTeam}, 2021 week {pl.week}: {desc[:110]}\n"
                f"{dname}'s attention G: {gvals['snap']:.2f} at the snap -> {gvals['pre_throw']:.2f} one second before "
                f"the throw (pull {row.GOE:+.2f} defenders vs expected)")
    P.play_panels(f"{OUT}/play_example.png", panels, xlim, title, subtitle)

    seq = []
    for k, fr in tp.groupby("frameId"):
        fr = fr.assign(moment="m")
        grav, att = gravity(fr, moment="m")
        g = grav.loc[grav.nflId == decoy, "G"].iloc[0]
        phase = "THROW" if k == pl.throw else ("ball in the air" if k > pl.throw else "")
        seq.append(dict(fr=fr, att=att, grav=grav, sep=_nearest_defender(fr) if k >= pl.throw else None,
                        label=f"{(k - pl.snap) / 10:.1f} s after the snap   {dname} G = {g:.2f}   {phase}"))
    seq += [seq[-1]] * 8  # hold the last frame
    P.play_gif(f"{OUT}/play_example.gif", seq, xlim, los, decoy, title=f"{dname} drags the defense")

    # 4. one-page summary
    cov = c["cov"].set_index(["metric", "coverage"])
    man, zone = cov.loc[("Pull", "Man"), "coef"], cov.loc[("Pull", "Zone"), "coef"]
    p_int = cov.loc[("Pull", "Man minus Zone (interaction)"), "p"]
    st = c["stab"].set_index(["level", "metric", "split"]).r
    team_r, player_r = st[("team", "GOE 1 s before throw", "weeks 1-4 vs 5-8")], st[("player", "GOE 1 s before throw", "weeks 1-4 vs 5-8")]
    tb = c["tb"]
    hero = c["heroes"].iloc[0]
    callouts = [
        ("Decoys matter more against zone",
         f"+{zone:.2f} yd of target separation per defender pulled vs zone, +{man:.2f} yd vs man. "
         f"Suggestive (p = {p_int:.2f}, one of several tests), not proven.", P.C_DECOY),
        ("Scheme matters",
         f"Team-level attention is as stable as player-level (r = {team_r:.2f} vs {player_r:.2f}), led by "
         + ", ".join(f"{r.possessionTeam} ({r.goe_pre:+.3f})" for r in tb.head(3).itertuples())
         + " defenders per decoy route.", "#009E73"),
        (f"Hidden hero: {hero.player} ({hero.team} {hero.pos})",
         f"A top-{int(hero['rank'])} decoy (pull {hero.pull:+.2f} per route) but targeted on only {hero.tprr:.0%} of his routes "
         f"(median {lb.tprr.median():.0%}).", C_HERO),
    ]
    b_pull = c["series"]["Pull: attention gained after the snap"][0]
    below, above = b_pull[b_pull.x_mean < 0]["mean"], b_pull[b_pull.x_mean > 0.5]["mean"]
    c["hockey"] = bool(below.max() - below.min() < 0.25 and above.min() > below.max() + 0.3)
    print(f"hockey stick for pull: {c['hockey']} (bins below 0: {below.round(2).tolist()}, above +0.5: {above.round(2).tolist()})")
    chart_sub = f"+{c['eff_pull'].coef:.2f} yd per extra defender pulled (attention level: +{c['eff_pre'].coef:.2f} yd)"
    if c["hockey"]:
        chart_sub += "\nDecoys only help when they genuinely pull coverage; average decoy routes add little."
    P.summary_page(
        f"{OUT}/summary.png",
        title="Gravity: who drags defenders without touching the ball",
        definition=("Pull = defenders' worth of attention a non-targeted receiver gains from the snap to 1 s before "
                    "the throw, compared with what is normal for his alignment and the situation."),
        series=c["series"], chart_title="Decoys that drag defenders open space for the target",
        chart_subtitle=chart_sub, lb=lb, callouts=callouts,
        footnote=(f"NFL Big Data Bowl 2023 tracking, 2021 weeks 1-8, {len(plays_df):,} targeted passes. Attention = softmax "
                  f"of defender-receiver distance (tau = 2 yd). Limits: proximity is not assignment; 8 weeks; player "
                  f"stability r = {st[('player', 'Pull', 'weeks 1-4 vs 5-8')]:.2f}; no detectable lift in completions or yards."))
    print("saved leaderboard.png, play_example.png, play_example.gif, summary.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    c = core()
    step5(c)
    block_a(c)
    visuals(c)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return c


if __name__ == "__main__":
    main()
