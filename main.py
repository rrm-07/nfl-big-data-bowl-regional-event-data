"""Gravity: end-to-end pipeline. Run from the repo root:  python main.py"""
import os
import time

import pandas as pd

from src import plots as P
from src.analysis import binned, leaderboard, ols, play_table, split_half
from src.clean import build
from src.metric import gravity
from src.model import features, oof_gbm, score

pd.set_option("display.width", 250)
OUT = "outputs"
BASELINE = "v2"  # chosen over v1: higher out-of-fold R2 and (slightly) more stable player GOE
MIN_ROUTES = 50  # minimum non-targeted routes for the leaderboard (20 let small-sample players into the top 10)


def step(msg):
    print(f"\n{'=' * 8} {msg} {'=' * 8}")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    step("Step 1: clean + standardise")
    plays_df, f, info = build()
    print(f"{len(plays_df)} plays | target matched on {info['target_match_rate']:.1%} of {info['n_thrown']} thrown passes")

    step("Step 2-3: gravity at the throw, expected gravity, GOE")
    g, _ = gravity(f, moment="throw")
    d = features(f, g, plays_df).reset_index(drop=True)
    d["xG_v1"] = oof_gbm(d, "v1")
    d["xG"] = oof_gbm(d, BASELINE)
    d["GOE"] = d.G - d.xG
    def fmt(dct):
        return ", ".join(f"{k}={float(v):.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in dct.items())

    print(f"out-of-fold            v1: {fmt(score(d.G, d.xG_v1))} | {BASELINE}: {fmt(score(d.G, d.xG))}")
    print(f"split-half stability   v1: {fmt(split_half(d.assign(GOE=d.G - d.xG_v1)))} | {BASELINE}: {fmt(split_half(d))}")

    # Robustness: the same metric measured 1.0 s before the throw (its own expected-gravity fit)
    g_pre, _ = gravity(f, moment="pre_throw")
    pre_ok = plays_df.loc[plays_df.pre_ok, ["gameId", "playId"]]  # drops 6 throws made < 1 s after the snap
    d_pre = features(f, g_pre, plays_df).merge(pre_ok, on=["gameId", "playId"]).reset_index(drop=True)
    d_pre["GOE"] = d_pre.G - oof_gbm(d_pre, BASELINE)

    step("Step 4: does decoy gravity create space?")
    p = play_table(d, f, plays_df)
    p = p.merge(play_table(d_pre, f, plays_df)[["gameId", "playId", "decoy_goe"]].rename(columns={"decoy_goe": "decoy_goe_pre"}),
                on=["gameId", "playId"])
    p_pre = p[p.pre_ok]
    print("play table:", p.shape)
    print(p[["decoy_goe", "decoy_goe_pre", "target_goe", "sep_throw", "sep_post_throw", "target_depth", "complete", "yards"]]
          .describe().round(2).loc[["mean", "std", "min", "max"]].to_string())

    b_throw = binned(p, "decoy_goe", "sep_throw")
    b_pre = binned(p_pre, "decoy_goe_pre", "sep_throw")
    print("\nTarget separation at the throw by quintile of decoy GOE (at the throw):")
    print(b_throw.round(2).to_string(index=False))
    print("\n... by quintile of decoy GOE measured 1.0 s before the throw:")
    print(b_pre.round(2).to_string(index=False))
    for y in ["complete", "yards"]:
        print(f"\n{y} by quintile of decoy GOE (at the throw):")
        print(binned(p, "decoy_goe", y).round(3).to_string(index=False))

    rows = []
    for x, df in [("decoy_goe", p), ("decoy_goe_pre", p_pre)]:
        for y in ["sep_throw", "sep_post_throw", "complete", "yards"]:
            rows.append({"decoy GOE measured": "at throw" if x == "decoy_goe" else "throw - 1.0 s", **ols(df, y, x)})
    reg = pd.DataFrame(rows)
    print("\nOLS: outcome ~ decoy GOE + target depth (SEs clustered by game). coef = per +1 defender of decoy attention")
    print(reg.round(4).to_string(index=False))

    s = reg.set_index(["decoy GOE measured", "outcome"])
    eff, eff_pre = s.loc[("at throw", "sep_throw")], s.loc[("throw - 1.0 s", "sep_throw")]
    P.binned_effect(
        {"Decoy GOE 1.0 s before the throw": (b_pre, P.C_DECOY, "-"),
         "Decoy GOE at the throw": (b_throw, P.C_OTHER, "--")},
        f"{OUT}/decoy_goe_vs_separation.png",
        title="Decoys that hold defenders open space for the target",
        subtitle=(f"+{eff_pre.coef:.2f} yd target separation per extra defender pulled 1 s before the throw "
                  f"(+{eff.coef:.2f} yd measured at the throw)"),
        xlabel="Total decoy GOE on the play (defenders' worth of attention above expected)",
        ylabel="Target separation at the throw (yd)",
    )
    print(f"\nsaved {OUT}/decoy_goe_vs_separation.png")

    step("Step 5: player leaderboard (non-targeted routes)")
    lb = leaderboard(d_pre, d, plays_df, min_n=MIN_ROUTES)
    lb.to_csv(f"{OUT}/leaderboard.csv", index=False)
    print(f"{len(lb)} players with >= {MIN_ROUTES} non-targeted routes | routes per player: "
          f"median {lb.decoy_routes.median():.0f}, min {lb.decoy_routes.min()}, max {lb.decoy_routes.max()}")
    cols = ["player", "team", "pos", "decoy_routes", "goe_pre", "ci95", "goe_throw"]
    print("\nTOP 10 (ranked by decoy GOE 1 s before the throw)")
    print(lb[cols].head(10).round(3).to_string(index=False))
    print("\nBOTTOM 5")
    print(lb[cols].tail(5).round(3).to_string(index=False))

    stab = []
    for name, df in [("GOE 1 s before throw (ranking)", d_pre), ("GOE at throw", d)]:
        for split in ["weeks", "odd_even"]:
            r = split_half(df, split=split)
            stab.append({"metric": name, "split": "wk 1-4 vs 5-8" if split == "weeks" else "odd vs even wk",
                         **r, "full-sample reliability (Spearman-Brown)": 2 * r["r"] / (1 + r["r"])})
    print("\nStability: player mean non-targeted GOE, >= 10 routes in each half")
    print(pd.DataFrame(stab).round(3).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")
    return plays_df, f, d, d_pre, p, reg, lb


if __name__ == "__main__":
    main()
