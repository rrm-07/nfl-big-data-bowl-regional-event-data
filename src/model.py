"""Expected gravity from pre-snap alignment + play context, and Gravity Over Expected (GOE).

Every model is scored out-of-fold by game (a game's plays never predict themselves).
Only things the receiver does not create after the snap are allowed as features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

KEY = ["gameId", "playId"]

# PFF alignment labels, left/right mirrors folded together
ALIGN = {
    "LWR": "Wide", "RWR": "Wide",
    "SLWR": "Slot", "SRWR": "Slot",
    "SLoWR": "Slot-outer", "SRoWR": "Slot-outer",
    "SLiWR": "Slot-inner", "SRiWR": "Slot-inner",
    "TE-L": "TE-inline", "TE-R": "TE-inline",
    "TE-oL": "TE-outer", "TE-oR": "TE-outer",
    "TE-iL": "TE-inner", "TE-iR": "TE-inner",
    "HB": "HB", "HB-L": "HB-offset", "HB-R": "HB-offset",
    "FB": "FB", "FB-L": "FB", "FB-R": "FB",
}

# v1: alignment at the snap + how many route runners / coverage defenders on the play
V1_CAT = ["alignment", "pos_group"]
V1_NUM = ["depth", "width", "n_rec", "n_def"]
# v2: + game situation and play/defense context (none of it created by the receiver)
CTX_CAT = ["offenseFormation", "pff_passCoverage", "pff_passCoverageType"]
CTX_NUM = ["down", "yardsToGo", "yards_to_endzone", "quarter", "score_diff", "defendersInBox",
           "pff_playAction", "motion", "snap_to_throw", "n_near5"]
FEATURE_SETS = {"v1": (V1_CAT, V1_NUM), "v2": (V1_CAT + CTX_CAT, V1_NUM + CTX_NUM)}
MIN_CELL = 30


def features(frames, grav, plays_df):
    """Join snap alignment + play context onto a gravity table (one row per receiver per play)."""
    snap = frames[frames.moment == "snap"]
    ball = snap[snap.side == "ball"].set_index(KEY)[["x", "y"]].add_prefix("ball_")
    rec = snap.loc[snap.is_receiver, KEY + ["nflId", "x", "y", "pff_positionLinedUp"]].join(ball, on=KEY)
    rec["depth"] = rec.x - rec.ball_x          # yards past the ball at the snap (negative = backfield)
    rec["width"] = (rec.y - rec.ball_y).abs()  # lateral yards from the ball at the snap
    rec["alignment"] = rec.pff_positionLinedUp.map(ALIGN).fillna("Other")

    # offensive teammates within 5 yards at the snap (bunches, stacks, inline TEs)
    off = snap.loc[snap.side == "off", KEY + ["nflId", "x", "y"]]
    pr = rec[KEY + ["nflId", "x", "y"]].merge(off, on=KEY, suffixes=("", "_o"))
    pr = pr[pr.nflId != pr.nflId_o]
    pr["near"] = np.hypot(pr.x - pr.x_o, pr.y - pr.y_o) < 5
    rec = rec.join(pr.groupby(KEY + ["nflId"]).near.sum().rename("n_near5"), on=KEY + ["nflId"])

    n_def = snap[snap.is_defender].groupby(KEY).size().rename("n_def")  # coverage defenders (PFF role)
    d = grav.merge(rec[KEY + ["nflId", "alignment", "depth", "width", "n_near5"]], on=KEY + ["nflId"]).join(n_def, on=KEY)
    d["n_rec"] = d.groupby(KEY).nflId.transform("size")                 # route runners on the play
    return d.merge(plays_df[KEY + ["week"] + CTX_CAT + [c for c in CTX_NUM if c in plays_df]], on=KEY)


def _X(d, cat, num):
    X = pd.DataFrame(index=d.index)
    for c in cat:  # categorical -> integer codes, missing -> NaN (HGB handles it)
        codes = pd.Categorical(d[c]).codes.astype(float)
        codes[codes < 0] = np.nan
        X[c] = codes
    for c in num:
        X[c] = d[c].astype(float)
    return X


def oof_gbm(d, feature_set="v1", y="G", n_splits=5):
    """Out-of-fold gradient-boosting predictions, folds grouped by game."""
    cat, num = FEATURE_SETS[feature_set]
    X = _X(d, cat, num)
    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=d.gameId):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                          categorical_features=list(range(len(cat))), random_state=0)
        pred[te] = m.fit(X.iloc[tr], d[y].iloc[tr]).predict(X.iloc[te])
    return pred


def oof_binned(d, y="G", n_splits=5):
    """Reference baseline: mean G by alignment x position group (alignment mean for small cells)."""
    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(d, groups=d.gameId):
        train, test = d.iloc[tr], d.iloc[te]
        cell = train.groupby(V1_CAT)[y].agg(["mean", "size"])
        cell = cell[cell["size"] >= MIN_CELL]["mean"].rename("cell")
        by_align = train.groupby("alignment")[y].mean().rename("align_mean")
        t = test[V1_CAT].join(cell, on=V1_CAT).join(by_align, on="alignment")
        pred[te] = t.cell.fillna(t.align_mean).fillna(train[y].mean()).to_numpy()
    return pred


def score(y, pred):
    resid = y - pred
    return {"R2": float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()), "MAE": float(resid.abs().mean())}
