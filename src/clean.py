"""Load, standardise and join the Big Data Bowl 2023 data.

Outputs two tables:
  plays_df  - one row per usable pass play (target nflId, key frame ids, week)
  frames_df - one row per player per key moment (snap / pre_throw = throw - 1 s / throw /
              post_throw = throw + 0.5 s), with offense always moving left -> right.
Tracking stops <= 0.5 s after the throw, so there is no true pass-arrival frame.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

DATA = "data"
CACHE = "cache"
SKILL = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}  # FB folded into RB group

# "pass [incomplete] [short|deep] [left|middle|right] to|intended for X.Lastname"
_NAME = r"([A-Z][a-zA-Z]*\.\s?[A-Z][\w'\-]*(?:\.\s?(?!Penalty|Coverage|Pressure)[A-Z][a-z][\w'\-]*)?)"
_TARGET_PAT = r"pass (?:incomplete )?(?:short |deep )?(?:left |middle |right )?(?:to|intended for) " + _NAME
_SUFFIX = r"\s(Jr\.?|Sr\.?|II|III|IV|V)$"


def load_tracking():
    """All tracking files, cached as a pickle after the first read (~6 s)."""
    path = f"{CACHE}/tracking_all.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    files = sorted(glob.glob(f"{DATA}/tracking/*.csv"))
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    t.to_pickle(path)
    return t


def standardise(t):
    """Flip left-moving plays so offense always moves left -> right."""
    t = t.copy()
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    for c in ["o", "dir"]:
        t.loc[left, c] = (t.loc[left, c] + 180) % 360
    return t


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def parse_targets(plays, pff, players):
    """There is no target column in this dataset, so recover it from playDescription.

    Parse 'X.Lastname' after 'pass ... to' and match it to a WR/TE/RB on that play
    (last name must match; first initial breaks ties).
    """
    p = plays[["gameId", "playId", "playDescription"]].copy()
    p["tgt_name"] = p.playDescription.str.extract(_TARGET_PAT)[0].str.rstrip(".")
    p = p.dropna(subset=["tgt_name"])
    p["t_init"] = p.tgt_name.str[0]
    p["t_last"] = p.tgt_name.str.split(".", n=1).str[1].str.replace(_SUFFIX, "", regex=True).map(_norm)

    cand = pff[["gameId", "playId", "nflId"]].merge(players[["nflId", "displayName", "officialPosition"]], on="nflId")
    cand = cand[cand.officialPosition.isin(SKILL)].copy()
    cand["last"] = cand.displayName.str.replace(_SUFFIX, "", regex=True).str.split(" ", n=1).str[1].map(_norm)
    cand["init"] = cand.displayName.str[0]

    m = p.merge(cand, on=["gameId", "playId"])
    m = m[[a.endswith(b) for a, b in zip(m["last"], m.t_last)]].copy()
    m["init_ok"] = m.t_init == m.init
    m = m.sort_values("init_ok", ascending=False).drop_duplicates(["gameId", "playId"])
    return m[["gameId", "playId", "nflId"]].rename(columns={"nflId": "targetNflId"})


def key_frames(t):
    """Snap, throw and arrival frame per play from the ball's event tags."""
    ev = t.loc[t.nflId.isna() & t.event.notna(), ["gameId", "playId", "frameId", "event"]]

    def first(names):
        return ev[ev.event.isin(names)].groupby(["gameId", "playId"]).frameId.min()

    kf = pd.DataFrame({
        "snap": first(["ball_snap"]).combine_first(first(["autoevent_ballsnap"])),
        "throw": first(["pass_forward"]).combine_first(first(["autoevent_passforward"])),
    })
    last = t.groupby(["gameId", "playId"]).frameId.max()
    kf = kf.reindex(last.index)
    kf["snap"] = kf.snap.fillna(1)
    kf = kf.dropna(subset=["throw"])
    # Tracking stops <= 0.5 s after the throw, so there is no true arrival frame.
    kf["post_throw"] = np.minimum(kf.throw + 5, last.reindex(kf.index))  # throw + 0.5 s
    kf["pre_throw"] = np.maximum(kf.throw - 10, kf.snap)                  # throw - 1.0 s
    kf["pre_ok"] = kf.throw - 10 > kf.snap                                # False on very quick throws
    tagged = ev[ev.event.isin(["man_in_motion", "shift"])].groupby(["gameId", "playId"]).size()
    kf["motion_tag"] = kf.index.isin(tagged.index)
    ints = {c: int for c in ["snap", "throw", "post_throw", "pre_throw"]}
    return kf.astype(ints).reset_index()


def build():
    games = pd.read_csv(f"{DATA}/games.csv")
    plays = pd.read_csv(f"{DATA}/plays.csv")
    players = pd.read_csv(f"{DATA}/players.csv")
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv")
    t = load_tracking()

    # --- play table: thrown passes with a recoverable target and a throw frame
    tg = parse_targets(plays, pff, players)
    kf = key_frames(t)
    thrown = plays[plays.passResult.isin(["C", "I", "IN"])]
    plays_df = (
        thrown[["gameId", "playId", "possessionTeam", "defensiveTeam", "passResult", "playResult",
                "prePenaltyPlayResult", "down", "yardsToGo", "quarter", "preSnapHomeScore", "preSnapVisitorScore",
                "offenseFormation", "defendersInBox", "pff_passCoverage", "pff_passCoverageType",
                "pff_playAction", "playDescription"]]
        .merge(tg, on=["gameId", "playId"])
        .merge(kf, on=["gameId", "playId"])
        .merge(games[["gameId", "week", "homeTeamAbbr"]], on="gameId")
    )
    home = plays_df.possessionTeam == plays_df.homeTeamAbbr
    diff = plays_df.preSnapHomeScore - plays_df.preSnapVisitorScore
    plays_df["score_diff"] = np.where(home, diff, -diff)  # offense minus defense
    plays_df["snap_to_throw"] = (plays_df.throw - plays_df.snap) / 10
    info = {"n_thrown": len(thrown), "target_match_rate": len(tg.merge(thrown[["gameId", "playId"]])) / len(thrown)}

    # --- tracking rows at the key moments only (keeps everything small)
    moments = plays_df.melt(id_vars=["gameId", "playId"], value_vars=["snap", "pre_throw", "throw", "post_throw"],
                            var_name="moment", value_name="frameId")
    f = standardise(t).merge(moments, on=["gameId", "playId", "frameId"])
    f = (
        f.merge(plays_df[["gameId", "playId", "possessionTeam", "defensiveTeam", "targetNflId"]], on=["gameId", "playId"])
        .merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
        .merge(pff[["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"]], on=["gameId", "playId", "nflId"], how="left")
    )
    f["side"] = np.select([f.team == "football", f.team == f.possessionTeam], ["ball", "off"], "def")
    f["pos_group"] = f.officialPosition.map(SKILL).where(f.side == "off")
    f["is_target"] = f.nflId == f.targetNflId
    # Eligible receivers: WR/TE/RB actually running a route. Defenders: PFF coverage players
    # (the 15-yard rule alone keeps ~95% of pass rushers, so we use PFF's role label).
    f["is_receiver"] = f.pos_group.notna() & (f.pff_role == "Pass Route")
    f["is_defender"] = (f.side == "def") & (f.pff_role == "Coverage")

    # Only keep plays where the parsed target is an eligible receiver
    tgt_ok = f[(f.moment == "throw") & f.is_target & f.is_receiver][["gameId", "playId"]]
    plays_df = plays_df.merge(tgt_ok, on=["gameId", "playId"])
    f = f.merge(tgt_ok, on=["gameId", "playId"])

    # Field position and motion from the (standardised) snap frame
    snap = f[f.moment == "snap"]
    los = snap[snap.side == "ball"].set_index(["gameId", "playId"]).x.rename("los_x")
    moving = snap[snap.is_receiver & (snap.s >= 2.0)].groupby(["gameId", "playId"]).size()  # in motion at the snap
    plays_df = plays_df.join(los, on=["gameId", "playId"])
    plays_df["yards_to_endzone"] = 110 - plays_df.los_x
    plays_df["motion"] = plays_df.motion_tag | plays_df.set_index(["gameId", "playId"]).index.isin(moving.index)

    keep = ["gameId", "playId", "moment", "frameId", "nflId", "displayName", "officialPosition", "pos_group",
            "side", "pff_role", "pff_positionLinedUp", "is_target", "is_receiver", "is_defender",
            "x", "y", "s", "a", "o", "dir"]
    return plays_df, f[keep], info
