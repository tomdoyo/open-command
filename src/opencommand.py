"""Step 4 of the OpenCommand pipeline.

Functions:
 - command_scores: naive/inferred miss distances
 - funnel:         how many pitches get dropped at each stage in the pipeline
 - pose_accuracy:  reprojection check against Statcast 9-param trajectory
 - distribution:   command distributions
 - validity:       correlations to BB%, xERA, Stuff+

Reads:      data/<year>/targets.csv.gz + data/<year>/pbp_info.csv.gz
            data/<year>/camera_poses.csv.gz
            data/fg_pitching_<season>.csv.gz
Writes:     data/<year>/command_scores.csv (plain csv, the one output small enough
            for GitHub uncompressed), per pitcher x pitch type (+ an ALL row per
            pitcher): n, naive_in, inferred_in
            artifacts/validations_<season>.txt
Run:        python src/opencommand.py [year=2026]
            The argument is a PATH FRAGMENT under data/, not a year, so a tree that
            keeps its scorable season somewhere deeper is named in full.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ocl import common_parse_args

SRC = Path(__file__).resolve().parent
DATA = SRC.parent / "data"
ART = SRC.parent / "artifacts"

LEADERBOARD_MIN_N = 100
MIN_IP = 50
MIN_N_PT = 50
N_BOOT = 2000
SEED = 0

PLATE_FR = 2.5
FLIGHT_FR = 24  # a flight is ~0.4 s = 24 frames at 60 fps

# distribution table's pitch-type rows
PITCH_TYPES = [("FF four-seam", ["FF"]), ("SI sinker", ["SI"]), ("FC cutter", ["FC"]),
               ("SL slider", ["SL"]), ("ST sweeper", ["ST"]), ("CU+KC curve", ["CU", "KC"]),
               ("CH change", ["CH"]), ("FS split", ["FS"])]

# validity table
VALIDITY_ROWS = [("BB%",           "BB%",       None),
                 ("Stuff+",        "sp_stuff",  None),
                 ("xERA",          "xERA",      None),
                 ("xERA | Stuff+", "xERA",      "sp_stuff")]


def miss(g, kind):  # per-pitch distance from the actual location to one target pair
    return np.hypot(g["plate_x_in"] - g[f"{kind}_x_in"],
                    g["plate_z_in"] - g[f"{kind}_z_in"]).to_numpy()


def scorable(targets, pbp, cols):
    """The plausible targets joined to the pbp columns a caller names."""
    pbp = pbp.assign(video=pbp["game_pk"].astype(str) + "_" + pbp["play_id"] + ".mp4")
    return targets[targets["plausible"]].merge(pbp[["video"] + cols], on="video")


def command_scores(targets, pbp):
    """Per pitcher x pitch type: n, naive and inferred median miss (inches)."""
    sc = scorable(targets, pbp, ["pitcher_id", "pitcher", "pitch_type"])
    sc["naive"], sc["inferred"] = miss(sc, "naive"), miss(sc, "inferred")
    stats = {"pitcher": ("pitcher", "first"), "n": ("naive", "size"),
             "naive_in": ("naive", "median"), "inferred_in": ("inferred", "median")}
    per_pt = sc.groupby(["pitcher_id", "pitch_type"]).agg(**stats).reset_index()
    alls = sc.groupby("pitcher_id").agg(**stats).assign(pitch_type="ALL").reset_index()
    out = pd.concat([per_pt, alls], ignore_index=True)
    return (out.sort_values(["pitcher", "pitch_type"])
               [["pitcher", "pitch_type", "n", "naive_in", "inferred_in"]]
               .reset_index(drop=True))


def funnel(targets, pbp, poses):
    """Pipeline coverage funnel."""
    ok = targets[targets["status"] == "ok"]
    n_pbp, n_pose = len(pbp), len(poses)
    n_ok, n_pl = len(ok), int(targets["plausible"].sum())
    L = ["PIPELINE FUNNEL", "-" * 64]
    for label, kept, prev, why in [
            ("pitches", n_pbp, None, "every pitch of the season, clip or no clip"),
            ("detections available", n_pose, n_pbp, "No video, strikezone box, ball release or CF picture"),
            ("glove target", n_ok, n_pose, "Too few glove detections in the target window"),
            ("plausible", n_pl, n_ok, "Implausible target")]:
        drop = "" if prev is None else f"  -{prev - kept}"
        L.append(f"  {label:<22}{kept:>8}{drop:>9}   {why}")
    L += [f"  end-to-end: {n_pl} command targets = {100 * n_pl / n_pbp:.2f}% of pitches", ""]
    return L


def pose_accuracy(poses):
    """Reprojection error of the shipped poses, read off the columns solve_camera_pose stamps."""
    pv = poses.dropna(subset=["reproj_px"])
    L = ["CAMERA-POSE ACCURACY (trajectory reprojection through shipped poses)", "-" * 64,
         f"  clips with a flight run under the pose: {len(pv)} / {len(poses)}",
         f"  reproj_px: median {pv['reproj_px'].median():.2f}, p90 {pv['reproj_px'].quantile(0.9):.2f}, "
         f"<5px {100 * (pv['reproj_px'] < 5).mean():.1f}%"]
    tp = pv.dropna(subset=["balltp_err_x_in"])
    for label, w in [("within 24 frames (the whole flight)", FLIGHT_FR),
                     ("within 2.5 frames (at the plate)", PLATE_FR)]:
        b = tp[tp["plate_gap_fr"].abs() <= w]
        L.append(f"  ball@plate vs Statcast, {label}: n={len(b)} ({100 * len(b) / len(pv):.1f}% of "
                 f"posed clips), |err| median X {b['balltp_err_x_in'].abs().median():.2f} in, "
                 f"Z {b['balltp_err_z_in'].abs().median():.2f} in")
    L.append("")
    return L


def distribution(targets, pbp):
    """Command distributions."""
    sc = scorable(targets, pbp, ["pitcher_id", "pitch_type"])
    sc["naive"], sc["inferred"] = miss(sc, "naive"), miss(sc, "inferred")
    groups = [("ALL", sc, LEADERBOARD_MIN_N)]
    groups += [(label, sc[sc["pitch_type"].isin(codes)], MIN_N_PT) for label, codes in PITCH_TYPES]

    L = ["COMMAND DISTRIBUTION (percentiles of the per-pitcher median miss, inches)", "-" * 64]
    for metric in ("naive", "inferred"):
        L += ["  " + f"{metric.upper():<20}{'pitchers':>9}" + "".join(f"{q:>8}" for q in
              ("min", "p10", "p25", "median", "p75", "p90", "max")), "  " + "-" * 85]
        for label, g, bar in groups:
            m = g.groupby("pitcher_id")[metric].agg(["size", "median"])
            m = m[m["size"] >= bar]["median"]
            L.append(f"  {label:<20}{len(m):>9}" + "".join(f"{v:>8.2f}" for v in
                     np.percentile(m, [0, 10, 25, 50, 75, 90, 100])))
        L.append("")
    return L


def corr(x, y, rank, control=None):
    """Correlation."""
    X = np.column_stack([x, y] if control is None else [x, y, control]).astype(float)
    if rank:
        X = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, X)
    if control is not None:
        z = np.column_stack([np.ones(len(X)), X[:, 2]])
        X = X[:, :2] - z @ np.linalg.lstsq(z, X[:, :2], rcond=None)[0]
    return float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])


def corr_ci(x, y, rank, control=None):
    """The correlation plus a 95% percentile bootstrap interval."""
    idx = np.random.default_rng(SEED).integers(0, len(x), size=(N_BOOT, len(x)))
    boot = [corr(x[i], y[i], rank, None if control is None else control[i]) for i in idx]
    return corr(x, y, rank, control), *np.percentile(boot, [2.5, 97.5])


def validity(targets, pbp, fg):
    """Correlations to BB%, xERA, Stuff+."""
    sc = scorable(targets, pbp, ["pitcher_id"])
    sc["naive"], sc["inferred"] = miss(sc, "naive"), miss(sc, "inferred")
    cmd = sc.groupby("pitcher_id").agg(n=("naive", "size"), naive_in=("naive", "median"),
                                       inferred_in=("inferred", "median")).reset_index()

    d = cmd.merge(fg[["xMLBAMID", "IP", "xERA", "BB%", "sp_stuff"]],
                  left_on="pitcher_id", right_on="xMLBAMID", how="left")
    assert d["xMLBAMID"].notna().all(), "a scored pitcher has no Fangraphs row"
    d = d[(d["n"] >= LEADERBOARD_MIN_N) & (d["IP"] >= MIN_IP)].reset_index(drop=True)
    assert d[[c for _, c, _ in VALIDITY_ROWS]].notna().all().all(), "missing Fangraphs values"

    L = ["External validity",
         f"  {len(d)} pitchers: >= {LEADERBOARD_MIN_N} scored pitches and >= {MIN_IP} IP.",
         ""]
    for rank, name in [(True, "SPEARMAN"), (False, "PEARSON")]:
        L += [f"  {name:<24}{'naive median miss':>26}{'inferred median miss':>26}", "  " + "-" * 76]
        for label, col, ctrl in VALIDITY_ROWS:
            cells = ["{:+.3f} [{:+.3f}, {:+.3f}]".format(
                        *corr_ci(d[m].to_numpy(), d[col].to_numpy(), rank,
                                 None if ctrl is None else d[ctrl].to_numpy()))
                     for m in ("naive_in", "inferred_in")]
            L.append(f"  {label:<24}{cells[0]:>26}{cells[1]:>26}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":

    args = common_parse_args("opencommand")
    year = str(args.year)

    base = DATA / year
    adjusted = base / "exports" / "targets.csv.gz"
    assert not adjusted.exists(), f"score the adjusted tree instead: opencommand.py {year}/exports"
    season = Path(year).parts[0]
    fg_file = DATA / f"fg_pitching_{season}.csv.gz"
    assert fg_file.exists(), f"the Fangraphs season file ships beside the trees: {fg_file}"

    targets = pd.read_csv(base / "targets.csv.gz")
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    scores = command_scores(targets, pbp)
    scores.to_csv(base / "command_scores.csv", index=False, lineterminator="\n")

    lb = scores[(scores["pitch_type"] == "ALL") & (scores["n"] >= LEADERBOARD_MIN_N)]
    lb = lb.sort_values("inferred_in")
    w = max(lb["pitcher"].str.len(), default=0)
    board = [f"Best command (inferred median miss, min {LEADERBOARD_MIN_N} scored pitches):"]
    board += [f"  {r['pitcher']:<{w}}  {r['inferred_in']:>6.2f} in  (naive {r['naive_in']:.2f}, n={r['n']})"
              for _, r in lb.iterrows()]

    poses = pd.read_csv(base / "camera_poses.csv.gz")
    L = funnel(targets, pbp, poses) + pose_accuracy(poses) + distribution(targets, pbp)
    L += [validity(targets, pbp, pd.read_csv(fg_file)), ""] + board
    print("\n".join(L))
    ART.mkdir(exist_ok=True)
    (ART / f"validations_{season}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote data/{year}/command_scores.csv, artifacts/validations_{season}.txt")
