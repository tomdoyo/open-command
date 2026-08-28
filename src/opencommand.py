"""Step 4 of the OpenCommand pipeline.

Functions:
 - val_median_miss:   median miss distance, pooled and per pitcher
 - val_heldout:       50/50 random heldout rmse
 - val_flatness:      check if E[miss] is flat across n random pitches
 - val_confidence:    CI on season-end miss with n random pitches
 - val_correlations:  BB%, Location+, Stuff+, xERA, and next season
 - val_stabilization: Cronbach's alpha
 - val_stickiness:    yoy
 - funnel:          how many pitches get dropped at each stage in the pipeline
 - pose_accuracy:   reprojection check against Statcast 9-param trajectory
 - distribution:    command distributions
 - command_scores:  per pitcher x pitch type miss distances & leaderboard

Reads:      data/<year>/targets.csv.gz + data/<year>/pbp_info.csv.gz
            data/<year>/camera_poses.csv.gz
            data/fangraphs/fg_pitching_*.csv.gz
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

from target_inference import CELL, infer_targets

SRC = Path(__file__).resolve().parent
DATA = SRC.parent / "data"
ART = SRC.parent / "artifacts"

LEADERBOARD_MIN_N = 100
MIN_N_PT = 50
MIN_N_SEASON = 500
N_BOOT = 2000
SEED = 0

PLATE_FR = 2.5
FLIGHT_FR = 24  # a flight is ~0.4 s = 24 frames at 60 fps

NS = [10, 30, 100, 300, 1000]
COLS = NS + [None]
HALF_NS = [10, 500]
ALPHAS = [0.5, 0.7]

# distribution table's pitch-type rows
PITCH_TYPES = [("FF four-seam", ["FF"]), ("SI sinker", ["SI"]), ("FC cutter", ["FC"]),
               ("SL slider", ["SL"]), ("ST sweeper", ["ST"]), ("CU+KC curve", ["CU", "KC"]),
               ("CH change", ["CH"]), ("FS split", ["FS"])]

VALIDITY_ROWS = [("BB%",           "BB%",         None),
                 ("Location+",     "sp_location", None),
                 ("Stuff+",        "sp_stuff",    None),
                 ("xERA",          "xERA",        None),
                 ("xERA | Stuff+", "xERA",        "sp_stuff")]

EXTERNAL = [("Location+", "sp_location"), ("Stuff+", "sp_stuff")]
MIN_N_PAIR = 25

STICKY = EXTERNAL + [("BB%", "BB%"), ("K%", "K%"), ("HR%", "HR%")]


# ─────────────────────────────────────────────  candidate targets

def naive(tr, te):
    return te.naive_x_in.to_numpy(), te.naive_z_in.to_numpy()


def fixedoffset(tr, te):
    """Naive plus one mean residual per (pitcher, pitch type)."""
    o = tr.assign(ox=tr.plate_x_in - tr.naive_x_in, oz=tr.plate_z_in - tr.naive_z_in)
    m = o.groupby(CELL)[["ox", "oz"]].mean()
    j = te[CELL].merge(m, left_on=CELL, right_index=True, how="left").fillna(0.0)
    return te.naive_x_in.to_numpy() + j.ox.to_numpy(), te.naive_z_in.to_numpy() + j.oz.to_numpy()


METHODS = [("naive", naive), ("fixed offset", fixedoffset), ("inferred", infer_targets)]


def missed(fn, tr, te):
    """Per-pitch miss, fit on tr and scored on te."""
    tx, tz = fn(tr, te)
    return pd.Series(np.hypot(te.plate_x_in - tx, te.plate_z_in - tz), index=te.index)


def per_pitcher(m, pid, floor):
    """Each pitcher's median miss."""
    s = m.groupby(pid).agg(["median", "size"])
    return s[s["size"] >= floor]["median"]


# ─────────────────────────────────────────────  validations

def val_median_miss(d, whole):
    """Median miss over all pitches, and over pitchers."""
    L = ["MEDIAN MISS", "-" * 64,
         f"  {'':16s}{'pooled':>10s}{'per pitcher':>14s}{'pitchers':>10s}"]
    pp = {}
    for name, _ in METHODS:
        pp[name] = per_pitcher(whole[name], d.pitcher_id, LEADERBOARD_MIN_N)
        L.append(f"  {name:16s}{whole[name].median():10.2f}"
                 f"{pp[name].median():14.2f}{len(pp[name]):10d}")
    got, base = METHODS[-1][0], METHODS[-2][0]
    dl = (pp[got] - pp[base]).dropna()
    L += ["", f"  {got} vs {base}: {dl.median():+.2f} in per pitcher, "
              f"better for {(dl < 0).mean():.0%} of {len(dl)} pitchers", ""]
    return L


def rms(x):
    """Root mean square of a per-pitch miss."""
    return float(np.sqrt(np.nanmean(x ** 2)))


def val_heldout(d):
    """50/50 random heldout rmse.

    This is a proxy for overfitting. If target inference logic is fit on random 50% of pitches,
    the logic should ideally translate perfectly to the other 50% and both 50% should have the
    same average misses.
    """
    rng = np.random.default_rng(SEED)
    rank = pd.Series(rng.random(len(d)), index=d.index).groupby(d.pitcher_id).rank(method="first")
    n = d.pitcher_id.map(d.pitcher_id.value_counts())
    h1, h2 = d[rank <= n // 2], d[rank > n // 2]
    test = {name: missed(fn, h1, h2) for name, fn in METHODS}    # one fit per method
    train = {name: missed(fn, h1, h1) for name, fn in METHODS}
    smaller_half = pd.concat([h1.pitcher_id.value_counts(), h2.pitcher_id.value_counts()],
                             axis=1).min(axis=1)

    heads = [f"{f}+ each half ({(smaller_half >= f).sum()})" for f in HALF_NS]
    L = ["HELD-OUT ACCURACY (rmse, random 50/50 train/test split)",
         "-" * 64,
         f"  {'':16s}" + "".join(f"{h:>26s}" for h in heads),
         f"  {'':16s}" + f"{'train':>10s}{'test (gap)':>16s}" * len(HALF_NS)]
    for name, _ in METHODS:
        te = test[name].groupby(h2.pitcher_id.to_numpy()).apply(rms)
        tr = train[name].groupby(h1.pitcher_id.to_numpy()).apply(rms)
        row = ""
        for floor in HALF_NS:
            keep = smaller_half[smaller_half >= floor].index
            row += f"{tr.loc[keep].median():10.2f}"
            row += f"{f'{te.loc[keep].median():.2f} ({(te - tr).loc[keep].median():+.2f})':>16s}"
        L.append(f"  {name:16s}" + row)
    L.append("")
    return L


def val_flatness(d, whole, early):
    """Check if E[miss] is flat across n pitches.

    Fixed offset has a huge flaw where it overfits at small n (e.g. at n=1 the average miss
    is 0 inches). This is a problem early in the season & for debutees.
    """
    pools = [per_pitcher(early[n][METHODS[0][0]], d.pitcher_id, n) for n in NS]
    L = ["FLATNESS (per pitcher median miss on n random pitches)", "-" * 64,
         "  " + f"{'':16s}" + "".join(f"{c:>12s}" for c in [f"n={n}" for n in NS] + ["full"]),
         "  " + f"{'pitchers':16s}" + "".join(f"{len(p):12d}" for p in pools)
         + f"{len(per_pitcher(whole[METHODS[0][0]], d.pitcher_id, LEADERBOARD_MIN_N)):12d}"]
    for name, _ in METHODS:
        row = [per_pitcher(early[n][name], d.pitcher_id, n).median() for n in NS]
        row.append(per_pitcher(whole[name], d.pitcher_id, LEADERBOARD_MIN_N).median())
        L.append(f"  {name:16s}" + "".join(f"{v:12.2f}" for v in row))
    L.append("")
    return L


def val_confidence(d, whole, early):
    """CI on season-end miss with n random pitches.

    Similar to val_flatness, but this measures how close the average miss estimate (on n
    random pitches) is to the end-of-season average miss.
    """
    L = ["CONFIDENCE INTERVAL (median miss calculated on n random pitches vs end of season)",
         "-" * 64,
         "  " + f"{'':22s}" + "".join(f"{c:>10s}" for c in [f"n={n}" for n in NS] + ["full"])]
    sizes = d.groupby("pitcher_id").size()
    for name, _ in METHODS:
        final = whole[name].groupby(d["pitcher_id"]).median()
        gap = {None: final - final}                       # the full column
        for n in NS:
            pm = early[n][name].groupby(d["pitcher_id"]).median()
            gap[n] = (pm[sizes.loc[pm.index] >= n] - final).dropna()
        L.append(f"  {name + ', pitchers':22s}" + "".join(f"{len(gap[n]):10d}" for n in COLS))
        for lab, f in [("avg deviation", lambda s: s.abs().median()),
                       ("90%", lambda s: s.abs().quantile(0.9)),
                       ("bias", lambda s: s.median())]:
            L.append(f"  {'  ' + lab:22s}" + "".join(f"{f(gap[n]):10.2f}" for n in COLS))
        L.append("")
    return L


def corr(x, y, control=None):
    """Pearson correlation, with a control variable residualized out of both sides."""
    X = np.column_stack([x, y] if control is None else [x, y, control]).astype(float)
    if control is not None:
        z = np.column_stack([np.ones(len(X)), X[:, 2]])
        X = X[:, :2] - z @ np.linalg.lstsq(z, X[:, :2], rcond=None)[0]
    return float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])


def corr_ci(x, y, control=None):
    """Correlation plus a 95% percentile bootstrap interval."""
    idx = np.random.default_rng(SEED).integers(0, len(x), size=(N_BOOT, len(x)))
    boot = [corr(x[i], y[i], None if control is None else control[i]) for i in idx]
    return corr(x, y, control), *np.percentile(boot, [2.5, 97.5])


def cell(x, y, control=None):
    """One table cell: correlation and interval."""
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(*corr_ci(x, y, control))


def val_correlations(d, whole, fg, fg_next, season):
    """BB%, Location+, Stuff+, xERA, and next season."""
    t = pd.DataFrame({name: per_pitcher(whole[name], d.pitcher_id, 1)   # every pitcher we scored
                      for name, _ in METHODS})
    cols = ["BB%", "sp_location", "sp_stuff", "xERA", "Pitches"]
    t = t.join(fg.set_index("xMLBAMID")[cols], how="inner")
    t = t[t.Pitches >= MIN_N_SEASON]        # pitches THROWN, so the pool does not move with coverage
    assert len(t) > 100, f"the Fangraphs join found only {len(t)} pitchers"

    L = ["CORRELATIONS (Pearson, whole season)", "-" * 64,
         f"  Min. {MIN_N_SEASON} pitches, N = {len(t)}", "",
         f"  {'':22s}" + "".join(f"{n:>26s}" for n, _ in METHODS)]
    for lab, col, ctrl in VALIDITY_ROWS:
        s = t.dropna(subset=[col] + ([ctrl] if ctrl else []))
        ctl = s[ctrl].to_numpy() if ctrl else None
        L.append(f"  {lab:22s}" + "".join(f"{cell(s[n].to_numpy(), s[col].to_numpy(), ctl):>26}"
                                          for n, _ in METHODS))
    s = t.dropna(subset=["sp_location", "BB%"])
    L += ["",
          f"  For reference, Location+ correlation to BB%: "
          f"{cell(s['sp_location'].to_numpy(), s['BB%'].to_numpy())}",
          ""]

    nxt = int(season) + 1
    L += ["  PREDICTIVENESS", "  " + "-" * 62]
    if fg_next is None:
        return L + [f"  skipped: no {nxt} Fangraphs file on disk", ""]
    p = t.join(fg_next.set_index("xMLBAMID")[cols], how="inner", rsuffix="_next")
    p = p[p["Pitches_next"] >= MIN_N_SEASON]
    L += [f"  Min. {MIN_N_SEASON} pitches, N = {len(p)}", "",
          f"  {'':22s}" + "".join(f"{n:>26s}" for n, _ in METHODS)
          + f"{'the outcome itself':>26s}"]
    for lab, col, ctrl in VALIDITY_ROWS:
        # predictors are all THIS season, so the control is what a forecaster would know today
        s = p.dropna(subset=[f"{col}_next", col] + ([ctrl] if ctrl else []))
        ctl, y = (s[ctrl].to_numpy() if ctrl else None), s[f"{col}_next"].to_numpy()
        L.append(f"  {f'next {lab}':22s}"
                 + "".join(f"{cell(s[n].to_numpy(), y, ctl):>26}" for n, _ in METHODS)
                 + f"{cell(s[col].to_numpy(), y, ctl):>26}")
    L.append("")
    return L


def external_variances(fg_all):
    """Fangraphs ships one number per pitcher-season, so vw cannot be measured per-pitch like
    the 3 target rows. It is fitted instead: across consecutive seasons the squared gap
    has expectation vw*(1/n1 + 1/n2) + 2*vy, so a line through that gap against (1/n1 + 1/n2)
    carries vw as its slope.
    """
    a = fg_all.rename(columns=lambda c: c + "1")
    b = fg_all.assign(year=fg_all.year - 1).rename(columns=lambda c: c + "2")
    p = a.merge(b, left_on=["xMLBAMID1", "year1"], right_on=["xMLBAMID2", "year2"])
    p = p[(p.Pitches1 >= MIN_N_PAIR) & (p.Pitches2 >= MIN_N_PAIR)]
    w = 1 / p.Pitches1 + 1 / p.Pitches2
    q = fg_all[fg_all.Pitches >= LEADERBOARD_MIN_N]
    out = {}
    for lab, col in EXTERNAL:
        vw = np.polyfit(w, (p[f"{col}1"] - p[f"{col}2"]) ** 2, 1)[0]
        out[lab] = (vw, q[col].var() - (vw / q.Pitches).mean())
    return len(p), out


def val_stabilization(d, whole, fg_all):
    """Cronbach's alpha."""
    L = ["STABILIZATION (cronbach's alpha, using mean miss)", "-" * 64,
         "  " + f"{'':16s}" + "".join(f"{f'n={n}':>12s}" for n in NS)
         + "".join(f"{f'n at {a}':>12s}" for a in ALPHAS)]
    for name, _ in METHODS:
        g = whole[name].groupby(d["pitcher_id"]).agg(["mean", "var", "size"])
        g = g[g["size"] >= 2]
        vw = np.average(g["var"], weights=g["size"] - 1)
        q = g[g["size"] >= LEADERBOARD_MIN_N]
        vb = q["mean"].var() - (vw / q["size"]).mean()
        row = f"  {name:16s}" + "".join(f"{vb / (vb + vw / n):12.3f}" for n in NS)
        L.append(row + "".join(f"{a / (1 - a) * vw / vb:12.0f}" for a in ALPHAS))
    L.append("")
    if fg_all.year.nunique() < 3:      # one gap alone cannot separate slope from intercept
        return L + ["  Location+ and Stuff+ skipped: fewer than three Fangraphs seasons on disk", ""]
    n_pair, ext = external_variances(fg_all)
    for lab, _ in EXTERNAL:
        vw, vb = ext[lab]
        L.append(f"  {lab:16s}" + "".join(f"{vb / (vb + vw / n):12.3f}" for n in NS)
                 + "".join(f"{a / (1 - a) * vw / vb:12.0f}" for a in ALPHAS))
    L += ["",
          f"  Location+ and Stuff+ ship one number per pitcher-season, so their vw is fitted",
          f"  across {n_pair} consecutive-season pairs and not measured per pitch.", ""]
    return L


def val_stickiness(d, whole, prev, prev_year, fg, fg_prev):
    """Year-over-year correlations."""
    L = ["STICKINESS", "-" * 64]
    if prev is None:
        return L + [f"  skipped: no {prev_year} tree on disk", ""]
    prev_miss = {name: missed(fn, prev, prev) for name, fn in METHODS}   # one fit, both rows
    labels = [n for n, _ in METHODS] + ([lab for lab, _ in STICKY] if fg_prev is not None else [])
    L += ["  year-over-year", "",
          f"  {'':16s}" + "".join(f"{n:>14s}" for n in labels) + f"{'pitchers':>14s}"]
    for floor in (MIN_N_SEASON, LEADERBOARD_MIN_N):
        rows, pool = {}, None
        for name, _ in METHODS:
            j = pd.DataFrame({"prev": per_pitcher(prev_miss[name], prev.pitcher_id, floor),
                              "cur": per_pitcher(whole[name], d.pitcher_id, floor)}).dropna()
            rows[name], pool = j.prev.corr(j.cur), j.index
        if fg_prev is not None:     # same pitchers again, scored by Fangraphs
            cols = [c for _, c in STICKY]
            e = (fg_prev.set_index("xMLBAMID")[cols]
                 .join(fg.set_index("xMLBAMID")[cols], lsuffix="_prev").reindex(pool).dropna())
            for lab, col in STICKY:
                rows[lab] = e[f"{col}_prev"].corr(e[col])
            if len(e) < len(pool):
                L.append(f"  {floor}+: the Fangraphs columns read {len(e)} of the {len(pool)}; "
                         f"the rest miss a Fangraphs row in one season.")
        L.append(f"  {f'{floor}+ pitches':16s}" + "".join(f"{rows[n]:+14.3f}" for n in labels)
                 + f"{len(pool):14d}")
    L.append("")
    return L


def read_fg(path):
    fg = pd.read_csv(path)
    fg["HR%"] = fg["HR"] / fg["TBF"]
    return fg


def fg_seasons(season):
    """Every Fangraphs season file."""
    files = {f.name.removeprefix("fg_pitching_").removesuffix(".csv.gz"): f
             for f in (DATA / "fangraphs").glob("fg_pitching_*.csv.gz")}
    cols = ["xMLBAMID", "Pitches"] + [c for _, c in EXTERNAL]
    return pd.concat([pd.read_csv(files[y], usecols=cols).assign(year=int(y))
                      for y in sorted(y for y in files if y <= season)])


# ─────────────────────────────────────────────  the four that describe the season

def miss(g, kind):  # per-pitch distance from the actual location to one shipped target pair
    return np.hypot(g["plate_x_in"] - g[f"{kind}_x_in"],
                    g["plate_z_in"] - g[f"{kind}_z_in"]).to_numpy()


def scorable(targets, pbp, cols):
    """The plausible targets joined to the pbp columns a caller names."""
    return targets[targets["plausible"]].merge(pbp[["play_id"] + cols], on="play_id")


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
    n_pbp, n_pose = len(pbp), len(poses)
    n_pl = int(targets["plausible"].sum())
    L = ["PIPELINE FUNNEL", "-" * 64]
    for label, kept, prev, why in [
            ("pitches", n_pbp, None, "every pitch of the season, clip or no clip"),
            ("detections available", n_pose, n_pbp, "No video, strikezone box, ball release or CF picture"),
            ("plausible", n_pl, n_pose, "No usable glove target")]:
        drop = "" if prev is None else f"  -{prev - kept}"
        L.append(f"  {label:<22}{kept:>8}{drop:>9}   {why}")
    L += [f"  end-to-end: {n_pl} command targets = {100 * n_pl / n_pbp:.2f}% of pitches", ""]
    return L


def pose_accuracy(poses):
    """Reprojection error of the shipped poses, read off the columns solve_camera_pose stamps."""
    pv = poses.dropna(subset=["reproj_px"])
    L = ["CAMERA-POSE ACCURACY (trajectory reprojection)", "-" * 64,
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


# ─────────────────────────────────────────────  entry point

def frame(targets, pbp):
    """Prepare table for each validation."""
    d = targets[targets["plausible"]].merge(
        pbp[["game_pk", "play_id", "date", "pitcher_id", "pitch_type"]],
        on=["game_pk", "play_id"], how="inner")
    d = d.dropna(subset=["pitcher_id", "pitch_type", "plate_x_in", "naive_x_in"])
    d = d.sort_values(["date", "play_id"]).reset_index(drop=True)
    d["pitcher_id"] = d.pitcher_id.astype(int)
    d["hand"] = np.where(d.pitcher_id.map(pbp.groupby("pitcher_id").x0.median()) < 0, "R", "L")  # release side
    assert d.play_id.is_unique
    return d


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    base = DATA / year
    nested = sorted(base.glob("*/targets.csv.gz"))
    assert not nested, f"this tree nests another season; score that one: opencommand.py {year}/{nested[0].parent.name}"
    season = Path(year).parts[0]
    prev_year = year.replace(season, str(int(season) - 1), 1)   # stickiness reads the season BEFORE
    fg_file = DATA / "fangraphs" / f"fg_pitching_{season}.csv.gz"
    fg_prev_file = DATA / "fangraphs" / f"fg_pitching_{int(season) - 1}.csv.gz"
    fg_next_file = DATA / "fangraphs" / f"fg_pitching_{int(season) + 1}.csv.gz"
    assert fg_file.exists(), f"the Fangraphs season file ships beside the trees: {fg_file}"

    targets = pd.read_csv(base / "targets.csv.gz")
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    poses = pd.read_csv(base / "camera_poses.csv.gz")
    scores = command_scores(targets, pbp)
    scores.to_csv(base / "command_scores.csv", index=False, lineterminator="\n")

    lb = scores[(scores["pitch_type"] == "ALL") & (scores["n"] >= LEADERBOARD_MIN_N)]
    lb = lb.sort_values("inferred_in")
    w = max(lb["pitcher"].str.len(), default=0)
    board = [f"Best command (inferred median miss, min {LEADERBOARD_MIN_N} scored pitches):"]
    board += [f"  {r['pitcher']:<{w}}  {r['inferred_in']:>6.2f} in  (naive {r['naive_in']:.2f}, n={r['n']})"
              for _, r in lb.iterrows()]

    d = frame(targets, pbp)
    prev = (frame(pd.read_csv(DATA / prev_year / "targets.csv.gz"),
                  pd.read_csv(DATA / prev_year / "pbp_info.csv.gz"))
            if (DATA / prev_year / "targets.csv.gz").exists() else None)
    print(f"{year}: {len(d)} scorable pitches, {d.pitcher_id.nunique()} pitchers"
          + (f"   previous season {prev_year}: {len(prev)} pitches" if prev is not None else ""),
          flush=True)
    whole = {name: missed(fn, d, d) for name, fn in METHODS}
    rank = pd.Series(np.random.default_rng(SEED).random(len(d)), index=d.index).groupby(d.pitcher_id).rank(method="first")
    early = {n: {name: missed(fn, t, t) for name, fn in METHODS}
             for n, t in ((n, d[rank <= n]) for n in NS)}  # reused by confidence

    fg = read_fg(fg_file)
    fg_prev = read_fg(fg_prev_file) if fg_prev_file.exists() else None
    fg_next = read_fg(fg_next_file) if fg_next_file.exists() else None

    L = [f"OPENCOMMAND VALIDATIONS {year}", "=" * 64, ""]
    for make_block in (lambda: val_median_miss(d, whole), lambda: val_heldout(d),
                       lambda: val_flatness(d, whole, early),
                       lambda: val_confidence(d, whole, early),
                       lambda: val_correlations(d, whole, fg, fg_next, season),
                       lambda: val_stabilization(d, whole, fg_seasons(season)),
                       lambda: val_stickiness(d, whole, prev, prev_year, fg, fg_prev),
                       lambda: funnel(targets, pbp, poses), lambda: pose_accuracy(poses),
                       lambda: distribution(targets, pbp)):
        block = make_block()                 # one at a time, printed on completion, so late
        L += block                           # failure can't eat finished sections
        print("\n".join(block), flush=True)
    L += board
    print("\n".join(board))
    ART.mkdir(exist_ok=True)
    (ART / f"validations_{season}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote data/{year}/command_scores.csv, artifacts/validations_{season}.txt")
