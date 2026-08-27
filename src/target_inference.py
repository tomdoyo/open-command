"""Step 3 of the OpenCommand pipeline.

Notes:
 - Target detection takes the glove peak
    1. Window: glove detections in [release - 2.0 s, release - 0.3 s]
    2. Peak: choose highest glove location (penalized by being far from release)
 - Target inference takes detected glove_xz at peak
   Both parts are a 2 level hierarchical model fit by empirical Bayes
     - Each by its own standard error, each level's variance adjusted by DerSimonian-Laird
    1. Glove dependence:
       - Some pitchers don't look at the glove (e.g. Misiorowski) or make their
         catchers set up depending on their miss patterns that day (or even previous
         pitch). Some pitchers adjust more than 1 inch per inch of glove movement
         (e.g. Skenes).
       - Fit 4 slopes (without intercepts): xx, xz, zx, zz
         (E.g. xx is how much target_x moves for each inch glove_x moves)
    2. Offset:
       - Some pitchers end pitches at the glove, some pitchers let pitches start
         at the glove and break away, etc.
       - Shrink offsets to get more robust values at low n.
         (Fixed pitcher × pitch type offset overfits at low n. Instead of getting
          0 inch miss at n=1, shrink offsets to league distribution prior.)
       - Pitch type × handedness has its own distribution (e.g. a changeup lands
         4 inches below the pitcher's average, a four-seam 4 above), so a pitch type
         starts there instead of at its pitcher.
       - The point of xz and zx terms in glove dependence is in case offsets are
         different between target clusters for a pitcher's pitch type
         (e.g. Kyle Hendricks' SI lands lower than the glove for glove side, but
          not for arm side).
         (This solution was shown to be simpler but as good as clustering approaches.)

Reads:      data/<year>/glove_locations/<game_pk>.csv.gz +
            data/<year>/pbp_info.csv.gz (pitch type for the screen, pitcher + actual
            location for the offset, x0 for the handedness group)
Writes:     data/<year>/targets.csv.gz, one row per posed clip, both target pairs;
            `status` is "ok" or "no target"
Run:        python src/target_inference.py [year=2026]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

WINDOW_S = 2.0              # target search window length before release
END_BEFORE_RELEASE_S = 0.3  # window end: release - 0.3 s (catch-lock-safe)
LATENESS_IN_PER_S = 15.0    # penalty factor peak being far from release

CELL = ["pitcher_id", "pitch_type"]
AXES = {"x": ("naive_x_in", "kx", "plate_x_in"), "z": ("naive_z_in", "kz", "plate_z_in")}
WX, WZ, WCROSS = (0.0, 1.3), (-0.2, 1.1), (-1.5, 1.5)   # own-x, own-z and cross-term weight bounds


def select_target(g):
    """Finds the targeting peak in one clip's glove-location rows (see module docstring).
    Returns a dict. status and release_s are always set; the target fields are set
    only when status is "ok"."""
    fps, release_s = float(g["fps"].iloc[0]), float(g["release_s"].iloc[0])
    out = {"release_s": release_s}
    lo = (release_s - WINDOW_S) * fps
    hi = (release_s - END_BEFORE_RELEASE_S) * fps
    # between() is False on NaN, so a glove-less clip's one NaN sentinel row drops here
    win = g[g["frame_idx"].between(lo, hi)]

    if len(win) == 0:
        return {**out, "status": "no target"}

    # world-space peak: highest glove, discounted by how long before the window's last
    # detection it sits, so a late target beats an earlier and higher one
    frames = win["frame_idx"].to_numpy()
    x_in, z_in = win["x_in"].to_numpy(), win["z_in"].to_numpy()
    i = int(np.argmax(z_in - LATENESS_IN_PER_S * (frames.max() - frames) / fps))
    return {**out, "status": "ok", "target_frame": int(frames[i]),
            "naive_x_in": float(x_in[i]), "naive_z_in": float(z_in[i])}


def targets_for_game(job):
    """One game's glove-location file → per-clip target rows (see select_target)."""
    f, info_rows = job
    rows = []
    for play_id, g in pd.read_csv(f, float_precision="round_trip").groupby("play_id", sort=False):
        t = select_target(g)
        p = info_rows[play_id]
        rows.append({"game_pk": int(g["game_pk"].iloc[0]), "play_id": play_id, "park": p["home_team"],
                     "y_depth_ft": float(g["y_depth_ft"].iloc[0]),  # travels with the glove rows
                     "plate_x_in": p["plate_x"] * 12, "plate_z_in": p["plate_z"] * 12, **t})
    return rows


# ─────────────────────────────────────────────  target inference

def predict_random_effect(est, se2, parent):
    """Shrinkage toward prior based on se.

    DerSimonian-Laird: adjust each point in prior distribution by n to adjust for noise.
    """
    if len(est) < 2: return parent.copy(), 0.0
    prec = 1 / se2
    S1, S2 = float(prec.sum()), float((prec ** 2).sum())
    Q = float((prec * (est - parent) ** 2).sum())
    den = S1 - S2 / S1 if S1 > 0 else 0.0     # zero on a slice too thin to hold a spread at all,
    if den <= 0: return parent.copy(), 0.0    # e.g. one pitch per pitcher in the season's first week
    tau2 = max((Q - (len(est) - 1)) / den, 0.0)
    return ((parent + tau2 * (est - parent) / (tau2 + se2)) if tau2 > 0 else parent.copy()), tau2


def solve_glove_coefs(s):
    """Least squares for wx, wz. ball deviation = wx * glove_dx + wz * glove_dz.

    No intercept. Returns wx, wz (correspond to xx, xz OR zx, zz) and their squared 
    standard error.
    """
    flat = s.xx * s.zz - s.xz ** 2                        # zero or below: fewer than 2 glove positions
    det = flat.clip(lower=1e-9)                           # keeps the division finite; masked off below
    wx, wz = (s.zz * s.xb - s.xz * s.zb) / det, (s.xx * s.zb - s.xz * s.xb) / det
    return wx, wz, (s.zz / det).mask(flat <= 0, np.inf), (s.xx / det).mask(flat <= 0, np.inf)


def shrink(est, se2, grp):
    """One level: shrink value towards group mean using standard error."""
    centre = grp.map((est / se2).groupby(grp).sum() / (1 / se2).groupby(grp).sum())
    tau2 = grp.map(pd.Series({g: predict_random_effect(est.loc[i], se2.loc[i], centre.loc[i])[1]
                              for g, i in est.groupby(grp).groups.items()}))
    return centre + tau2 * (est - centre) / (tau2 + se2)


def fit_glove_weights(r):
    """Calculate glove dependence weights: xx, xz, zx, zz.

    1. Least squares on xx, xz, zx, zz at pitcher, pitcher × pitch type
    2. 2 level empirical Bayes on each of xx, xz, zx, zz, each shrunk by its own standard error
        - Pitcher-level: shrink to league distribution of pitcher xx, xz, zx, zz
        - Pitch type-level: pitch type subtracted by shrunk pitcher-level, shrink to pitch type × hand
                            distribution of pitch type avg subtracted by pitcher avg
    xx clipped to 0..1.3, zz to -0.2..1.1; xz, zx to -1.5..1.5
    """
    gx, gz = (r.naive_x_in - r.kx).to_numpy(), (r.naive_z_in - r.kz).to_numpy()
    hand = r.groupby("pitcher_id").hand.first()
    out = {}
    for ax, (nv, kc, pl) in AXES.items():
        b = (r[pl] - r[kc]).to_numpy()
        # 1. least squares
        f = pd.DataFrame({"xx": gx * gx, "xz": gx * gz, "zz": gz * gz,
                          "xb": gx * b, "zb": gz * b, "bb": b * b}, index=r.index)
        s, n = f.groupby(r.pitcher_id).sum(), f.groupby(r.pitcher_id).size()
        px, pz, ix, iz = solve_glove_coefs(s)
        sig2 = ((s.bb - px * s.xb - pz * s.zb) / (n - 2).clip(lower=1)).clip(lower=1e-9)  # scatter about the fit
        # 2a. pitcher level
        league = pd.Series("league", index=s.index)
        wpx, wpz = shrink(px, sig2 * ix, league), shrink(pz, sig2 * iz, league)
        # 2b. pitch type level (one group per pitch type × hand)
        c = f.groupby([r.pitcher_id, r.pitch_type]).sum()
        pid = c.index.get_level_values(0)
        grp = pd.Series(c.index.get_level_values(1) + "-" + pid.map(hand), index=c.index)
        cx, cz, icx, icz = solve_glove_coefs(c)
        csig = pd.Series(pid.map(sig2).to_numpy(), index=c.index)
        parx = pd.Series(pid.map(wpx).to_numpy(), index=c.index)
        parz = pd.Series(pid.map(wpz).to_numpy(), index=c.index)
        wcx, wcz = parx + shrink(cx - parx, csig * icx, grp), parz + shrink(cz - parz, csig * icz, grp)
        # 3. clip
        bx, bz = (WX, WCROSS) if ax == "x" else (WCROSS, WZ)
        out[ax] = pd.DataFrame({"wx": wcx.clip(*bx), "wz": wcz.clip(*bz)})
    return out


def apply_glove_weights(f, W):
    """Target_x/z = mean_x/z + wx * (glove_x - mean_x) + wz * (glove_z - mean_z) + offset.

    wx, wz fit separately for x/z.
    """
    gx, gz = (f.naive_x_in - f.kx).to_numpy(), (f.naive_z_in - f.kz).to_numpy()
    for ax, (nv, kc, pl) in AXES.items():
        cell = f[CELL].merge(W[ax], left_on=CELL, right_index=True, how="left")
        f["t" + ax] = f[kc].to_numpy() + cell.wx.to_numpy() * gx + cell.wz.to_numpy() * gz
    return f


def fit_and_apply_offsets(a, e):
    """Target_x/z = mean_x/z + wx * (glove_x - mean_x) + wz * (glove_z - mean_z) + offset.

    1. Get mean offset per league, pitcher, pitcher × pitch type
    2. 2 level empirical Bayes on each axis, each mean shrunk by its own standard error
       (season per-pitch variance / pitch count)
        - Pitcher-level: shrink to league distribution of pitcher mean leftover, centred on
                         the league mean
        - Pitch type-level: pitch type subtracted by shrunk pitcher-level, shrink to pitch type × hand
                            distribution of pitch type mean subtracted by pitcher mean
    3. Look up each row's e: its pitch type's offset
    """
    hand = a.groupby("pitcher_id").hand.first()
    out = {}
    for col in ("rx", "rz"):
        # 1. league mean and per-pitch variance
        g0, S = float(a[col].mean()), float(a[col].var())
        # 2a. pitcher level
        p = a.groupby("pitcher_id")[col].agg(["mean", "count"])
        pp = shrink(p["mean"], S / p["count"], pd.Series("league", index=p.index))
        # 2b. pitch type level (one group per pitch type × hand)
        c = a.groupby(CELL)[col].agg(["mean", "count"])
        pid = c.index.get_level_values(0)
        grp = pd.Series(c.index.get_level_values(1) + "-" + pid.map(hand), index=c.index)
        par = pd.Series(pid.map(pp).to_numpy(), index=c.index)
        cp = par + shrink(c["mean"] - par, S / c["count"], grp)
        # 3. look up e
        out[col] = e[CELL].merge(cp.rename("v"), left_on=CELL, right_index=True, how="left").v.to_numpy()
    return out["rx"], out["rz"]


def infer_targets(tr, te):
    """Infer targets with glove weights & offsets. Fit on tr, applied to te.

    Target_x = mean_glove_x + xx * (glove_x - mean_glove_x) + xz * (glove_z - mean_glove_z) + offset_x
    Target_z = mean_glove_z + zx * (glove_x - mean_glove_x) + zz * (glove_z - mean_glove_z) + offset_z

    mean_glove is per pitcher × pitch type from tr. The chain passes the same frame twice;
    opencommand.py passes half a season and scores the other half.
    """
    cen = tr.groupby(CELL)[["naive_x_in", "naive_z_in"]].mean().rename(
        columns={"naive_x_in": "kx", "naive_z_in": "kz"})
    a = tr.merge(cen, left_on=CELL, right_index=True, how="left")
    e = te.merge(cen, left_on=CELL, right_index=True, how="left")
    W = fit_glove_weights(a)
    a, e = apply_glove_weights(a, W), apply_glove_weights(e, W)
    a["rx"], a["rz"] = a.plate_x_in - a.tx, a.plate_z_in - a.tz
    ox, oz = fit_and_apply_offsets(a, e)
    return e.tx.to_numpy() + ox, e.tz.to_numpy() + oz


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    base = DATA / year
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    info = pbp.set_index("play_id")[["game_pk", "home_team", "pitch_type", "pitcher_id",
                                     "plate_x", "plate_z"]]
    hand = pbp.groupby("pitcher_id").x0.median().lt(0).map({True: "R", False: "L"})   # release side

    fields = info[["game_pk", "home_team", "plate_x", "plate_z"]]
    by_game = {str(g): d.drop(columns="game_pk").to_dict("index")
               for g, d in fields.groupby("game_pk")}
    jobs = [(f, by_game[f.name.split(".")[0]]) for f in sorted((base / "glove_locations").glob("*.csv.gz"))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("OC_WORKERS", max(1, os.cpu_count() - 2)))) as ex:
        rows = [r for part in ex.map(targets_for_game, jobs, chunksize=4) for r in part]
    tg = pd.DataFrame(rows)

    # plausibility screen 
    pt = tg["play_id"].map(info["pitch_type"])
    z_lo = np.where(pt.isin(["FF", "SI", "FC"]), 10, 6)
    z_hi = np.where(pt == "FF", 50, 44)
    tg["plausible"] = ((tg["status"] == "ok") & (tg["naive_x_in"].abs() <= 20)
                       & (tg["naive_z_in"] > z_lo) & (tg["naive_z_in"] < z_hi))

    # inferred targets
    a = tg.assign(pitcher_id=tg["play_id"].map(info["pitcher_id"]), pitch_type=pt)
    a = a[tg["plausible"]].dropna(subset=["pitcher_id", "pitch_type"])
    a["pitcher_id"] = a["pitcher_id"].astype(int)
    a["hand"] = a["pitcher_id"].map(hand)
    ix, iz = infer_targets(a, a)
    tg["inferred_x_in"], tg["inferred_z_in"] = np.nan, np.nan
    tg.loc[a.index, ["inferred_x_in", "inferred_z_in"]] = np.c_[ix, iz]

    tg.to_csv(base / "targets.csv.gz", index=False, lineterminator="\n",
              compression={"method": "gzip", "compresslevel": 6})
    ok = int((tg["status"] == "ok").sum())
    print(f"targets: {ok} ok / {int(tg['plausible'].sum())} plausible of {len(tg)} posed clips")
