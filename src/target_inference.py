"""Step 3 of the OpenCommand pipeline.

Notes:
 - Target detection takes the glove peak
    1. Window: glove detections in [release - 2.0 s, release - 0.3 s].
        - lead_s is time between first glove detection & release.
          (Used to filter late camera cuts)
    2. Peak: highest glove location whose ±0.05 s neighborhood is tight
        - neighborhood spread ≤ SUPPORT_IN inches

Reads:      data/<year>/glove_locations/<game_pk>.csv.gz +
            data/<year>/pbp_info.csv.gz (pitch type for the screen, pitcher + actual
            location for the offset)
Writes:     data/<year>/targets.csv.gz, one row per posed clip, both target pairs;
            `status` says why a clip has no target (too few detections / no
            supported peak)
Run:        python src/target_inference.py [year=2026]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ocl import common_parse_args

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

WINDOW_S = 2.0              # target search window length before release
END_BEFORE_RELEASE_S = 0.3  # window end: release - 0.3 s (catch-lock-safe)
PEAK_HALF_S = 0.05          # neighborhood half-width around a peak candidate
SUPPORT_IN = 4.0            # max neighborhood spread (inches) for a stable peak
MIN_WINDOW_DETS = 10        # fewer glove detections than this → no target
SPEED_HALF_S = 0.10         # catcher_speed: half-width of the local velocity fit, seconds
SPEED_MIN_FIT = 4           # a slope on fewer points than this is the noise, not the motion
LEAD_DETS = 3               # k=3 worked best


def catcher_speed(frames, x_in, z_in, fps, target_frame):
    """Glove speed at the target frame, in world inches per second."""
    t = frames / fps
    near = np.abs(frames - target_frame) <= SPEED_HALF_S * fps
    if near.sum() < SPEED_MIN_FIT or np.ptp(t[near]) <= 1e-6:
        return float("nan")
    return float(np.hypot(np.polyfit(t[near], x_in[near], 1)[0],
                          np.polyfit(t[near], z_in[near], 1)[0]))


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
    if len(win) < MIN_WINDOW_DETS:
        return {**out, "status": f"too few glove detections in window ({len(win)})"}

    # n_window_max is how many integer frame slots the window holds. Neither bound is an
    # integer, so it comes out 101 or 102 depending on fps and on where release_s * fps
    # falls between frames. n_window means something only against this ceiling: the raw
    # count is the ceiling minus the missed frames, the two parts mean opposite things,
    # and the count alone is not a quality signal.
    # This line sits below the guard because derive_release returns NaN on a release-less
    # clip and int() of a NaN bound raises. Those clips drop above, on the empty window.
    n_window_max = int(np.floor(hi) - np.ceil(lo) + 1)

    # how long the glove was under observation before release. 
    # safe to index because MIN_WINDOW_DETS above already guaranteed 10 rows.
    lead_s = release_s - np.sort(win["frame_idx"].to_numpy())[LEAD_DETS - 1] / fps

    # world-space peak: highest glove first, first tight neighborhood wins
    frames = win["frame_idx"].to_numpy()
    x_in, z_in = win["x_in"].to_numpy(), win["z_in"].to_numpy()
    for i in np.argsort(-z_in):
        near = np.abs(frames - frames[i]) <= PEAK_HALF_S * fps
        if max(np.ptp(x_in[near]), np.ptp(z_in[near])) <= SUPPORT_IN:
            return {**out, "status": "ok", "target_frame": int(frames[i]),
                    "naive_x_in": float(np.median(x_in[near])),
                    "naive_z_in": float(np.median(z_in[near])),
                    # gap to the window END; ~0 = pinned at the edge, glove still rising
                    "gap_to_release_s": float((hi - frames[i]) / fps),
                    "n_window": len(win), "n_window_max": n_window_max,
                    "n_peak": int(near.sum()), "lead_s": float(lead_s),
                    "catcher_speed": catcher_speed(frames, x_in, z_in, fps, frames[i]),
                    "peak_spread_x_in": float(np.subtract(*np.percentile(x_in[near], [75, 25]))),
                    "peak_spread_z_in": float(np.subtract(*np.percentile(z_in[near], [75, 25])))}
    return {**out, "status": "no supported peak"}


def targets_for_game(job):
    """One game's glove-location file → per-clip target rows (see select_target)."""
    f, info_rows = job
    rows = []
    for video, g in pd.read_csv(f, float_precision="round_trip").groupby("video", sort=False):
        t = select_target(g)
        p = info_rows[video]
        rows.append({"video": video, "game_pk": int(video.split("_")[0]), "park": p["home_team"],
                     "y_depth_ft": float(g["y_depth_ft"].iloc[0]),  # travels with the glove rows
                     "plate_x_in": p["plate_x"] * 12, "plate_z_in": p["plate_z"] * 12, **t})
    return rows


if __name__ == "__main__":

    args = common_parse_args("target_inference")
    year = str(args.year)

    base = DATA / year
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    video_key = pbp["game_pk"].astype(str) + "_" + pbp["play_id"] + ".mp4"
    info = pbp.set_index(video_key)[["home_team", "pitch_type", "pitcher_id", "plate_x", "plate_z"]]

    fields = info[["home_team", "plate_x", "plate_z"]]
    by_game = {g: d.to_dict("index")
               for g, d in fields.groupby(fields.index.str.split("_").str[0])}
    jobs = [(f, by_game[f.name.split(".")[0]]) for f in sorted((base / "glove_locations").glob("*.csv.gz"))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("OC_WORKERS", max(1, os.cpu_count() - 2)))) as ex:
        rows = [r for part in ex.map(targets_for_game, jobs, chunksize=4) for r in part]
    tg = pd.DataFrame(rows)

    # plausibility screen 
    pt = tg["video"].map(info["pitch_type"])
    z_lo = np.where(pt.isin(["FF", "SI", "FC"]), 10, 6)
    z_hi = np.where(pt == "FF", 50, 44)
    tg["plausible"] = ((tg["status"] == "ok") & (tg["naive_x_in"].abs() <= 20)
                       & (tg["naive_z_in"] > z_lo) & (tg["naive_z_in"] < z_hi))

    # inferred targets
    offset_key = [tg["video"].map(info["pitcher_id"]), pt]
    for ax in ("x", "z"):
        resid = (tg[f"plate_{ax}_in"] - tg[f"naive_{ax}_in"]).where(tg["plausible"])
        offset = resid.groupby(offset_key).transform("mean")
        tg[f"inferred_{ax}_in"] = (tg[f"naive_{ax}_in"] + offset).where(tg["plausible"])

    tg.to_csv(base / "targets.csv.gz", index=False, lineterminator="\n",
              compression={"method": "gzip", "compresslevel": 6})
    ok = int((tg["status"] == "ok").sum())
    print(f"targets: {ok} ok / {int(tg['plausible'].sum())} plausible of {len(tg)} posed clips")
