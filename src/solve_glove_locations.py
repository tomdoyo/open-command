"""Step 2 of the OpenCommand pipeline.

Notes:
 - Glove plane is fixed to be y = -1.75ft (median catch depth)

Reads:      data/<year>/camera_poses.csv.gz +
            data/<year>/raw/gloveball_tracks/<game_pk>.csv.gz
Writes:     data/<year>/glove_locations/<game_pk>.csv.gz, one row per detection:
            video, frame_idx, x_in, z_in, y_depth_ft, fps, release_s
Run:        python src/solve_glove_locations.py [year=2026] [game_pk ...]
            (game subsets need the year spelled out; default: every game)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poselib

MAX_GLOVE_WIDTH_PX = 120
MAX_STEP_SPEED_IN_S = 150.0   # a catcher's glove tops out near 180 in/s; misfires jump 15-40 in
MAX_STEP_JUMP_IN = 20.0       # radius cap, so a long occlusion cannot buy an unbounded jump


def glove_pixels(df):
    """Catcher-glove detection rows (see module docstring)."""
    w = df.get("glove_width", pd.Series(0, index=df.index)).fillna(0)
    m = df[df["glove_center_x"].notna() & (w < MAX_GLOVE_WIDTH_PX)]
    return m[["frame_idx", "glove_center_x", "glove_center_y"]]


def _reach(fr, x, z, fps, seed, forward, keep):
    """Mark every detection reachable from `seed`, walking forward or backward from it.

    The anchor advances ONLY on an accepted detection, so a run of consecutive misfires is
    measured against the last good glove rather than against the previous misfire, and the
    whole run falls instead of just its first frame."""
    af, ax, az = fr[seed], x[seed], z[seed]
    keep[seed] = True
    for i in (range(seed + 1, len(fr)) if forward else range(seed - 1, -1, -1)):
        budget = MAX_STEP_SPEED_IN_S * abs(fr[i] - af) / fps
        dx, dz = x[i] - ax, z[i] - az
        if dx * dx + dz * dz > min(budget, MAX_STEP_JUMP_IN) ** 2:
            continue
        keep[i] = True
        af, ax, az = fr[i], x[i], z[i]


def clean_teleports(xz, fps):
    """Detections the glove could physically have reached (see module docstring).

    Keeps the largest mutually-consistent set. Seeds are tried until every detection has
    been covered by some seed's set, so a clip whose opening frames are misfires still
    finds the real glove; ties go to the earliest seed, which keeps the output byte-stable.
    The real glove is the majority of any clip this detector poses, so it wins from any
    seed inside it."""
    if len(xz) < 2:
        return xz
    fr, x, z = (xz[c].tolist() for c in ("frame_idx", "x_in", "z_in"))
    best, todo = np.zeros(len(fr), bool), set(range(len(fr)))
    while todo:
        keep = np.zeros(len(fr), bool)
        _reach(fr, x, z, fps, min(todo), True, keep)
        _reach(fr, x, z, fps, min(todo), False, keep)
        if keep.sum() > best.sum():
            best = keep
        todo -= set(np.flatnonzero(keep).tolist())
    return xz[best]


def glove_xz(rows, pose):
    """Glove pixel rows → world (x_in, z_in) on the setup-depth plane. 
    The plane depth travels with the rows (y_depth_ft)."""
    pts = poselib.unproject(pose[:3], *pose[3:6], pose[6],
                            rows[["glove_center_x", "glove_center_y"]].to_numpy(),
                            poselib.SETUP_DEPTH_FT)
    out = rows[["frame_idx"]].copy()
    out["x_in"], out["z_in"] = pts[:, 0] * 12, pts[:, 1] * 12
    out["y_depth_ft"] = poselib.SETUP_DEPTH_FT
    return out


def locate_game(args):
    """Solves one game. Unprojects every posed clip's glove detections into world
    inches, stamps fps and release_s, and writes the game its own file at
    data/glove_locations/<game_pk>.csv.gz."""
    game_pk, poses, tracks_dir, out_dir = args
    tracks = pd.read_csv(tracks_dir / f"{game_pk}.csv.gz")
    parts = []
    for video, df in tracks.groupby("video", sort=True):
        if video not in poses.index:
            continue
        pose = poses.loc[video][["Cx", "Cy", "Cz", "pan", "tilt", "roll", "f_px"]].to_numpy(float)
        fps = float(df["fps"].iloc[0])
        release_s = poselib.derive_release(df, fps)
        xz = clean_teleports(glove_xz(glove_pixels(df), pose), fps)
        if not len(xz):  # posed but glove-less clip: keep one NaN row so it stays visible
            xz = pd.DataFrame([{"frame_idx": np.nan, "x_in": np.nan, "z_in": np.nan,
                                "y_depth_ft": poselib.SETUP_DEPTH_FT}])
        xz.insert(0, "video", video)
        xz["fps"], xz["release_s"] = fps, release_s
        parts.append(xz)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_dir / f"{game_pk}.csv.gz", index=False, lineterminator="\n",
              compression={"method": "gzip", "compresslevel": 6})
    return df["video"].nunique(), len(df)


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    base = poselib.DATA / year
    poses = pd.read_csv(base / "camera_poses.csv.gz").set_index("video")
    games = [int(g) for g in sys.argv[2:]] or sorted(poses["game_pk"].unique())
    poses_by = dict(list(poses.groupby("game_pk")))
    out_dir = base / "glove_locations"
    out_dir.mkdir(exist_ok=True)
    tracks = base / "raw" / "gloveball_tracks"
    jobs = [(game_pk, poses_by[game_pk], tracks, out_dir) for game_pk in games if game_pk in poses_by]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("OC_WORKERS", 10))) as ex:
        results = list(ex.map(locate_game, jobs, chunksize=4))
    clips, rows = sum(r[0] for r in results), sum(r[1] for r in results)
    print(f"glove_locations: {rows} rows across {clips} clips ({len(jobs)} games) -> {out_dir}")
