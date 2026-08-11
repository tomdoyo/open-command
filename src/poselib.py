"""Shared geometry + timing library.

Notes:
 - World coordinates: feet, plate-centered, with x = catcher's right, 
   y = toward the pitcher (origin at the plate's back tip), z = up. 
 - Strikezone box drawing moved from 17/12 ft to 8.5/12 ft in 2026 (ABS).
   BUT plate_x/z stays at 17/12 ft.
 - Broadcast cam (CY_FT) is assumed to be 400ft.

What lives here:
  camera/project/unproject:  the pinhole model
  make_traj:                 Statcast 9-param ball trajectory
  derive_release:            release time (s): the tracker's clock-fit hint when
                             present, else the first coherent ball string, else NaN
                             (no prior fallback, so a release-less clip drops)
  ball_runs:                 one clip's ball detections, split into contiguous runs;
                             the shared first step of the two scans below
  candidate_runs:            contiguous ball-detection runs whose timing is
                             physically plausible for the pitch (release +/-
                             RUN_END_HALF_S), so warm-ups/replays can't
                             masquerade as flight
  fit_position:              per-clip 7-param (Cx, Cz, pan, tilt, roll, f, t0) solve
                             with Cy pinned; votes per-game camera positions
  solve_ptz:                 pose from the 4 box corners alone with C frozen; the
                             per-clip pose, every boxed clip gets one
  load_game:                 joins one game's tracks + K-zone boxes + pbp into clip
                             dicts the solvers and target inference consume
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parents[1] / "data"

W, H = 1280, 720          # broadcast clip resolution
CY_FT = 400.0             # camera distance is PINNED, never solved: a free Cy is
                          # degenerate against focal length (validated on a 140-clip
                          # harness: it rails at bounds while targets move 0.06 in)
SETUP_DEPTH_FT = -1.75    # depth plane the catcher sets up at (median catch depth)

def zone_y(year):
    """Depth plane (ft) the strike zone lives at."""
    return 17 / 12 if int(Path(str(year)).parts[0]) <= 2025 else 8.5 / 12

RELEASE_LO_S, RELEASE_HI_S = 2.5, 3.7
RUN_END_HALF_S = 0.60
RELEASE_HALF_S = 0.60

BALL_COLS = ["frame_idx", "baseball_center_x", "baseball_center_y"]
MAX_RUN_GAP_FR = 4   # a hole longer than this ends a contiguous ball run

# --- pinhole camera model -----------------------------------------------------------

_I3 = np.eye(3)

def _skew(a):
    """Cross-product matrix of a 3-vector."""
    K = np.zeros((3, 3))
    K[0, 1], K[0, 2] = -a[2], a[1]
    K[1, 0], K[1, 2] = a[2], -a[0]
    K[2, 0], K[2, 1] = -a[1], a[0]
    return K

def camera(C, pan, tilt, roll):
    """Camera basis vectors (right, up, fwd) for a pose at C looking at the plate."""
    # base frame: aim the camera at a point just above the plate, (0, 0, 3 ft)
    fwd = np.array([0, 0, 3.0]) - C
    fwd /= np.linalg.norm(fwd)
    # cross(fwd, z_hat) written out: np.cross on a 3-vector costs more than the 3 products
    # it performs, and with b = (0, 0, 1) the same multiply-subtracts reduce to this exactly.
    right = np.array([fwd[1], -fwd[0], 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)   # unit: right and fwd are unit and perpendicular

    def rot(axis, angle):  # Rodrigues rotation of `angle` radians about `axis`
        axis = axis / np.linalg.norm(axis)
        K = _skew(axis)
        return _I3 + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    # apply pan (about up), then tilt (about right), then roll (about fwd)
    R = rot(fwd, roll) @ rot(right, tilt) @ rot(up, pan)
    return R @ right, R @ up, R @ fwd

def project_on(basis, C, f, X):
    """World point(s) X → pixel(s) against an already-solved camera basis."""
    right, up, fwd = basis
    rel = np.atleast_2d(X) - C   # camera-to-point vector(s)
    depth = rel @ fwd            # distance along the optical axis
    return np.stack([W / 2 + f * (rel @ right) / depth,
                     H / 2 - f * (rel @ up) / depth], axis=-1)

def project(C, pan, tilt, roll, f, X):
    """World point(s) X → pixel(s). Accepts a single point or an (N, 3) array."""
    px = project_on(camera(C, pan, tilt, roll), C, f, X)
    return px[0] if np.ndim(X) == 1 else px

def unproject(C, pan, tilt, roll, f, pix, y_plane):
    """Pixel(s) → world (x, z) where the back-projected ray crosses the depth plane
    y=y_plane. Accepts a single pixel (returns a tuple) or an (N, 2) array."""
    right, up, fwd = camera(C, pan, tilt, roll)
    p = np.atleast_2d(pix)
    ray = (p[:, :1] - W / 2) / f * right + (H / 2 - p[:, 1:]) / f * up + fwd
    scale = (y_plane - C[1]) / ray[:, 1]   # march along each ray until it reaches y_plane
    pt = C + scale[:, None] * ray
    return (pt[0, 0], pt[0, 2]) if np.ndim(pix) == 1 else pt[:, [0, 2]]

# --- ball trajectory ----------------------------------------------------------------

def make_traj(p):
    """Statcast 9-param trajectory closure: time t (s) → ball world position (x, y, z)."""
    def traj(t):
        t = np.asarray(t)
        return np.stack([p["x0"] + p["vx0"] * t + 0.5 * p["ax"] * t**2,
                         p["y0"] + p["vy0"] * t + 0.5 * p["ay"] * t**2,
                         p["z0"] + p["vz0"] * t + 0.5 * p["az"] * t**2], axis=-1)
    return traj

# --- release time -------------------------------------------------------------------

def derive_release(df, fps):
    """Release time (s) for glove-window placement, or NaN when release can't be
    detected; a release-less clip drops (no window to place)."""
    if "release_s_hint" in df and df["release_s_hint"].notna().any():
        return float(df["release_s_hint"].dropna().iloc[0])
    ball, runs = ball_runs(df[df["frame_idx"] >= RELEASE_LO_S * fps])
    frames = ball["frame_idx"].to_numpy()
    for r in runs:
        if len(r) < 3:
            continue
        xy = ball.iloc[r][["baseball_center_x", "baseball_center_y"]].to_numpy()
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(frames[r])
        if 0.5 <= np.median(step) <= 30:  # moving, not teleporting between false positives
            return float(np.clip(frames[r[0]] / fps, RELEASE_LO_S, RELEASE_HI_S))
    return np.nan

# --- ball runs + solves -------------------------------------------------------------

def ball_runs(df):
    """One clip's detected-ball rows sorted by frame, split into contiguous runs.

    Returns (ball, runs): the rows, and a list of positional index arrays into them, one
    per run. An empty clip yields one empty run, which every caller's length test drops.
    Both the release scan and candidate_runs start here."""
    ball = df[df["baseball_center_x"].notna()][BALL_COLS].sort_values("frame_idx")
    frames = ball["frame_idx"].to_numpy()
    breaks = np.where(np.diff(frames) > MAX_RUN_GAP_FR)[0] + 1
    return ball, np.split(np.arange(len(frames)), breaks)

def candidate_runs(df, fps, T_plate):
    """Contiguous detected-ball runs (gap ≤ 4, ≥ 8 pts) whose END lies in the physically
    plausible plate-crossing window around this clip's OWN release; clipped to flight 
    duration."""
    ball, runs = ball_runs(df)
    if len(ball) < 8:
        return []
    rel = derive_release(df, fps)
    lo, hi = ((RELEASE_LO_S, RELEASE_HI_S) if np.isnan(rel)
              else (rel - RUN_END_HALF_S, rel + RUN_END_HALF_S))
    end_lo = (lo + T_plate - 0.15) * fps
    end_hi = (hi + T_plate + 0.35) * fps
    out = []
    for r in runs:
        seg = ball.iloc[r]
        if not end_lo <= seg["frame_idx"].max() <= end_hi:
            continue
        seg = seg[seg["frame_idx"] >= seg["frame_idx"].max() - int((T_plate + 0.10) * fps)]
        if len(seg) >= 8:
            out.append(seg)
    return sorted(out, key=len, reverse=True)[:4]

def fit_position(c, seg):
    """Phase-1 position solve on one candidate run: (Cx, Cz, pan, tilt, roll, f, t0)
    with Cy pinned at CY_FT. t0 is a FITTED NUISANCE parameter, solved for and then
    discarded (only Cx/Cz are returned). Box-only position was degenerate in Cz."""
    obs = seg[["baseball_center_x", "baseball_center_y"]].to_numpy()
    frames = seg["frame_idx"].to_numpy()
    f0 = c["f0"]
    t0_init = float(np.clip(c["T_plate"] - frames.max() / c["fps"], -RELEASE_HI_S, -RELEASE_LO_S))
    t_frames = frames / c["fps"]
    traj, box_world, box_obs = c["traj"], c["box_world"], c["box_obs"]

    def res(x):  # x = Cx, Cz, pan, tilt, roll, f, t0; residual = ball reproj + 3x box reproj
        C = np.array([x[0], CY_FT, x[1]])
        basis = camera(C, x[2], x[3], x[4]) 
        r_ball = (project_on(basis, C, x[5], traj(x[6] + t_frames)) - obs).ravel()
        r_box = 3.0 * (project_on(basis, C, x[5], box_world) - box_obs).ravel()
        return np.concatenate([r_ball, r_box])

    n_ball = 2 * len(frames)

    x0 = np.array([0, 20, 0, 0, 0, f0, t0_init])          # initial guess
    lo = [-200, 5, -0.15, -0.15, -0.15, f0 / 2, -RELEASE_HI_S]   # lower bounds
    hi = [200, 120, 0.15, 0.15, 0.15, f0 * 2, -RELEASE_LO_S]     # upper bounds
    try:
        fit = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0,
                            x_scale=[30, 15, 0.02, 0.02, 0.02, f0 * 0.3, 0.05])
    except Exception:
        return None
    return {"Cx": float(fit.x[0]), "Cz": float(fit.x[1]), "n_fit": len(frames),
            "ball_rmse_px": float(np.sqrt((fit.fun[:n_ball] ** 2).mean()))}

def solve_ptz(c, C):
    """Phase-2 per-clip pose: pan/tilt/roll/f from the 4 box corners alone, C frozen
    to the game position. The box was detected on the clip's own peak-glove frame. 
    Returns [C, pan, tilt, roll, f] or None."""
    f0 = c["f0"]

    def res(q):
        return (project(C, *q[:3], q[3], c["box_world"]) - c["box_obs"]).ravel()

    try:
        fit = least_squares(res, np.array([0, 0, 0, f0]), bounds=([-0.15, -0.15, -0.15, f0 / 2], [0.15, 0.15, 0.15, f0 * 2]),
                            x_scale=[0.02, 0.02, 0.02, f0 * 0.3])
    except Exception:
        return None
    return np.concatenate([C, fit.x])

# --- clip loading -------------------------------------------------------------------

def load_game(game_pk, pbp, zones, tracks_dir, zone_y_ft):
    """One game's clip dicts: join tracks (per-game csv.gz) + K-zone boxes + pbp.

    pbp/zones are the season DataFrames (pass pre-filtered or whole, since it is filtered here);
    tracks_dir is the season's data/<year>/raw/gloveball_tracks; zone_y_ft is the
    era's zone depth plane (zone_y(year)).
    Only boxed clips load: a clip with no K-zone box gets no pose and no target."""
    f = Path(tracks_dir) / f"{game_pk}.csv.gz"
    if not f.exists():
        return []
    tracks = pd.read_csv(f)
    # split the game's tracks by clip ONCE: the per-clip mask this replaces re-scanned the
    # whole (object-dtype) video column for every box, and at ~250 boxes a game that scan
    # was a sixth of the whole pose stage
    by_video = dict(tuple(tracks.groupby("video")))
    boxes = zones[zones["video"].str.startswith(f"{game_pk}_")].dropna(subset=["x_left", "y_top", "x_right", "y_bottom"])
    pbp = pbp[pbp["game_pk"] == game_pk]
    pbp = pbp.set_index(pbp["game_pk"].astype(str) + "_" + pbp["play_id"] + ".mp4")
    clips = []
    for _, box in boxes.iterrows():
        video = box["video"]
        if video not in pbp.index:
            continue
        p = pbp.loc[video]
        df = by_video.get(video)
        if df is None:
            continue
        fps = float(df["fps"].iloc[0])
        # T_plate: flight time to the era's zone plane, positive root of the y(t) quadratic
        roots = np.roots([0.5 * p["ay"], p["vy0"], p["y0"] - zone_y_ft])
        T_plate = float(min(t for t in roots if np.isreal(t) and t > 0))
        # 4 box corners: world points (plate width 17 in, sz_top..sz_bot tall) and their pixels.
        # The REFERENCE zone, never this pitch's own sz: Statcast re-measures sz_top/sz_bot on
        # takes and leaves the batter's default on swings (pre-ABS), but strikezone box doesn't
        # change, so the raw value would fit the pose against the pitch's own outcome
        box_world = np.array([[x, zone_y_ft, z] for z in [p["sz_top_ref"], p["sz_bot_ref"]] for x in [-17 / 24, 17 / 24]])
        box_obs = np.array([[box["x_right"], box["y_top"]], [box["x_left"], box["y_top"]],
                            [box["x_right"], box["y_bottom"]], [box["x_left"], box["y_bottom"]]])
        clips.append({"video": video, "game_pk": int(p["game_pk"]), "park": p["home_team"], "p": p,
                      "fps": fps, "T_plate": T_plate, "df": df,
                      # f0 initializes/scales the focal solve (bounds f0/2..2f0): CY_FT is the
                      # pinned camera distance the solve fixes Cy at, so the seed matches the
                      # frame f converges in (Cy and f are degenerate for a planar target)
                      "f0": (box["x_right"] - box["x_left"]) * (CY_FT * 12) / 17,
                      "box_world": box_world, "box_obs": box_obs})
    return clips

def load_season(year):
    """(pbp, zones) season DataFrames: the two compact inputs every step joins against."""
    base = DATA / str(year)
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    zones = pd.read_csv(base / "raw" / "strikezone_tracking.csv.gz")
    assert {"game_pk", "play_id", "home_team", "pitcher_id", "pitch_type",
            "sz_top", "sz_bot", "sz_top_ref", "sz_bot_ref",
            "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
            "plate_x", "plate_z"} <= set(pbp.columns), "pbp_info is missing chain columns"
    assert {"video", "x_left", "y_top", "x_right", "y_bottom"} <= set(zones.columns), \
        "strikezone_tracking is missing box columns"
    return pbp, zones
