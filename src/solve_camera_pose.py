"""Step 1 of the OpenCommand pipeline.

Notes:
 - Broadcast CF camera is a fixed mount (per game) that pans/tilts/zooms per pitch
 - Camera pose solve runs in 2 phases each game:
    1. Mount solve (per-game)
        - Every clip with a plausible ball run gets a 7-param solve
          (ball pixels + strikezone box corners, Cy pinned, clock t0 fitted)
        - C = the median of the votes, minimum 2 votes
    2. PTZ solve (per-clip)
        - C frozen, so 4-param solve
        - Use strikezone 4 corners only
          (stikezone detected at peak-glove)
 - A game is ~250 pitches so running it serially (even with numpy) takes too long.
   Instead run all at once through ONE Levenberg-Marquardt (_lm).
 - vmap supplies the batch and jacfwd supplies the Jacobian.

Reads:      data/<year>/pbp_info.csv.gz, data/<year>/raw/strikezone_tracking.csv.gz,
            data/<year>/raw/gloveball_tracks/<game_pk>.csv.gz
Writes:     data/<year>/camera_poses.csv.gz (one row per posed clip: C,
            pan/tilt/roll/f_px, ball_rmse_px + n_fit of the clip's own position vote
            when it had one, and the two accuracy numbers under the SHIPPED pose and
            clock: reproj_px, balltp_err_x_in, balltp_err_z_in and plate_gap_fr,
            which opencommand.py reports)
Run:        python src/solve_camera_pose.py [year=2026] [game_pk ...]
            (game subsets need the year spelled out; default: every game)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
jax.config.update("jax_enable_x64", True)      # float32 loses the geometry this solve needs
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poselib

MIN_VOTES = 2
VOTE_RMSE_PX = 3   # a position solve votes only if the ball reprojects this tightly, and
                   # the retry pass gates against it too.
F_SCALE = 2.0
MAX_ITER = 100

# K=32 measured fastest
K_BUCKET, N_BUCKET = 32, 16


# --- the model, differentiated rather than derived -------------------------------------

def _camera(C, pan, tilt, roll):
    """Camera basis rows (right, up, fwd) for a pose at C looking at the plate."""
    a = jnp.array([0.0, 0.0, 3.0]) - C           # aim at a point just above the plate
    fwd = a / jnp.linalg.norm(a)
    right = jnp.array([fwd[1], -fwd[0], 0.0])    # cross(fwd, z_hat), written out
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(right, fwd)

    def rot(axis, angle):        # Rodrigues; cross(I, a) IS the cross-product matrix of a
        K = jnp.cross(jnp.eye(3), axis)
        return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)

    R = rot(fwd, roll) @ rot(right, tilt) @ rot(up, pan)   # pan, then tilt, then roll
    return jnp.stack([R @ right, R @ up, R @ fwd])


def _reproject(C, pan, tilt, roll, f, X, obs, w):
    """Weighted pixel residual for one problem, raveled. Both phases end here."""
    s = (X - C) @ _camera(C, pan, tilt, roll).T          # rel in camera components
    px = jnp.stack([poselib.W / 2 + f * s[:, 0] / s[:, 2],
                    poselib.H / 2 - f * s[:, 1] / s[:, 2]], axis=-1)
    return ((px - obs) * w[:, None]).ravel()


def _res_position(x, obs, w, tf, tp, bw):
    """Phase-1 residual (2P,) for one candidate ball run.

    x = Cx, Cz, pan, tilt, roll, f, t0, with Cy pinned at CY_FT. t0 is a FITTED
    NUISANCE parameter, solved for and discarded. Box-only position is degenerate in
    Cz, because a coplanar box cannot pin camera height. Rows are the ball run, then the
    padding the bucket added (weight 0), then the 4 box corners at weight 3."""
    t = x[6] + tf
    ball = jnp.stack([tp[0] + tp[1] * t + 0.5 * tp[2] * t ** 2,
                      tp[3] + tp[4] * t + 0.5 * tp[5] * t ** 2,
                      tp[6] + tp[7] * t + 0.5 * tp[8] * t ** 2], axis=-1)
    C = jnp.array([x[0], poselib.CY_FT, x[1]])
    return _reproject(C, x[2], x[3], x[4], x[5], jnp.concatenate([ball, bw]), obs, w)


def _res_ptz(q, C, bw, obs):
    """Phase-2 residual (8,): the 4 box corners only, C frozen. q = pan, tilt, roll, f."""
    return _reproject(C, q[0], q[1], q[2], q[3], bw, obs, jnp.ones(4))


def _soft_l1(r, f_scale):
    """Same thing as scipy's soft_l1 rescaling into an ordinary least squares, or plain
    when f_scale is None. Same lines as its loss_function + scale_for_robust_loss_function,
    so this converges where least_squares(loss="soft_l1") converges."""
    if f_scale is None:
        return 0.5 * (r ** 2).sum(axis=-1), r, jnp.ones_like(r)
    t = 1 + (r / f_scale) ** 2
    rho1, rho2 = t ** -0.5, -0.5 * t ** -1.5 / f_scale ** 2
    js = jnp.maximum(rho1 + 2 * rho2 * r ** 2, jnp.finfo(jnp.float64).eps) ** 0.5
    return f_scale ** 2 * (jnp.sqrt(t) - 1).sum(axis=-1), r * rho1 / js, js


@partial(jax.jit, static_argnums=(0, 1))
def _lm(resfn, f_scale, x, lo, hi, xs, args):
    """Levenberg-Marquardt over K independent problems at once, every shape static.

       A step that leaves the bounds is clipped per component, never shortened as a
       whole. Shortening the whole step lets one component at its bound stall all the
       others, and most clips then never converge.
       Convergence tests the parameter step, not the cost. Cost is quadratic at the
       minimum, so a cost tolerance of eps pins the parameters only to sqrt(eps).
       That scattered votes by ~1e-3 ft.
       A converged problem is frozen (`live`) instead of dropped from the batch,
       because jit forbids a shrinking shape. This costs arithmetic and changes no
       answer."""
    res, jacf = jax.vmap(resfn), jax.vmap(jax.jacfwd(resfn))
    n = x.shape[1]

    def step(_, state):
        x, lam, live = state
        c, rs, js = _soft_l1(res(x, *args), f_scale)
        Jh = jacf(x, *args) * js[..., None] * xs[:, None, :]
        g = jnp.einsum("kmn,km->kn", Jh, rs)
        A = jnp.einsum("kmi,kmj->kij", Jh, Jh)
        damp = jnp.maximum(jnp.diagonal(A, axis1=1, axis2=2), 1e-12)
        p = jnp.linalg.solve(A + lam[:, None, None] * damp[:, None, :] * jnp.eye(n),
                             -g[..., None])[..., 0]
        x_new = jnp.clip(x + p * xs, lo, hi)
        c2 = _soft_l1(res(x_new, *args), f_scale)[0]
        better = (c2 < c) & live
        moved = jnp.abs((x_new - x) / xs).max(axis=1)
        lam = jnp.where(live, jnp.where(better, jnp.maximum(lam / 3.0, 1e-12), lam * 5.0), lam)
        return (jnp.where(better[:, None], x_new, x), lam,
                live & ~((moved < 1e-10) | (lam > 1e12)))

    K = x.shape[0]
    x, _, _ = jax.lax.fori_loop(0, MAX_ITER, step,
                                (x, jnp.full(K, 1e-3), jnp.ones(K, bool)))
    return x


# --- phase 1: position votes -----------------------------------------------------------

def _pack(pairs):
    """A game's (clip, run) pairs as the rectangular arrays the solver takes.

    Same layout: ball rows, bucket padding at weight 0, 4 box corners. 
    Returns the arrays plus n_fit and the real-ball mask."""
    K, N = len(pairs), max(len(seg) for _, seg in pairs)
    Kb, Nb = -(-K // K_BUCKET) * K_BUCKET, -(-N // N_BUCKET) * N_BUCKET
    obs, w = np.zeros((K, Nb + 4, 2)), np.zeros((K, Nb + 4))
    tf, mask = np.zeros((K, Nb)), np.zeros((K, Nb))
    tp, bw = np.empty((K, 9)), np.empty((K, 4, 3))
    x0, lo, hi, xs = (np.empty((K, 7)) for _ in range(4))
    n_fit = np.empty(K, int)

    for i, (c, seg) in enumerate(pairs):
        n = n_fit[i] = len(seg)
        frames = seg["frame_idx"].to_numpy()
        obs[i, :n] = seg[["baseball_center_x", "baseball_center_y"]].to_numpy()
        obs[i, Nb:] = c["box_obs"]
        w[i, :n], w[i, Nb:] = 1.0, 3.0
        tf[i, :n], mask[i, :n] = frames / c["fps"], 1.0
        bw[i] = c["box_world"]
        p, f0 = c["p"], c["f0"]
        tp[i] = [p["x0"], p["vx0"], p["ax"], p["y0"], p["vy0"], p["ay"],
                 p["z0"], p["vz0"], p["az"]]
        rel = poselib.derive_release(c["df"], c["fps"])
        t_lo, t_hi = ((-poselib.RELEASE_HI_S, -poselib.RELEASE_LO_S) if np.isnan(rel)
                      else (-rel - poselib.RELEASE_HALF_S, -rel + poselib.RELEASE_HALF_S))
        t0 = np.clip(c["T_plate"] - frames.max() / c["fps"], t_lo, t_hi)
        x0[i] = [0, 20, 0, 0, 0, f0, t0]
        lo[i] = [-200, 5, -0.15, -0.15, -0.15, f0 / 2, t_lo]
        hi[i] = [200, 120, 0.15, 0.15, 0.15, f0 * 2, t_hi]
        xs[i] = [30, 15, 0.02, 0.02, 0.02, f0 * 0.3, 0.05]

    grow = lambda v: np.concatenate([v, np.repeat(v[:1], Kb - K, axis=0)])   # well-posed filler
    packed = tuple(grow(v) for v in (obs, w, tf, tp, bw, x0, lo, hi, xs, mask))
    return K, packed, n_fit


@jax.jit
def _positions(obs, w, tf, tp, bw, mask, real, x0, lo, hi, xs):
    """Every candidate run in the game solved twice, second-best discarded.

    Pass 1 starts from the generic guess. Pass 2 restarts from the median position
    that pass 1 voted. The reason is that the solve is nonconvex.
    Only a run that failed the reprojection gate takes the restart. The bad basin
    is usually a wrong clock, so the pose it lands on sits near the median and a
    distance test cannot see it. Failing to reproject is what defines it.
    A restart is kept only when its cost is lower, so a retry can only improve a fit.
    Both passes solve every problem, because jit forbids solving a subset."""
    args = (obs, w, tf, tp, bw)
    rmse = lambda x: _ball_rmse(x, args, mask)
    x = _lm(_res_position, F_SCALE, x0, lo, hi, xs, args)
    med = jnp.nanmedian(jnp.where(real[:, None], x[:, :2], jnp.nan), axis=0)
    x2 = _lm(_res_position, F_SCALE, x0.at[:, 0].set(med[0]).at[:, 1].set(med[1]),
             lo, hi, xs, args)
    cost = lambda q: _soft_l1(jax.vmap(_res_position)(q, *args), F_SCALE)[0]
    take = (rmse(x) >= VOTE_RMSE_PX) & (cost(x2) < cost(x))
    x = jnp.where(take[:, None], x2, x)
    return x, rmse(x)


def _ball_rmse(x, args, mask):
    """Ball reprojection RMSE per run, over that run's real points only."""
    r = jax.vmap(_res_position)(x, *args).reshape(x.shape[0], -1, 2)
    ball = r[:, :mask.shape[1]] * mask[..., None]
    return jnp.sqrt((ball ** 2).sum(axis=(1, 2)) / (2 * mask.sum(axis=1)))


# --- phase 2: per-clip PTZ -------------------------------------------------------------

def _ptz(clips, C):
    """pan/tilt/roll/f for every clip at once with C frozen. (N, 4); the caller already
    holds C and composes the full pose."""
    K = len(clips)
    bw = np.stack([c["box_world"] for c in clips])
    obs = np.stack([c["box_obs"] for c in clips])
    f0 = np.array([c["f0"] for c in clips])
    Kb = -(-K // K_BUCKET) * K_BUCKET
    grow = lambda v: np.concatenate([v, np.repeat(v[:1], Kb - K, axis=0)])
    z = np.zeros(Kb)
    Cb = np.repeat(np.asarray(C, float)[None, :], Kb, axis=0)
    bw, obs, f0 = grow(bw), grow(obs), grow(f0)
    x = _lm(_res_ptz, None,
            jnp.stack([z, z, z, f0], axis=1),
            jnp.stack([z - 0.15, z - 0.15, z - 0.15, f0 / 2], axis=1),
            jnp.stack([z + 0.15, z + 0.15, z + 0.15, f0 * 2], axis=1),
            jnp.stack([z + 0.02, z + 0.02, z + 0.02, f0 * 0.3], axis=1),
            (jnp.asarray(Cb), jnp.asarray(bw), jnp.asarray(obs)))
    return np.asarray(x)[:K]


def _shipped_ball_check(c, pose, t0, seg):
    """Accuracy numbers for one clip:
    reproj_px, the ball@plate x and z errors in inches, and plate_gap_fr, the frames
    between the nearest ball sample and the plate crossing. 
    All NaN on an empty run."""
    px = seg[["baseball_center_x", "baseball_center_y"]].to_numpy()
    fi = seg["frame_idx"].to_numpy()
    if not len(fi):
        return np.nan, np.nan, np.nan, np.nan
    t = t0 + fi / c["fps"]
    d = poselib.project(pose[:3], *pose[3:6], pose[6], c["traj"](t)) - px
    reproj = float(np.sqrt((d ** 2).mean()))
    j = int(np.abs(t - c["T_plate"]).argmin())
    truth = c["traj"](t[j])
    bx, bz = poselib.unproject(pose[:3], *pose[3:6], pose[6], px[j], truth[1])
    return (reproj, float(bx * 12 - truth[0] * 12), float(bz * 12 - truth[2] * 12),
            float((t[j] - c["T_plate"]) * c["fps"]))


# --- one game --------------------------------------------------------------------------

def solve_game(args):
    """One game: position votes → game C → per-clip PTZ poses. Returns pose rows."""
    game_pk, pbp, zones, tracks_dir, zone_y_ft = args
    clips = poselib.load_game(game_pk, pbp, zones, tracks_dir, zone_y_ft)
    pairs, owner = [], []
    for i, c in enumerate(clips):
        c["traj"] = poselib.make_traj(c["p"])
        for seg in poselib.candidate_runs(c["df"], c["fps"], c["T_plate"]):
            pairs.append((c, seg))
            owner.append(i)
    if not pairs:
        return []

    k_real, packed, n_fit = _pack(pairs)
    obs, w, tf, tp, bw, x0, lo, hi, xs, mask = (jnp.asarray(v) for v in packed)
    real = jnp.arange(obs.shape[0]) < k_real
    x, rmse = _positions(obs, w, tf, tp, bw, mask, real, x0, lo, hi, xs)
    x, rmse = np.asarray(x)[:k_real], np.asarray(rmse)[:k_real]

    fits = [None] * len(clips)                      # best run per clip, by ball RMSE
    for i, j in enumerate(owner):
        if fits[j] is None or rmse[i] < fits[j][1]:
            fits[j] = (x[i], rmse[i], n_fit[i], pairs[i][1])   # ... and the run that won

    votes = [f[0][:2] for f in fits if f is not None and f[1] < VOTE_RMSE_PX]
    if len(votes) < MIN_VOTES:
        return []
    med = np.median(np.array(votes), axis=0)
    C = np.array([med[0], poselib.CY_FT, med[1]])

    rows = []
    for c, f, q in zip(clips, fits, _ptz(clips, C)):
        pose = np.concatenate([C, q])
        chk = _shipped_ball_check(c, pose, f[0][6], f[3]) if f else (np.nan,) * 4
        rows.append({"video": c["video"], "game_pk": game_pk, "park": c["park"],
                     "Cx": pose[0], "Cy": pose[1], "Cz": pose[2], "pan": pose[3],
                     "tilt": pose[4], "roll": pose[5], "f_px": pose[6],
                     "ball_rmse_px": f[1] if f else np.nan,
                     "reproj_px": chk[0], "balltp_err_x_in": chk[1],
                     "balltp_err_z_in": chk[2], "plate_gap_fr": chk[3],
                     "n_fit": int(f[2]) if f else 0})
    return rows


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    base = poselib.DATA / year
    pbp, zones = poselib.load_season(year)
    games = [int(g) for g in sys.argv[2:]] or sorted(pbp["game_pk"].unique())
    # slice the season tables once (games x full-table scans would dwarf the solve)
    want = set(games)
    zones_by = {game_pk: g for game_pk, g in zones.groupby(zones["video"].str.split("_").str[0].astype(int)) if game_pk in want}
    pbp_by = {game_pk: g for game_pk, g in pbp.groupby("game_pk") if game_pk in want}
    tracks = base / "raw" / "gloveball_tracks"
    zy = poselib.zone_y(year)  # era zone plane: plate front ≤2025, ABS mid-plate ≥2026
    jobs = [(game_pk, pbp_by[game_pk], zones_by[game_pk], tracks, zy) for game_pk in games if game_pk in pbp_by and game_pk in zones_by]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("OC_WORKERS", 10))) as ex:
        results = list(ex.map(solve_game, jobs, chunksize=4))
    rows = [r for game_rows in results for r in game_rows]
    poses = pd.DataFrame(rows)
    poses.to_csv(base / "camera_poses.csv.gz", index=False, lineterminator="\n",
                 compression={"method": "gzip", "compresslevel": 6})
    solved_games = poses["game_pk"].nunique() if len(poses) else 0
    print(f"camera_poses: {len(poses)} clips posed across {solved_games}/{len(games)} games")
