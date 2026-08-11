<div align="center">

<img src="artifacts/banner.png" alt="OpenCommand" width="100%">

[![Version](https://img.shields.io/badge/version-1.0.0-6E7681?style=for-the-badge&labelColor=24292F)](https://github.com/tomdoyo/open-command/releases)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-6E7681?style=for-the-badge&labelColor=24292F)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub](https://img.shields.io/badge/github.com%2Ftomdoyo%2Fopen--command-00852E?style=for-the-badge&labelColor=24292F)](https://github.com/tomdoyo/open-command)

[How it Works](#how-it-works) · [Data](#data) · [Topics](#topics) · [Citation](#license--citation)

</div>

> [!IMPORTANT]  
> `data/` is currently empty due to large file storage issues.  
> This will be fixed soon!

## Let's Measure Command

OpenCommand scores **command** using the pitch location's distance from target.

This repo contains 2025/2026 computer vision object detections and the full inference pipeline for producing **target estimates** and resulting **command scores**.

<p align="center">
  <img src="artifacts/rogers_sinker.gif" alt="Tyler Rogers sinker">
</p>
<p align="center"><sub>Tyler Rogers dots a backdoor sinker (TB @ TOR, 2026/05/13). <b>Yellow box:</b> broadcast strikezone detection. <b>Thin white circle:</b> catcher glove detection. <b>Thick white circle:</b> glove detection projected onto strikezone plane.</sub></p>

## How it Works

### Summary
- Estimate camera position with broadcast strikezone & ball detection
- Estimate camera zoom/pan/tilt with broadcast strikezone & camera position
- Estimate glove location with camera position/zoom/pan/tilt/roll & glove detection
- Estimate target with glove location
- Estimate command with target & actual location

### Install

```
pip install -r requirements.txt          
# or:  conda env create -f environment.yml
```

### Pipeline

Every script in `src/` takes upstream CSVs and writes **one** output.  
And they're standalone: `python src/<script>.py [year=2026] ...`  
(This means you can work on a single stage by regenerating just that stage's file!)  

```
raw/gloveball_tracks  raw/strikezone_tracking
     │       │              │
     │       └──────┬───────┘
     │              ▼                            
     │  1. solve_camera_pose.py ──► camera_poses.csv.gz
     │              │                            
     └──────┬───────┘                            
            ▼                                    
  2. solve_glove_locations.py ──► glove_locations/ 
            │                                    
            ▼                                    
  3. target_inference.py ──► targets.csv.gz
            │                                    
            ▼                                   
  4. opencommand.py  ──► command_scores.csv
                        (+ artifacts/validations_<year>.txt)

```

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `solve_camera_pose.py` | gloveball_tracks, strikezone_tracking, pbp_info | `camera_poses.csv.gz` |
| 2 | `solve_glove_locations.py` | gloveball_tracks, camera_poses | `glove_locations/<game_pk>.csv.gz` |
| 3 | `target_inference.py` | glove_locations, pbp_info | `targets.csv.gz` |
| 4 | `opencommand.py` | targets, pbp_info, camera_poses, fg_pitching | `command_scores.csv` + `artifacts/validations_<year>.txt` |
| — | `poselib.py` | (library, not a stage) | imported by steps 1 and 2 |

> Step 1 is particularly heavy (hours); other steps take minutes.


### In detail

#### **1. Solving camera pose** (every pitch)
- The CF camera is a fixed mount per game that pans/tilts/zooms per pitch.
- Estimate *where* the camera is:
  - Statcast's 9-parameter equation `(xyz_0, xyz_velo, xyz_acc)` gives us ball position in time (through pitch trajectory).
  - Broadcasts draw strikezone as `(17in width, sz_top/sz_bot)` at the front of the plate (middle for 2026).
  - These give us **12+** datapoints per pitch (8+ ball pixels, 4 box corners) to fit<sup>1</sup> **7** parameters: `(Cx, Cy=400`<sup>2</sup>`, Cz, pan, tilt, roll, f, t0)`.
  - Just keep the game median `Cx`/`Cz`<sup>3</sup>.
- Fit (pan, tilt, roll, f) separately with fixed `(Cx, Cy, Cz)`.
  - Use the drawn strikezone at a snapshot pre-pitch<sup>4</sup>.
  - Don't use ball positions because camera often moves mid-ball flight.

<sub><sup>1</sup> Levenberg-Marquardt on the pixel reprojection error, with a soft_l1 loss.<br>
<sup>2</sup> Camera depth (`Cy`) is degenerate against focal length (`f`): moving camera back and zooming in produce nearly the same pixels. Not a big deal down the line so `Cy` is fixed at 400.<br>
<sup>3</sup> Others are nuisance parameters.<br>
<sup>4</sup> Snapshot is taken when glove is at the highest point in the [release-2.0s, release-0.3s] window.</sub>

#### **2. Solving glove location** (every frame)
- `glove_px/pz` is a 2D projection of glove onto the camera.
- Use camera pose `(Cx, Cy, Cz, pan, tilt, roll, f)` to unproject detected `glove_px/pz` into *global* `glove_xyz`<sup>1</sup>.

<sub><sup>1</sup> Like `Cy`, glove depth (`glove_y`) is really hard to estimate. So we assume `glove_y` to be -1.75ft (median catch depth).<br>

#### **3. Inferring target with glove locations**
- Take the highest `glove_xz` in the [release-2.0s, release-0.3s] window<sup>1</sup>. This is the **naive** target.
- Many pitchers like to "start the pitch from the glove and let the ball break away from it". To account for this, add `pitcher × pitch type × season` offset. This is the **inferred target**.
  - This assumes every pitcher is **perfectly calibrated** on a pitch type level.
- Use plausibility filter<sup>2</sup> to filter out extreme targets.

<sub><sup>1</sup> Median in the ±0.05s around highest `glove_xz` frame; dispersed neighborhoods (jitter, detection flips) are skipped to the next stable candidate.<br>
<sup>2</sup> (`|x|` over 20in, `z` outside the pitch-type floor/cap)</sub>

#### **4. Scoring command**
- `miss` = distance from the actual location to target.
- For leaderboards, **median** miss is used, after plausible target filter.

## Data

Each season lives under `data/<year>/`. 

**Keys:** `(game_pk, play_id)` identify a pitch. Five of the seven files below join on `video` instead, which is `"<game_pk>_<play_id>.mp4"`.

Raw detections (in `data/<year>/raw/`) are produced using YOLO11 glove/ball/strikezone detector models, with postprocessing<sup>1</sup> based on eight measured quality axes.

| File (per season) | One row per | Contents |
|---|---|---|
| `pbp_info.csv.gz` | pitch | Statcast 9-parameter trajectory, `sz_top`/`sz_bot`, plate location, pitcher, pitch type, and the game_date/type/venue |
| `raw/gloveball_tracks/<game_pk>.csv.gz` | frame | glove + ball detections (pixels on screen) |
| `raw/strikezone_tracking.csv.gz` | clip | broadcast strikezone detections (pixels on screen) |
| `camera_poses.csv.gz` | clip | 16 columns: the solved pose (position, focal length<sup>2</sup>/pan/tilt/roll), the clip's own position-vote diagnostics, and the reprojection accuracy of the shipped pose |
| `glove_locations/<game_pk>.csv.gz` | detection | solved glove location (real-world), clip's fps<sup>3</sup>, `release_s` (the sub-frame offset of release from frame 0). A posed clip with no glove keeps one all-NaN row |
| `targets.csv.gz` | clip | two targets per clip — `naive_*` (the glove track alone, with no pitch outcome in it) and `inferred_*` (naive + the pitcher × pitch-type season offset) — quality columns |
| `command_scores.csv` | pitcher, pitch type | n, naive and inferred median miss |

<sub><sup>1</sup> Clip quality is graded on eight axes — `zone_conf`, `n_window`, `lead_s`, `peak_conf`, `h_ratio`, `glove_px_width`, `glove_width_steadiness`, `glove_box_steadiness` — of which only two are detector confidences; the rest grade the scene and the glove track. Every axis sets a hard cutoff, and six of them also carry a soft correction that moves a lower-quality clip's glove pixels toward what a high-quality clip would have shown. The correction touches 96.19% of 2025 glove rows (64.4M matched detections) and 93.78% of 2026 rows (33.3M), moving each by a median of 3.7 px.<br>
<sup>2</sup> Means zoom<br>
<sup>3</sup> `frame_idx` is relative to release: frame 0 is the release frame, so the pre-pitch glove window sits at negative frames. `fps` converts frame counts to seconds.</sub>

### Coverage

OpenCommand tracks nearly all the pitches that it *can*, with most clips lost being due to *no strikezone detected*<sup>1</sup> and *late center field camera cut*<sup>2</sup>.

**For 2025:** 90.02 / 92.70% possible

| Funnel loss | Clips Lost (%) | Remaining | Coverage |
|---|---:|---:|---:|
| All pitches | — | 724,005 | 100.00% |
| Clip never published | 1,244 (-0.17%) | 722,761 | 99.83% |
| No strikezone detected | 30,984 (-4.28%) | 691,777 | 95.55% |
| No ball release detected | 11,710 (-1.62%) | 680,067 | 93.93% |
| Late center field camera cut | 20,653 (-2.85%) | 659,414 | 91.08% |
| Low detection quality | 5,615 (-0.78%) | 653,799 | 90.30% |
| Implausible target | 2,069 (-0.29%) | **651,730** | **90.02%** |

<sub><sup>1</sup> Sometimes broadcasts don't draw a strikezone box on the screen<br>
<sup>2</sup> Sometimes camera cuts to CF-cam (i.e. pitcher-batter view) too late</sub>

## Topics

### Target maps

A nice feature of this is that you can tell where the pitcher was *trying* to throw, which is really hard just looking at the final location.

<p align="center">
  <img src="artifacts/degrom_target_map_2025_example.png" alt="Jacob deGrom inferred targets and actual four-seam locations, 2025" width="720">
</p>

### Command distribution

Couple notes:
- Inferring targets (pitcher × pitch-type offset) shaves off about **1 inch** off of naive miss. 
- MLB pitchers miss by 9-11 inches
- Pitchers command fastballs ~1 inch better
- The miss distribution has some right skew

**2025, naive median miss** — min. 50 pitches

| Pitch type | Pitchers | Min | p10 | p25 | Median | p75 | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All pitches | 716 | 8.41 | 9.87 | 10.43 | 11.07 | 11.90 | 12.76 | 16.02 |
| Four-seam (FF) | 581 | 7.35 | 8.39 | 9.24 | 10.09 | 10.91 | 11.85 | 14.24 |
| Sinker (SI) | 379 | 6.20 | 8.52 | 9.20 | 9.93 | 10.97 | 12.13 | 18.36 |
| Cutter (FC) | 215 | 6.85 | 8.44 | 9.05 | 9.84 | 10.76 | 11.74 | 15.91 |
| Slider (SL) | 393 | 7.00 | 9.69 | 10.56 | 11.58 | 13.13 | 14.33 | 20.58 |
| Sweeper (ST) | 230 | 8.72 | 10.04 | 10.95 | 11.96 | 13.40 | 14.55 | 19.64 |
| Curveball (CU+KC) | 248 | 8.35 | 10.80 | 11.73 | 13.11 | 14.63 | 16.23 | 21.33 |
| Changeup (CH) | 301 | 8.27 | 10.51 | 11.61 | 12.79 | 14.45 | 16.48 | 29.00 |
| Splitter (FS) | 95 | 7.58 | 10.66 | 12.14 | 13.71 | 15.85 | 17.30 | 22.29 |

**2025, inferred median miss** — naive + pitcher × pitch-type offset

| Pitch type | Pitchers | Min | p10 | p25 | Median | p75 | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All pitches | 716 | 7.38 | 8.93 | 9.45 | 9.99 | 10.57 | 11.14 | 14.97 |
| Four-seam (FF) | 581 | 6.78 | 8.19 | 8.72 | 9.40 | 10.15 | 10.89 | 12.78 |
| Sinker (SI) | 379 | 6.11 | 7.98 | 8.52 | 9.12 | 9.90 | 10.73 | 13.42 |
| Cutter (FC) | 215 | 6.88 | 8.09 | 8.63 | 9.37 | 10.15 | 10.79 | 13.30 |
| Slider (SL) | 393 | 7.18 | 8.98 | 9.71 | 10.40 | 11.28 | 12.07 | 14.60 |
| Sweeper (ST) | 230 | 8.12 | 9.40 | 9.91 | 10.67 | 11.53 | 12.44 | 15.02 |
| Curveball (CU+KC) | 248 | 8.38 | 9.86 | 10.63 | 11.46 | 12.38 | 13.38 | 17.97 |
| Changeup (CH) | 301 | 7.25 | 9.09 | 9.82 | 10.57 | 11.34 | 12.05 | 14.88 |
| Splitter (FS) | 95 | 7.44 | 9.32 | 9.92 | 11.00 | 12.08 | 13.05 | 15.58 |

### Some correlations

**2025** — 339 pitchers, min. 50 innings

| | Naive | Inferred |
|---|---:|---:|
| BB% | +0.456 [+0.367, +0.543] | +0.548 [+0.466, +0.628] |
| Stuff+ | +0.194 [+0.092, +0.294] | +0.250 [+0.147, +0.345] |
| xERA | -0.068 [-0.177, +0.039] | -0.069 [-0.173, +0.041] |
| xERA \| Stuff+ | +0.085 [-0.020, +0.190] | +0.137 [+0.029, +0.243] |

In particular, we can see a strong correlation between command and walk rates.
<p align="center">
  <img src="artifacts/bb_vs_command_2025.png" alt="2025 inferred median miss against walk rate, 339 pitchers" width="380">
</p>

#### **Why (~~mean~~) median miss?** 
- Median is more robust to extreme values (e.g. due to bad inferred targets/glove detections/etc.)
- Median (50th percentile) better answers "what's pitcher x's *typical* miss?". A pitcher can't miss by less than 0 in, but can spike one and get a 100 inch miss, which takes 100 pitches with 1 inch above mean miss to make up for it. So coloquially, median makes more sense as an "average".

#### **How accurate is OpenCommand at measuring command?**
- This is really hard to tell because there's no *ground truth* (unless we ask "hey where did you aim?" every pitch).
- True median miss for **fastballs** is probably [7 to 10 inches](https://x.com/tomdoyo/status/2082066794404294671?s=20).
- Glove detections get post-hoc adjustments based on detection accuracies. This adjustment makes miss distances unbiased, but doesn't remove the pitch-level variance. 
- Inferred miss assumes every pitcher perfectly calibrates his pitches, but most pitchers are probably an inch or two off. At the same time, most pitchers fine tune their targets (beyond the catcher's glove) every pitch, depending on the situation. Perhaps these two cancel off on a season-level. 
- So, on a season-level, OpenCommand has a good chance of being accurate within <1 inch. On a pitch-level, certainly not. 

#### **Which pitchers are represented well/worse?**
- Some pitchers see-glove-hit-glove (especially ones that throw down the middle). These pitchers have the best OpenCommand representations.
- Some pitchers do the **opposite** of micro-adjustment. They make their catchers adjust the glove depending on their miss patterns that day so that their end locations stay the same.
- Some pitchers **don't look at the glove at all** (e.g. Misiorowski). These pitchers are heavily misrepresented. 

## License & citation

Everything in this repository, data and code, is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): use it, build on it, publish with it, with **attribution** (cite OpenCommand; see [CITATION.cff](CITATION.cff)) and **not commercially**. 

Data derived from MLB broadcast video and Statcast public feeds. MLB and Statcast are trademarks of MLB Advanced Media, L.P.; this project is not affiliated with MLB.
