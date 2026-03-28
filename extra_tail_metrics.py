"""
Tail and shape metrics from normalized centerlines.

Outputs (wide DataFrames, one row per animal, Time_*** columns):
  Shape / kinematics
    tailbeat_cycles_count_wide      — zero-crossing tailbeat count (scalar)
    tailbeat_cycles_rate_hz_wide    — tailbeat rate Hz, zero-crossing (scalar)
    tailbeat_rate_thresh_hz_wide    — tailbeat rate Hz, loose Δ-frame threshold (scalar)
    tail_tip_angle_rad_wide         — tail-tip segment angle vs +X (timeseries)
    tail_mean_angle_rad_wide        — mean angle over last N segments (timeseries)
    curvature_abs_tail_wide         — mean |κ| over tail interior (timeseries)
    angles_centered_for_pca_wide    — frame-centered segment angles, one row per segment

  Posture / frequency (active-masked)
    tailtip_freq_active_hz_wide     — dominant tail-tip freq, active-only blocks (scalar)
    tailtip_freq_all_hz_wide        — dominant tail-tip freq, longest finite run (scalar)
    y_energy_rms_p95_wide           — RMS of lateral y (p95 across time) (scalar)
    symmetry_y_bias_wide            — |mean y| whole body (scalar)
    symmetry_tail_y_bias_wide       — |mean y| tail region (scalar)
    wave_index_wide                 — spatial wave count from Hilbert phase (scalar)

  Curvature
    avg_curvature_extrap_wide       — mean |κ| whole body, hard cap 0.2 (timeseries)
    avg_tail_curvature_extrap_wide  — mean |κ| tail region, robust clip (timeseries)

"""

import os
import numpy as np
import pandas as pd

try:
    from scipy.signal import hilbert, savgol_filter
    _SCIPY = True
except ImportError:
    hilbert = savgol_filter = None
    _SCIPY = False


# =======================
# Config
# =======================

TAIL_LAST_SEG   = 3       # last N segments define the tail region
CROSS_EPS       = 0.01    # deadband (normalized y) for zero-crossing detection
MED_WIN         = 3       # median filter window along time (frames)
SMOOTH_S_WIN    = 7       # SavGol / moving-avg window along body axis s
TAIL_FRAC       = 0.35    # fraction of body length counted as tail (curvature)
CURV_CLIP_Q     = 99.5    # robust percentile cap for curvature values
FFT_MIN_SEC     = 1.0     # minimum contiguous seconds for FFT frequency estimate
VEL_ACTIVE_Q    = 0.95    # per-animal velocity quantile threshold for active mask
TB_THRESH_MODE  = "percentile"  # "percentile" | "mad" | "value"
TB_THRESH_VALUE = 50.0          # parameter for the chosen threshold mode
TB_MIN_PAIRS    = 10            # minimum qualifying frame pairs to report Hz

KCOLS = ["Experiment_ID", "Stage", "Condition", "Individual"]

FS   = float(globals().get("FS_RS", 6.0))
DT   = 1.0 / FS


def _tcols(T: int) -> list[str]:
    return [f"Time_{i:03d}" for i in range(T)]


def _meta(key: tuple) -> dict:
    return {"Experiment_ID": key[0], "Stage": key[1],
            "Condition": key[2], "Individual": key[3]}


def _row(key, metric, values) -> pd.DataFrame:
    """Pack one wide row: meta + Metric name + Time_*** values."""
    T = len(values)
    return pd.DataFrame([_meta(key) | {"Metric": metric} | dict(zip(_tcols(T), values))])


def _concat(rows: list) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =======================
# Signal Helpers
# =======================

def _median_smooth(x: np.ndarray, k: int = MED_WIN) -> np.ndarray:
    """NaN-aware median filter along a 1-D array."""
    if k <= 1 or x.size < k:
        return x
    from numpy.lib.stride_tricks import sliding_window_view
    out = x.copy()
    med = np.nanmedian(sliding_window_view(x, k), axis=1)
    out[k // 2: k // 2 + med.size] = med
    return out


def _neg_to_pos_crossings(y: np.ndarray, eps: float = CROSS_EPS) -> int:
    
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    if not finite.any():
        return 0
    # locate contiguous finite runs
    pad = np.r_[False, finite, False]
    edges = np.flatnonzero(np.diff(pad.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    count = 0
    for s, e in zip(starts, ends):
        seg = y[s:e]
        sgn = np.where(seg > eps, 1, np.where(seg < -eps, -1, 0))
        prev = 0
        for v in sgn:
            if v == 0:
                continue
            if prev == -1 and v == 1:
                count += 1
            prev = v
    return count


def _pair_threshold_mask(y: np.ndarray,
                         mode: str = TB_THRESH_MODE,
                         value: float = TB_THRESH_VALUE) -> np.ndarray:
    
    dy      = np.abs(np.diff(np.asarray(y, float)))
    finite  = np.isfinite(y)
    valid   = finite[:-1] & finite[1:]
    dy_use  = dy[valid]
    if dy_use.size == 0:
        return np.zeros_like(dy, dtype=bool)

    if mode == "percentile":
        thr = float(np.nanpercentile(dy_use, float(value)))
    elif mode == "mad":
        med = float(np.nanmedian(dy_use))
        mad = float(np.nanmedian(np.abs(dy_use - med))) or float(np.nanstd(dy_use)) or 1e-6
        thr = med + float(value) * mad
    else:  # "value"
        thr = float(value)

    return valid & (dy >= thr)


def _negpos_on_masked_pairs(y: np.ndarray, mask: np.ndarray,
                             eps: float = CROSS_EPS) -> tuple[int, int]:
    
    y   = np.asarray(y, float)
    sgn = np.where(y > eps, 1, np.where(y < -eps, -1, 0))
    cross = (sgn[:-1] == -1) & (sgn[1:] == 1) & mask
    return int(cross.sum()), int(mask.sum())


# =======================
# Geometry Helpers
# =======================

def _segment_angles(CLn: np.ndarray) -> np.ndarray:
    
    seg = CLn[:, 1:, :] - CLn[:, :-1, :]
    return np.arctan2(seg[..., 1], seg[..., 0])


def _curvature_tail(CLn: np.ndarray, tail_last: int = TAIL_LAST_SEG) -> np.ndarray:
    
    T, K, _ = CLn.shape
    if K < 3:
        return np.full(T, np.nan)
    h = 1.0 / (K - 1)
    X, Y   = CLn[..., 0], CLn[..., 1]
    Xs     = (X[:, 2:] - X[:, :-2]) / (2.0 * h)
    Ys     = (Y[:, 2:] - Y[:, :-2]) / (2.0 * h)
    Xss    = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / (h * h)
    Yss    = (Y[:, 2:] - 2 * Y[:, 1:-1] + Y[:, :-2]) / (h * h)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (Xs * Yss - Ys * Xss) / (Xs * Xs + Ys * Ys) ** 1.5   # (T, K-2)
    n_tail = max(1, min(tail_last, kappa.shape[1]))
    return np.nanmean(np.abs(kappa[:, -n_tail:]), axis=1)


def _curvature_body(CLn: np.ndarray,
                    tail_frac: float = TAIL_FRAC,
                    clip_q: float = CURV_CLIP_Q,
                    smooth_win: int = SMOOTH_S_WIN) -> np.ndarray:
    
    T, K, _ = CLn.shape
    if K < 3:
        return np.full(T, np.nan)

    XY = _smooth_along_s(CLn, win=smooth_win)
    h  = 1.0 / (K - 1)
    X, Y = XY[..., 0], XY[..., 1]
    Xs   = (X[:, 2:] - X[:, :-2]) / (2.0 * h)
    Ys   = (Y[:, 2:] - Y[:, :-2]) / (2.0 * h)
    Xss  = (X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]) / (h * h)
    Yss  = (Y[:, 2:] - 2.0 * Y[:, 1:-1] + Y[:, :-2]) / (h * h)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (Xs * Yss - Ys * Xss) / (Xs * Xs + Ys * Ys) ** 1.5   # (T, K-2)

    D     = kappa.shape[1]
    start = max(0, int(np.floor((1.0 - tail_frac) * D)))
    tail_k = np.abs(kappa[:, start:])

    finite = tail_k[np.isfinite(tail_k)]
    if finite.size:
        cap    = float(np.percentile(finite, clip_q))
        tail_k = np.clip(tail_k, 0, cap)

    return np.nanmean(tail_k, axis=1)


def _smooth_along_s(XY: np.ndarray, win: int = SMOOTH_S_WIN) -> np.ndarray:
   
    if win is None or win <= 1:
        return XY
    T, K, _ = XY.shape
    w   = min(K - (1 - K % 2), int(win))   # ensure odd and <= K
    if w < 3:
        return XY
    out = XY.copy()

    for t in range(T):
        for d in (0, 1):
            row = out[t, :, d]
            fin = np.isfinite(row)
            if _SCIPY and fin.sum() >= w:
                s = np.arange(K)
                filled = row.copy()
                if not fin.all():
                    filled[~fin] = np.interp(s[~fin], s[fin], row[fin])
                try:
                    out[t, :, d] = savgol_filter(
                        filled, window_length=w,
                        polyorder=min(2, w - 1), axis=0, mode="interp")
                    continue
                except Exception:
                    pass
            # NaN-aware moving-average fallback
            r0   = np.nan_to_num(row, nan=0.0)
            m    = fin.astype(float)
            k    = np.ones(w)
            num  = np.convolve(r0, k, "same")
            den  = np.convolve(m,  k, "same")
            out[t, :, d] = np.where(den > 0, num / den, np.nan)

    return out


# =======================
# Frequency / Posture Helpers
# =======================

def _dominant_freq_fft(y: np.ndarray, fs: float) -> float:
    
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 8 or np.nanstd(y) == 0:
        return np.nan
    y  = y - y.mean()
    P  = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    f  = np.fft.rfftfreq(y.size, d=1.0 / fs)
    P[0] = 0.0
    k  = int(np.argmax(P))
    if 1 <= k < P.size - 1:
        a, b, c = P[k - 1], P[k], P[k + 1]
        denom = a - 2 * b + c
        if denom != 0 and np.isfinite(denom):
            return float((k + 0.5 * (a - c) / denom) * (fs / y.size))
    return float(f[k])


def _wave_index(CLn: np.ndarray) -> float:
    """
    Mean number of spatial waves along the body per frame.
    Fits a line to the unwrapped Hilbert phase of frame-centered segment angles;
    wave_index = |slope| / 2π.  Falls back to unwrapped angle directly if no SciPy.
    """
    ANG  = _segment_angles(CLn)                              # (T, K-1)
    ANG -= np.nanmean(ANG, axis=1, keepdims=True)           # center per frame
    T, D = ANG.shape
    if D < 3:
        return np.nan
    s    = np.linspace(0.0, 1.0, D)
    vals = []
    for t in range(T):
        row = ANG[t]
        if not np.all(np.isfinite(row)):
            continue
        phi = np.unwrap(np.angle(hilbert(row))) if _SCIPY else np.unwrap(row)
        a, _ = np.polyfit(s, phi, 1)
        wi = abs(a) / (2.0 * np.pi)
        if np.isfinite(wi):
            vals.append(wi)
    return float(np.nanmedian(vals)) if vals else np.nan


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) pairs for True-runs; end is exclusive."""
    m   = np.asarray(mask, bool)
    pad = np.r_[False, m, False]
    e   = np.flatnonzero(np.diff(pad.astype(int)))
    return list(zip(e[::2], e[1::2]))


def _tail_y(CLn: np.ndarray, tail_last: int = TAIL_LAST_SEG) -> np.ndarray:
    """Mean lateral y of the last tail_last+1 centerline points. Returns (T,)."""
    n = max(2, tail_last + 1)
    return np.nanmean(CLn[:, -n:, 1], axis=1)


# =======================
# Shape Metrics
# =======================

def build_shape_metrics_wide(ex_store: dict) -> dict[str, pd.DataFrame]:
    
    rows = {k: [] for k in ["cnt", "hz", "tip", "mean", "curv", "ang"]}

    for key, entry in ex_store.items():
        CLn = entry.get("centerline_K_norm")
        if CLn is None or CLn.ndim != 3 or CLn.shape[-1] != 2:
            continue
        T, K, _ = CLn.shape

        # tail lateral signal → tailbeat count and rate
        y = _median_smooth(_tail_y(CLn), k=MED_WIN)
        n_cross = _neg_to_pos_crossings(y, eps=CROSS_EPS)
        n_fin   = int(np.isfinite(y).sum())
        dur     = (n_fin - 1) * DT if n_fin >= 2 else np.nan
        rate_hz = (n_cross / dur) if (dur and np.isfinite(dur) and dur > 0) else np.nan

        rows["cnt"].append(_row(key, "TailbeatCycles_count",   [n_cross] * T))
        rows["hz"].append( _row(key, "TailbeatCycles_rate_Hz", [rate_hz] * T))

        # segment angles
        ANG      = _segment_angles(CLn)                               # (T, K-1)
        tip_ang  = ANG[:, -1]                                        # tip segment
        mean_ang = np.nanmean(ANG[:, -TAIL_LAST_SEG:], axis=1)

        rows["tip"].append( _row(key, "TailTipAngle_rad",  tip_ang.tolist()))
        rows["mean"].append(_row(key, "TailMeanAngle_rad", mean_ang.tolist()))

        # tail curvature
        curv = _curvature_tail(CLn, tail_last=TAIL_LAST_SEG)
        rows["curv"].append(_row(key, "CurvatureAbsTail", curv.tolist()))

        # frame-centered angles for PCA (one row per segment)
        ANGc = ANG - np.nanmean(ANG, axis=1, keepdims=True)
        for seg_i in range(ANGc.shape[1]):
            r = _meta(key) | {"Metric": "ShapeAnglesCentered", "Segment": seg_i} | \
                dict(zip(_tcols(T), ANGc[:, seg_i].tolist()))
            rows["ang"].append(pd.DataFrame([r]))

    return {
        "tailbeat_cycles_count":    _concat(rows["cnt"]),
        "tailbeat_cycles_rate_hz":  _concat(rows["hz"]),
        "tail_tip_angle_rad":       _concat(rows["tip"]),
        "tail_mean_angle_rad":      _concat(rows["mean"]),
        "curvature_abs_tail":       _concat(rows["curv"]),
        "angles_centered_for_pca":  _concat(rows["ang"]),
    }


# =======================
# Tailbeat Rate — Loose Threshold
# =======================

def build_tailbeat_rate_threshold_wide(ex_store: dict) -> dict[str, pd.DataFrame]:
    
    rows_hz, rows_cnt = [], []

    for key, entry in ex_store.items():
        CLn = entry.get("centerline_K_norm")
        if CLn is None or CLn.ndim != 3 or CLn.shape[-1] != 2:
            continue
        T = CLn.shape[0]

        y    = _median_smooth(_tail_y(CLn), k=MED_WIN)
        mask = _pair_threshold_mask(y, mode=TB_THRESH_MODE, value=TB_THRESH_VALUE)
        n_cross, n_pairs = _negpos_on_masked_pairs(y, mask, eps=CROSS_EPS)

        active_s = n_pairs * DT
        rate_hz  = (n_cross / active_s
                    if active_s > 0 and n_pairs >= TB_MIN_PAIRS else np.nan)

        rows_cnt.append(_row(key, "TailbeatCycles_thresh_count", [n_cross]  * T))
        rows_hz.append( _row(key, "TailbeatRate_thresh_Hz",      [rate_hz]  * T))

    return {
        "tailbeat_rate_thresh_hz":      _concat(rows_hz),
        "tailbeat_cycles_thresh_count": _concat(rows_cnt),
    }


# =======================
# Posture & Frequency (active-masked)
# =======================

def _build_active_mask(velocity_wide: pd.DataFrame) -> pd.DataFrame:
    
    tcols_v = [c for c in velocity_wide.columns
               if isinstance(c, str) and c.startswith("Time_")]
    rows = []
    for _, row in velocity_wide[KCOLS + tcols_v].iterrows():
        v   = pd.to_numeric(row[tcols_v], errors="coerce").to_numpy(float)
        fin = v[np.isfinite(v)]
        thr = np.nanpercentile(fin, VEL_ACTIVE_Q * 100.0) if fin.size else np.nan
        act = (v >= thr).astype(int) if np.isfinite(thr) else np.zeros_like(v, int)
        r   = {k: row[k] for k in KCOLS}
        r  |= {"Metric": "ActiveByVel_q95"} | dict(zip(tcols_v, act.tolist()))
        rows.append(pd.DataFrame([r]))
    return _concat(rows)


def build_posture_frequency_wide(ex_store: dict,
                                 velocity_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    
    active_mask_wide = _build_active_mask(velocity_wide)
    tcols_v = [c for c in velocity_wide.columns
               if isinstance(c, str) and c.startswith("Time_")]
    min_block = int(np.ceil(FFT_MIN_SEC * FS))

    rows = {k: [] for k in ["f_act", "f_all", "yE", "sym", "sym_t", "wave"]}

    for key, entry in ex_store.items():
        CLn = entry.get("centerline_K_norm")
        if CLn is None or CLn.ndim != 3 or CLn.shape[-1] != 2:
            continue
        T = CLn.shape[0]

        # align active mask to centerline timeline
        mrow = active_mask_wide[
            (active_mask_wide["Experiment_ID"] == key[0]) &
            (active_mask_wide["Stage"]         == key[1]) &
            (active_mask_wide["Condition"].astype(str) == str(key[2])) &
            (active_mask_wide["Individual"]    == key[3])
        ]
        if mrow.empty:
            active = np.zeros(T, dtype=int)
        else:
            m = pd.to_numeric(mrow.iloc[0][tcols_v], errors="coerce") \
                  .fillna(0).astype(int).to_numpy()
            active = np.zeros(T, dtype=int)
            active[:min(T, m.size)] = m[:min(T, m.size)]

        y = _tail_y(CLn)

        # tail-tip frequency — active blocks only
        freqs = [_dominant_freq_fft(y[s:e], FS)
                 for s, e in _contiguous_runs(active > 0)
                 if (e - s) >= min_block]
        f_act = float(np.nanmedian([f for f in freqs if np.isfinite(f)])) \
                if freqs else np.nan

        # tail-tip frequency — longest finite run
        runs  = _contiguous_runs(np.isfinite(y))
        f_all = np.nan
        if runs:
            s0, e0 = max(runs, key=lambda ab: ab[1] - ab[0])
            if (e0 - s0) >= min_block:
                f_all = _dominant_freq_fft(y[s0:e0], FS)

        # lateral y-energy: per-frame RMS over all body points, then p95 across time
        rms_t = np.sqrt(np.nanmean(CLn[..., 1] ** 2, axis=1))
        rms_t = rms_t[np.isfinite(rms_t)]
        y_E   = float(np.nanpercentile(rms_t, 95)) if rms_t.size else np.nan

        # y-symmetry: |mean y| (0 = balanced oscillation)
        sym_body = float(np.abs(np.nanmean(CLn[..., 1])))
        sym_tail = float(np.abs(np.nanmean(y)))

        # spatial wave index from Hilbert phase of centered angles
        wi = _wave_index(CLn)

        for k_out, metric, val in [
            ("f_act",  "TailTipFreq_active_Hz",  f_act),
            ("f_all",  "TailTipFreq_all_Hz",      f_all),
            ("yE",     "y_energy_rms_p95",        y_E),
            ("sym",    "symmetry_y_bias",          sym_body),
            ("sym_t",  "symmetry_tail_y_bias",     sym_tail),
            ("wave",   "wave_index",               wi),
        ]:
            rows[k_out].append(_row(key, metric, [val] * T))

    return {
        "tailtip_freq_active_hz":    _concat(rows["f_act"]),
        "tailtip_freq_all_hz":       _concat(rows["f_all"]),
        "y_energy_rms_p95":          _concat(rows["yE"]),
        "symmetry_y_bias":           _concat(rows["sym"]),
        "symmetry_tail_y_bias":      _concat(rows["sym_t"]),
        "wave_index":                _concat(rows["wave"]),
    }


# =======================
# Curvature — Whole Body & Tail-Focused
# =======================

def build_curvature_wide(ex_store: dict) -> dict[str, pd.DataFrame]:
    
    rows_body, rows_tail = [], []

    for key, entry in ex_store.items():
        CLn = entry.get("centerline_K_norm")
        if CLn is None or CLn.ndim != 3 or CLn.shape[-1] != 2:
            continue
        T, K, _ = CLn.shape
        if K < 3:
            continue

        # whole-body curvature, simple hard cap
        h = 1.0 / (K - 1)
        X, Y = CLn[..., 0], CLn[..., 1]
        Xs   = (X[:, 2:] - X[:, :-2]) / (2.0 * h)
        Ys   = (Y[:, 2:] - Y[:, :-2]) / (2.0 * h)
        Xss  = (X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]) / (h * h)
        Yss  = (Y[:, 2:] - 2.0 * Y[:, 1:-1] + Y[:, :-2]) / (h * h)
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa = (Xs * Yss - Ys * Xss) / (Xs * Xs + Ys * Ys) ** 1.5
        kappa_capped = np.where(np.abs(kappa) <= 0.2, np.abs(kappa), np.nan)
        curv_body = np.nanmean(kappa_capped, axis=1)

        # tail-focused curvature with smoothing and robust clip
        curv_tail = _curvature_body(CLn, tail_frac=TAIL_FRAC,
                                    clip_q=CURV_CLIP_Q, smooth_win=SMOOTH_S_WIN)

        rows_body.append(_row(key, "Avg_Curvature_extrap",      curv_body.tolist()))
        rows_tail.append(_row(key, "Avg_Curvature_tail_extrap", curv_tail.tolist()))

    return {
        "avg_curvature_extrap":      _concat(rows_body),
        "avg_tail_curvature_extrap": _concat(rows_tail),
    }


# =======================
# Run
# =======================

os.makedirs(PLOT_OUTDIR, exist_ok=True)

shape_wides   = build_shape_metrics_wide(ex_store)
thresh_wides  = build_tailbeat_rate_threshold_wide(ex_store)
posture_wides = build_posture_frequency_wide(ex_store, velocity_wide)
curv_wides    = build_curvature_wide(ex_store)

# promote to workspace variables
tailbeat_cycles_count_wide      = shape_wides["tailbeat_cycles_count"]
tailbeat_cycles_rate_hz_wide    = shape_wides["tailbeat_cycles_rate_hz"]
tail_tip_angle_rad_wide         = shape_wides["tail_tip_angle_rad"]
tail_mean_angle_rad_wide        = shape_wides["tail_mean_angle_rad"]
curvature_abs_tail_wide         = shape_wides["curvature_abs_tail"]
angles_centered_for_pca_wide    = shape_wides["angles_centered_for_pca"]

tailbeat_rate_thresh_hz_wide    = thresh_wides["tailbeat_rate_thresh_hz"]
tailbeat_cycles_thresh_count_wide = thresh_wides["tailbeat_cycles_thresh_count"]

tailtip_freq_active_hz_wide     = posture_wides["tailtip_freq_active_hz"]
tailtip_freq_all_hz_wide        = posture_wides["tailtip_freq_all_hz"]
y_energy_rms_p95_wide           = posture_wides["y_energy_rms_p95"]
symmetry_y_bias_wide            = posture_wides["symmetry_y_bias"]
symmetry_tail_y_bias_wide       = posture_wides["symmetry_tail_y_bias"]
wave_index_wide                 = posture_wides["wave_index"]

avg_curvature_extrap_wide       = curv_wides["avg_curvature_extrap"]
avg_tail_curvature_extrap_wide  = curv_wides["avg_tail_curvature_extrap"]

# save CSVs
_all_outputs = {
    "tailbeat_cycles_count_wide":         tailbeat_cycles_count_wide,
    "tailbeat_cycles_rate_hz_wide":       tailbeat_cycles_rate_hz_wide,
    "tail_tip_angle_rad_wide":            tail_tip_angle_rad_wide,
    "tail_mean_angle_rad_wide":           tail_mean_angle_rad_wide,
    "curvature_abs_tail_wide":            curvature_abs_tail_wide,
    "angles_centered_for_pca_wide":       angles_centered_for_pca_wide,
    "tailbeat_rate_thresh_hz_wide":       tailbeat_rate_thresh_hz_wide,
    "tailbeat_cycles_thresh_count_wide":  tailbeat_cycles_thresh_count_wide,
    "tailtip_freq_active_hz_wide":        tailtip_freq_active_hz_wide,
    "tailtip_freq_all_hz_wide":           tailtip_freq_all_hz_wide,
    "y_energy_rms_p95_wide":              y_energy_rms_p95_wide,
    "symmetry_y_bias_wide":               symmetry_y_bias_wide,
    "symmetry_tail_y_bias_wide":          symmetry_tail_y_bias_wide,
    "wave_index_wide":                    wave_index_wide,
    "avg_curvature_extrap_wide":          avg_curvature_extrap_wide,
    "avg_tail_curvature_extrap_wide":     avg_tail_curvature_extrap_wide,
}

for name, df in _all_outputs.items():
    if df is not None and not df.empty:
        df.to_csv(os.path.join(PLOT_OUTDIR, f"{name}.csv"), index=False)

print(f"[tail_metrics] {len(_all_outputs)} wide tables saved → {PLOT_OUTDIR}")