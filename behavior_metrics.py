"""
Module for calculating behavioral metrics from Oikopleura tracking data.

This module provides functions for computing:
- Center of mass (COM) from trunk coordinates
- Skeleton-based metrics (curvature, tangent angles, quirkiness)
- Velocity and acceleration
- Tail-beat frequency and amplitude
- Angular velocity (omega)
- Window-based features (tortuosity, path complexity, MSD slope)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, detrend
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, Optional

# =======================
# Skeleton Structure Helpers
# =======================

def split_trunk_tail(n_bodyparts: int, trunk_length: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    
    trunk = np.arange(min(trunk_length, n_bodyparts))
    tail = np.arange(trunk_length, n_bodyparts) if n_bodyparts > trunk_length else np.array([], dtype=int)
    return trunk, tail


def skeleton_indices(n_bodyparts: int, trunk_length: int = 4) -> np.ndarray:
    
    if n_bodyparts == 0:
        return np.array([], dtype=int)
    
    _, tail = split_trunk_tail(n_bodyparts, trunk_length)
    
    if tail.size == 0:
        return np.array([0], dtype=int)
    
    return np.concatenate(([0], tail))


# =======================
# Center of Mass (COM)
# =======================

def calculate_center_of_mass(cx: np.ndarray, cy: np.ndarray, 
                            indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    
    if indices is not None:
        cx = cx[:, indices]
        cy = cy[:, indices]
    
    com_x = pd.DataFrame(cx).mean(axis=1, skipna=True).to_numpy()
    com_y = pd.DataFrame(cy).mean(axis=1, skipna=True).to_numpy()
    
    return com_x, com_y


# =======================
# Curvature & Shape Metrics
# =======================

def calculate_curvature(skeleton: np.ndarray, 
                       window_length: int = 15, 
                       polyorder: int = 2,
                       max_curvature: float = 0.2) -> np.ndarray:
    
    T, n_points = skeleton.shape[:2]
    
    if n_points < 3:
        return np.full(T, np.nan)
    
    
    w = window_length if n_points >= window_length else (n_points if n_points % 2 == 1 else n_points - 1)
    if w < 3:
        w = 3
   
    if n_points >= 3:
        dx = savgol_filter(np.nan_to_num(skeleton[:, :, 0]), 
                          window_length=w, polyorder=polyorder, deriv=1, mode='nearest', axis=1)
        dy = savgol_filter(np.nan_to_num(skeleton[:, :, 1]), 
                          window_length=w, polyorder=polyorder, deriv=1, mode='nearest', axis=1)
        ddx = savgol_filter(np.nan_to_num(skeleton[:, :, 0]), 
                           window_length=w, polyorder=polyorder, deriv=2, mode='nearest', axis=1)
        ddy = savgol_filter(np.nan_to_num(skeleton[:, :, 1]), 
                           window_length=w, polyorder=polyorder, deriv=2, mode='nearest', axis=1)
    else:
        dx = np.gradient(np.nan_to_num(skeleton[:, :, 0]), axis=1)
        dy = np.gradient(np.nan_to_num(skeleton[:, :, 1]), axis=1)
        ddx = np.gradient(dx, axis=1)
        ddy = np.gradient(dy, axis=1)
    
    # Curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    eps = 1e-6
    num = dx[:, :-2] * ddy[:, 2:] - dy[:, :-2] * ddx[:, 2:]
    den = (dx[:, :-2]**2 + dy[:, :-2]**2 + eps)**1.5
    curv = np.abs(num / den)
    
    
    curv = np.where(curv <= max_curvature, curv, np.nan)
    
    
    return pd.DataFrame(curv).mean(axis=1, skipna=True).to_numpy()


def calculate_tangent_angles(skeleton: np.ndarray, mean_center: bool = True) -> np.ndarray:
    
    dy = np.diff(skeleton[:, :, 1], axis=1)
    dx = np.diff(skeleton[:, :, 0], axis=1)
    angles = np.arctan2(dy, dx)
    
    if mean_center:
       
        row_means = pd.DataFrame(angles).mean(axis=1, skipna=True).to_numpy()[:, None]
        row_means = np.where(np.isfinite(row_means), row_means, 0.0)
        angles = angles - row_means
        
        
        angles = np.where(angles < 0, angles + 2*np.pi, angles)
    
    return pd.DataFrame(angles).mean(axis=1, skipna=True).to_numpy()


def calculate_quirkiness(skeleton: np.ndarray, head_range: int = 10, tail_range: int = 10) -> np.ndarray:
    
    x = skeleton[:, :, 0]
    y = skeleton[:, :, 1]
    n_points = x.shape[1]
    
    
    head_rng = min(head_range, n_points)
    tail_rng = min(tail_range, n_points)
    
    head_x = pd.DataFrame(x[:, :head_rng]).mean(axis=1, skipna=True).to_numpy()
    head_y = pd.DataFrame(y[:, :head_rng]).mean(axis=1, skipna=True).to_numpy()
    tail_x = pd.DataFrame(x[:, -tail_rng:]).mean(axis=1, skipna=True).to_numpy()
    tail_y = pd.DataFrame(y[:, -tail_rng:]).mean(axis=1, skipna=True).to_numpy()
    
    
    axis_angle = np.arctan2(tail_y - head_y, tail_x - head_x)
    
    
    dy = np.diff(y, axis=1)
    dx = np.diff(x, axis=1)
    local_angles = np.arctan2(dy, dx)
    
    deviation = local_angles - axis_angle[:, None]
    deviation = (deviation + np.pi) % (2*np.pi) - np.pi  # Wrap to [-π, π]
    
   
    return np.nansum(np.abs(deviation), axis=1)


def calculate_body_axis_angle(skeleton: np.ndarray, 
                              head_range: int = 10, 
                              tail_range: int = 10) -> np.ndarray:
    
    x = skeleton[:, :, 0]
    y = skeleton[:, :, 1]
    n_points = x.shape[1]
    
    head_rng = min(head_range, n_points)
    tail_rng = min(tail_range, n_points)
    
    head_x = pd.DataFrame(x[:, :head_rng]).mean(axis=1, skipna=True).to_numpy()
    head_y = pd.DataFrame(y[:, :head_rng]).mean(axis=1, skipna=True).to_numpy()
    tail_x = pd.DataFrame(x[:, -tail_rng:]).mean(axis=1, skipna=True).to_numpy()
    tail_y = pd.DataFrame(y[:, -tail_rng:]).mean(axis=1, skipna=True).to_numpy()
    
    return np.arctan2(tail_y - head_y, tail_x - head_x)


# =======================
# Velocity & Acceleration
# =======================

def calculate_velocity(com_x: np.ndarray, com_y: np.ndarray, 
                      dt: float, pixel_to_um: float = 1.0) -> np.ndarray:
   
    dx = np.diff(com_x)
    dy = np.diff(com_y)
    
    displacement = np.sqrt(dx**2 + dy**2)
    velocity = (displacement / dt) * pixel_to_um
    
    
    return np.concatenate([[np.nan], velocity])


def calculate_acceleration(velocity: np.ndarray, dt: float) -> np.ndarray:
    
    dv = np.diff(velocity)
    acceleration = dv / dt
    
   
    return np.concatenate([[np.nan], acceleration])


# =======================
# Angular Velocity (Omega)
# =======================

def calculate_angular_velocity(angles: np.ndarray, dt: float, unwrap: bool = True) -> np.ndarray:
    
    if unwrap:
        angles_unwrapped = np.unwrap(angles)
    else:
        angles_unwrapped = angles
    
    omega = np.full_like(angles, np.nan)
    omega[1:] = np.diff(angles_unwrapped) / dt
    
    return omega


def calculate_omega_per_run(angles: np.ndarray, dt: float) -> np.ndarray:
    
    out = np.full_like(angles, np.nan)
    
   
    valid = np.isfinite(angles).astype(int)
    padded = np.concatenate([[0], valid, [0]])
    diff = np.diff(padded)
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    
    for s, e in zip(starts, ends):
        segment = angles[s:e]
        segment_unwrapped = np.unwrap(segment)
        
        if len(segment_unwrapped) >= 2:
            omega_segment = np.full_like(segment_unwrapped, np.nan)
            omega_segment[1:] = np.diff(segment_unwrapped) / dt
            out[s:e] = omega_segment
    
    return out


# =======================
# Tail-beat Metrics
# =======================

def tail_trunk_vectors(skeleton: np.ndarray, 
                       trunk_indices: np.ndarray, 
                       tail_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    T = skeleton.shape[0]
    trunk_angle = np.full(T, np.nan)
    tail_lateral = np.full(T, np.nan)
    
    if len(trunk_indices) == 0:
        return trunk_angle, tail_lateral
    
    anchor = 0 if 0 in trunk_indices else trunk_indices[0]
    base = skeleton[:, anchor, :]
    others = [j for j in trunk_indices if j != anchor] or [anchor]
    trunk_tip = np.nanmean(skeleton[:, others, :], axis=1)
    
    trunk_vec = trunk_tip - base
    trunk_angle = np.arctan2(trunk_vec[:, 1], trunk_vec[:, 0])
    
    trunk_len = np.linalg.norm(trunk_vec, axis=1, keepdims=True)
    trunk_len = np.where(trunk_len == 0, np.nan, trunk_len)
    u_trunk = trunk_vec / trunk_len
    n_trunk = np.stack([-u_trunk[:, 1], u_trunk[:, 0]], axis=1)  # perpendicular
    
    if len(tail_indices) > 0:
        tail_tip = skeleton[:, tail_indices[-1], :]
        tail_rel = tail_tip - base
        tail_lateral = np.sum(tail_rel * n_trunk, axis=1)
    
    return trunk_angle, tail_lateral


def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    
    if signal.size < 4 or np.nanstd(signal) == 0:
        return np.nan
        
    if signal.size >= 8:
        x = detrend(signal, type='linear')
    else:
        x = signal - np.nanmean(signal)
    
    xw = x * np.hanning(x.size)
    n = xw.size
    nfft = 1
    while nfft < 8 * n:
        nfft <<= 1
    
    X = np.fft.rfft(xw, n=nfft)
    P = np.abs(X)**2
    f = np.fft.rfftfreq(nfft, d=1.0/fs)
    
    if P.size == 0:
        return np.nan
    
    P[0] = 0.0
    
    k = int(np.argmax(P))
    if k <= 0 or k >= P.size - 1:
        return float(f[k])
    
    a, b, c = P[k-1], P[k], P[k+1]
    denom = (a - 2*b + c)
    
    if denom == 0 or not np.isfinite(denom):
        return float(f[k])
    
    delta = 0.5 * (a - c) / denom
    return float((k + delta) * fs / nfft)


def peak_to_peak_amplitude(signal: np.ndarray) -> float:
    
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    
    if signal.size == 0:
        return np.nan
    
    return 0.5 * (np.nanmax(signal) - np.nanmin(signal))


# =======================
# Window-based Features
# =======================

def sliding_windows(arr: np.ndarray, window_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    
    T = arr.shape[0]
    if T < window_size:
        return None, None
    
    view = sliding_window_view(arr, window_shape=window_size, axis=0)
    starts = np.arange(0, T - window_size + 1, stride)
    
    return view[starts], starts


def calculate_tortuosity(x_window: np.ndarray, y_window: np.ndarray, 
                        min_displacement: float = 5.0) -> float:
    
    dx = np.diff(x_window)
    dy = np.diff(y_window)
    
    path_length = np.nansum(np.hypot(dx, dy))
    displacement = np.hypot(x_window[-1] - x_window[0], y_window[-1] - y_window[0])
    
    if not np.isfinite(path_length) or not np.isfinite(displacement) or displacement < min_displacement:
        return np.nan
    
    return path_length / displacement


def calculate_path_complexity(x_window: np.ndarray, y_window: np.ndarray) -> float:
    
    x_std = (x_window - np.nanmean(x_window)) / np.nanstd(x_window)
    y_std = (y_window - np.nanmean(y_window)) / np.nanstd(y_window)
    
    r = np.nanmean(x_std * y_std)
    
    p1 = np.clip((1.0 + r) * 0.5, 0.0, 1.0)
    p2 = np.clip((1.0 - r) * 0.5, 0.0, 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -(p1 * np.log2(p1) + p2 * np.log2(p2))
    
    return H if np.isfinite(H) else np.nan


def calculate_msd_slope(x_window: np.ndarray, y_window: np.ndarray, 
                       dt: float, max_lag: int = 3) -> float:
    
    n = len(x_window)
    max_lag = min(max_lag, n - 1)
    
    if max_lag < 1:
        return np.nan
    
    msd_values = []
    for lag in range(1, max_lag + 1):
        dx = x_window[lag:] - x_window[:-lag]
        dy = y_window[lag:] - y_window[:-lag]
        msd = np.nanmean(dx**2 + dy**2)
        msd_values.append(msd)
    
    msd_values = np.array(msd_values)
    
    if not np.all(np.isfinite(msd_values)):
        return np.nan
    
    t = np.arange(1, max_lag + 1) * dt
    t_centered = t - t.mean()
    
    slope = np.sum(msd_values * t_centered) / np.sum(t_centered**2)
    
    return slope if np.isfinite(slope) else np.nan


if __name__ == '__main__':
   
    print("Behavior metrics module loaded successfully")
    print("\nAvailable functions:")
    print("- Center of mass: calculate_center_of_mass")
    print("- Curvature: calculate_curvature")
    print("- Velocity: calculate_velocity, calculate_acceleration")
    print("- Angular velocity: calculate_angular_velocity, calculate_omega_per_run")
    print("- Tail-beat: tail_trunk_vectors, dominant_frequency")
    print("- Window features: calculate_tortuosity, calculate_path_complexity, calculate_msd_slope")


