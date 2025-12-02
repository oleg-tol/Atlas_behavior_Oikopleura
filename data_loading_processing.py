"""
Module for loading and processing DLC (DeepLabCut) output CSV files for Oikopleura tracking data.

This module handles:
- Reading filtered CSV files with individual/bodypart structure
- Likelihood-based filtering
- Outlier removal (IQR method)
- Teleport detection (large step filtering)
- Continuity enforcement (minimum run length)
- Bracketed interpolation
- Temporal resampling
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# =======================
# File Discovery & Metadata
# =======================

def collect_csv_files(root_dir: str) -> List[str]:
    
    paths = []
    if not root_dir or not os.path.isdir(root_dir):
        return paths
        
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith("_filtered.csv"):
                paths.append(os.path.join(dirpath, fn))
    
    have = set(paths)
        
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".csv"):
                fp = os.path.join(dirpath, fn)
                if fp not in have:
                    paths.append(fp)
    
    return sorted(set(paths))


def extract_stage_from_path(file_path: str, 
                           adult_dir: Optional[str] = None,
                           juvenile_dir: Optional[str] = None,
                           larvae_dir: Optional[str] = None) -> Optional[str]:
   
    d = os.path.normpath(os.path.dirname(file_path))
        
    if adult_dir and (d == os.path.normpath(adult_dir) or d.startswith(os.path.normpath(adult_dir) + os.sep)):
        return "adult"
    if juvenile_dir and (d == os.path.normpath(juvenile_dir) or d.startswith(os.path.normpath(juvenile_dir) + os.sep)):
        return "juvenile"
    if larvae_dir and (d == os.path.normpath(larvae_dir) or d.startswith(os.path.normpath(larvae_dir) + os.sep)):
        return "larva"
    
    parts = [s.lower() for s in os.path.normpath(file_path).split(os.sep)]
    if any(re.match(r"^adult", s) for s in parts) or "adult" in parts:
        return "adult"
    if any(re.match(r"^juv", s) for s in parts) or "juvenile" in parts or "juv" in parts:
        return "juvenile"
    if any(re.match(r"^larv", s) for s in parts) or "larva" in parts or "larvae" in parts:
        return "larva"
    
    return None


def extract_condition_from_path(file_path: str, stage_dir: str) -> Optional[str]:
    
    d = os.path.normpath(os.path.dirname(file_path))
    stage_base = os.path.normpath(stage_dir)
    
    if d == stage_base or d.startswith(stage_base + os.sep):
        rel = os.path.relpath(d, stage_base)
        parts = [] if rel == "." else rel.split(os.sep)
        return parts[0] if parts else None
    
    return None


# =======================
# CSV Reading
# =======================

def read_filtered_csv(file_path: str) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    
    hdr = pd.read_csv(file_path, nrows=3, header=None)
    
    individuals = hdr.iloc[1, 1::3].fillna("Unknown").astype(str).tolist()
    bodyparts = hdr.iloc[2, 1::3].fillna("").astype(str).tolist()
    
    raw = pd.read_csv(file_path, header=None, skiprows=3, low_memory=False)
    
    cx = pd.to_numeric(raw.iloc[:, 1::3].stack(), errors='coerce').values.reshape(raw.shape[0], -1)
    cy = pd.to_numeric(raw.iloc[:, 2::3].stack(), errors='coerce').values.reshape(raw.shape[0], -1)
    likelihood = pd.to_numeric(raw.iloc[:, 3::3].stack(), errors='coerce').values.reshape(raw.shape[0], -1)
    
    bodyparts = [
        (bp.strip() if isinstance(bp, str) and bp.strip() else f"BP_{i+1}")
        for i, bp in enumerate(bodyparts)
    ]
    
    return individuals, bodyparts, cx, cy, likelihood


def safe_to_numeric(df_like: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to numeric, coercing errors to NaN."""
    return df_like.apply(pd.to_numeric, errors="coerce")


# =======================
# Filtering & Cleaning
# =======================

def apply_likelihood_threshold(cx: np.ndarray, cy: np.ndarray, 
                              likelihood: np.ndarray, 
                              min_likelihood: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    cx_out = cx.copy()
    cy_out = cy.copy()
    
    mask = likelihood < min_likelihood
    cx_out[mask] = np.nan
    cy_out[mask] = np.nan
    
    return cx_out, cy_out


def iqr_outlier_removal(arr_2d: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    
    if arr_2d.size == 0:
        return arr_2d
    
    q25 = np.nanpercentile(arr_2d, 25, axis=0)
    q75 = np.nanpercentile(arr_2d, 75, axis=0)
    iqr = q75 - q25
    
    low = q25 - threshold * iqr
    high = q75 + threshold * iqr
    
    out = arr_2d.copy()
    out[(out < low) | (out > high)] = np.nan
    
    return out


def filter_large_steps(cx: np.ndarray, cy: np.ndarray, 
                      max_distance: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    
    if cx.shape[0] <= 1:
        return cx, cy
    
    dx = np.diff(cx, axis=0)
    dy = np.diff(cy, axis=0)
    dist = np.sqrt(dx**2 + dy**2)
    
    keep = np.vstack([np.ones((1, dist.shape[1]), dtype=bool), dist <= max_distance])
    
    cx_out = cx.copy()
    cy_out = cy.copy()
    cx_out[~keep] = np.nan
    cy_out[~keep] = np.nan
    
    return cx_out, cy_out


def mask_short_runs(arr_2d: np.ndarray, min_length: int) -> np.ndarray:
    
    if min_length <= 1 or arr_2d.size == 0:
        return arr_2d
    
    out = arr_2d.copy()
    T, N = out.shape
    
    for j in range(N):
        
        valid = np.isfinite(out[:, j]).astype(int)
        padded = np.concatenate([[0], valid, [0]])
        diff = np.diff(padded)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for s, e in zip(starts, ends):
            if (e - s) < min_length:
                out[s:e, j] = np.nan
    
    return out


# =======================
# Interpolation
# =======================

def interpolate_bracketed(arr_2d: np.ndarray) -> np.ndarray:
    
    out = np.empty_like(arr_2d, dtype=float)
    
    for j in range(arr_2d.shape[1]):
        series = pd.Series(arr_2d[:, j], dtype="float64")
        interpolated = series.interpolate(method="linear", limit_area="inside")
        out[:, j] = interpolated.values
    
    return out


# =======================
# Resampling
# =======================

def resample_temporal(arr: np.ndarray, rate: int) -> np.ndarray:
    
    if rate is None or rate <= 1:
        return arr
    return arr[::rate]


# =======================
# Full Processing Pipeline
# =======================

def process_csv_file(file_path: str,
                    likelihood_min: float = 0.99,
                    iqr_threshold: float = 2.0,
                    distance_threshold: float = 100.0,
                    min_run_frames: int = 0,
                    apply_resample: bool = True,
                    resample_rate: int = 5) -> dict:
    
    individuals, bodyparts, cx, cy, likelihood = read_filtered_csv(file_path)
   
    cx, cy = apply_likelihood_threshold(cx, cy, likelihood, likelihood_min)
    cx = iqr_outlier_removal(cx, iqr_threshold)
    cy = iqr_outlier_removal(cy, iqr_threshold)
    cx, cy = filter_large_steps(cx, cy, distance_threshold)
    
    if min_run_frames > 1:
        cx = mask_short_runs(cx, min_run_frames)
        cy = mask_short_runs(cy, min_run_frames)
    
    cx = interpolate_bracketed(cx)
    cy = interpolate_bracketed(cy)
    
    if apply_resample and resample_rate > 1:
        cx_rs = resample_temporal(cx, resample_rate)
        cy_rs = resample_temporal(cy, resample_rate)
    else:
        cx_rs, cy_rs = cx, cy
    
    return {
        'individuals': individuals,
        'bodyparts': bodyparts,
        'cx_native': cx,
        'cy_native': cy,
        'cx_resampled': cx_rs,
        'cy_resampled': cy_rs,
        'T_native': cx.shape[0],
        'T_resampled': cx_rs.shape[0]
    }


if __name__ == '__main__':
    
    example_dir = "/path/to/csv/files"
    files = collect_csv_files(example_dir)
    print(f"Found {len(files)} CSV files")
    
    if files:
        
        result = process_csv_file(
            files[0],
            likelihood_min=0.99,
            iqr_threshold=2.0,
            distance_threshold=100.0,
            min_run_frames=90,  # e.g., 3s @ 30fps
            apply_resample=True,
            resample_rate=5
        )
        print(f"Processed: {len(result['individuals'])} individuals, {len(result['bodyparts'])} bodyparts")
        print(f"Native frames: {result['T_native']}, Resampled frames: {result['T_resampled']}")