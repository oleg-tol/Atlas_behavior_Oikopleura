"""
Module for centerline extraction and shape analysis from skeleton data.

Provides: centerline extraction/interpolation, normalization (head-tail orientation, length),
shape metrics computation, quality filtering, aggregation by behavioral clusters.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import re


# =======================
# Geometry Helpers
# =======================

def find_bodypart_indices(bodyparts: List[str], target_patterns: Dict[str, str]) -> Dict[str, Optional[int]]:
    
    result = {key: None for key in target_patterns.keys()}
    
    if not bodyparts:
        return result
    
    names_lower = [str(name or "").lower() for name in bodyparts]
    
    for key, pattern in target_patterns.items():
        regex = re.compile(pattern, re.IGNORECASE)
        for j, name in enumerate(names_lower):
            if regex.search(name):
                result[key] = j
                break
    
    return result


def compute_arc_length_parameterization(points: np.ndarray, 
                                       normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    
    if points.shape[0] <= 1:
        if points.shape[0] == 1:
            return np.array([0.0, 1.0]), np.vstack([points[0], points[0]])
        return np.array([0.0, 1.0]), np.zeros((2, 2))
    
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segments)])
    
    total_length = s[-1]
    if total_length <= 0:
        return np.array([0.0, 1.0]), np.vstack([points[0], points[0]])
    
    if normalize:
        s = s / total_length
    
    tol = 1e-6
    for i in range(1, s.size):
        if s[i] <= s[i-1] + tol:
            s[i] = s[i-1] + tol
    
    if normalize:
        s = (s - s[0]) / (s[-1] - s[0])
    
    return s, points


def interpolate_curve_linear(control_points: np.ndarray, n_points: int) -> np.ndarray:
    
    s, ctrl = compute_arc_length_parameterization(control_points)
    s_new = np.linspace(0.0, 1.0, n_points)
    
    x = np.interp(s_new, s, ctrl[:, 0])
    y = np.interp(s_new, s, ctrl[:, 1])
    
    return np.column_stack([x, y])


def interpolate_curve_pchip(control_points: np.ndarray, n_points: int) -> np.ndarray:
    
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        raise ImportError("PCHIP interpolation requires scipy")
    
    s, ctrl = compute_arc_length_parameterization(control_points)
    s_new = np.linspace(0.0, 1.0, n_points)
    
    fx = PchipInterpolator(s, ctrl[:, 0], extrapolate=False)
    fy = PchipInterpolator(s, ctrl[:, 1], extrapolate=False)
    
    return np.column_stack([fx(s_new), fy(s_new)])


# =======================
# Centerline Normalization
# =======================

def normalize_centerline(centerline: np.ndarray) -> Optional[np.ndarray]:
    
    if centerline is None or centerline.shape[0] < 2:
        return None
    
    head = centerline[0]
    centered = centerline - head
    
    tail = centered[-1]
    length = float(np.hypot(tail[0], tail[1]))
    
    if length <= 0:
        return None
    
    angle = np.arctan2(tail[1], tail[0])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated = centered @ rotation.T
    
    normalized = rotated / length
    
    return normalized


def compute_centerline_length(centerline: np.ndarray) -> float:
    
    if centerline is None or centerline.shape[0] < 2:
        return 0.0
    
    diffs = np.diff(centerline, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    
    return float(np.sum(lengths))


# =======================
# Temporal Smoothing
# =======================

def smooth_trajectory(track: np.ndarray, 
                     window_length: int = 3,
                     polyorder: int = 2) -> np.ndarray:
   
    if window_length is None or window_length <= 1:
        return track
    
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("[WARNING] scipy not available, skipping smoothing")
        return track
    
    T = track.shape[0]
    w = min(window_length, T if T % 2 == 1 else T - 1)
    
    if w < 3:
        return track
    
    out = track.copy()
    idx = np.arange(T)
    
    for dim in (0, 1):
        col = out[:, dim]
        mask = np.isfinite(col)
        
        if mask.sum() >= w:
            
            filled = col.copy()
            if not mask.all():
                filled[~mask] = np.interp(idx[~mask], idx[mask], col[mask])
            
            out[:, dim] = savgol_filter(
                filled,
                window_length=w,
                polyorder=min(polyorder, w - 1),
                mode="interp"
            )
    
    return out


# =======================
# Centerline Extraction
# =======================

def extract_centerlines_from_skeleton(
    skeleton_store: Dict,
    anchor_patterns: Dict[str, str],
    mid_pattern: Optional[str] = None,
    extra_before: Optional[List[str]] = None,
    extra_after: Optional[List[str]] = None,
    n_interpolated: int = 21,
    interpolation_method: str = "pchip",
    min_length_px: float = 8.0,
    max_length_px: float = 250.0,
    smooth_window: int = 3,
    smooth_anchors: bool = True
) -> Dict[Tuple, Dict]:
    
    interpolator = interpolate_curve_pchip if interpolation_method == "pchip" else interpolate_curve_linear
    
    result = {}
    
    for key, meta in skeleton_store.items():
        full_skeleton = meta.get("full_resampled")
        bodyparts = meta.get("bodyparts", [])
        
        if full_skeleton is None or full_skeleton.size == 0 or not bodyparts:
            continue
        
        all_patterns = anchor_patterns.copy()
        if mid_pattern:
            all_patterns["mid"] = mid_pattern
        if extra_before:
            for i, pat in enumerate(extra_before):
                all_patterns[f"extra_before_{i}"] = pat
        if extra_after:
            for i, pat in enumerate(extra_after):
                all_patterns[f"extra_after_{i}"] = pat
        
        indices = find_bodypart_indices(bodyparts, all_patterns)
        
        anchor_keys = list(anchor_patterns.keys())
        if any(indices[k] is None for k in anchor_keys):
            T = full_skeleton.shape[0]
            empty = np.full((T, n_interpolated, 2), np.nan)
            result[key] = {"centerline_K": empty, "centerline_K_norm": empty.copy()}
            continue
        
        skeleton = full_skeleton.copy()
        if smooth_anchors:
            for anchor_key in anchor_keys:
                idx = indices[anchor_key]
                if idx is not None:
                    skeleton[:, idx, :] = smooth_trajectory(
                        skeleton[:, idx, :],
                        window_length=smooth_window
                    )
        
        T = skeleton.shape[0]
        centerlines = np.full((T, n_interpolated, 2), np.nan, dtype=float)
        centerlines_norm = np.full((T, n_interpolated, 2), np.nan, dtype=float)
        
        for t in range(T):
            frame = skeleton[t]
            
            anchor_pts = [frame[indices[k]] for k in anchor_keys]
            
            if not all(np.all(np.isfinite(pt)) for pt in anchor_pts):
                continue
            
            anchor_vec = anchor_pts[1] - anchor_pts[0]
            anchor_dist = float(np.hypot(anchor_vec[0], anchor_vec[1]))
            
            if anchor_dist < min_length_px:
                continue
            
            control = [anchor_pts[0]]
            
            if extra_before:
                for i in range(len(extra_before)):
                    idx = indices.get(f"extra_before_{i}")
                    if idx is not None and np.all(np.isfinite(frame[idx])):
                        control.append(frame[idx])
            
            if mid_pattern and indices.get("mid") is not None:
                mid_pt = frame[indices["mid"]]
                if np.all(np.isfinite(mid_pt)):
                    control.append(mid_pt)
                else:
                    
                    control.append(anchor_pts[0] + 0.5 * anchor_vec)
            else:
                control.append(anchor_pts[0] + 0.5 * anchor_vec)
           
            if extra_after:
                for i in range(len(extra_after)):
                    idx = indices.get(f"extra_after_{i}")
                    if idx is not None and np.all(np.isfinite(frame[idx])):
                        control.append(frame[idx])
            
            control.append(anchor_pts[1])
            
            control = np.vstack(control)
            axis_unit = anchor_vec / anchor_dist
            t_vals = (control - anchor_pts[0]) @ axis_unit
            control = control[np.argsort(t_vals)]
            
            try:
                centerline = interpolator(control, n_interpolated)
            except Exception:
                
                centerline = interpolate_curve_linear(control, n_interpolated)
            
            length = compute_centerline_length(centerline)
            if length > max_length_px:
                continue
            
            centerlines[t] = centerline
            
            normalized = normalize_centerline(centerline)
            if normalized is not None:
                centerlines_norm[t] = normalized
        
        result[key] = {
            "centerline_K": centerlines,
            "centerline_K_norm": centerlines_norm
        }
    
    return result


# =======================
# Quality Filtering
# =======================

def filter_centerlines_by_quality(
    centerlines_dict: Dict,
    max_lateral_deviation: float = 0.6,
    max_segment_angle: float = 2.0
) -> Dict:
    
    result = {}
    
    for key, data in centerlines_dict.items():
        centerlines_norm = data.get("centerline_K_norm")
        
        if centerlines_norm is None:
            continue
        
        T = centerlines_norm.shape[0]
        valid_frames = []
        
        for t in range(T):
            frame = centerlines_norm[t]
            
            
            if not np.all(np.isfinite(frame)):
                continue
            
            
            max_y = np.max(np.abs(frame[:, 1]))
            if max_y > max_lateral_deviation:
                continue
            
            if frame.shape[0] >= 3:
                vectors = np.diff(frame, axis=0)
                norms = np.linalg.norm(vectors, axis=1)
                
                valid_vecs = norms > 1e-9
                if np.sum(valid_vecs) < 2:
                    valid_frames.append(frame)
                    continue
                
                unit_vecs = np.zeros_like(vectors)
                unit_vecs[valid_vecs] = (vectors[valid_vecs].T / norms[valid_vecs]).T
                
                dots = np.clip(np.sum(unit_vecs[:-1] * unit_vecs[1:], axis=1), -1.0, 1.0)
                crosses = unit_vecs[:-1, 0] * unit_vecs[1:, 1] - unit_vecs[:-1, 1] * unit_vecs[1:, 0]
                angles = np.abs(np.arctan2(crosses, dots))
                
                if np.max(angles) <= max_segment_angle:
                    valid_frames.append(frame)
            else:
                valid_frames.append(frame)
        
        if valid_frames:
            #
            filtered = np.full_like(centerlines_norm, np.nan)
            for i, frame in enumerate(valid_frames):
               
                if i < T:
                    filtered[i] = frame
            
            result[key] = {
                "centerline_K_norm": filtered,
                "n_valid": len(valid_frames)
            }
    
    return result


# =======================
# Aggregation by Cluster
# =======================

def aggregate_centerlines_by_cluster(
    centerlines_dict: Dict,
    frame_labels: Dict[Tuple, np.ndarray],
    metadata_cols: List[str] = ["Stage", "Condition"]
) -> Dict:
    
    result = {}
    
    # Metadata extraction map
    key_to_meta = {
        "Experiment_ID": 0,
        "Stage": 1,
        "Condition": 2,
        "Individual": 3
    }
    
    meta_indices = [key_to_meta.get(col, None) for col in metadata_cols]
    
    for key, data in centerlines_dict.items():
        centerlines = data.get("centerline_K_norm")
        labels = frame_labels.get(key)
        
        if centerlines is None or labels is None:
            continue
        
        if len(labels) != centerlines.shape[0]:
            continue
        
        meta_tuple = tuple(key[i] if i is not None else None for i in meta_indices)
        
        for t in range(centerlines.shape[0]):
            frame = centerlines[t]
            
            if not np.all(np.isfinite(frame)):
                continue
            
            cluster_id = int(labels[t])
            
            result.setdefault(meta_tuple, {}).setdefault(cluster_id, []).append(
                (key, t, frame)
            )
    
    return result


if __name__ == '__main__':
    print("Centerline analysis module loaded successfully")
    print("\nAvailable functions:")
    print("- Geometry: compute_arc_length_parameterization, normalize_centerline")
    print("- Interpolation: interpolate_curve_linear, interpolate_curve_pchip")
    print("- Extraction: extract_centerlines_from_skeleton")
    print("- Quality control: filter_centerlines_by_quality")
    print("- Aggregation: aggregate_centerlines_by_cluster")
