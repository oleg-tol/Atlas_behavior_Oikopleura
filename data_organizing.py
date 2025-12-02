"""
Module for organizing and structuring processed tracking data.

This module handles:
- Building skeleton data structures
- Creating wide-format dataframes for time series
- Managing metadata (experiment ID, stage, condition, individual)
- Constructing time column names
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# =======================
# Time Column Utilities
# =======================

def build_time_columns(n_frames: int) -> List[str]:
    
    return [f"Time_{i:03d}" for i in range(n_frames)]


def extract_time_columns(df: pd.DataFrame) -> List[str]:
    
    return [c for c in df.columns if isinstance(c, str) and c.startswith("Time_")]


# =======================
# Skeleton Data Structure
# =======================

class SkeletonStore:
        
    def __init__(self):
        
        self._data = {}
    
    def add_individual(self,
                      experiment_id: str,
                      stage: Optional[str],
                      condition: Optional[str],
                      individual: str,
                      bodyparts: List[str],
                      cx_native: np.ndarray,
                      cy_native: np.ndarray,
                      cx_resampled: np.ndarray,
                      cy_resampled: np.ndarray,
                      trunk_indices: np.ndarray,
                      tail_indices: np.ndarray,
                      skeleton_indices: np.ndarray):
        
        key = (experiment_id, stage, condition, individual)
        
        # Stack coordinates into (T, n, 2) format
        full_native = np.stack((cx_native, cy_native), axis=-1)
        full_resampled = np.stack((cx_resampled, cy_resampled), axis=-1)
        
        self._data[key] = {
            'bodyparts': bodyparts,
            'full_native': full_native,
            'full_resampled': full_resampled,
            'trunk_idx': trunk_indices.tolist() if isinstance(trunk_indices, np.ndarray) else trunk_indices,
            'tail_idx': tail_indices.tolist() if isinstance(tail_indices, np.ndarray) else tail_indices,
            'skel_idx': skeleton_indices.tolist() if isinstance(skeleton_indices, np.ndarray) else skeleton_indices,
            'T_native': int(full_native.shape[0]),
            'T_resampled': int(full_resampled.shape[0]),
            'n_bodyparts': int(full_native.shape[1])
        }
    
    def get_individual(self, experiment_id: str, stage: Optional[str], 
                      condition: Optional[str], individual: str) -> Optional[Dict]:
       
        key = (experiment_id, stage, condition, individual)
        return self._data.get(key)
    
    def iter_individuals(self):
        
        for key, data in self._data.items():
            yield key, data
    
    def get_summary(self) -> pd.DataFrame:
        
        rows = []
        for (exp_id, stage, cond, indiv), data in self._data.items():
            rows.append({
                'Experiment_ID': exp_id,
                'Stage': stage,
                'Condition': cond,
                'Individual': indiv,
                'T_native': data['T_native'],
                'T_resampled': data['T_resampled'],
                'N_bodyparts': data['n_bodyparts']
            })
        
        return pd.DataFrame(rows)
    
    def __len__(self):
        
        return len(self._data)
    
    def __contains__(self, key):
        
        return key in self._data


# =======================
# Wide-Format Dataframe Construction
# =======================

def build_coordinates_wide(experiment_id: str,
                          stage: Optional[str],
                          condition: Optional[str],
                          individual: str,
                          bodyparts: List[str],
                          cx: np.ndarray,
                          cy: np.ndarray,
                          trunk_length: int = 4) -> pd.DataFrame:
    
    T, n_parts = cx.shape
    time_cols = build_time_columns(T)
    
    rows = []
    for j in range(n_parts):
        bp_name = bodyparts[j] if j < len(bodyparts) else f"BP_{j+1}"
        segment = "trunk" if j < trunk_length else "tail"
        
        # X coordinate row
        meta_x = {
            'Experiment_ID': experiment_id,
            'Stage': stage,
            'Condition': condition,
            'Individual': individual,
            'Coordinate': 'x',
            'Bodypart': bp_name,
            'Segment': segment
        }
        row_x = meta_x | dict(zip(time_cols, cx[:, j]))
        rows.append(row_x)
        
        # Y coordinate row
        meta_y = meta_x.copy()
        meta_y['Coordinate'] = 'y'
        row_y = meta_y | dict(zip(time_cols, cy[:, j]))
        rows.append(row_y)
    
    return pd.DataFrame(rows)


def build_metric_wide(experiment_id: str,
                     stage: Optional[str],
                     condition: Optional[str],
                     individual: str,
                     metric_name: str,
                     values: np.ndarray) -> pd.DataFrame:
    
    time_cols = build_time_columns(len(values))
    
    meta = {
        'Experiment_ID': experiment_id,
        'Stage': stage,
        'Condition': condition,
        'Individual': individual,
        'Metric': metric_name
    }
    
    row = meta | dict(zip(time_cols, values))
    return pd.DataFrame([row])


def build_com_wide(experiment_id: str,
                  stage: Optional[str],
                  condition: Optional[str],
                  individual: str,
                  com_x: np.ndarray,
                  com_y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df_x = build_metric_wide(experiment_id, stage, condition, individual, 
                            "COM_X_trunk", com_x)
    df_y = build_metric_wide(experiment_id, stage, condition, individual, 
                            "COM_Y_trunk", com_y)
    
    return df_x, df_y


def concatenate_wide_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    
    valid_dfs = [df for df in df_list if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    return pd.concat(valid_dfs, axis=0, ignore_index=True)


# =======================
# Metadata Extraction
# =======================

def align_xy_coordinates(com_x_wide: pd.DataFrame, 
                        com_y_wide: pd.DataFrame,
                        metadata_cols: List[str] = ['Experiment_ID', 'Stage', 'Condition', 'Individual']
                        ) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    
    keys = [k for k in metadata_cols if k in com_x_wide.columns and k in com_y_wide.columns]
        
    tcols_x = extract_time_columns(com_x_wide)
    tcols_y = extract_time_columns(com_y_wide)
    
    if tcols_x == tcols_y:
        common_time = tcols_x
    else:
        common_time = sorted(
            set(tcols_x).intersection(tcols_y),
            key=lambda s: int(s.split("_")[-1])
        )
    
    
    merged = com_x_wide[keys + common_time].merge(
        com_y_wide[keys + common_time],
        on=keys,
        suffixes=("_x", "_y")
    )
    
    X = merged[[f"{c}_x" for c in common_time]].to_numpy(float)
    Y = merged[[f"{c}_y" for c in common_time]].to_numpy(float)
    
    return merged[keys].reset_index(drop=True), common_time, X, Y


def group_by_metadata(df: pd.DataFrame, 
                     group_cols: List[str] = ['Stage', 'Condition']) -> pd.DataFrame:
    
    valid_groups = [col for col in group_cols if col in df.columns]
    
    if not valid_groups:
        return df
    
    time_cols = extract_time_columns(df)
    
    if time_cols:
        
        grouped = df.groupby(valid_groups, dropna=False)[time_cols].mean()
    else:
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        grouped = df.groupby(valid_groups, dropna=False)[numeric_cols].mean()
    
    return grouped.reset_index()


# =======================
# Validation Utilities
# =======================

def validate_skeleton_dimensions(cx: np.ndarray, cy: np.ndarray, 
                                bodyparts: List[str]) -> bool:
   
    if cx.shape != cy.shape:
        return False
    
    if cx.shape[1] != len(bodyparts):
        return False
    
    return True


def check_data_quality(cx: np.ndarray, cy: np.ndarray, 
                      min_valid_ratio: float = 0.1) -> Tuple[bool, float]:
    
    total_values = cx.size + cy.size
    valid_values = np.sum(np.isfinite(cx)) + np.sum(np.isfinite(cy))
    
    valid_ratio = valid_values / total_values if total_values > 0 else 0.0
    is_valid = valid_ratio >= min_valid_ratio
    
    return is_valid, valid_ratio


if __name__ == '__main__':
   
    print("Data organizing module loaded successfully")
    print("\nAvailable classes and functions:")
    print("- SkeletonStore: storage container for skeleton data")
    print("- build_coordinates_wide: create wide-format coordinate tables")
    print("- build_metric_wide: create wide-format metric tables")
    print("- align_xy_coordinates: align X and Y dataframes")
    print("- Time column utilities: build_time_columns, extract_time_columns")

