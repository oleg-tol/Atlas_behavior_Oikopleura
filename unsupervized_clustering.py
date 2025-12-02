"""
Module for unsupervised behavior space embedding and clustering.

Provides: window feature extraction, normalization/preprocessing, dimensionality reduction
(PCA, t-SNE), HDBSCAN clustering with auto-tuning, stage-balanced sampling, label propagation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from openTSNE import TSNE as OTSNE
from openTSNE import affinity, initialization
import hdbscan


# =======================
# Window Feature Extraction
# =======================

def extract_time_columns(df: pd.DataFrame) -> List[str]:
    
    return [c for c in df.columns if isinstance(c, str) and c.startswith("Time_")]


def sort_time_columns(time_cols: List[str]) -> List[str]:
    
    import re
    def extract_index(col):
        m = re.search(r'(\d+)$', str(col))
        return int(m.group(1)) if m else 0
    return sorted(time_cols, key=extract_index)


def compute_window_starts(n_frames: int, window_size: int, stride: int) -> np.ndarray:
    
    if n_frames < window_size:
        return np.array([], dtype=int)
    return np.arange(0, n_frames - window_size + 1, stride, dtype=int)


def window_statistic(values: np.ndarray, stat: str = "mean", q: Optional[float] = None) -> float:
    
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan
    
    if stat == "mean":
        return float(np.mean(valid))
    elif stat == "percentile":
        q_val = q if q is not None else 95.0
        return float(np.percentile(valid, q_val))
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def wide_to_windows(df: pd.DataFrame, 
                   feature_name: str,
                   window_size: int,
                   stride: int,
                   stat: str = "mean",
                   q: Optional[float] = None,
                   metadata_cols: List[str] = ['Experiment_ID', 'Stage', 'Condition', 'Individual']
                   ) -> pd.DataFrame:
   
    if df is None or df.empty:
        cols = metadata_cols + ["WindowStart", feature_name]
        return pd.DataFrame(columns=cols)
    
    time_cols = extract_time_columns(df)
    if not time_cols:
        cols = metadata_cols + ["WindowStart", feature_name]
        return pd.DataFrame(columns=cols)
    
    time_cols = sort_time_columns(time_cols)
    meta_cols = [c for c in metadata_cols if c in df.columns]
    
    rows = []
    for _, row in df.iterrows():
        values = pd.to_numeric(row[time_cols], errors='coerce').to_numpy(dtype=float)
        starts = compute_window_starts(len(values), window_size, stride)
        
        for start in starts:
            window = values[start:start + window_size]
            value = window_statistic(window, stat=stat, q=q)
            
            if not np.isfinite(value):
                continue
            
            record = {col: row[col] for col in meta_cols}
            record["WindowStart"] = int(start)
            record[feature_name] = float(value)
            rows.append(record)
    
    return pd.DataFrame(rows)


# =======================
# Feature Normalization
# =======================

def robust_clip(df: pd.DataFrame, 
               columns: List[str],
               quantiles: Tuple[float, float] = (0.005, 0.995)) -> pd.DataFrame:
    
    for col in columns:
        if col in df.columns and df[col].notna().any():
            lo, hi = df[col].quantile(list(quantiles))
            df[col] = df[col].clip(lo, hi)
    return df


def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(pd.to_numeric(df[col], errors='coerce').astype(float))
    return df


def normalize_within_individual(df: pd.DataFrame, 
                                feature_cols: List[str],
                                method: str = "robust") -> pd.DataFrame:
    
    if "Individual" not in df.columns:
        return df
    
    df[feature_cols] = df[feature_cols].astype(float)
    
    if method == "robust":
        def robust_normalize(group):
            median = group.median()
            q25 = group.quantile(0.25)
            q75 = group.quantile(0.75)
            iqr = (q75 - q25).replace(0, np.nan)
            return (group - median) / iqr
        
        df[feature_cols] = df.groupby("Individual", group_keys=False)[feature_cols].apply(robust_normalize)
    else:
        # Standard z-score
        df[feature_cols] = df.groupby("Individual", group_keys=False)[feature_cols].apply(
            lambda g: (g - g.mean()) / g.std()
        )
    
    # Handle infinities and NaNs
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df


def normalize_globally(df: pd.DataFrame, 
                      feature_cols: List[str],
                      method: str = "standard") -> pd.DataFrame:
    
    df[feature_cols] = df[feature_cols].astype(float)
    
    if method == "robust":
        median = df[feature_cols].median()
        q25 = df[feature_cols].quantile(0.25)
        q75 = df[feature_cols].quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        df[feature_cols] = (df[feature_cols] - median) / iqr
    else:
        scaler = StandardScaler()
        df[feature_cols] = pd.DataFrame(
            scaler.fit_transform(df[feature_cols].values),
            index=df.index,
            columns=feature_cols
        )
    
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df


# =======================
# Dimensionality Reduction
# =======================

def run_pca(X: np.ndarray, 
           n_components: int = 7,
           random_state: int = 139) -> Tuple[np.ndarray, PCA]:
    
    n_comp = min(n_components, X.shape[1], max(2, X.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    print(f"[PCA] Variance explained: {pca.explained_variance_ratio_.sum():.3f} "
          f"({n_comp} components)")
    
    return X_pca, pca


def run_tsne_opentsne(X: np.ndarray,
                     perplexity: float = 30.0,
                     random_state: int = 139) -> Tuple[np.ndarray, object]:
          
    n = X.shape[0]
    
    
    perp_max = (n - 1) / 3.0
    if perplexity >= perp_max:
        perplexity = max(5.0, perp_max - 1.0)
        print(f"[t-SNE] Adjusted perplexity to {perplexity:.1f} (n={n})")
    
    aff = affinity.PerplexityBasedNN(
        X, perplexity=float(perplexity),
        metric="euclidean",
        n_jobs=-1,
        random_state=random_state,
        method="approx"
    )
    
    init = initialization.pca(X, random_state=random_state)
    
    tsne = OTSNE(
        n_components=2,
        negative_gradient_method="fft",
        n_jobs=-1,
        random_state=random_state,
        verbose=False
    )
    
    
    embedding = tsne.fit(affinities=aff, initialization=init)
       
    embedding = embedding.optimize(n_iter=750, exaggeration=1.0, momentum=0.8, inplace=True)
    
    Y = np.asarray(embedding)
    
    return Y, embedding


# =======================
# HDBSCAN Clustering
# =======================

def estimate_distance_scale(Y: np.ndarray, sample_size: int = 5000) -> float:
    
    n = Y.shape[0]
    if n <= 2:
        return 1.0
    
    if n > sample_size:
        idx = np.random.RandomState(42).choice(n, size=sample_size, replace=False)
        Y_sample = Y[idx]
    else:
        Y_sample = Y
    
    D = pairwise_distances(Y_sample, metric="euclidean")
    
    tri = D[np.triu_indices_from(D, k=1)]
    tri = tri[tri > 0]
    
    if tri.size == 0:
        return 1.0
    
    median_dist = np.median(tri)
    
    return float(median_dist) if np.isfinite(median_dist) and median_dist > 0 else 1.0


def run_hdbscan_autotune(Y: np.ndarray, 
                        target_k: Optional[int] = None,
                        min_cluster_size_range: Optional[Tuple[int, int]] = None,
                        random_state: int = 42) -> Tuple[np.ndarray, Dict]:
        
    n = Y.shape[0]
    scale = estimate_distance_scale(Y)
    
    if min_cluster_size_range:
        mcs_min, mcs_max = min_cluster_size_range
        min_cluster_sizes = sorted(set([mcs_min, (mcs_min + mcs_max) // 2, mcs_max]))
    else:
        min_cluster_sizes = sorted(set([
            max(15, n // 400),
            max(20, n // 300),
            max(30, n // 200)
        ]))
    
    min_samples_list = [None, 1, 5]
    epsilon_list = [0.0, 0.05 * scale, 0.10 * scale, 0.20 * scale]
    
    best_result = None
    best_score = float('inf')
    
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for eps in epsilon_list:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_epsilon=eps,
                    core_dist_n_jobs=-1
                )
                
                labels = clusterer.fit_predict(Y)
                k = len(set(labels)) - (1 if -1 in labels else 0)
                
                if target_k is not None:
                   
                    score = abs(k - target_k)
                    
                    if score < best_score or (score == best_score and k > best_result[2]):
                        best_result = (labels, k, mcs, ms, eps)
                        best_score = score
                else:
                    
                    n_noise = np.sum(labels == -1)
                    score = n_noise - k * 10  # Favor more clusters, penalize noise
                    if score < best_score:
                        best_result = (labels, k, mcs, ms, eps)
                        best_score = score
    
    labels, k, mcs, ms, eps = best_result
    
    params = {
        "min_cluster_size": mcs,
        "min_samples": ms,
        "cluster_selection_epsilon": eps,
        "achieved_k": k,
        "distance_scale": scale,
        "n_noise": int(np.sum(labels == -1))
    }
    
    if target_k is not None:
        params["target_k"] = target_k
    
    return labels, params


# =======================
# Stage-balanced Sampling
# =======================

def balance_by_stage(metadata: pd.DataFrame,
                    X: np.ndarray,
                    min_per_stage: Optional[int] = None,
                    random_state: int = 139) -> Tuple[pd.DataFrame, np.ndarray]:
    
    if "Stage" not in metadata.columns or metadata["Stage"].dropna().empty:
        print("[WARNING] Stage column missing or empty - skipping balancing")
        return metadata, X
    
    stage_counts = metadata["Stage"].value_counts(dropna=True)
    target = stage_counts.min() if min_per_stage is None else int(min_per_stage)
    
    print(f"[Balancing] Stage counts: {stage_counts.to_dict()}")
    print(f"[Balancing] Target per stage: {target}")
    
    rng = np.random.RandomState(random_state)
    indices_keep = []
    
    for stage in stage_counts.index:
        stage_indices = metadata.index[metadata["Stage"] == stage].to_numpy()
        n_stage = len(stage_indices)
        
        if n_stage > target:
            sampled = rng.choice(stage_indices, size=target, replace=False)
            indices_keep.append(sampled)
        else:
            indices_keep.append(stage_indices)
    
    indices_keep = np.concatenate(indices_keep)
    indices_keep.sort()
    
    metadata_balanced = metadata.loc[indices_keep].copy().reset_index(drop=True)
    X_balanced = X[indices_keep].copy()
    
    return metadata_balanced, X_balanced


# =======================
# Label Propagation
# =======================

def propagate_labels_knn(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_full: np.ndarray,
                        n_neighbors: int = 7,
                        weights: str = "distance") -> np.ndarray:
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="euclidean"
    )
    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_full)
    
    return y_pred


def transform_new_data_tsne(embedding_obj,
                           X_new_pca: np.ndarray,
                           perplexity: float) -> np.ndarray:
    
    Y_new_embedding = embedding_obj.transform(X_new_pca, perplexity=float(perplexity))
    return np.asarray(Y_new_embedding)


# =======================
# Pipeline Helpers
# =======================

def build_feature_matrix(window_features: pd.DataFrame,
                        feature_columns: List[str],
                        within_individual: bool = True,
                        clip_quantiles: Tuple[float, float] = (0.005, 0.995),
                        log_transform_cols: Optional[List[str]] = None
                        ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    
    df = window_features.copy()
    
    used_columns = [c for c in feature_columns if c in df.columns]
    
    if not used_columns:
        raise ValueError("No specified feature columns found in dataframe")
    
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing features (skipped): {missing}")
    
    df = robust_clip(df, used_columns, clip_quantiles)
    
    
    if log_transform_cols:
        log_cols = [c for c in log_transform_cols if c in used_columns]
        df = apply_log_transform(df, log_cols)
    
    
    if within_individual and "Individual" in df.columns:
        df = normalize_within_individual(df, used_columns, method="robust")
    else:
        df = normalize_globally(df, used_columns, method="standard")
    
    
    X = df[used_columns].to_numpy(dtype=float)
    
   
    metadata_cols = ["Experiment_ID", "Stage", "Condition", "Individual", "WindowStart"]
    meta_present = [c for c in metadata_cols if c in df.columns]
    df_normalized = df[meta_present + used_columns].copy()
    
    return df_normalized, X, used_columns


if __name__ == '__main__':
    print("Unsupervised clustering module loaded successfully")
    print("\nAvailable functions:")
    print("- Feature extraction: wide_to_windows, window_statistic")
    print("- Normalization: robust_clip, normalize_within_individual, normalize_globally")
    print("- Dimensionality reduction: run_pca, run_tsne_opentsne")
    print("- Clustering: run_hdbscan_autotune, estimate_distance_scale")
    print("- Utilities: balance_by_stage, propagate_labels_knn")

