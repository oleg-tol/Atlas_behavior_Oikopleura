"""
Module for Hidden Markov Model (HMM) analysis of behavioral sequences.

Provides: sequence preparation, robust HMM fitting with multiple restarts, model selection
(BIC, held-out likelihood), rare state pruning/merging, Viterbi decoding, dwell time analysis,
state occupancy and transition matrices by experimental conditions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# =======================
# Sequence Preparation
# =======================

def prepare_sequences(df: pd.DataFrame,
                     feature_cols: List[str],
                     min_length: int = 30,
                     downsample: int = 1,
                     groupby_cols: List[str] = ['Experiment_ID', 'Individual']
                     ) -> Tuple[List[np.ndarray], List[Tuple]]:
    
    if downsample and downsample > 1:
        df_work = df.groupby(groupby_cols, group_keys=False).apply(
            lambda g: g.iloc[::downsample]
        ).reset_index(drop=True)
    else:
        df_work = df.copy()
    
    df_work['_seq_key'] = df_work[groupby_cols].astype(str).apply(
        lambda r: '||'.join(r), axis=1
    )
    
    sequences = []
    metadata = []
    
    for seq_key, group in df_work.groupby('_seq_key'):
        if len(group) < min_length:
            continue
        
        features = group[feature_cols].to_numpy(dtype=float)
        
        exp_id = group['Experiment_ID'].iat[0]
        individual = group['Individual'].iat[0]
        stage = group['Stage'].iat[0] if 'Stage' in group.columns else None
        condition = group['Condition'].iat[0] if 'Condition' in group.columns else None
        frame_indices = group.index.to_numpy()
        
        sequences.append(features)
        metadata.append((seq_key, exp_id, individual, stage, condition, frame_indices))
    
    return sequences, metadata


def split_sequences(n_sequences: int,
                   test_size: float = 0.2,
                   min_train: int = 5,
                   random_state: int = 139) -> Tuple[np.ndarray, np.ndarray]:
    
    if n_sequences < min_train:
        return np.arange(n_sequences), np.array([], dtype=int)
    
    return train_test_split(
        np.arange(n_sequences),
        test_size=test_size,
        random_state=random_state
    )


# =======================
# HMM Fitting
# =======================

def initialize_hmm_parameters(X: np.ndarray,
                             K: int,
                             covariance_type: str = 'full',
                             random_state: int = 139
                             ) -> Dict[str, np.ndarray]:
    
    rng = np.random.RandomState(random_state)
    D = X.shape[1]
    
    try:
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        means = kmeans.cluster_centers_
    except Exception:
        
        labels = rng.randint(0, K, size=X.shape[0])
        means = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) 
            else X[rng.choice(len(X))]
            for k in range(K)
        ])
    
    eps = 1e-6
    if covariance_type == 'full':
        covars = np.zeros((K, D, D), dtype=float)
        for k in range(K):
            points = X[labels == k]
            if points.shape[0] <= 1:
                cov = np.eye(D) * (eps * 10.0)
            else:
                cov = np.cov(points, rowvar=False) + np.eye(D) * eps
            covars[k] = cov
    else:  # diagonal
        covars = np.zeros((K, D), dtype=float)
        for k in range(K):
            points = X[labels == k]
            if points.shape[0] <= 1:
                var = np.ones(D) * (eps * 10.0)
            else:
                var = np.var(points, axis=0)
            covars[k] = np.clip(var, eps, None)
    
    counts = np.bincount(labels, minlength=K).astype(float)
    counts[counts == 0] = 1e-8
    startprob = counts / counts.sum()
    
    if K == 1:
        transmat = np.array([[1.0]])
    else:
        self_p = 0.8
        off_diag = (1.0 - self_p) / (K - 1)
        transmat = np.full((K, K), off_diag)
        np.fill_diagonal(transmat, self_p)
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
    
    return {
        'means': means,
        'covars': covars,
        'startprob': startprob,
        'transmat': transmat
    }


def fit_hmm_with_restarts(X_train: np.ndarray,
                         lengths_train: List[int],
                         K: int,
                         n_restarts: int = 8,
                         covariance_type: str = 'full',
                         n_iter: int = 500,
                         tol: float = 1e-4,
                         random_state: int = 139,
                         verbose: bool = False) -> Tuple[Optional[Any], float, List[Dict]]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn required. Install with: pip install hmmlearn")
    
    best_model = None
    best_ll = -np.inf
    restart_info = []
    
    for r in range(n_restarts):
        seed = int(random_state + r)
        
        init_params = initialize_hmm_parameters(
            X_train, K, covariance_type, random_state=seed
        )
        
        model = GaussianHMM(
            n_components=K,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            verbose=False
        )
        
        model.startprob_ = init_params['startprob']
        model.transmat_ = init_params['transmat']
        model.means_ = init_params['means']
        model.covars_ = init_params['covars']
        model.init_params = '' 
       
        try:
            model.fit(X_train, lengths=lengths_train)
            ll = model.score(X_train, lengths=lengths_train)
        except Exception as e:
            if verbose:
                print(f"[WARN] Restart {r} failed: {e}")
            restart_info.append({'restart': r, 'seed': seed, 'll': np.nan, 'error': str(e)})
            continue
        
        restart_info.append({'restart': r, 'seed': seed, 'll': float(ll), 'error': ''})
        
        if verbose:
            print(f"[INFO] Restart {r}: LL = {ll:.4f}")
        
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_model = model
    
    return best_model, best_ll, restart_info


# =======================
# Model Evaluation
# =======================

def compute_bic(model: Any,
               X_train: np.ndarray,
               lengths_train: List[int]) -> float:
    
    K = model.n_components
    D = model.means_.shape[1]
    N = X_train.shape[0]
    
    
    if model.covariance_type == 'full':
        cov_params = K * (D * (D + 1) / 2.0)
    else:  # diagonal
        cov_params = K * D
    
   
    p = (K - 1) + K * (K - 1) + K * D + cov_params
    
    ll = model.score(X_train, lengths=lengths_train)
    bic = -2.0 * ll + p * np.log(N)
    
    return float(bic)


def evaluate_model(model: Any,
                  X_train: np.ndarray,
                  lengths_train: List[int],
                  X_test: Optional[np.ndarray] = None,
                  lengths_test: Optional[List[int]] = None) -> Dict:
    
    K = model.n_components
    train_ll = model.score(X_train, lengths=lengths_train)
    
    test_ll = np.nan
    if X_test is not None and lengths_test and len(lengths_test) > 0:
        try:
            test_ll = model.score(X_test, lengths=lengths_test)
        except Exception:
            pass
    
    bic = compute_bic(model, X_train, lengths_train)
    
    D = model.means_.shape[1]
    if model.covariance_type == 'full':
        cov_params = K * (D * (D + 1) / 2.0)
    else:
        cov_params = K * D
    n_params = (K - 1) + K * (K - 1) + K * D + cov_params
    
    return {
        'K': K,
        'train_ll': float(train_ll),
        'test_ll': float(test_ll),
        'bic': float(bic),
        'n_params': int(n_params)
    }


# =======================
# Rare State Handling
# =======================

def identify_rare_states(state_sequences: List[np.ndarray],
                        prune_fraction: float = 0.01) -> List[int]:
   
    all_states = np.concatenate(state_sequences)
    total_frames = all_states.size
    
    unique, counts = np.unique(all_states, return_counts=True)
    state_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    min_frames = max(1, int(prune_fraction * total_frames))
    rare = [s for s, c in state_counts.items() if c < min_frames]
    
    return rare


def merge_rare_states(model: Any,
                     state_sequences: List[np.ndarray],
                     rare_states: List[int]) -> Tuple[Dict[int, int], List[int]]:
    
    K = model.n_components
    means = model.means_
    
    keep_states = [s for s in range(K) if s not in rare_states]
    
    if len(keep_states) == 0:
        raise ValueError("Cannot merge: all states are rare")
    
    mapping = {s: s for s in range(K)}
    
    for rare_s in rare_states:
        distances = np.linalg.norm(means[keep_states] - means[rare_s], axis=1)
        nearest = keep_states[int(np.argmin(distances))]
        mapping[rare_s] = nearest
    
    return mapping, keep_states


def apply_state_mapping(state_sequences: List[np.ndarray],
                       mapping: Dict[int, int]) -> List[np.ndarray]:
    
    mapped_seqs = [
        np.array([mapping[int(s)] for s in seq], dtype=int)
        for seq in state_sequences
    ]
    
    all_mapped = np.concatenate(mapped_seqs)
    unique_mapped = np.sort(np.unique(all_mapped))
    remap = {old: new for new, old in enumerate(unique_mapped)}
    
    remapped_seqs = [
        np.array([remap[int(s)] for s in seq], dtype=int)
        for seq in mapped_seqs
    ]
    
    return remapped_seqs


# =======================
# Viterbi Decoding
# =======================

def decode_sequences(model: Any,
                    sequences: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    
    states = [model.predict(seq) for seq in sequences]
    posteriors = [model.predict_proba(seq) for seq in sequences]
    
    return states, posteriors


# =======================
# Dwell Time Analysis
# =======================

def compute_dwell_times(state_sequences: List[np.ndarray],
                       n_states: int) -> Dict[int, List[int]]:
    
    dwell_times = {s: [] for s in range(n_states)}
    
    for seq in state_sequences:
        if len(seq) == 0:
            continue
        
        current_state = seq[0]
        run_length = 1
        
        for s in seq[1:]:
            if s == current_state:
                run_length += 1
            else:
                dwell_times[current_state].append(run_length)
                current_state = s
                run_length = 1
        
        dwell_times[current_state].append(run_length)
    
    return dwell_times


def summarize_dwell_times(dwell_times: Dict[int, List[int]]) -> pd.DataFrame:
    
    rows = []
    for state, dwells in dwell_times.items():
        if not dwells:
            continue
        rows.append({
            'state': state,
            'n_runs': len(dwells),
            'median': int(np.median(dwells)),
            'mean': float(np.mean(dwells)),
            'std': float(np.std(dwells)),
            'min': int(np.min(dwells)),
            'max': int(np.max(dwells))
        })
    
    return pd.DataFrame(rows)


# =======================
# State Occupancy & Transitions
# =======================

def compute_state_occupancy(state_sequences: List[np.ndarray],
                           metadata: List[Tuple],
                           group_by: List[str] = ['Stage', 'Condition']
                           ) -> pd.DataFrame:
    
    rows = []
    meta_keys = {0: 'seq_key', 1: 'Experiment_ID', 2: 'Individual', 
                 3: 'Stage', 4: 'Condition'}
    
    for meta, states in zip(metadata, state_sequences):
        for s in states:
            row = {meta_keys[i]: meta[i] for i in range(min(len(meta), 5))}
            row['state'] = int(s)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not all(col in df.columns for col in group_by):
        return pd.DataFrame()
    
    occupancy = df.groupby(group_by + ['state']).size().reset_index(name='count')
    totals = occupancy.groupby(group_by)['count'].transform('sum')
    occupancy['fraction'] = occupancy['count'] / totals
    
    return occupancy


def compute_transition_matrix(state_sequences: List[np.ndarray],
                             n_states: int,
                             normalize: bool = True) -> np.ndarray:
    
    trans = np.zeros((n_states, n_states), dtype=float)
    
    for seq in state_sequences:
        if len(seq) < 2:
            continue
        
        for s_from, s_to in zip(seq[:-1], seq[1:]):
            trans[int(s_from), int(s_to)] += 1
    
    if normalize:
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans = trans / row_sums
    
    return trans


def compute_transitions_by_group(state_sequences: List[np.ndarray],
                                metadata: List[Tuple],
                                n_states: int,
                                group_by: List[str] = ['Stage', 'Condition']
                                ) -> pd.DataFrame:
   
    groups = {}
    meta_keys = {0: 'seq_key', 1: 'Experiment_ID', 2: 'Individual', 
                 3: 'Stage', 4: 'Condition'}
    
    for meta, states in zip(metadata, state_sequences):
        if len(meta) < 5:
            continue
        
        group_key = tuple(meta[3:5])  # (Stage, Condition)
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(states)
    
    rows = []
    for group_key, seqs in groups.items():
        trans_mat = compute_transition_matrix(seqs, n_states, normalize=True)
        
        for i in range(n_states):
            for j in range(n_states):
                row = dict(zip(group_by, group_key))
                row['from_state'] = i
                row['to_state'] = j
                row['prob'] = float(trans_mat[i, j])
                rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("HMM temporal modeling module loaded successfully")
    print("\nAvailable functions:")
    print("- Sequence prep: prepare_sequences, split_sequences")
    print("- HMM fitting: fit_hmm_with_restarts, initialize_hmm_parameters")
    print("- Evaluation: evaluate_model, compute_bic")
    print("- Rare states: identify_rare_states, merge_rare_states, apply_state_mapping")
    print("- Decoding: decode_sequences")
    print("- Analysis: compute_dwell_times, compute_state_occupancy, compute_transition_matrix")
