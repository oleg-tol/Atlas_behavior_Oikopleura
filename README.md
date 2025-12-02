# Oikopleura Developmental Behavior Atlas

Analysis pipeline for constructing an unsupervised atlas of behavioral states across developmental stages in *Oikopleura dioica*, using DeepLabCut tracking data.

## Overview

This repository provides modular tools for processing high-throughput behavioral tracking data and constructing behavior atlases through unsupervised learning. The pipeline handles data from larval through adult stages, enabling comparative developmental analyses.

**Key features:**
- Robust data loading and preprocessing for DLC output
- Comprehensive behavioral metric extraction (motion, shape, tail-beat)
- Unsupervised behavior space embedding and clustering
- Temporal sequence modeling with Hidden Markov Models
- Developmental stage-aware analysis and visualization

## Installation

### Requirements
```bash
# Core dependencies
pip install numpy pandas scipy scikit-learn

# For clustering and embedding
pip install openTSNE hdbscan

# For HMM analysis
pip install hmmlearn

# For visualization (optional)
pip install matplotlib seaborn
```

### Python Version
Tested with Python 3.8+

## Module Structure

### Core Analysis Modules

#### 1. `data_loading_processing.py`
Data loading and preprocessing pipeline.

**Functions:**
- `collect_csv_files()` - Discover CSV files recursively
- `read_filtered_csv()` - Parse DLC multi-level headers
- `extract_stage_from_path()` - Extract developmental stage
- `apply_likelihood_threshold()` - Filter low-confidence tracking
- `iqr_outlier_removal()` - Statistical outlier removal
- `filter_large_steps()` - Teleportation artifact removal
- `mask_short_runs()` - Enforce minimum continuity
- `interpolate_bracketed()` - Fill internal gaps only
- `resample_temporal()` - Temporal downsampling
- `process_csv_file()` - Complete pipeline

**Example:**
```python
from data_loading_processing import process_csv_file

result = process_csv_file(
    'path/to/file.csv',
    likelihood_min=0.99,
    iqr_threshold=2.0,
    distance_threshold=100.0,
    min_run_frames=90,  # 3s @ 30fps
    resample_rate=5     # downsample to 6 Hz
)
```

---

#### 2. `behavior_metrics.py`
Behavioral metric calculations.

**Functions:**
- `split_trunk_tail()` - Segment bodyparts
- `calculate_center_of_mass()` - From trunk coordinates
- `calculate_curvature()` - Savitzky-Golay-based curvature
- `calculate_tangent_angles()` - Mean-centered tangent directions
- `calculate_quirkiness()` - Deviation from body axis
- `calculate_velocity()`, `calculate_acceleration()` - With spatial units
- `calculate_omega_per_run()` - Angular velocity with per-run unwrapping
- `tail_trunk_vectors()`, `dominant_frequency()` - Tail-beat analysis
- `calculate_tortuosity()`, `calculate_path_complexity()`, `calculate_msd_slope()` - Window features

**Example:**
```python
from behavior_metrics import calculate_velocity, dominant_frequency

# Motion metrics
velocity = calculate_velocity(com_x, com_y, dt=0.167, pixel_to_um=11.56)

# Tail-beat frequency
trunk_angle, tail_lateral = tail_trunk_vectors(skeleton, trunk_idx, tail_idx)
freq = dominant_frequency(tail_lateral, fs=30.0)
```

---

#### 3. `data_organizing.py`
Data structure management.

**Classes:**
- `SkeletonStore` - Organized storage with metadata

**Functions:**
- `build_time_columns()` - Standardized column naming
- `build_coordinates_wide()`, `build_metric_wide()` - Wide-format tables
- `align_xy_coordinates()` - Pair X/Y for analysis
- `validate_skeleton_dimensions()`, `check_data_quality()` - QC

**Example:**
```python
from data_organizing import SkeletonStore, build_metric_wide

store = SkeletonStore()
store.add_individual(
    experiment_id='exp_001',
    stage='adult',
    condition='control',
    individual='ind_01',
    bodyparts=bodypart_names,
    cx_native=cx, cy_native=cy,
    cx_resampled=cx_rs, cy_resampled=cy_rs,
    trunk_indices=trunk_idx,
    tail_indices=tail_idx,
    skeleton_indices=skel_idx
)
```

---

#### 4. `unsupervised_clustering.py`
Behavior space embedding and clustering.

**Functions:**
- `wide_to_windows()` - Extract window features from time series
- `build_feature_matrix()` - Preprocessing pipeline (clip, transform, normalize)
- `run_pca()` - Dimensionality reduction
- `run_tsne_opentsne()` - Scalable t-SNE embedding
- `run_hdbscan_autotune()` - Flexible cluster discovery
- `balance_by_stage()` - Equal stage representation
- `propagate_labels_knn()` - Extend clusters to full dataset

**Example:**
```python
from unsupervised_clustering import (
    build_feature_matrix, run_pca, run_tsne_opentsne, 
    run_hdbscan_autotune, balance_by_stage
)

# Preprocessing
features_norm, X, cols = build_feature_matrix(
    window_features,
    feature_columns=['vel_p95', 'acc_p95', 'omega_p95', ...],
    within_individual=True
)

# Stage balancing for fair representation
meta_balanced, X_balanced = balance_by_stage(metadata, X)

# Dimensionality reduction
X_pca, pca = run_pca(X_balanced, n_components=7)
Y, embedding = run_tsne_opentsne(X_pca, perplexity=750)

# Flexible clustering
# Option 1: Target-guided (exploratory)
labels, params = run_hdbscan_autotune(Y, target_k=7)

# Option 2: Data-driven (no target)
labels, params = run_hdbscan_autotune(Y, target_k=None)

print(f"Found {params['achieved_k']} clusters")
```

The `target_k` parameter can be:
- Specified for exploratory analysis with different counts
- Set to `None` for automatic selection based on cluster stability
- Fine-tuned via `min_cluster_size_range` parameter

---

#### 5. `centerline_analysis.py`
Skeleton shape extraction and analysis.

**Functions:**
- `find_bodypart_indices()` - Regex-based pattern matching
- `interpolate_curve_pchip()`, `interpolate_curve_linear()` - Curve interpolation
- `normalize_centerline()` - Head→origin, tail→+X, length=1
- `extract_centerlines_from_skeleton()` - Full extraction pipeline
- `filter_centerlines_by_quality()` - Artifact removal
- `aggregate_centerlines_by_cluster()` - Group by behavior states

**Example:**
```python
from centerline_analysis import extract_centerlines_from_skeleton, aggregate_centerlines_by_cluster

# Define anchor patterns
patterns = {
    'bp5': r'bodypart\s*5',
    'bp6': r'bodypart\s*6'
}

# Extract centerlines
centerlines = extract_centerlines_from_skeleton(
    skeleton_store,
    anchor_patterns=patterns,
    mid_pattern=r'bodypart\s*7',
    n_interpolated=21,
    interpolation_method='pchip'
)

# Aggregate by cluster
grouped = aggregate_centerlines_by_cluster(
    centerlines,
    frame_labels,
    metadata_cols=['Stage', 'Condition']
)
```

---

#### 6. `hmm_temporal_analysis.py`
Hidden Markov Model analysis for temporal sequences.

**Functions:**
- `prepare_sequences()` - Extract sequences from windowed features
- `fit_hmm_with_restarts()` - Robust fitting with multiple initializations
- `evaluate_model()` - BIC, held-out likelihood
- `identify_rare_states()`, `merge_rare_states()` - State pruning
- `decode_sequences()` - Viterbi decoding
- `compute_dwell_times()` - Run length analysis
- `compute_state_occupancy()`, `compute_transition_matrix()` - Summary statistics

**Example:**
```python
from hmm_temporal_analysis import (
    prepare_sequences, fit_hmm_with_restarts, 
    evaluate_model, decode_sequences
)

# Prepare sequences
sequences, metadata = prepare_sequences(
    features_df,
    feature_cols=['PC1', 'PC2', ..., 'PC7'],
    min_length=30,
    downsample=1
)

# Fit HMM
model, train_ll, info = fit_hmm_with_restarts(
    X_train, lengths_train,
    K=10,
    n_restarts=8,
    covariance_type='full'
)

# Evaluate
metrics = evaluate_model(model, X_train, lengths_train, X_test, lengths_test)
print(f"BIC: {metrics['bic']:.1f}, Test LL: {metrics['test_ll']:.2f}")

# Decode sequences
states, posteriors = decode_sequences(model, sequences)
```

---

## Typical Workflow

### 1. Data Loading and Processing
```python
from data_loading_processing import collect_csv_files, process_csv_file

# Discover files
files = collect_csv_files('/path/to/data')

# Process each file
results = []
for file in files:
    result = process_csv_file(
        file,
        likelihood_min=0.99,
        min_run_frames=90,
        resample_rate=5
    )
    results.append(result)
```

### 2. Build Skeleton Store and Compute Metrics
```python
from data_organizing import SkeletonStore
from behavior_metrics import (
    calculate_center_of_mass, calculate_velocity,
    calculate_curvature, dominant_frequency
)

store = SkeletonStore()

for result in results:
    # Extract trunk/tail indices
    trunk_idx, tail_idx = split_trunk_tail(result['cx_resampled'].shape[1])
    
    # Compute COM
    com_x, com_y = calculate_center_of_mass(
        result['cx_resampled'], 
        result['cy_resampled'],
        indices=trunk_idx
    )
    
    # Velocity
    velocity = calculate_velocity(com_x, com_y, dt=0.167, pixel_to_um=11.56)
    
    # Store for later use
    # ... (add to skeleton store)
```

### 3. Window Feature Extraction
```python
from unsupervised_clustering import wide_to_windows

# Extract percentile features over sliding windows
vel_p95 = wide_to_windows(
    velocity_df,
    'vel_p95',
    window_size=18,  # 3s @ 6Hz
    stride=6,        # 1s stride
    stat='percentile',
    q=95
)

# Combine multiple features
window_features = merge_features([vel_p95, acc_p95, omega_mean, ...])
```

### 4. Unsupervised Behavior Atlas Construction
```python
from unsupervised_clustering import (
    build_feature_matrix, run_pca, run_tsne_opentsne,
    run_hdbscan_autotune, balance_by_stage, propagate_labels_knn
)

# Preprocess features
features_norm, X, cols = build_feature_matrix(
    window_features,
    feature_columns=selected_features,
    within_individual=True
)

# Balance by developmental stage
meta_balanced, X_balanced = balance_by_stage(metadata, X)

# Embed in 2D
X_pca, pca = run_pca(X_balanced, n_components=7)
Y_balanced, embedding = run_tsne_opentsne(X_pca, perplexity=750)

# Cluster (data-driven or target-guided)
labels_balanced, params = run_hdbscan_autotune(Y_balanced, target_k=None)
print(f"Discovered {params['achieved_k']} behavioral states")

# Propagate to full dataset
X_pca_full = pca.transform(X)
Y_full = embedding.transform(X_pca_full, perplexity=750)
labels_full = propagate_labels_knn(Y_balanced, labels_balanced, Y_full)
```

### 5. Shape Analysis per Cluster
```python
from centerline_analysis import (
    extract_centerlines_from_skeleton,
    aggregate_centerlines_by_cluster
)

# Extract normalized centerlines
centerlines = extract_centerlines_from_skeleton(
    skeleton_store,
    anchor_patterns={'bp5': r'bodypart\s*5', 'bp6': r'bodypart\s*6'},
    n_interpolated=21
)

# Group by behavioral cluster
grouped = aggregate_centerlines_by_cluster(
    centerlines,
    frame_labels,
    metadata_cols=['Stage', 'Condition']
)

# Analyze mean shapes per cluster
for (stage, condition), clusters in grouped.items():
    for cluster_id, centerline_list in clusters.items():
        # centerline_list: [(key, frame_idx, centerline), ...]
        shapes = np.array([cl for _, _, cl in centerline_list])
        mean_shape = np.nanmean(shapes, axis=0)
        # ... plot or analyze
```

### 6. Temporal Sequence Modeling (HMM)
```python
from hmm_temporal_analysis import (
    prepare_sequences, fit_hmm_with_restarts,
    identify_rare_states, merge_rare_states,
    compute_dwell_times, compute_state_occupancy
)

# Prepare sequences from PCA features
sequences, metadata = prepare_sequences(
    features_df,
    feature_cols=['PC1', 'PC2', ..., 'PC7'],
    min_length=30
)

# Fit initial model
model, ll, info = fit_hmm_with_restarts(
    X_train, lengths_train,
    K=14,
    n_restarts=8
)

# Decode all sequences
states, posteriors = decode_sequences(model, sequences)

# Identify and merge rare states (< 1% occupancy)
rare = identify_rare_states(states, prune_fraction=0.01)
if rare:
    mapping, kept = merge_rare_states(model, states, rare)
    states = apply_state_mapping(states, mapping)
    # Refit with reduced K
    # ...

# Analyze dwell times
dwell_times = compute_dwell_times(states, model.n_components)
dwell_summary = summarize_dwell_times(dwell_times)

# Occupancy by stage/condition
occupancy = compute_state_occupancy(states, metadata, group_by=['Stage', 'Condition'])
```

---
## File Formats

### Input: DLC Filtered CSV
Expected structure:
```
Row 0: scorer
Row 1: individual_id_1, individual_id_1, individual_id_1, ...
Row 2: bodypart_1, bodypart_1, bodypart_1, ...
Row 3: x, y, likelihood, x, y, likelihood, ...
Row 4+: tracking data
```

### Output: Wide-format Tables
Standardized format for time series:
```
Experiment_ID | Stage | Condition | Individual | Time_000 | Time_001 | ...
exp_001       | adult | control   | ind_01     | 145.3    | 147.1    | ...
```

---

## Configuration Examples

### Stage-specific Parameters
Different developmental stages may require different processing:

```python
# Larvae/Juvenile: enforce 3s minimum continuity
min_run_frames_larva = 90  # 3s @ 30fps

# Adult: more permissive
min_run_frames_adult = 30  # 1s @ 30fps

# Apply stage-specific filtering
if stage == 'larva' or stage == 'juvenile':
    min_run = min_run_frames_larva
else:
    min_run = min_run_frames_adult
```

### Window Parameters
Standard 3-second windows with 1-second stride:

```python
# Native rate: 30 fps
# Resampled rate: 6 fps (1:5 downsample)

WINDOW_SECONDS = 3.0
STRIDE_SECONDS = 1.0

WINDOW_FRAMES_RESAMPLED = int(3.0 * 6)  # 18 frames
STRIDE_FRAMES_RESAMPLED = int(1.0 * 6)  # 6 frames
```

---

## Citation

If you use these tools in your research, please cite:

```
[    ]
```

---

## Acknowledgments

- DeepLabCut team for pose estimation framework
- openTSNE developers for scalable t-SNE implementation
- hmmlearn and scikit-learn communities
