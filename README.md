# Oikopleura Developmental Behaviour Atlas — *Oikopleura dioica*
Pipeline for constructing an unsupervised atlas of behavioural states across developmental stages using DeepLabCut tracking data.

## Modules

| File | Description |
|------|-------------|
| `data_loading_processing.py` | Load DLC `*filtered.csv`, apply likelihood threshold, IQR + teleportation filtering, gap interpolation, temporal resampling |
| `behavior_metrics.py` | COM velocity, acceleration, curvature, tangent angles, quirkiness, omega, tortuosity, path complexity, MSD slope, tail-beat frequency |
| `data_organizing.py` | `SkeletonStore` class; wide-format table builders for coordinates and metrics; data quality checks |
| `centerline_analysis.py` | PCHIP/linear centerline extraction, head→origin normalization, quality filtering, aggregation by behavioural cluster |
| `extra_tail_metrics.py` | Tailbeat count and rate (zero-crossing and adaptive threshold), segment angles, tail curvature, posture metrics (y-energy, symmetry, wave index), active-masked tail-tip frequency |
| `unsupervised_clustering.py` | Window feature extraction, PCA, t-SNE (openTSNE), HDBSCAN autotune, stage balancing, KNN label propagation |
| `hmm_temporal_analysis.py` | Sequence preparation, GaussianHMM fitting with restarts, BIC evaluation, Viterbi decoding, dwell times, state occupancy and transition matrices |
| `size_correction.py` | Body-length growth curve (logistic + 95 % bootstrap CI), BL normalisation of speed and window metrics, outside-house developmental trajectory, pairwise MWU stat table |

## Pipeline

```
DLC *filtered.csv
  → data_loading_processing.py     →  resampled coordinates, per-file quality report
  → behavior_metrics.py            →  velocity, curvature, omega, path metrics (wide CSVs)
  → data_organizing.py             →  SkeletonStore, coordinate/metric wide tables
  → centerline_analysis.py         →  centerline_K_norm per animal per frame
  → extra_tail_metrics.py          →  tailbeat rate, angles, curvature, posture wide CSVs
  → unsupervised_clustering.py     →  PCA → t-SNE embedding, HDBSCAN cluster labels
  → hmm_temporal_analysis.py       →  decoded states, dwell summaries, transition matrices
  → size_correction.py             →  BL-corrected metrics, supplement figures, MWU table
```

## Key parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Likelihood threshold | 0.99 | `data_loading_processing.py` |
| Teleportation filter | 100 px max step | `data_loading_processing.py` |
| Min run length — larva/juvenile | 90 frames (3 s @ 30 fps) | `data_loading_processing.py` |
| Min run length — adult | 30 frames (1 s @ 30 fps) | `data_loading_processing.py` |
| Resample rate | 1:5 → 6 Hz | `data_loading_processing.py` |
| Pixel to µm | 11.56 µm/px | `behavior_metrics.py` |
| Window / stride | 18 / 6 frames (3 s / 1 s @ 6 Hz) | `unsupervised_clustering.py` |
| PCA components | 7 | `unsupervised_clustering.py` |
| t-SNE perplexity | 750 | `unsupervised_clustering.py` |
| Centerline points | 21 | `centerline_analysis.py` |
| Tail region | last 3 segments / last 35 % of body | `extra_tail_metrics.py` |
| Tailbeat threshold mode | percentile 50 of \|Δy\| | `extra_tail_metrics.py` |
| Curvature robust cap | 99.5th percentile | `extra_tail_metrics.py` |
| Body-length correction | divide by mean\_length\_um | `size_correction.py` |
| House condition label | `"house"` (excluded from developmental trajectory) | `size_correction.py` |
| HMM states | 10–14 (BIC-selected) | `hmm_temporal_analysis.py` |

## Requirements

```
numpy pandas scipy scikit-learn
opentsne hdbscan hmmlearn
matplotlib seaborn
```

Set `PLOT_OUTDIR`, `PIXEL_TO_UM`, and `FS_RS` in the calling scope before running individual modules.

- DeepLabCut team for pose estimation framework
- openTSNE developers for scalable t-SNE implementation
- hmmlearn and scikit-learn communities
