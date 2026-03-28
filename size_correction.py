"""
Size correction for locomotor metrics across developmental stages.

Outputs (supplement figures):
  A. Body-length growth curve (logistic fit + 95% bootstrap CI)
  B. Speed raw vs. BL-corrected side-by-side (all stages)
  C. Propagation metrics BL-corrected, outside-house animals by stage
  D. Pairwise Mann-Whitney U stat table (CSV + PNG)

Assumes: PLOT_OUTDIR, PIXEL_TO_UM, time_cols_from() defined in calling scope.
         velocity_wide, tailbeat_amp_wide, window_features, length_per_animal
         available as DataFrames.
HOUSE_CONDITION: condition label for non-propagating indoor animals; excluded
                 from developmental trajectory comparisons (panels C).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
from itertools import combinations


# =======================
# Config
# =======================

SIZE_CORR_OUTDIR = os.path.join(os.path.dirname(PLOT_OUTDIR), "size_correction")
os.makedirs(SIZE_CORR_OUTDIR, exist_ok=True)

STAGE_ORDER     = ["larva", "juvenile", "adult"]
STAGE_RANK      = {s: i for i, s in enumerate(STAGE_ORDER)}
STAGE_PAL       = dict(zip(STAGE_ORDER, sns.color_palette("Set2", n_colors=3)))
KCOLS           = ["Experiment_ID", "Stage", "Condition", "Individual"]
HOUSE_CONDITION = "house"

# Accumulates all pairwise test records for the final stat table
_STAT_RECORDS: list[dict] = []


# =======================
# Statistics Helpers
# =======================

def _p_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _pairwise_mwu(df: pd.DataFrame, group_col: str, groups: list,
                  metric: str, context: str) -> list[dict]:
    """All pairwise two-sided MWU; appends to _STAT_RECORDS."""
    records = []
    for g1, g2 in combinations(groups, 2):
        a = df.loc[df[group_col] == g1, "Value"].dropna().to_numpy(float)
        b = df.loc[df[group_col] == g2, "Value"].dropna().to_numpy(float)
        if a.size < 2 or b.size < 2:
            continue
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        rec  = dict(metric=metric, context=context,
                    group_A=g1, group_B=g2,
                    n_A=a.size, n_B=b.size,
                    U=U, p_value=p, stars=_p_stars(p))
        records.append(rec)
        _STAT_RECORDS.append(rec)
    return records


def _add_brackets(ax: plt.Axes, groups_order: list, records: list[dict],
                  y_top: float, y_range: float,
                  step_frac: float = 0.12) -> None:
    """Stacked significance brackets; shorter spans drawn lower."""
    if not records:
        return
    pos  = {g: i for i, g in enumerate(groups_order)}
    step = y_range * step_frac
    ceil = {i: y_top for i in range(len(groups_order))}

    for rec in sorted(records,
                      key=lambda r: abs(pos.get(r["group_B"], 0) -
                                        pos.get(r["group_A"], 0))):
        g1, g2 = rec["group_A"], rec["group_B"]
        if g1 not in pos or g2 not in pos:
            continue
        x1, x2 = sorted([pos[g1], pos[g2]])
        base    = max(ceil[c] for c in range(x1, x2 + 1)) + step * 0.4
        tip     = base + step * 0.35
        ax.plot([x1, x1, x2, x2], [base, tip, tip, base],
                lw=1.0, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, tip + step * 0.05,
                rec["stars"], ha="center", va="bottom", fontsize=8)
        new = tip + step * 0.35
        for c in range(x1, x2 + 1):
            ceil[c] = new

    ax.set_ylim(ax.get_ylim()[0], max(ceil.values()) + step * 0.5)


# =======================
# Growth Curve
# =======================

def _logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def plot_growth_curve(length_per_animal: pd.DataFrame, outdir: str) -> None:
    """Scatter + logistic fit of body length across stages, pairwise MWU brackets."""
    df = length_per_animal.copy()
    df = df[df["Stage"].isin(STAGE_ORDER) & np.isfinite(df["mean_length_um"])]
    if df.empty:
        print("[growth_curve] no data"); return

    df["stage_rank"] = df["Stage"].map(STAGE_RANK).astype(float)
    x, y   = df["stage_rank"].to_numpy(float), df["mean_length_um"].to_numpy(float)
    p0     = [y.max(), 2.0, 1.0]
    bounds = ([0, 0, -1], [y.max() * 3, 20, 3])

    try:
        popt, pcov = curve_fit(_logistic, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError as e:
        print(f"[growth_curve] logistic failed ({e}), using poly2")
        popt, pcov = np.polyfit(x, y, 2), None

    xs = np.linspace(-0.3, 2.3, 300)
    if pcov is not None:
        ys  = _logistic(xs, *popt)
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(x), len(x))
            try:
                pb, _ = curve_fit(_logistic, x[idx], y[idx],
                                  p0=popt, bounds=bounds, maxfev=2000)
                boots.append(_logistic(xs, *pb))
            except Exception:
                pass
        ci_lo, ci_hi = (np.percentile(np.array(boots), [2.5, 97.5], axis=0)
                        if boots else (ys, ys))
    else:
        ys = np.polyval(popt, xs)
        ci_lo = ci_hi = ys

    agg     = df[["Stage"]].assign(Value=df["mean_length_um"])
    present = [s for s in STAGE_ORDER if s in set(agg["Stage"])]
    pw      = _pairwise_mwu(agg, "Stage", present, "body_length_um", "growth_curve")

    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in STAGE_ORDER:
        sub = df[df["Stage"] == stage]
        if sub.empty: continue
        ax.scatter(STAGE_RANK[stage] + np.random.normal(0, 0.05, len(sub)),
                   sub["mean_length_um"],
                   color=STAGE_PAL[stage], alpha=0.75, s=28,
                   edgecolors="none", label=stage, zorder=3)

    ax.plot(xs, ys, color="black", lw=2,
            label="Logistic fit" if pcov is not None else "Poly-2 fit")
    if ci_lo is not ci_hi:
        ax.fill_between(xs, ci_lo, ci_hi, color="black", alpha=0.12,
                        label="95% bootstrap CI")

    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax.set_xlabel("Developmental stage", fontsize=12)
    ax.set_ylabel("Mean body length (µm)", fontsize=12)
    ax.set_title("Body length across developmental stages", fontsize=13)
    ax.legend(framealpha=0.7, fontsize=9)
    if pcov is not None:
        L, k, x0 = popt
        ax.text(0.03, 0.97, f"L={L:.0f} µm  k={k:.2f}  x₀={x0:.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color="gray")

    _add_brackets(ax, present, pw,
                  y_top=y.max(), y_range=y.max() - y.min())
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(outdir, f"body_length_growth_curve.{ext}"), dpi=300)
    plt.show()


# =======================
# Body-Length Correction
# =======================

def _build_lookup(length_per_animal: pd.DataFrame) -> dict:
    """(Experiment_ID, Individual) → mean_length_um."""
    return {(r["Experiment_ID"], r["Individual"]): r["mean_length_um"]
            for _, r in length_per_animal.iterrows()
            if np.isfinite(r["mean_length_um"])}


def correct_wide(wide_df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Divide all Time_ columns by individual body length (µm → BL units)."""
    if wide_df is None or wide_df.empty: return pd.DataFrame()
    tcols = time_cols_from(wide_df)
    if not tcols: return wide_df.copy()
    out   = wide_df.copy()
    vals  = out[tcols].to_numpy(float)
    scale = np.array([lookup.get((r.get("Experiment_ID"), r.get("Individual")), np.nan)
                      for _, r in out.iterrows()])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[tcols] = vals / scale[:, np.newaxis]
    return out


# Columns normalised; msd_slope divided by BL_px² to keep BL²/s units
_WINDOW_COLS = ["vel_p95", "vel_max", "acc_p95", "msd_slope"]

def correct_window_features(window_features: pd.DataFrame, lookup: dict,
                             pixel_to_um: float = PIXEL_TO_UM) -> pd.DataFrame:
    """Add <col>_bl columns (BL-normalised) to window_features."""
    if window_features is None or window_features.empty: return pd.DataFrame()
    out = window_features.copy()
    out["_bl_um"] = out.apply(
        lambda r: lookup.get((r.get("Experiment_ID"), r.get("Individual")), np.nan),
        axis=1)
    out["_bl_px"] = out["_bl_um"] / pixel_to_um
    for col in _WINDOW_COLS:
        if col not in out.columns: continue
        with np.errstate(invalid="ignore", divide="ignore"):
            if col == "msd_slope":
                out[f"{col}_bl"] = out[col] / out["_bl_px"] ** 2
            else:
                out[f"{col}_bl"] = out[col] / out["_bl_um"]
    return out.drop(columns=["_bl_um", "_bl_px"], errors="ignore")


# =======================
# Plot Helpers
# =======================

def _per_animal_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df[KCOLS + [col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d[col])]
    if d.empty: return pd.DataFrame(columns=KCOLS + ["Value"])
    return (d.groupby(KCOLS, dropna=False, observed=False)[col]
             .mean().reset_index().rename(columns={col: "Value"}))


def _add_jitter(ax, df, group_col, groups_order):
    for i, g in enumerate(groups_order):
        yv = df.loc[df[group_col] == g, "Value"].to_numpy(float)
        if yv.size:
            ax.scatter(np.random.normal(i, 0.05, yv.size), yv,
                       s=18, alpha=0.9, edgecolors="none", c="black")


def _y_limits(ax, arr, pad_frac=0.05):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return
    y0, y1 = arr.min(), arr.max()
    pad = pad_frac * (y1 - y0) if y1 > y0 else max(abs(y1) * 0.1, 0.01)
    ax.set_ylim(y0 - pad, y1 + pad)


_STAR_LEGEND = "* p<0.05   ** p<0.01   *** p<0.001   ns = n.s."


def _boxplot(agg_df: pd.DataFrame, group_col: str, groups_order: list,
             palette_map: dict, title: str, ylabel: str, fname: str,
             outdir: str, metric: str = "", context: str = "") -> None:
    """Single-panel boxplot with all pairwise MWU brackets."""
    d = agg_df[agg_df[group_col].isin(groups_order)].copy()
    d = d[np.isfinite(d["Value"])]
    if d.empty: print(f"[skip] {fname}"); return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=d, x=group_col, y="Value",
                order=groups_order, hue=group_col, hue_order=groups_order,
                showfliers=False, linewidth=1.2, palette=palette_map, ax=ax)
    leg = ax.get_legend()
    if leg: leg.remove()
    _add_jitter(ax, d, group_col, groups_order)

    yall    = d["Value"].to_numpy(float)
    _y_limits(ax, yall)
    present = [g for g in groups_order if g in set(d[group_col].dropna())]
    pw      = _pairwise_mwu(d, group_col, present, metric or ylabel, context or fname)
    yall    = yall[np.isfinite(yall)]
    if yall.size:
        y_rng = yall.max() - yall.min() or max(abs(yall.max()) * 0.2, 0.05)
        _add_brackets(ax, present, pw, y_top=yall.max(), y_range=y_rng)

    ax.text(0.02, 0.01, _STAR_LEGEND,
            transform=ax.transAxes, fontsize=7, color="gray", va="bottom")
    ax.set_xlabel(group_col); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=300)
    plt.show()


def _side_by_side_box(raw_agg: pd.DataFrame, corr_agg: pd.DataFrame,
                      stage_order: list, palette_map: dict,
                      title: str, ylabel_raw: str, ylabel_corr: str,
                      fname: str, outdir: str,
                      metric: str = "") -> None:
    """Two-panel raw | BL-corrected boxplot with pairwise MWU on each panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [(raw_agg,  ylabel_raw,  "Raw",                f"{metric}_raw"),
             (corr_agg, ylabel_corr, "Size-corrected (BL)", f"{metric}_corrected")]

    for ax, (agg, ylabel, suffix, ctx) in zip(axes, pairs):
        d = agg[agg["Stage"].isin(stage_order)].copy()
        d = d[np.isfinite(d["Value"])]
        if d.empty: ax.set_title(f"{suffix} — no data"); continue

        sns.boxplot(data=d, x="Stage", y="Value",
                    order=stage_order, hue="Stage", hue_order=stage_order,
                    showfliers=False, linewidth=1.2, palette=palette_map, ax=ax)
        leg = ax.get_legend()
        if leg: leg.remove()
        _add_jitter(ax, d, "Stage", stage_order)

        yall    = d["Value"].to_numpy(float)
        _y_limits(ax, yall)
        present = [s for s in stage_order if s in set(d["Stage"].dropna())]
        pw      = _pairwise_mwu(d, "Stage", present, metric or ylabel, ctx)
        yall    = yall[np.isfinite(yall)]
        if yall.size:
            y_rng = yall.max() - yall.min() or max(abs(yall.max()) * 0.2, 0.05)
            _add_brackets(ax, present, pw, y_top=yall.max(), y_range=y_rng)

        ax.text(0.02, 0.01, _STAR_LEGEND,
                transform=ax.transAxes, fontsize=7, color="gray", va="bottom")
        ax.set_xlabel("Stage"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n({suffix})", fontsize=10)

    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=300)
    plt.show()


# =======================
# Stat Table Export
# =======================

def _save_stat_table(outdir: str) -> pd.DataFrame:
    """Collect all MWU records, apply Bonferroni correction, save CSV + PNG."""
    if not _STAT_RECORDS:
        print("[stat_table] no records"); return pd.DataFrame()

    tbl              = pd.DataFrame(_STAT_RECORDS)
    n                = len(tbl)
    tbl["p_bonf"]    = (tbl["p_value"] * n).clip(upper=1.0)
    tbl["stars_bonf"] = tbl["p_bonf"].apply(_p_stars)
    tbl              = tbl.sort_values(["metric", "context", "group_A", "group_B"]).reset_index(drop=True)

    csv_path = os.path.join(outdir, "pairwise_mwu_stats.csv")
    tbl.to_csv(csv_path, index=False)
    print(f"[stat_table] {len(tbl)} tests → {csv_path}")

    disp = tbl[["metric", "context", "group_A", "group_B",
                "n_A", "n_B", "U", "p_value", "stars", "p_bonf", "stars_bonf"]].copy()
    disp["p_value"] = disp["p_value"].map(lambda v: f"{v:.3e}")
    disp["p_bonf"]  = disp["p_bonf"].map(lambda v: f"{v:.3e}")
    disp["U"]       = disp["U"].map(lambda v: f"{v:.0f}")

    col_labels = ["Metric", "Context", "Group A", "Group B",
                  "n A", "n B", "U", "p (raw)", "★", "p (Bonf.)", "★ Bonf."]
    nrows, ncols = len(disp), len(col_labels)
    fig, ax = plt.subplots(figsize=(ncols * 1.45, max(3.0, nrows * 0.35 + 1.2)))
    ax.axis("off")
    t = ax.table(cellText=disp.values, colLabels=col_labels,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8)
    t.auto_set_column_width(list(range(ncols)))
    for j in range(ncols):
        t[(0, j)].set_facecolor("#2c3e50")
        t[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, row in tbl.iterrows():
        bg = "#fffde7" if row["p_value"] < 0.05 else "white"
        for j in range(ncols):
            t[(i + 1, j)].set_facecolor(bg)

    plt.title("Pairwise Mann-Whitney U  (Bonferroni corrected)",
              fontsize=10, pad=10, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(outdir, "pairwise_mwu_stats_table.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    return tbl


# =======================
# Run
# =======================

def _wide_agg(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Per-row mean across Time_ columns, retaining KCOLS."""
    if wide_df is None or wide_df.empty:
        return pd.DataFrame(columns=KCOLS + ["Value"])
    tcols = time_cols_from(wide_df)
    if not tcols: return pd.DataFrame(columns=KCOLS + ["Value"])
    out          = wide_df[KCOLS].copy()
    out["Value"] = np.nanmean(wide_df[tcols].to_numpy(float), axis=1)
    return out[np.isfinite(out["Value"])]


# A. Growth curve
plot_growth_curve(length_per_animal, SIZE_CORR_OUTDIR)

# Build lookup and correct all metrics
bl_lookup          = _build_lookup(length_per_animal)
velocity_wide_bl   = correct_wide(velocity_wide,     bl_lookup)
tbamp_wide_bl      = correct_wide(tailbeat_amp_wide, bl_lookup)
window_features_bl = correct_window_features(window_features, bl_lookup)
print(f"[size_correction] lookup: {len(bl_lookup)} individuals")

# B. Speed raw vs corrected — all stages
raw_vel  = _wide_agg(velocity_wide)
corr_vel = _wide_agg(velocity_wide_bl)
stages   = [s for s in STAGE_ORDER if s in set(raw_vel["Stage"].dropna())]
if stages:
    _side_by_side_box(raw_vel, corr_vel, stage_order=stages,
                      palette_map=STAGE_PAL,
                      title="Locomotion speed",
                      ylabel_raw="Speed (µm/s)", ylabel_corr="Speed (BL/s)",
                      fname="velocity_raw_vs_corrected_by_stage",
                      outdir=SIZE_CORR_OUTDIR, metric="speed")

# C. Propagation metrics — outside-house animals, BL-corrected, by stage
# (house adults excluded: minimal propagation skews the developmental trajectory)
WINDOW_METRICS = [
    ("vel_p95_bl",   "Speed p95 (BL/s)"),
    ("vel_max_bl",   "Speed max (BL/s)"),
    ("acc_p95_bl",   "Acceleration p95 (BL/s²)"),
    ("msd_slope_bl", "MSD slope (BL²/s)"),
]

for col, label in WINDOW_METRICS:
    if col not in window_features_bl.columns:
        print(f"[skip] {col}"); continue
    agg = _per_animal_mean(window_features_bl, col)
    agg = agg[agg["Value"] >= 0]
    if agg.empty: continue

    out_agg = agg[agg["Condition"] != HOUSE_CONDITION].copy()
    present = [s for s in STAGE_ORDER if s in set(out_agg["Stage"].dropna())]
    if len(present) < 2: continue

    _boxplot(out_agg, "Stage", present, STAGE_PAL,
             title=f"{label} — outside-house by Stage",
             ylabel=label,
             fname=f"{col}_outside_house_by_stage",
             outdir=SIZE_CORR_OUTDIR,
             metric=col, context="outside_house_by_stage")

# Tail-beat amplitude — raw vs corrected, then outside-house trajectory
raw_tb  = _wide_agg(tailbeat_amp_wide)
corr_tb = _wide_agg(tbamp_wide_bl)
tb_stgs = [s for s in STAGE_ORDER if s in set(raw_tb["Stage"].dropna())]
if tb_stgs:
    _side_by_side_box(raw_tb, corr_tb, stage_order=tb_stgs,
                      palette_map=STAGE_PAL,
                      title="Tail-beat amplitude",
                      ylabel_raw="Amplitude (px)",
                      ylabel_corr="Amplitude (BL fraction)",
                      fname="tbamp_raw_vs_corrected_by_stage",
                      outdir=SIZE_CORR_OUTDIR, metric="tailbeat_amplitude")

    tb_out   = corr_tb[corr_tb["Condition"] != HOUSE_CONDITION].copy()
    tb_pres  = [s for s in STAGE_ORDER if s in set(tb_out["Stage"].dropna())]
    if len(tb_pres) >= 2:
        _boxplot(tb_out, "Stage", tb_pres, STAGE_PAL,
                 title="Tail-beat amplitude (BL fraction) — outside-house by Stage",
                 ylabel="Amplitude (BL fraction)",
                 fname="tbamp_corrected_outside_house_by_stage",
                 outdir=SIZE_CORR_OUTDIR,
                 metric="tailbeat_amplitude_bl", context="outside_house_by_stage")

# D. Stat table
stat_table = _save_stat_table(SIZE_CORR_OUTDIR)

# Save corrected DataFrames
for df, fn in [(velocity_wide_bl,   "velocity_corrected_bl.csv"),
               (tbamp_wide_bl,       "tailbeat_amp_corrected_bl.csv"),
               (window_features_bl,  "window_features_corrected_bl.csv")]:
    if df is not None and not df.empty:
        df.to_csv(os.path.join(SIZE_CORR_OUTDIR, fn), index=False)

print(f"\n[size_correction] outputs → {SIZE_CORR_OUTDIR}")