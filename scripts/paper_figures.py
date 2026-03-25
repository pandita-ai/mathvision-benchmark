#!/usr/bin/env python3
"""
paper_figures.py — Generate publication-quality figures, tables, and data exports
for the LLM mathematical diagram generation benchmarking paper.

Reads eval_results.json + generation_log.json from each model's output directory.
Produces LaTeX tables, PDF figures, and CSV exports.

Usage:
    python scripts/paper_figures.py --outputs-dir outputs-gcp
    python scripts/paper_figures.py --outputs-dir outputs-gcp --figures fig1 fig2
    python scripts/paper_figures.py --outputs-dir outputs-gcp --paper-dir paper
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "deepseek-v3": "DeepSeek V3",
    "deepseek-r1": "DeepSeek R1",
    "gpt-5.4": "GPT-5.4",
    "gpt-oss": "GPT-OSS",
    "claude-opus-4.6": "Claude Opus 4.6",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "qwen3.5-35b": "Qwen3.5-35B",
    "llama-4-maverick": "Llama 4 Maverick",
}

MODEL_ORDER = list(MODEL_NAMES.keys())

METRICS = [
    ("dists", "DISTS", "lower"),
    ("clip_sim", "CLIP Sim", "higher"),
    ("edge_iou", "Edge IoU", "higher"),
    ("edge_f1", "Edge F1", "higher"),
]

PALETTE = sns.color_palette("colorblind", n_colors=len(MODEL_NAMES))
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_ORDER)}

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_models(outputs_dir):
    """Load eval_results.json and generation_log.json for all discovered models."""
    all_data = {}
    for model in MODEL_ORDER:
        model_dir = os.path.join(outputs_dir, model)
        eval_path = os.path.join(model_dir, "eval_results.json")
        gen_path = os.path.join(model_dir, "generation_log.json")

        if not os.path.exists(eval_path):
            print(f"  SKIP {model}: no eval_results.json")
            continue

        with open(eval_path) as f:
            eval_data = json.load(f)

        gen_data = None
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen_data = json.load(f)

        all_data[model] = {"eval": eval_data, "gen_log": gen_data}
        n = len(eval_data.get("per_image", []))
        print(f"  {model}: {n} images")

    return all_data


def _infer_dominant_format(gen_log):
    """Infer the dominant code format from a generation log's format_counts.

    For models with high cache rates, many images have 'unknown' code_language
    because they were generated in an earlier run. We use the known format
    distribution to label unknowns as the dominant format (with a flag).
    """
    if not gen_log:
        return None
    fmt_counts = gen_log.get("format_counts", {})
    if not fmt_counts:
        return None
    # Return the most common format
    return max(fmt_counts, key=fmt_counts.get)


def build_dataframe(all_data):
    """Convert per-image results into a single pandas DataFrame."""
    rows = []
    for model, data in all_data.items():
        # Build format lookup from generation log entries
        format_lookup = {}
        if data["gen_log"]:
            for entry in data["gen_log"].get("entries", []):
                iid = entry.get("image_id")
                fmt = entry.get("format_detected")
                if iid and fmt and fmt != "unknown":
                    format_lookup[iid] = fmt

        dominant_fmt = _infer_dominant_format(data["gen_log"])

        for item in data["eval"]["per_image"]:
            iid = item["image_id"]
            # Resolve code_language: try eval data → log entries → dominant format
            lang = item.get("code_language", "unknown")
            if lang == "unknown" and iid in format_lookup:
                lang = format_lookup[iid]
            if lang == "unknown" and dominant_fmt:
                lang = dominant_fmt

            row = {
                "model": model,
                "model_name": MODEL_NAMES.get(model, model),
                "image_id": iid,
                "category": item.get("category", "unknown"),
                "code_language": lang,
            }
            for mk, _, _ in METRICS:
                row[mk] = item.get(mk)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_common_subset(df):
    """Return DataFrame filtered to image_ids present in ALL models."""
    models = df["model"].unique()
    common_ids = None
    for model in models:
        ids = set(df[df["model"] == model]["image_id"])
        common_ids = ids if common_ids is None else common_ids & ids
    print(f"  Common subset: {len(common_ids)} images across {len(models)} models")
    return df[df["image_id"].isin(common_ids)].copy()


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------

def safe_stat(values):
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if len(valid) < 2:
        return {"mean": np.mean(valid) if valid else None, "std": None, "ci95": None, "n": len(valid)}
    m = float(np.mean(valid))
    sd = float(np.std(valid, ddof=1))
    se = scipy_stats.sem(valid)
    ci = float(se * scipy_stats.t.ppf(0.975, len(valid) - 1))
    return {"mean": m, "std": sd, "ci95": ci, "n": len(valid)}


# ---------------------------------------------------------------------------
# Table 1: Main Leaderboard
# ---------------------------------------------------------------------------

def tab1_leaderboard(df, common_df, gen_stats, paper_dir):
    """Generate LaTeX leaderboard table (both all-pairs and common-subset)."""

    for suffix, data in [("all", df), ("common", common_df)]:
        models_present = [m for m in MODEL_ORDER if m in data["model"].unique()]

        # Collect stats per model
        rows = []
        for model in models_present:
            mdf = data[data["model"] == model]
            gs = gen_stats.get(model, {})
            total = gs.get("success", 0) + gs.get("cached", 0) + gs.get("compile_error", 0) + gs.get("api_error", 0)
            success = gs.get("success", 0) + gs.get("cached", 0)
            rate = success / total * 100 if total > 0 else 0

            row = {"model": MODEL_NAMES[model], "n": len(mdf), "compile_rate": rate}
            for mk, _, _ in METRICS:
                s = safe_stat(mdf[mk].dropna().tolist())
                row[f"{mk}_mean"] = s["mean"]
                row[f"{mk}_ci"] = s["ci95"]
            rows.append(row)

        # Find best/second-best per metric
        for mk, _, direction in METRICS:
            vals = [(i, r[f"{mk}_mean"]) for i, r in enumerate(rows) if r[f"{mk}_mean"] is not None]
            if not vals:
                continue
            if direction == "lower":
                vals.sort(key=lambda x: x[1])
            else:
                vals.sort(key=lambda x: -x[1])
            if len(vals) >= 1:
                rows[vals[0][0]][f"{mk}_best"] = True
            if len(vals) >= 2:
                rows[vals[1][0]][f"{mk}_second"] = True

        # Also find best compile rate
        cr_vals = [(i, r["compile_rate"]) for i, r in enumerate(rows)]
        cr_vals.sort(key=lambda x: -x[1])
        if cr_vals:
            rows[cr_vals[0][0]]["cr_best"] = True
        if len(cr_vals) >= 2:
            rows[cr_vals[1][0]]["cr_second"] = True

        # Build LaTeX
        def fmt_cell(row, mk, direction):
            m = row.get(f"{mk}_mean")
            ci = row.get(f"{mk}_ci")
            if m is None:
                return "N/A"
            val = f"{m:.3f}"
            if ci is not None:
                val += f" {{\\scriptsize $\\pm${ci:.3f}}}"
            if row.get(f"{mk}_best"):
                val = f"\\textbf{{{val}}}"
            elif row.get(f"{mk}_second"):
                val = f"\\underline{{{val}}}"
            return val

        def fmt_cr(row):
            val = f"{row['compile_rate']:.1f}\\%"
            if row.get("cr_best"):
                val = f"\\textbf{{{val}}}"
            elif row.get("cr_second"):
                val = f"\\underline{{{val}}}"
            return val

        header = "Model & n & Compile & DISTS $\\downarrow$ & CLIP Sim $\\uparrow$ & Edge IoU $\\uparrow$ & Edge F1 $\\uparrow$"
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Model performance {'(common subset)' if suffix == 'common' else '(all evaluated pairs)'}. "
            "Bold = best, underline = second best. Mean $\\pm$ 95\\% CI.}}",
            f"\\label{{tab:leaderboard_{suffix}}}",
            "\\begin{tabular}{l r r r r r r}",
            "\\toprule",
            header + " \\\\",
            "\\midrule",
        ]

        for row in rows:
            cells = [
                row["model"],
                str(row["n"]),
                fmt_cr(row),
            ]
            for mk, _, direction in METRICS:
                cells.append(fmt_cell(row, mk, direction))
            lines.append(" & ".join(cells) + " \\\\")

        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]

        out_path = os.path.join(paper_dir, "tables", f"tab1_leaderboard_{suffix}.tex")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 1: Violin Plots
# ---------------------------------------------------------------------------

def fig1_distributions(df, common_df, paper_dir):
    """Metric distribution violin plots on common subset."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.5))
    models_present = [m for m in MODEL_ORDER if m in common_df["model"].unique()]
    model_names = [MODEL_NAMES[m] for m in models_present]

    for ax, (mk, label, direction) in zip(axes, METRICS):
        plot_df = common_df[common_df[mk].notna()][["model", mk]].copy()
        plot_df["model_name"] = plot_df["model"].map(MODEL_NAMES)

        sns.violinplot(
            data=plot_df, x="model_name", y=mk, order=model_names,
            palette=[MODEL_COLORS[m] for m in models_present],
            inner="box", linewidth=0.5, ax=ax, cut=0,
        )
        arrow = "↓" if direction == "lower" else "↑"
        ax.set_title(f"{label} {arrow}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig1_distributions.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Compile Rates
# ---------------------------------------------------------------------------

def fig2_compile_rates(gen_stats, paper_dir):
    """Stacked bar chart of compile rates per model."""
    models_present = [m for m in MODEL_ORDER if m in gen_stats]
    names = [MODEL_NAMES[m] for m in models_present]

    # Count cached as success (they were generated successfully in a prior run)
    success = [gen_stats[m].get("success", 0) + gen_stats[m].get("cached", 0) for m in models_present]
    compile_err = [gen_stats[m].get("compile_error", 0) for m in models_present]
    api_err = [gen_stats[m].get("api_error", 0) for m in models_present]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(names))
    w = 0.6

    ax.bar(x, success, w, label="Success", color="#2ecc71")
    ax.bar(x, compile_err, w, bottom=success, label="Compile Error", color="#e74c3c")
    ax.bar(x, api_err, w, bottom=[s + c for s, c in zip(success, compile_err)],
           label="API Error", color="#95a5a6")

    # Annotate compile rate
    for i, m in enumerate(models_present):
        total = success[i] + compile_err[i] + api_err[i]
        rate = success[i] / total * 100 if total > 0 else 0
        ax.text(i, total + 20, f"{rate:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Prompts")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Generation Success Rates")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig2_compile_rates.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Category Heatmap
# ---------------------------------------------------------------------------

def fig3_category_heatmap(common_df, paper_dir):
    """Heatmap of CLIP Similarity by category × model."""
    models_present = [m for m in MODEL_ORDER if m in common_df["model"].unique()]
    categories = sorted(common_df["category"].unique())

    matrix = []
    for cat in categories:
        row = []
        for model in models_present:
            vals = common_df[(common_df["category"] == cat) & (common_df["model"] == model)]["clip_sim"].dropna()
            row.append(vals.mean() if len(vals) > 0 else np.nan)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=[MODEL_NAMES[m] for m in models_present],
        yticklabels=categories,
        ax=ax, vmin=0.5, vmax=1.0,
        cbar_kws={"label": "CLIP Similarity ↑", "shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("CLIP Similarity by Category and Model")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig3_category_heatmap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Code Format Analysis
# ---------------------------------------------------------------------------

def fig4_format_analysis(df, paper_dir):
    """Format distribution per model + metrics by format."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Panel A: Format distribution per model
    formats = ["tikz", "python", "svg", "unknown"]
    format_colors = {"tikz": "#3498db", "python": "#f1c40f", "svg": "#2ecc71", "unknown": "#bdc3c7"}

    bottoms = np.zeros(len(models_present))
    for fmt in formats:
        counts = []
        for model in models_present:
            mdf = df[df["model"] == model]
            counts.append((mdf["code_language"] == fmt).sum())
        counts = np.array(counts)
        if counts.sum() > 0:
            ax1.bar(range(len(models_present)), counts, bottom=bottoms,
                    label=fmt.upper(), color=format_colors[fmt], width=0.6)
            bottoms += counts

    ax1.set_xticks(range(len(models_present)))
    ax1.set_xticklabels([MODEL_NAMES[m] for m in models_present], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Images")
    ax1.set_title("(a) Format Distribution")
    ax1.legend(fontsize=7, loc="upper right")

    # Panel B: Metrics by format (pooled across models)
    format_df = df[df["code_language"].isin(["tikz", "python", "svg"])].copy()
    if len(format_df) > 0:
        sns.boxplot(
            data=format_df, x="code_language", y="clip_sim",
            order=["tikz", "python", "svg"],
            palette=[format_colors[f] for f in ["tikz", "python", "svg"]],
            ax=ax2, linewidth=0.8, fliersize=2,
        )
        ax2.set_xlabel("Code Format")
        ax2.set_ylabel("CLIP Similarity ↑")
        ax2.set_title("(b) Quality by Format")

        # Kruskal-Wallis test
        groups = [format_df[format_df["code_language"] == f]["clip_sim"].dropna() for f in ["tikz", "python", "svg"]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            stat, p = scipy_stats.kruskal(*groups)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax2.text(0.95, 0.95, f"KW p={p:.2e} ({sig})", transform=ax2.transAxes,
                     ha="right", va="top", fontsize=7, style="italic")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig4_format_analysis.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Metric Correlations
# ---------------------------------------------------------------------------

def fig5_correlations(df, paper_dir):
    """Metric correlation matrix with scatter plots."""
    metric_keys = [mk for mk, _, _ in METRICS]
    metric_labels = [ml for _, ml, _ in METRICS]

    corr_df = df[metric_keys].dropna()
    if len(corr_df) < 10:
        print("  SKIP fig5: not enough data")
        return

    # Pearson correlation
    pearson = corr_df.corr(method="pearson")
    spearman = corr_df.corr(method="spearman")

    fig, axes = plt.subplots(len(metric_keys), len(metric_keys), figsize=(6.0, 6.0))

    for i in range(len(metric_keys)):
        for j in range(len(metric_keys)):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram
                ax.hist(corr_df[metric_keys[i]], bins=30, color=PALETTE[i], alpha=0.7, edgecolor="white")
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: scatter
                sample = corr_df.sample(min(2000, len(corr_df)), random_state=42)
                ax.scatter(sample[metric_keys[j]], sample[metric_keys[i]], s=1, alpha=0.3, color="#34495e")
            else:
                # Upper triangle: correlation values
                r_p = pearson.iloc[i, j]
                r_s = spearman.iloc[i, j]
                ax.text(0.5, 0.6, f"r={r_p:.2f}", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
                ax.text(0.5, 0.35, f"ρ={r_s:.2f}", transform=ax.transAxes, ha="center", fontsize=8, color="#666")
                ax.set_xticks([])
                ax.set_yticks([])

            if i == len(metric_keys) - 1:
                ax.set_xlabel(metric_labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(metric_labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Metric Correlations (Pearson r, Spearman ρ)", fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig5_correlations.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Significance Heatmap
# ---------------------------------------------------------------------------

def fig6_significance(common_df, paper_dir):
    """Pairwise significance heatmap using Wilcoxon signed-rank."""
    models_present = [m for m in MODEL_ORDER if m in common_df["model"].unique()]
    n_models = len(models_present)

    # Use CLIP Sim as primary metric
    mk = "clip_sim"
    image_ids = sorted(common_df["image_id"].unique())

    # Build pivot
    pivot = common_df.pivot_table(index="image_id", columns="model", values=mk)

    p_matrix = np.ones((n_models, n_models))
    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1, m2 = models_present[i], models_present[j]
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            paired = pivot[[m1, m2]].dropna()
            if len(paired) < 10:
                continue
            try:
                _, p = scipy_stats.wilcoxon(paired[m1], paired[m2])
            except Exception:
                p = 1.0
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    # Holm-Bonferroni correction
    upper_ps = []
    upper_idx = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            upper_ps.append(p_matrix[i, j])
            upper_idx.append((i, j))
    if upper_ps:
        sorted_order = np.argsort(upper_ps)
        m = len(upper_ps)
        adjusted = np.ones(m)
        for rank, idx in enumerate(sorted_order):
            adjusted[idx] = min(upper_ps[idx] * (m - rank), 1.0)
        for k, (i, j) in enumerate(upper_idx):
            p_matrix[i, j] = adjusted[k]
            p_matrix[j, i] = adjusted[k]

    # Determine direction (which model is better)
    mean_scores = {m: common_df[common_df["model"] == m][mk].mean() for m in models_present}
    direction_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            if p_matrix[i, j] < 0.05:
                direction_matrix[i, j] = 1 if mean_scores[models_present[i]] > mean_scores[models_present[j]] else -1

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    names = [MODEL_NAMES[m] for m in models_present]

    # Custom colormap: green (row better), red (col better), gray (ns)
    display = np.where(p_matrix < 0.05, -np.log10(p_matrix), 0)
    display = display * np.sign(direction_matrix)
    np.fill_diagonal(display, 0)

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    vmax = max(abs(display.max()), abs(display.min()), 1)
    sns.heatmap(
        display, xticklabels=names, yticklabels=names,
        cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=np.vectorize(lambda x: f"{x:.1f}" if abs(x) > 0 else "")(display),
        fmt="", ax=ax, linewidths=0.5,
        cbar_kws={"label": "−log₁₀(p) × direction", "shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Pairwise Significance (CLIP Sim, Holm-corrected)")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig6_significance.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 8: Category Difficulty
# ---------------------------------------------------------------------------

def fig8_difficulty(common_df, paper_dir):
    """Category difficulty ranking by average CLIP Sim across all models."""
    cat_means = common_df.groupby("category")["clip_sim"].agg(["mean", "std", "count"]).reset_index()
    cat_means = cat_means.sort_values("mean")

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(cat_means)))

    ax.barh(range(len(cat_means)), cat_means["mean"], xerr=cat_means["std"],
            color=colors, edgecolor="white", capsize=3, linewidth=0.5)
    ax.set_yticks(range(len(cat_means)))
    ax.set_yticklabels(cat_means["category"], fontsize=8)
    ax.set_xlabel("Mean CLIP Similarity ↑")
    ax.set_title("Category Difficulty (all models, common subset)")

    # Annotate n per category
    for i, (_, row) in enumerate(cat_means.iterrows()):
        ax.text(row["mean"] + row["std"] + 0.01, i, f"n={int(row['count'])}", va="center", fontsize=6, color="#666")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig8_difficulty.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 9: Radar Chart
# ---------------------------------------------------------------------------

def fig9_radar(df, gen_stats, paper_dir):
    """Radar chart comparing models across normalized metrics."""
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Compute values per model
    radar_data = {}
    for model in models_present:
        mdf = df[df["model"] == model]
        gs = gen_stats.get(model, {})
        total = gs.get("success", 0) + gs.get("compile_error", 0) + gs.get("api_error", 0)
        cr = gs.get("success", 0) / total if total > 0 else 0

        radar_data[model] = {
            "Compile Rate": cr,
            "1 - DISTS": 1 - (mdf["dists"].mean() if mdf["dists"].notna().any() else 1),
            "CLIP Sim": mdf["clip_sim"].mean() if mdf["clip_sim"].notna().any() else 0,
            "Edge IoU": mdf["edge_iou"].mean() if mdf["edge_iou"].notna().any() else 0,
            "Edge F1": mdf["edge_f1"].mean() if mdf["edge_f1"].notna().any() else 0,
        }

    axes_labels = list(next(iter(radar_data.values())).keys())
    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))

    for model in models_present:
        values = [radar_data[model][a] for a in axes_labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=MODEL_NAMES[model], color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.08, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7, color="#999")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    ax.set_title("Model Comparison Radar", y=1.08)

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig9_radar.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Table 2: Per-Category Breakdown
# ---------------------------------------------------------------------------

def tab2_categories(common_df, paper_dir):
    """LaTeX table of per-category metrics averaged across models."""
    categories = sorted(common_df["category"].unique())

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-category performance averaged across all models (common subset).}",
        "\\label{tab:categories}",
        "\\small",
        "\\begin{tabular}{l r r r r r}",
        "\\toprule",
        "Category & n & DISTS $\\downarrow$ & CLIP Sim $\\uparrow$ & Edge IoU $\\uparrow$ & Edge F1 $\\uparrow$ \\\\",
        "\\midrule",
    ]

    cat_stats = []
    for cat in categories:
        cdf = common_df[common_df["category"] == cat]
        n = len(cdf) // len(common_df["model"].unique())
        row = {"cat": cat, "n": n}
        for mk, _, _ in METRICS:
            row[mk] = cdf[mk].mean()
        cat_stats.append(row)

    # Sort by CLIP Sim (easiest first)
    cat_stats.sort(key=lambda x: -(x.get("clip_sim") or 0))

    for row in cat_stats:
        cells = [
            row["cat"],
            str(row["n"]),
        ]
        for mk, _, _ in METRICS:
            v = row.get(mk)
            cells.append(f"{v:.3f}" if v is not None else "N/A")
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = os.path.join(paper_dir, "tables", "tab2_categories.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 11: Quality vs Compile Rate Scatter
# ---------------------------------------------------------------------------

def fig11_scatter(df, gen_stats, paper_dir):
    """Scatter plot: compile rate vs mean CLIP Similarity."""
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique() and m in gen_stats]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for model in models_present:
        gs = gen_stats[model]
        total = gs.get("success", 0) + gs.get("compile_error", 0) + gs.get("api_error", 0)
        cr = gs.get("success", 0) / total * 100 if total > 0 else 0
        clip_mean = df[df["model"] == model]["clip_sim"].mean()

        ax.scatter(cr, clip_mean, s=80, color=MODEL_COLORS[model], zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(MODEL_NAMES[model], (cr, clip_mean), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Compile Rate (%)")
    ax.set_ylabel("Mean CLIP Similarity ↑")
    ax.set_title("Quality vs. Compilation Success")

    plt.tight_layout()
    out = os.path.join(paper_dir, "figures", "fig11_scatter.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Data Exports
# ---------------------------------------------------------------------------

def export_data(df, common_df, gen_stats, paper_dir):
    """Export flat CSVs and summary JSON for reproducibility."""
    data_dir = os.path.join(paper_dir, "data")

    # Full results CSV
    df.to_csv(os.path.join(data_dir, "full_results.csv"), index=False)
    print(f"  Saved {data_dir}/full_results.csv ({len(df)} rows)")

    # Common subset CSV
    common_df.to_csv(os.path.join(data_dir, "common_subset.csv"), index=False)
    print(f"  Saved {data_dir}/common_subset.csv ({len(common_df)} rows)")

    # Summary stats JSON
    summary = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        gs = gen_stats.get(model, {})
        total = gs.get("success", 0) + gs.get("compile_error", 0) + gs.get("api_error", 0)

        summary[model] = {
            "display_name": MODEL_NAMES.get(model, model),
            "total_prompts": total,
            "compiled": gs.get("success", 0) + gs.get("cached", 0),
            "compile_rate": (gs.get("success", 0) + gs.get("cached", 0)) / total if total > 0 else 0,
            "evaluated": len(mdf),
        }
        for mk, _, _ in METRICS:
            s = safe_stat(mdf[mk].dropna().tolist())
            summary[model][mk] = s

    with open(os.path.join(data_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {data_dir}/summary_stats.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--outputs-dir", default="outputs-gcp", help="Directory with model outputs")
    parser.add_argument("--paper-dir", default="paper", help="Output directory for paper assets")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--figures", nargs="*", default=None, help="Specific figures to generate (e.g., fig1 tab1)")
    args = parser.parse_args()

    # Create output dirs
    for subdir in ["figures", "tables", "data"]:
        os.makedirs(os.path.join(args.paper_dir, subdir), exist_ok=True)

    # Load data
    print("Loading model results...")
    all_data = load_all_models(args.outputs_dir)
    if not all_data:
        print("ERROR: No model results found")
        return

    # Build DataFrames
    print("Building DataFrames...")
    df = build_dataframe(all_data)
    common_df = compute_common_subset(df)

    # Load generation stats
    gen_stats = {}
    for model, data in all_data.items():
        if data["gen_log"]:
            gen_stats[model] = data["gen_log"].get("stats", {})

    # Determine which outputs to generate
    all_generators = {
        "tab1": ("Table 1: Leaderboard", lambda: tab1_leaderboard(df, common_df, gen_stats, args.paper_dir)),
        "fig1": ("Figure 1: Distributions", lambda: fig1_distributions(df, common_df, args.paper_dir)),
        "fig2": ("Figure 2: Compile Rates", lambda: fig2_compile_rates(gen_stats, args.paper_dir)),
        "fig3": ("Figure 3: Category Heatmap", lambda: fig3_category_heatmap(common_df, args.paper_dir)),
        "fig4": ("Figure 4: Format Analysis", lambda: fig4_format_analysis(df, args.paper_dir)),
        "fig5": ("Figure 5: Correlations", lambda: fig5_correlations(df, args.paper_dir)),
        "fig6": ("Figure 6: Significance", lambda: fig6_significance(common_df, args.paper_dir)),
        "fig8": ("Figure 8: Difficulty", lambda: fig8_difficulty(common_df, args.paper_dir)),
        "fig9": ("Figure 9: Radar", lambda: fig9_radar(df, gen_stats, args.paper_dir)),
        "tab2": ("Table 2: Categories", lambda: tab2_categories(common_df, args.paper_dir)),
        "fig11": ("Figure 11: Scatter", lambda: fig11_scatter(df, gen_stats, args.paper_dir)),
        "data": ("Data Exports", lambda: export_data(df, common_df, gen_stats, args.paper_dir)),
    }

    targets = args.figures if args.figures else list(all_generators.keys())

    print(f"\nGenerating {len(targets)} outputs...")
    for key in targets:
        if key not in all_generators:
            print(f"  UNKNOWN: {key}")
            continue
        name, func = all_generators[key]
        print(f"\n--- {name} ---")
        try:
            func()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Outputs in {args.paper_dir}/")


if __name__ == "__main__":
    main()
