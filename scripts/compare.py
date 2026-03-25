#!/usr/bin/env python3
"""
compare.py — Cross-model comparison report for the benchmarking paper.

Reads eval_results.json from each model's output directory and produces:
  1. Summary table (models × metrics) with mean ± 95% CI
  2. Common-subset evaluation (only prompts all models compiled successfully)
  3. Pairwise significance tests (Wilcoxon signed-rank with Holm correction)
  4. Per-category heatmap
  5. HTML report

Usage:
    python scripts/compare.py
    python scripts/compare.py --models deepseek-v3 deepseek-r1 gpt-oss
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats


METRICS = [
    ("dists", "DISTS", "lower"),
    ("clip_sim", "CLIP Sim", "higher"),
    ("edge_iou", "Edge IoU", "higher"),
    ("edge_f1", "Edge F1", "higher"),
]


def load_model_results(model_dir):
    path = os.path.join(model_dir, "eval_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def safe_stat(values):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return {"mean": None, "std": None, "ci95": None, "n": len(valid)}
    m = float(np.mean(valid))
    sd = float(np.std(valid, ddof=1))
    se = scipy_stats.sem(valid)
    ci = float(se * scipy_stats.t.ppf(0.975, len(valid) - 1))
    return {"mean": m, "std": sd, "ci95": ci, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser(description="Cross-model comparison report")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to compare (default: all found in outputs/)")
    parser.add_argument("--output", default="outputs/comparison_report.html")
    args = parser.parse_args()

    # Discover models
    if args.models:
        models = args.models
    else:
        models = sorted(d for d in os.listdir("outputs")
                        if os.path.isfile(os.path.join("outputs", d, "eval_results.json")))

    if not models:
        print("No model results found in outputs/")
        return

    print(f"Comparing {len(models)} models: {', '.join(models)}")

    # Load all results
    all_results = {}
    for model in models:
        data = load_model_results(os.path.join("outputs", model))
        if data:
            all_results[model] = data
        else:
            print(f"  WARNING: No results for {model}, skipping")

    models = list(all_results.keys())
    if len(models) < 2:
        print("Need at least 2 models to compare")
        return

    # Build per-image lookup: {model: {image_id: scores}}
    per_image = {}
    for model, data in all_results.items():
        per_image[model] = {r["image_id"]: r for r in data["per_image"]}

    # --- Common subset: images ALL models compiled successfully ---
    common_ids = None
    for model in models:
        ids = set(per_image[model].keys())
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = sorted(common_ids)
    print(f"Common subset: {len(common_ids)} images (compiled by ALL {len(models)} models)")

    # --- Generation stats ---
    gen_stats = {}
    for model in models:
        log_path = os.path.join("outputs", model, "generation_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                gen_stats[model] = json.load(f).get("stats", {})
        else:
            gen_stats[model] = {}

    # --- Summary table (all pairs) ---
    summary_all = {}
    for model in models:
        summary_all[model] = {}
        for metric_key, _, _ in METRICS:
            vals = [per_image[model][iid].get(metric_key) for iid in per_image[model]]
            summary_all[model][metric_key] = safe_stat(vals)

    # --- Summary table (common subset only) ---
    summary_common = {}
    for model in models:
        summary_common[model] = {}
        for metric_key, _, _ in METRICS:
            vals = [per_image[model][iid].get(metric_key) for iid in common_ids]
            summary_common[model][metric_key] = safe_stat(vals)

    # --- Pairwise significance tests (on common subset) ---
    sig_tests = {}
    for metric_key, metric_name, direction in METRICS:
        sig_tests[metric_key] = {}
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i >= j:
                    continue
                v1 = [per_image[m1][iid].get(metric_key) for iid in common_ids]
                v2 = [per_image[m2][iid].get(metric_key) for iid in common_ids]
                # Remove pairs where either is None
                pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
                if len(pairs) < 10:
                    continue
                a, b = zip(*pairs)
                try:
                    stat, p = scipy_stats.wilcoxon(a, b)
                    sig_tests[metric_key][(m1, m2)] = {"stat": float(stat), "p": float(p), "n": len(pairs)}
                except Exception:
                    pass

        # Holm-Bonferroni correction
        all_pairs = list(sig_tests[metric_key].keys())
        p_values = [sig_tests[metric_key][k]["p"] for k in all_pairs]
        if p_values:
            sorted_idx = np.argsort(p_values)
            m = len(p_values)
            for rank, idx in enumerate(sorted_idx):
                adjusted = min(p_values[idx] * (m - rank), 1.0)
                sig_tests[metric_key][all_pairs[idx]]["p_adjusted"] = adjusted

    # --- Per-category breakdown (common subset) ---
    cat_breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model in models:
        for iid in common_ids:
            r = per_image[model][iid]
            cat = r.get("category", "unknown")
            for metric_key, _, _ in METRICS:
                v = r.get(metric_key)
                if v is not None:
                    cat_breakdown[cat][model][metric_key].append(v)

    # --- Language distribution per model ---
    lang_dist = {}
    for model in models:
        from collections import Counter
        langs = Counter(r.get("code_language", "unknown") for r in all_results[model]["per_image"])
        lang_dist[model] = dict(langs)

    # --- Build HTML ---
    def fv(val, d=4):
        return f"{val:.{d}f}" if val is not None else "N/A"

    def fmt_stat(s, d=4):
        if s["mean"] is None:
            return "N/A"
        if s["ci95"] is not None:
            return f"{s['mean']:.{d}f} <small>±{s['ci95']:.{d}f}</small>"
        return f"{s['mean']:.{d}f}"

    # Summary rows (all pairs)
    summary_rows_all = []
    for model in models:
        gs = gen_stats.get(model, {})
        total = gs.get("success", 0) + gs.get("compile_error", 0) + gs.get("api_error", 0)
        success = gs.get("success", 0)
        rate = f"{success/total*100:.0f}%" if total > 0 else "N/A"
        n_eval = summary_all[model][METRICS[0][0]]["n"]

        cols = f"<td>{model}</td><td>{total}</td><td>{success}</td><td>{rate}</td><td>{n_eval}</td>"
        for mk, _, direction in METRICS:
            s = summary_all[model][mk]
            cols += f"<td>{fmt_stat(s)}</td>"

        # Language
        ld = lang_dist.get(model, {})
        lang_str = ", ".join(f"{k}: {v}" for k, v in sorted(ld.items(), key=lambda x: -x[1]))
        cols += f"<td style='font-size:0.75rem'>{lang_str}</td>"

        summary_rows_all.append(f"<tr>{cols}</tr>")

    # Summary rows (common subset)
    summary_rows_common = []
    for model in models:
        cols = f"<td>{model}</td><td>{len(common_ids)}</td>"
        for mk, _, direction in METRICS:
            s = summary_common[model][mk]
            cols += f"<td>{fmt_stat(s)}</td>"
        summary_rows_common.append(f"<tr>{cols}</tr>")

    # Significance rows
    sig_rows = []
    for mk, mname, direction in METRICS:
        for (m1, m2), result in sorted(sig_tests.get(mk, {}).items()):
            p = result.get("p_adjusted", result["p"])
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            sig_rows.append(f"<tr><td>{mname}</td><td>{m1}</td><td>{m2}</td><td>{fv(result['p'], 6)}</td><td>{fv(p, 6)}</td><td>{stars}</td><td>{result['n']}</td></tr>")

    # Category rows
    all_cats = sorted(cat_breakdown.keys())
    cat_rows = []
    for cat in all_cats:
        for model in models:
            cols = f"<td>{cat}</td><td>{model}</td>"
            for mk, _, _ in METRICS:
                vals = cat_breakdown[cat][model].get(mk, [])
                cols += f"<td>{fv(np.mean(vals)) if vals else 'N/A'}</td>"
            cat_rows.append(f"<tr>{cols}</tr>")

    metric_headers_all = "".join(f"<th>{mn} {'↓' if d=='lower' else '↑'}</th>" for _, mn, d in METRICS)
    metric_headers = metric_headers_all

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cross-Model Comparison Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; color: #1a1a2e; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; color: #16213e; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #0f3460; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 2rem; }}
  th {{ background: #1e293b; color: white; padding: 0.75rem 0.6rem; text-align: center; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; position: sticky; top: 0; }}
  td {{ padding: 0.6rem; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem; text-align: center; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  tr:hover {{ background: #f8fafc; }}
  small {{ color: #94a3b8; }}
  .note {{ font-size: 0.82rem; color: #64748b; margin-bottom: 1rem; }}
  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; font-size: 0.75rem; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Cross-Model Comparison</h1>
<p class="subtitle">{len(models)} models &middot; {len(common_ids)} common images &middot; Generated: March 2026</p>

<h2>1. Full Results (All Pairs per Model)</h2>
<p class="note">Each model evaluated on its own successful compilations. Sample sizes differ — use common-subset table for fair comparison.</p>
<table>
<thead>
<tr><th>Model</th><th>Attempted</th><th>Compiled</th><th>Rate</th><th>Evaluated</th>{metric_headers_all}<th>Languages</th></tr>
</thead>
<tbody>
{"".join(summary_rows_all)}
</tbody>
</table>

<h2>2. Common Subset (Fair Comparison)</h2>
<p class="note">Only the {len(common_ids)} images that ALL models compiled successfully. Metrics directly comparable.</p>
<table>
<thead>
<tr><th>Model</th><th>n</th>{metric_headers}</tr>
</thead>
<tbody>
{"".join(summary_rows_common)}
</tbody>
</table>

<h2>3. Pairwise Significance Tests</h2>
<p class="note">Wilcoxon signed-rank test on common subset. p-values adjusted with Holm-Bonferroni correction. *p&lt;0.05, **p&lt;0.01, ***p&lt;0.001</p>
<table>
<thead>
<tr><th>Metric</th><th>Model A</th><th>Model B</th><th>p (raw)</th><th>p (adjusted)</th><th>Sig</th><th>n pairs</th></tr>
</thead>
<tbody>
{"".join(sig_rows)}
</tbody>
</table>

<h2>4. Per-Category Breakdown (Common Subset)</h2>
<table>
<thead>
<tr><th>Category</th><th>Model</th>{metric_headers}</tr>
</thead>
<tbody>
{"".join(cat_rows)}
</tbody>
</table>

<div class="footer">
  <p>Metrics: CMMD (CLIP MMD, unbiased), DISTS (Deep Image Structure &amp; Texture Similarity), CLIP Similarity (CLIP ViT-B/32 cosine), Edge IoU &amp; Edge F1 (Canny edge overlap). All means shown with 95% CI (Welch). Significance via Wilcoxon signed-rank with Holm-Bonferroni correction.</p>
</div>

</body>
</html>"""

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nComparison report saved to {args.output} ({size_kb:.0f} KB)")

    # Print summary table to console
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL COMPARISON (common subset: n={len(common_ids)})")
    print(f"{'='*80}")
    header = f"{'Model':25s}"
    for _, mn, d in METRICS:
        header += f"  {mn:>15s}"
    print(header)
    print("-" * 80)
    for model in models:
        row = f"{model:25s}"
        for mk, _, _ in METRICS:
            s = summary_common[model][mk]
            if s["mean"] is not None and s["ci95"] is not None:
                row += f"  {s['mean']:>7.4f}±{s['ci95']:.4f}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
