"""Generate HTML report for benchmarking evaluation results."""

import json
import csv
import base64
import os


def img_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def fmt(val, decimals=3):
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def pct(val):
    return f"{val:.0%}" if val is not None else "N/A"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument("--model", default="deepseek-v3", help="Model name (matches outputs/<model>/ directory)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images in side-by-side (default: all). Shows top N/2 and bottom N/2 by DISTS.")
    args = parser.parse_args()

    model = args.model
    output_dir = f"outputs/{model}"

    # Load results
    with open(f"{output_dir}/eval_results.json") as f:
        data = json.load(f)

    with open("data/concise_prompts.csv") as f:
        reader = csv.DictReader(f)
        meta = {r["image_id"]: r for r in reader}

    with open(f"{output_dir}/generation_log.json") as f:
        gen_log = json.load(f)

    overall = data["overall"]
    by_cat = data["by_category"]
    per_image = data["per_image"]
    gen_stats = gen_log["stats"]

    total_attempted = gen_stats.get("success", 0) + gen_stats.get("compile_error", 0) + gen_stats.get("api_error", 0)
    compile_rate = gen_stats["success"] / total_attempted * 100 if total_attempted > 0 else 0

    # Format distribution (from generation log)
    fmt_counts = gen_log.get("format_counts", {})
    fmt_str = ", ".join(f"{k}: {v}" for k, v in sorted(fmt_counts.items(), key=lambda x: -x[1])) or "N/A"

    # Language distribution (from eval results, only successfully evaluated)
    lang_dist = overall.get("code_language_distribution", {})
    lang_str = ", ".join(f"{k}: {v}" for k, v in sorted(lang_dist.items(), key=lambda x: -x[1])) if lang_dist else fmt_str

    # Per-image rows (sorted by DISTS, best first)
    display_images = sorted(per_image, key=lambda x: x.get("dists") or 999)

    if args.max_images and len(display_images) > args.max_images:
        half = args.max_images // 2
        best = display_images[:half]
        worst = display_images[-half:]
        display_images = best + worst
        image_note = f"Showing {len(display_images)} of {len(per_image)} images (top {half} best + top {half} worst by DISTS)."
    else:
        image_note = ""

    rows_html = []
    for item in display_images:
        iid = item["image_id"]
        prompt = meta.get(iid, {}).get("concise_prompt", "N/A")
        cat = item.get("category", "unknown")

        gen_b64 = img_b64(f"{output_dir}/{iid}.png")
        gt_b64 = img_b64(f"data/ground_truth/{iid}.png")

        gen_img = f'<img src="data:image/png;base64,{gen_b64}" />' if gen_b64 else '<span class="na">Failed</span>'
        gt_img = f'<img src="data:image/png;base64,{gt_b64}" />' if gt_b64 else '<span class="na">N/A</span>'

        dists_val = item.get("dists") if item.get("dists") is not None else 1.0
        clip_val = item.get("clip_sim") if item.get("clip_sim") is not None else 0.0
        dists_class = "good" if dists_val < 0.25 else ("mid" if dists_val < 0.35 else "bad")
        clip_class = "good" if clip_val > 0.9 else ("mid" if clip_val > 0.8 else "bad")

        lang = item.get("code_language", "unknown")
        lang_class = f"lang-{lang}" if lang in ("tikz", "python", "svg") else "lang-unknown"

        rows_html.append(f"""<tr>
            <td class="id">{iid}</td>
            <td class="cat">{cat}</td>
            <td><span class="lang-badge {lang_class}">{lang}</span></td>
            <td class="img-cell">{gt_img}</td>
            <td class="img-cell">{gen_img}</td>
            <td class="metric {dists_class}">{fmt(item.get('dists'))}</td>
            <td class="metric {clip_class}">{fmt(item.get('clip_sim'))}</td>
            <td class="metric">{fmt(item.get('edge_iou'))}</td>
            <td class="metric">{fmt(item.get('edge_f1'))}</td>
            <td class="prompt">{prompt}</td>
        </tr>""")

    # Category rows
    cat_rows = []
    for cat, sc in sorted(by_cat.items(), key=lambda x: x[1]["n"], reverse=True):
        cat_rows.append(f"""<tr>
            <td>{cat}</td>
            <td>{sc['n']}</td>
            <td>{fmt(sc['dists_mean'])}</td>
            <td>{fmt(sc['clip_sim_mean'])}</td>
            <td>{fmt(sc['edge_iou_mean'])}</td>
            <td>{fmt(sc['edge_f1_mean'])}</td>
        </tr>""")

    # Model display name
    model_display = model.replace("-", " ").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_display} Benchmarking Report (n={len(per_image)})</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; color: #1a1a2e; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; color: #16213e; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #0f3460; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}

  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
  .stat-card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }}
  .stat-card .value {{ font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }}
  .stat-card .detail {{ font-size: 0.75rem; color: #94a3b8; }}
  .stat-card .value.blue {{ color: #2563eb; }}
  .stat-card .value.green {{ color: #16a34a; }}
  .stat-card .value.amber {{ color: #d97706; }}
  .stat-card .value.red {{ color: #dc2626; }}

  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 2rem; }}
  th {{ background: #1e293b; color: white; padding: 0.75rem 0.6rem; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; position: sticky; top: 0; }}
  td {{ padding: 0.6rem; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem; vertical-align: middle; }}
  tr:hover {{ background: #f8fafc; }}

  .img-cell {{ width: 160px; min-width: 160px; }}
  .img-cell img {{ width: 150px; height: 150px; object-fit: contain; border: 1px solid #e2e8f0; border-radius: 6px; background: white; }}
  .id {{ font-weight: 600; font-family: monospace; white-space: nowrap; }}
  .cat {{ font-size: 0.78rem; color: #475569; white-space: nowrap; }}
  .prompt {{ font-size: 0.78rem; color: #64748b; max-width: 400px; }}
  .metric {{ font-family: monospace; font-weight: 600; text-align: center; }}
  .good {{ color: #16a34a; background: #f0fdf4; }}
  .mid {{ color: #d97706; background: #fffbeb; }}
  .bad {{ color: #dc2626; background: #fef2f2; }}
  .na {{ color: #94a3b8; font-style: italic; }}

  .lang-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }}
  .lang-tikz {{ background: #dbeafe; color: #1e40af; }}
  .lang-python {{ background: #fef3c7; color: #92400e; }}
  .lang-svg {{ background: #d1fae5; color: #065f46; }}
  .lang-unknown {{ background: #f1f5f9; color: #64748b; }}

  .cat-table td, .cat-table th {{ text-align: center; }}
  .cat-table td:first-child {{ text-align: left; }}

  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; font-size: 0.75rem; color: #94a3b8; }}
</style>
</head>
<body>

<h1>{model_display} Diagram Generation Report</h1>
<p class="subtitle">Model: <strong>{model}</strong> &middot; {total_attempted} prompts &middot; {len(per_image)} evaluated &middot; Generated: March 2026</p>

<h2>Generation Summary</h2>
<div class="stats-grid">
  <div class="stat-card">
    <div class="label">Total Prompts</div>
    <div class="value blue">{total_attempted}</div>
  </div>
  <div class="stat-card">
    <div class="label">Compiled Successfully</div>
    <div class="value green">{gen_stats['success'] + gen_stats.get('cached', 0)}</div>
    <div class="detail">{compile_rate:.0f}% compile rate</div>
  </div>
  <div class="stat-card">
    <div class="label">Compile Errors</div>
    <div class="value amber">{gen_stats['compile_error']}</div>
  </div>
  <div class="stat-card">
    <div class="label">API Errors</div>
    <div class="value green">{gen_stats['api_error']}</div>
  </div>
  <div class="stat-card">
    <div class="label">Code Languages</div>
    <div class="value blue" style="font-size:1rem;">{lang_str}</div>
  </div>
</div>

<h2>Overall Metrics</h2>
<div class="stats-grid">
  <div class="stat-card">
    <div class="label">CMMD</div>
    <div class="value blue">{fmt(overall['cmmd'], 4)}</div>
    <div class="detail">lower = better</div>
  </div>
  <div class="stat-card">
    <div class="label">DISTS (mean)</div>
    <div class="value amber">{fmt(overall.get('dists_mean'), 4)}</div>
    <div class="detail">lower = better &middot; &plusmn;{fmt(overall.get('dists_ci95'), 4)}</div>
  </div>
  <div class="stat-card">
    <div class="label">CLIP Similarity (mean)</div>
    <div class="value green">{fmt(overall.get('clip_sim_mean'), 4)}</div>
    <div class="detail">higher = better &middot; &plusmn;{fmt(overall.get('clip_sim_ci95'), 4)}</div>
  </div>
  <div class="stat-card">
    <div class="label">Edge IoU (mean)</div>
    <div class="value red">{fmt(overall.get('edge_iou_mean'), 4)}</div>
    <div class="detail">higher = better &middot; &plusmn;{fmt(overall.get('edge_iou_ci95'), 4)}</div>
  </div>
  <div class="stat-card">
    <div class="label">Edge F1 (mean)</div>
    <div class="value amber">{fmt(overall.get('edge_f1_mean'), 4)}</div>
    <div class="detail">higher = better &middot; &plusmn;{fmt(overall.get('edge_f1_ci95'), 4)}</div>
  </div>
</div>

<h2>Per-Category Breakdown</h2>
<table class="cat-table">
<thead>
<tr><th>Category</th><th>n</th><th>DISTS &darr;</th><th>CLIP Sim &uarr;</th><th>Edge IoU &uarr;</th><th>Edge F1 &uarr;</th></tr>
</thead>
<tbody>
{"".join(cat_rows)}
</tbody>
</table>

<h2>Side-by-Side Comparisons</h2>
<p style="font-size:0.85rem; color:#64748b; margin-bottom:1rem;">Sorted by DISTS (best matches first). Color coding: <span style="color:#16a34a">green</span> = good, <span style="color:#d97706">amber</span> = moderate, <span style="color:#dc2626">red</span> = poor.{f' <strong>{image_note}</strong>' if image_note else ''}</p>
<table>
<thead>
<tr><th>ID</th><th>Category</th><th>Lang</th><th>Ground Truth</th><th>Generated</th><th>DISTS&darr;</th><th>CLIP&uarr;</th><th>Edge IoU&uarr;</th><th>Edge F1&uarr;</th><th>Prompt</th></tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>

<div class="footer">
  <p>Generated by benchmarking pipeline &middot; Metrics: CMMD (CLIP MMD, unbiased), DISTS (Deep Image Structure &amp; Texture Similarity), CLIP Similarity (CLIP ViT-B/32 cosine), Edge IoU &amp; Edge F1 (Canny edge overlap). All means shown with 95% CI.</p>
</div>

</body>
</html>"""

    out_path = f"{output_dir}/report.html"
    with open(out_path, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Report saved to {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
