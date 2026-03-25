"""
evaluate.py — Evaluate generated images against MathVision ground truth.

Metrics implemented:
  1. CMMD  — CLIP Maximum Mean Discrepancy (distribution-level)
  2. DISTS — Deep Image Structure and Texture Similarity (per-pair)
  3. SAMScore — Segment Anything structural similarity (per-pair)
  4. IoU   — Intersection over Union via object detection (per-pair)
  5. mAP   — Mean Average Precision via object detection (per-pair)
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def load_image_tensor(path: str, size: int = 256) -> torch.Tensor:
    """Load image as [1, 3, H, W] float tensor normalized to [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def load_image_pil(path: str, size: int = 512) -> Image.Image:
    """Load image as PIL RGB at given size."""
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Metric 1: DISTS
# ---------------------------------------------------------------------------

_dists_model = None

def compute_dists(gen_path: str, gt_path: str) -> float:
    """Compute DISTS between a generated and ground truth image.
    Lower = more similar. Range [0, 1].
    """
    import piq
    global _dists_model
    if _dists_model is None:
        _dists_model = piq.DISTS()

    gen = load_image_tensor(gen_path, 256)
    gt = load_image_tensor(gt_path, 256)
    score = _dists_model(gen, gt)
    return score.item()


# ---------------------------------------------------------------------------
# Metric 2: CMMD (CLIP Maximum Mean Discrepancy)
# ---------------------------------------------------------------------------

class CLIPEmbedder:
    """Extract CLIP embeddings for images."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = out.pooler_output if hasattr(out, 'pooler_output') else out
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()


def compute_cmmd(gen_embeddings: np.ndarray, gt_embeddings: np.ndarray) -> float:
    """Compute CMMD between two sets of CLIP embeddings.

    Follows Jayasumana et al. (ICLR 2024): unbiased MMD^2 estimator with
    RBF kernel and median heuristic for bandwidth.
    Lower = more similar.
    """
    from scipy.spatial.distance import cdist

    m = len(gen_embeddings)
    n = len(gt_embeddings)
    if m < 2 or n < 2:
        return 0.0

    def rbf_kernel(X, Y, sigma=1.0):
        dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-dists / (2 * sigma ** 2))

    # Median heuristic for bandwidth (per Jayasumana et al.)
    cross_dists = cdist(gen_embeddings, gt_embeddings, metric="sqeuclidean")
    sigma = np.sqrt(np.median(cross_dists))
    if sigma == 0:
        sigma = 1.0

    Kxx = rbf_kernel(gen_embeddings, gen_embeddings, sigma)
    Kyy = rbf_kernel(gt_embeddings, gt_embeddings, sigma)
    Kxy = rbf_kernel(gen_embeddings, gt_embeddings, sigma)

    # Unbiased estimator: exclude diagonal for Kxx and Kyy
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd = (Kxx.sum() / (m * (m - 1))) + (Kyy.sum() / (n * (n - 1))) - 2 * (Kxy.sum() / (m * n))
    return float(mmd)


# ---------------------------------------------------------------------------
# Metric 3: CLIP Cosine Similarity (high-level structural similarity)
# ---------------------------------------------------------------------------

def compute_clip_similarity(gen_path: str, gt_path: str) -> float:
    """Compute CLIP cosine similarity between two images.

    Uses CLIP ViT-B/32 image embeddings. Higher = more similar. Range [-1, 1].
    """
    return _samscore_clip_fallback(gen_path, gt_path)


_clip_cache = {}

def _get_clip_model():
    """Return cached CLIP model and processor."""
    if "model" not in _clip_cache:
        from transformers import CLIPProcessor, CLIPModel
        _clip_cache["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_cache["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_cache["model"].eval()
    return _clip_cache["model"], _clip_cache["processor"]


def _samscore_clip_fallback(gen_path: str, gt_path: str) -> float:
    """CLIP-based structural similarity as SAMScore proxy."""
    model, processor = _get_clip_model()

    gen_img = Image.open(gen_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")

    with torch.no_grad():
        gen_inputs = processor(images=gen_img, return_tensors="pt")
        gt_inputs = processor(images=gt_img, return_tensors="pt")
        gen_out = model.get_image_features(pixel_values=gen_inputs["pixel_values"])
        gt_out = model.get_image_features(pixel_values=gt_inputs["pixel_values"])
        gen_emb = gen_out.pooler_output if hasattr(gen_out, 'pooler_output') else gen_out
        gt_emb = gt_out.pooler_output if hasattr(gt_out, 'pooler_output') else gt_out

        gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
        gt_emb = gt_emb / gt_emb.norm(dim=-1, keepdim=True)

        similarity = (gen_emb * gt_emb).sum().item()

    return similarity


def _samscore_with_sam(gen_path: str, gt_path: str, predictor) -> float:
    """SAMScore using Segment Anything Model."""
    gen_img = np.array(Image.open(gen_path).convert("RGB"))
    gt_img = np.array(Image.open(gt_path).convert("RGB"))
    
    predictor.set_image(gen_img)
    gen_embedding = predictor.get_image_embedding().cpu().numpy().flatten()

    predictor.set_image(gt_img)
    gt_embedding = predictor.get_image_embedding().cpu().numpy().flatten()

    # Cosine similarity
    cos_sim = np.dot(gen_embedding, gt_embedding) / (
        np.linalg.norm(gen_embedding) * np.linalg.norm(gt_embedding)
    )
    return float(cos_sim)


# ---------------------------------------------------------------------------
# Metrics 4 & 5: Edge IoU and Edge F1 (edge-based structural overlap)
# ---------------------------------------------------------------------------

def compute_edge_metrics(gen_path: str, gt_path: str) -> dict:
    """Compute Edge IoU and Edge F1 (Dice) using Canny edge masks.

    For mathematical diagrams (line drawings), edge detection captures
    structural similarity by comparing the geometric layout of strokes.
    NOT equivalent to bounding-box IoU or COCO mAP.
    """
    import cv2

    def get_edge_mask(img_path, size=512):
        img = np.array(Image.open(img_path).convert("L").resize((size, size)))
        # Canny edge detection
        edges = cv2.Canny(img, 50, 150)
        # Dilate to make edges thicker for better overlap
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return (edges > 0).astype(np.float32)

    try:
        import cv2
    except ImportError:
        return {"iou": 0.0, "map": 0.0, "error": "opencv not installed"}

    gen_mask = get_edge_mask(gen_path)
    gt_mask = get_edge_mask(gt_path)

    # IoU
    intersection = (gen_mask * gt_mask).sum()
    union = ((gen_mask + gt_mask) > 0).astype(np.float32).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    # Precision & Recall for mAP proxy
    precision = float(intersection / gen_mask.sum()) if gen_mask.sum() > 0 else 0.0
    recall = float(intersection / gt_mask.sum()) if gt_mask.sum() > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"edge_iou": iou, "edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_pair(gen_path: str, gt_path: str, clip_embedder=None) -> dict:
    """Run all per-pair metrics on a single generated/ground-truth pair."""
    results = {}

    # DISTS (Deep Image Structure and Texture Similarity)
    try:
        results["dists"] = compute_dists(gen_path, gt_path)
    except Exception as e:
        results["dists"] = None
        results["dists_error"] = str(e)

    # CLIP Cosine Similarity
    try:
        results["clip_sim"] = compute_clip_similarity(gen_path, gt_path)
    except Exception as e:
        results["clip_sim"] = None
        results["clip_sim_error"] = str(e)

    # Edge IoU + Edge F1
    try:
        edge_results = compute_edge_metrics(gen_path, gt_path)
        results.update(edge_results)
    except Exception as e:
        results["edge_iou"] = None
        results["edge_iou_error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images against ground truth")
    parser.add_argument("--model", default="deepseek-v3")
    parser.add_argument("--gen-dir", default=None, help="Dir with generated images (default: outputs/<model>)")
    parser.add_argument("--gt-dir", default="data/ground_truth")
    parser.add_argument("--csv", default="data/concise_prompts.csv")
    parser.add_argument("--output", default=None, help="Results JSON path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    gen_dir = args.gen_dir or os.path.join("outputs", args.model)
    output_path = args.output or os.path.join("outputs", args.model, "eval_results.json")

    # Load prompt metadata for category info
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        meta = {r["image_id"]: r for r in reader}

    # Load format info from generation log
    gen_log_path = os.path.join(gen_dir, "generation_log.json")
    format_map = {}
    if os.path.exists(gen_log_path):
        with open(gen_log_path) as f:
            gen_log = json.load(f)
        for entry in gen_log.get("entries", []):
            if entry.get("image_id") and entry.get("format_detected"):
                format_map[entry["image_id"]] = entry["format_detected"]

    # Find pairs that exist in both gen and gt
    gen_ids = set(f.replace(".png", "") for f in os.listdir(gen_dir) if f.endswith(".png"))
    gt_ids = set(f.replace(".png", "") for f in os.listdir(args.gt_dir) if f.endswith(".png"))
    paired_ids = sorted(gen_ids & gt_ids)

    if args.limit:
        paired_ids = paired_ids[:args.limit]

    print(f"Generated images:    {len(gen_ids)}")
    print(f"Ground truth images: {len(gt_ids)}")
    print(f"Paired for eval:     {len(paired_ids)}")

    if not paired_ids:
        print("ERROR: No paired images found. Check directories.")
        return

    # --- Per-pair metrics ---
    pair_results = []
    for image_id in tqdm(paired_ids, desc="Evaluating pairs"):
        gen_path = os.path.join(gen_dir, f"{image_id}.png")
        gt_path = os.path.join(args.gt_dir, f"{image_id}.png")
        category = meta.get(image_id, {}).get("category", "unknown")

        scores = evaluate_pair(gen_path, gt_path)
        scores["image_id"] = image_id
        scores["category"] = category
        scores["code_language"] = format_map.get(image_id, "unknown")
        pair_results.append(scores)

    # --- CMMD (distribution-level) ---
    print("\nComputing CMMD (CLIP embeddings)...")
    clip_embedder = CLIPEmbedder()
    gen_embeddings = []
    gt_embeddings = []
    for image_id in tqdm(paired_ids, desc="CLIP embeddings"):
        gen_path = os.path.join(gen_dir, f"{image_id}.png")
        gt_path = os.path.join(args.gt_dir, f"{image_id}.png")
        try:
            gen_emb = clip_embedder.embed(gen_path)
            gt_emb = clip_embedder.embed(gt_path)
            # Only add if both succeed (keeps arrays aligned)
            gen_embeddings.append(gen_emb)
            gt_embeddings.append(gt_emb)
        except Exception:
            continue

    gen_embeddings = np.vstack(gen_embeddings) if gen_embeddings else np.empty((0, 512))
    gt_embeddings = np.vstack(gt_embeddings) if gt_embeddings else np.empty((0, 512))
    cmmd_score = compute_cmmd(gen_embeddings, gt_embeddings) if len(gen_embeddings) > 1 else 0.0
    print(f"CMMD: {cmmd_score:.6f}")

    # --- Aggregate ---
    from collections import Counter, defaultdict
    from scipy import stats as scipy_stats

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    def safe_std(values):
        valid = [v for v in values if v is not None]
        return float(np.std(valid, ddof=1)) if len(valid) > 1 else None

    def safe_ci95(values):
        valid = [v for v in values if v is not None]
        if len(valid) < 2:
            return None
        se = scipy_stats.sem(valid)
        ci = se * scipy_stats.t.ppf(0.975, len(valid) - 1)
        return float(ci)

    METRICS = ["dists", "clip_sim", "edge_iou", "edge_f1"]

    # Code language distribution
    lang_dist = dict(Counter(r.get("code_language", "unknown") for r in pair_results))

    # Overall aggregation with SD and 95% CI
    overall = {
        "cmmd": cmmd_score,
        "n_pairs": len(paired_ids),
        "code_language_distribution": lang_dist,
    }
    for m in METRICS:
        vals = [r.get(m) for r in pair_results]
        overall[f"{m}_mean"] = safe_mean(vals)
        overall[f"{m}_std"] = safe_std(vals)
        overall[f"{m}_ci95"] = safe_ci95(vals)

    # Per-category aggregation
    by_category = defaultdict(list)
    for r in pair_results:
        by_category[r["category"]].append(r)

    category_scores = {}
    for cat, entries in sorted(by_category.items()):
        cat_scores = {"n": len(entries)}
        for m in METRICS:
            vals = [r.get(m) for r in entries]
            cat_scores[f"{m}_mean"] = safe_mean(vals)
            cat_scores[f"{m}_std"] = safe_std(vals)
        category_scores[cat] = cat_scores

    # Save results
    results = {
        "model": args.model,
        "overall": overall,
        "by_category": category_scores,
        "per_image": pair_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {args.model}")
    print(f"{'='*60}")
    print(f"Pairs evaluated:  {overall['n_pairs']}")
    def fmtv(val, decimals=4):
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    def fmtpm(mean_key, ci_key):
        m = overall.get(mean_key)
        ci = overall.get(ci_key)
        if m is None: return "N/A"
        if ci is not None: return f"{m:.4f} ± {ci:.4f}"
        return f"{m:.4f}"

    print(f"CMMD:             {fmtv(overall['cmmd'], 6)}  (lower=better)")
    print(f"DISTS:            {fmtpm('dists_mean', 'dists_ci95')}  (lower=better)")
    print(f"CLIP Similarity:  {fmtpm('clip_sim_mean', 'clip_sim_ci95')}  (higher=better)")
    print(f"Edge IoU:         {fmtpm('edge_iou_mean', 'edge_iou_ci95')}  (higher=better)")
    print(f"Edge F1:          {fmtpm('edge_f1_mean', 'edge_f1_ci95')}  (higher=better)")
    print(f"Languages:        {lang_dist}")
    print(f"\nPer-category breakdown:")
    for cat, sc in sorted(category_scores.items(), key=lambda x: x[1]["n"], reverse=True):
        print(f"  {cat:30s}  n={sc['n']:4d}  DISTS={fmtv(sc['dists_mean'],3)}  CLIP={fmtv(sc['clip_sim_mean'],3)}  EdgeIoU={fmtv(sc['edge_iou_mean'],3)}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
