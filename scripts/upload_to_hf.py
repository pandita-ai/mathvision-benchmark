#!/usr/bin/env python3
"""
upload_to_hf.py — Build and upload MathVision Diagram Benchmark to HuggingFace.

Runs on the GCP VM where all generated images live.

Usage:
    export HF_TOKEN=hf_xxx
    python scripts/upload_to_hf.py
    python scripts/upload_to_hf.py --dry-run  # validate without pushing
"""

import argparse
import csv
import json
import os
from pathlib import Path

from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi

REPO_ID = "diagramAI/mathvision-diagram-benchmark"
BASE = Path(__file__).resolve().parent.parent
PROMPTS_CSV = BASE / "data" / "concise_prompts.csv"
GROUND_TRUTH = BASE / "data" / "ground_truth"
OUTPUTS = BASE / "outputs"
FULL_RESULTS = BASE / "paper" / "data" / "full_results.csv"
COMMON_SUBSET = BASE / "paper" / "data" / "common_subset.csv"
SUMMARY_STATS = BASE / "paper" / "data" / "summary_stats.json"

PAPER_MODELS = [
    "deepseek-v3", "deepseek-r1", "gpt-5.4", "gpt-oss",
    "claude-opus-4.6", "gemini-3.1-pro", "qwen3.5-35b",
    "llama-4-maverick", "kimi-k2.5", "nano-banana-2", "nano-banana-pro",
]

FEATURES = Features({
    "image_id": Value("string"),
    "model": Value("string"),
    "model_name": Value("string"),
    "category": Value("string"),
    "concise_prompt": Value("string"),
    "original_question": Value("string"),
    "code_language": Value("string"),
    "ground_truth_image": HFImage(),
    "generated_image": HFImage(),
    "dists": Value("float32"),
    "clip_sim": Value("float32"),
    "edge_iou": Value("float32"),
    "edge_f1": Value("float32"),
})

GT_FEATURES = Features({
    "image_id": Value("string"),
    "category": Value("string"),
    "concise_prompt": Value("string"),
    "original_question": Value("string"),
    "ground_truth_image": HFImage(),
})


def load_prompts():
    """Load prompts CSV into {image_id: {question, category, concise_prompt}}."""
    prompts = {}
    with open(PROMPTS_CSV) as f:
        for row in csv.DictReader(f):
            prompts[row["image_id"]] = row
    return prompts


def generate_rows(results_csv, prompts):
    """Yield one record per row in results CSV, skipping missing images."""
    skipped = 0
    yielded = 0
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            image_id = row["image_id"]

            gen_path = OUTPUTS / model / f"{image_id}.png"
            gt_path = GROUND_TRUTH / f"{image_id}.png"

            if not gen_path.exists() or not gt_path.exists():
                skipped += 1
                continue

            # Skip images excluded from curated dataset
            if image_id not in prompts:
                skipped += 1
                continue

            prompt_info = prompts[image_id]
            yielded += 1
            yield {
                "image_id": image_id,
                "model": model,
                "model_name": row.get("model_name", model),
                "category": row.get("category", "unknown"),
                "concise_prompt": prompt_info.get("concise_prompt", ""),
                "original_question": prompt_info.get("question", ""),
                "code_language": row.get("code_language", "unknown"),
                "ground_truth_image": str(gt_path),
                "generated_image": str(gen_path),
                "dists": float(row["dists"]),
                "clip_sim": float(row["clip_sim"]),
                "edge_iou": float(row["edge_iou"]),
                "edge_f1": float(row["edge_f1"]),
            }

    print(f"  Yielded {yielded} rows, skipped {skipped} (missing images)")


def generate_gt_rows(prompts):
    """Yield one record per curated prompt with ground truth image."""
    for image_id, info in prompts.items():
        gt_path = GROUND_TRUTH / f"{image_id}.png"
        if not gt_path.exists():
            continue
        yield {
            "image_id": image_id,
            "category": info.get("category", "unknown"),
            "concise_prompt": info.get("concise_prompt", ""),
            "original_question": info.get("question", ""),
            "ground_truth_image": str(gt_path),
        }


def validate_prerequisites():
    """Check all required files and directories exist before starting."""
    errors = []

    if not PROMPTS_CSV.exists():
        errors.append(f"Missing: {PROMPTS_CSV}")
    if not GROUND_TRUTH.exists():
        errors.append(f"Missing: {GROUND_TRUTH}")
    if not FULL_RESULTS.exists():
        errors.append(f"Missing: {FULL_RESULTS}")
    if not COMMON_SUBSET.exists():
        errors.append(f"Missing: {COMMON_SUBSET}")

    gt_count = len(list(GROUND_TRUTH.glob("*.png"))) if GROUND_TRUTH.exists() else 0
    print(f"Ground truth images: {gt_count}")

    for model in PAPER_MODELS:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            errors.append(f"Missing model dir: {model_dir}")
            continue
        png_count = len(list(model_dir.glob("*.png")))
        print(f"  {model}: {png_count} PNGs")
        if png_count == 0:
            errors.append(f"No PNGs in {model_dir}")

    if errors:
        print("\nFATAL — cannot proceed:")
        for e in errors:
            print(f"  - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and build datasets without pushing")
    args = parser.parse_args()

    print("=== Validating prerequisites ===")
    if not validate_prerequisites():
        return

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: Set HF_TOKEN environment variable")
        return

    prompts = load_prompts()
    print(f"\nLoaded {len(prompts)} prompts")

    # --- Config: full ---
    print("\n=== Building 'full' dataset ===")
    full_ds = Dataset.from_generator(
        lambda: generate_rows(FULL_RESULTS, prompts),
        features=FEATURES,
    )
    print(f"  Full dataset: {len(full_ds)} rows")

    # --- Config: common_subset ---
    print("\n=== Building 'common_subset' dataset ===")
    common_ds = Dataset.from_generator(
        lambda: generate_rows(COMMON_SUBSET, prompts),
        features=FEATURES,
    )
    print(f"  Common subset: {len(common_ds)} rows")

    # --- Config: ground_truth ---
    print("\n=== Building 'ground_truth' dataset ===")
    gt_ds = Dataset.from_generator(
        lambda: generate_gt_rows(prompts),
        features=GT_FEATURES,
    )
    print(f"  Ground truth: {len(gt_ds)} rows")

    if args.dry_run:
        print("\n=== Dry run — skipping push ===")
        print("Sampling 3 rows from full dataset:")
        for i in range(min(3, len(full_ds))):
            row = full_ds[i]
            print(f"  [{i}] model={row['model']} id={row['image_id']} "
                  f"dists={row['dists']:.3f} clip={row['clip_sim']:.3f}")
        return

    # --- Push ---
    print(f"\n=== Pushing to {REPO_ID} ===")

    print("Pushing 'full' config...")
    full_ds.push_to_hub(REPO_ID, config_name="full", split="test",
                        max_shard_size="500MB", token=token)

    print("Pushing 'common_subset' config...")
    common_ds.push_to_hub(REPO_ID, config_name="common_subset", split="test",
                          max_shard_size="500MB", token=token)

    print("Pushing 'ground_truth' config...")
    gt_ds.push_to_hub(REPO_ID, config_name="ground_truth", split="test",
                      max_shard_size="500MB", token=token)

    # --- Supplementary files ---
    print("\n=== Uploading supplementary files ===")
    api = HfApi(token=token)

    if SUMMARY_STATS.exists():
        api.upload_file(
            path_or_fileobj=str(SUMMARY_STATS),
            path_in_repo="supplementary/summary_stats.json",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print("  Uploaded summary_stats.json")

    for model in PAPER_MODELS:
        eval_file = OUTPUTS / model / "eval_results.json"
        if eval_file.exists():
            api.upload_file(
                path_or_fileobj=str(eval_file),
                path_in_repo=f"supplementary/eval_results/{model}.json",
                repo_id=REPO_ID, repo_type="dataset",
            )
            print(f"  Uploaded {model} eval_results.json")

    print(f"\n=== Done! Dataset at https://huggingface.co/datasets/{REPO_ID} ===")


if __name__ == "__main__":
    main()
