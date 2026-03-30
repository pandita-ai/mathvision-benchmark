---
language:
- en
pretty_name: MathVision Diagram Generation Benchmark
tags:
- mathematical-diagrams
- benchmarking
- code-generation
- tikz
- evaluation-metrics
- image-similarity
license: cc-by-4.0
task_categories:
- text-to-image
- image-to-image
size_categories:
- 10K<n<100K
configs:
- config_name: full
  default: true
  data_files:
  - split: test
    path: full/test-*.parquet
- config_name: common_subset
  data_files:
  - split: test
    path: common_subset/test-*.parquet
- config_name: ground_truth
  data_files:
  - split: test
    path: ground_truth/test-*.parquet
---

# MathVision Diagram Generation Benchmark

This dataset contains the outputs and evaluation results of 11 large language models and image generation models tasked with generating mathematical diagrams from natural language descriptions. Each model received the same set of 2,920 curated prompts derived from the [MathVision](https://arxiv.org/abs/2404.15910) dataset, and the resulting images were evaluated against ground truth using four automated metrics.

## Task

Given a textual description of a mathematical diagram (e.g., *"Draw a circle with 15 equally spaced dots on its circumference, connected by a triangle"*), the model must produce a visual rendering. Code-generating models output TikZ, Python, or SVG code that is then compiled; image generation models return images directly.

## Models

| Model | Type | Provider |
|---|---|---|
| Claude Opus 4.6 | Code LLM | Anthropic (via OpenRouter) |
| DeepSeek V3 | Code LLM | DeepSeek |
| DeepSeek R1 | Code LLM | DeepSeek |
| GPT-5.4 | Code LLM | OpenAI |
| GPT-OSS-120B | Code LLM | OpenAI (via Groq) |
| Gemini 3.1 Pro | Code LLM | Google (via OpenRouter) |
| Kimi K2.5 | Code LLM | Moonshot AI (via OpenRouter) |
| Qwen3.5-35B | Code LLM | Alibaba (via OpenRouter) |
| Llama 4 Maverick | Code LLM | Meta (via OpenRouter) |
| Nano Banana 2 | Image Gen | Google Gemini 3.1 Flash (via OpenRouter) |
| Nano Banana Pro | Image Gen | Google Gemini 3 Pro (via OpenRouter) |

## Metrics

All metrics are computed per image pair (generated vs. ground truth):

- **DISTS** (lower is better): Deep Image Structure and Texture Similarity. Measures perceptual distance using deep features.
- **CLIP Similarity** (higher is better): Cosine similarity of CLIP ViT-B/32 embeddings. Captures semantic alignment.
- **Edge IoU** (higher is better): Intersection-over-Union of Canny edge maps. Measures structural overlap.
- **Edge F1** (higher is better): F1 score of edge pixel matching. Balances precision and recall of structural features.

## Dataset Structure

### Configurations

- **`full`** (default): 28,589 rows across all 11 models. Every successfully compiled image with its ground truth and metrics.
- **`common_subset`**: 11,748 rows. The 1,068 prompts where all 11 models produced a valid image, enabling fair head-to-head comparison.
- **`ground_truth`**: 2,920 rows. The curated prompts with ground truth images only. Use this to benchmark new models.

### Schema

| Column | Type | Description |
|---|---|---|
| `image_id` | string | MathVision image identifier |
| `model` | string | Model slug |
| `model_name` | string | Display name |
| `category` | string | Mathematical category (16 categories) |
| `concise_prompt` | string | Natural language diagram description |
| `original_question` | string | Original MathVision exam question |
| `code_language` | string | Output format: tikz, python, svg, or image_gen |
| `ground_truth_image` | image | Reference diagram from MathVision |
| `generated_image` | image | Model-generated diagram |
| `dists` | float | DISTS score |
| `clip_sim` | float | CLIP cosine similarity |
| `edge_iou` | float | Edge IoU |
| `edge_f1` | float | Edge F1 |

### Mathematical Categories

The prompts span 16 categories: algebra, analytic geometry, arithmetic, combinatorial geometry, combinatorics, counting, descriptive geometry, graph theory, length, logic, metric geometry (area), metric geometry (length), solid geometry, statistics, transformation geometry, and trigonometry.

## Supplementary Files

The `supplementary/` folder contains raw evaluation outputs:

- `summary_stats.json`: Aggregated per-model statistics with confidence intervals.
- `eval_results/<model>.json`: Full per-image evaluation results including per-category breakdowns and dataset-level CMMD scores.

## Usage

```python
from datasets import load_dataset

# Load all results
ds = load_dataset("diagramAI/mathvision-diagram-benchmark", "full", split="test")

# Load only the common subset for fair comparison
common = load_dataset("diagramAI/mathvision-diagram-benchmark", "common_subset", split="test")

# Load ground truth prompts to benchmark a new model
gt = load_dataset("diagramAI/mathvision-diagram-benchmark", "ground_truth", split="test")

# Filter by model
claude_results = ds.filter(lambda x: x["model"] == "claude-opus-4.6")

# Streaming (no full download required)
ds_stream = load_dataset("diagramAI/mathvision-diagram-benchmark", "full", split="test", streaming=True)
```

## Source

Prompts are derived from the MathVision dataset (Wang et al., 2024). Ground truth images are the original exam figures. Generated images and evaluation metrics were produced by our benchmarking pipeline, available at [github.com/pandita-ai/mathvision-benchmark](https://github.com/pandita-ai/mathvision-benchmark).

## Citation

```bibtex
@dataset{mathvision_diagram_benchmark_2026,
  title={MathVision Diagram Generation Benchmark},
  author={Mistry, Aryan},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/diagramAI/mathvision-diagram-benchmark}
}
```

## License

CC-BY-4.0
