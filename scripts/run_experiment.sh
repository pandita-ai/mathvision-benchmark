#!/bin/bash
# run_experiment.sh — Full benchmarking pipeline across multiple models.
# Run from: ~/benchmarking_paper/ with venv activated and API keys exported.
#
# Usage:
#   bash scripts/run_experiment.sh                          # run all models
#   MODELS="gpt-4o deepseek-v3" bash scripts/run_experiment.sh  # run specific models
#   MODELS="gpt-4o" WORKERS=10 bash scripts/run_experiment.sh   # single model, custom workers
set -uo pipefail

WORKERS=${WORKERS:-15}

ALL_MODELS="deepseek-v3 deepseek-r1 gpt-5.4 gpt-oss claude-opus-4.6 gemini-3.1-pro qwen3.5-397b qwen3.5-35b llama-4-maverick"
MODELS_TO_RUN=${MODELS:-$ALL_MODELS}

echo "============================================"
echo "=== Multi-Model Benchmarking Experiment ==="
echo "============================================"
echo "Models:  $MODELS_TO_RUN"
echo "Workers: $WORKERS"
echo "Started: $(date)"
echo ""

for MODEL in $MODELS_TO_RUN; do
    LOG_DIR="outputs/${MODEL}"
    mkdir -p "$LOG_DIR"
    LOGFILE="${LOG_DIR}/run.log"

    echo "========================================" | tee -a "$LOGFILE"
    echo "=== Model: $MODEL ===" | tee -a "$LOGFILE"
    echo "Workers: $WORKERS" | tee -a "$LOGFILE"
    echo "Started: $(date)" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    # --- Phase 1: Generation ---
    echo "=== [1/3] Generation ===" | tee -a "$LOGFILE"
    START=$(date +%s)
    if ! python scripts/generate.py --model "$MODEL" --workers "$WORKERS" 2>&1 | tee -a "$LOGFILE"; then
        echo "WARNING: Generation failed for $MODEL, skipping to next model" | tee -a "$LOGFILE"
        continue
    fi
    END=$(date +%s)
    echo "Generation time: $(( (END - START) / 60 )) min" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    # --- Phase 2: Evaluation metrics ---
    echo "=== [2/3] Evaluation (DISTS, CLIP Sim, CMMD, Edge IoU/F1) ===" | tee -a "$LOGFILE"
    START=$(date +%s)
    if ! python scripts/evaluate.py --model "$MODEL" 2>&1 | tee -a "$LOGFILE"; then
        echo "WARNING: Evaluation failed for $MODEL, skipping report" | tee -a "$LOGFILE"
        continue
    fi
    END=$(date +%s)
    echo "Evaluation time: $(( (END - START) / 60 )) min" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    # --- Phase 3: Report ---
    echo "=== [3/3] Generating HTML report ===" | tee -a "$LOGFILE"
    python scripts/report.py --model "$MODEL" --max-images 200 2>&1 | tee -a "$LOGFILE" || true
    echo "" | tee -a "$LOGFILE"

    echo "=== $MODEL complete at $(date) ===" | tee -a "$LOGFILE"
    echo "Results: ${LOG_DIR}/eval_results.json" | tee -a "$LOGFILE"
    echo "Report:  ${LOG_DIR}/report.html" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
done

echo "============================================"
echo "=== All models complete ==="
echo "Finished: $(date)"
echo "============================================"

# --- Cross-model comparison ---
echo ""
echo "=== Generating cross-model comparison report ==="
python scripts/compare.py || echo "WARNING: Comparison report failed"
echo "============================================"
