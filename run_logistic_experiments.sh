#!/bin/bash
# Script to run data poisoning experiments using Logistic Regression
# Logistic Regression is more vulnerable to poisoning than RandomForest

set -e  # Exit on error

echo "=========================================="
echo "Data Poisoning - Logistic Regression"
echo "=========================================="
echo ""

# Detect Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Create directories
mkdir -p models mlruns

# Set experiment name
EXPERIMENT_NAME="iris-poisoning-logistic"

# Poison levels to test
POISON_LEVELS=(0.0 0.05 0.10 0.50)

echo "Running experiments for poison levels: ${POISON_LEVELS[@]}"
echo "Model: Logistic Regression (more vulnerable to poisoning)"
echo "Expected duration: ~1-2 minutes"
echo ""

# Track start time
START_TIME=$(date +%s)

# Run training for each poison level
for poison_ratio in "${POISON_LEVELS[@]}"
do
    pct=$(echo "$poison_ratio * 100" | bc 2>/dev/null || echo "$(awk "BEGIN {print $poison_ratio * 100}")")
    
    echo "----------------------------------------"
    echo "Training with ${poison_ratio} poison ratio (${pct}%)"
    echo "----------------------------------------"
    
    $PYTHON_CMD train_with_poisoning_logistic.py \
        --poison-ratio "$poison_ratio" \
        --experiment-name "$EXPERIMENT_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed training with ${poison_ratio} poison ratio"
    else
        echo "✗ Failed training with ${poison_ratio} poison ratio"
        exit 1
    fi
    echo ""
done

# Track end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total execution time: ${DURATION} seconds"
echo ""
echo "Next steps:"
echo "  1. View MLFlow UI: mlflow ui --host 0.0.0.0 --port 5000"
echo "  2. In Cloud Shell, use Web Preview on port 5000"
echo "  3. Compare with RandomForest results (experiment: iris-data-poisoning)"
echo ""
echo "Files generated:"
echo "  - models/logistic_poison_0pct/   (baseline)"
echo "  - models/logistic_poison_5pct/"
echo "  - models/logistic_poison_10pct/"
echo "  - models/logistic_poison_50pct/"
echo "  - mlruns/               (MLFlow tracking)"
echo ""


