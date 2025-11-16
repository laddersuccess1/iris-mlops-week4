#!/bin/bash
# Complete setup script for Data Poisoning demonstration
# Designed for Google Cloud Shell

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Data Poisoning Attack - Setup & Demonstration        ║"
echo "║  IRIS Classification with MLFlow Tracking             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python is not installed"
    exit 1
fi

echo "✓ Python detected: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
    echo "  Activating existing environment..."
    source venv/bin/activate
else
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "   Trying to install python3-venv..."
        sudo apt-get update && sudo apt-get install -y python3-venv
        $PYTHON_CMD -m venv venv
    fi
    
    echo "✓ Virtual environment created"
    source venv/bin/activate
fi

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🔍 Verifying installation..."
$PYTHON_CMD -c "import mlflow; import sklearn; import pandas; print('✓ Core packages verified')"

echo ""
echo "🏗️  Creating directory structure..."
mkdir -p models mlruns data

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Setup Complete! Ready to run experiments             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next Steps:"
echo ""
echo "  1️⃣  Run all experiments (0%, 5%, 10%, 50% poisoning):"
echo "     bash run_poisoning_experiments.sh"
echo ""
echo "  2️⃣  Compare results:"
echo "     python compare_poisoning_results.py"
echo ""
echo "  3️⃣  Launch MLFlow UI:"
echo "     mlflow ui --host 0.0.0.0 --port 5000"
echo "     Then: Web Preview → Preview on port 5000"
echo ""
echo "  OR run individual experiments:"
echo "     python train_with_poisoning.py --poison-ratio 0.05"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: QUICK_START.md"
echo "   - Full Guide:  DATA_POISONING_GUIDE.md"
echo "   - Summary:     DATA_POISONING_README.md"
echo ""


