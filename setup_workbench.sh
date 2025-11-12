#!/bin/bash

set -e

echo "============================================"
echo "Setting up Vertex AI Workbench for IRIS MLOps"
echo "============================================"
echo ""

# Install kubectl
echo "Installing kubectl..."
if ! command -v kubectl &> /dev/null; then
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl
    echo "✓ kubectl installed"
else
    echo "✓ kubectl already installed"
fi

# Install gke-gcloud-auth-plugin
echo ""
echo "Installing GKE authentication plugin..."
sudo apt-get update -qq
sudo apt-get install -y google-cloud-sdk-gke-gcloud-auth-plugin
echo "✓ GKE auth plugin installed"

# Install wrk
echo ""
echo "Installing wrk for stress testing..."
if ! command -v wrk &> /dev/null; then
    sudo apt-get install -y build-essential libssl-dev git
    cd /tmp
    git clone https://github.com/wg/wrk.git
    cd wrk
    make
    sudo cp wrk /usr/local/bin/
    cd -
    rm -rf /tmp/wrk
    echo "✓ wrk installed"
else
    echo "✓ wrk already installed"
fi

# Install jq for JSON processing
echo ""
echo "Installing jq..."
sudo apt-get install -y jq
echo "✓ jq installed"

# Verify installations
echo ""
echo "============================================"
echo "Verifying installations..."
echo "============================================"
echo "kubectl version:"
kubectl version --client --short
echo ""
echo "gcloud version:"
gcloud version | head -1
echo ""
echo "wrk version:"
wrk --version
echo ""
echo "jq version:"
jq --version

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x stress_test.sh 2>/dev/null || true
chmod +x monitor.sh 2>/dev/null || true
echo "✓ Scripts ready"

echo ""
echo "============================================"
echo "Setup completed successfully!"
echo "============================================"
