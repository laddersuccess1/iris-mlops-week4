#!/bin/bash

echo "Starting real-time monitoring..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "============================================"
    echo "IRIS API Kubernetes Monitoring"
    echo "Time: $(date)"
    echo "============================================"
    echo ""
    
    echo "--- Horizontal Pod Autoscaler Status ---"
    kubectl get hpa iris-api-hpa
    echo ""
    
    echo "--- Pod Status ---"
    kubectl get pods -l app=iris-api -o wide
    echo ""
    
    echo "--- Pod Resource Usage ---"
    kubectl top pods -l app=iris-api
    echo ""
    
    echo "--- Deployment Status ---"
    kubectl get deployment iris-api-deployment
    echo ""
    
    echo "--- Recent Pod Events ---"
    kubectl get events --sort-by='.lastTimestamp' | grep iris-api | tail -5
    echo ""
    
    sleep 5
done
