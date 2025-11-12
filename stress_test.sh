#!/bin/bash

# Get the LoadBalancer IP
SERVICE_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [ -z "$SERVICE_IP" ]; then
    echo "Error: Could not get service IP. Make sure the service is deployed."
    exit 1
fi

API_URL="http://${SERVICE_IP}/predict"
echo "API URL: $API_URL"
echo "Testing endpoint availability..."

# Test if endpoint is reachable
curl -s -o /dev/null -w "%{http_code}" $API_URL
echo ""

echo "============================================"
echo "Starting Stress Tests"
echo "============================================"

# Test 1: 1000 requests, 10 connections, 2 threads, 30 seconds
echo ""
echo "Test 1: 1000 requests with 10 concurrent connections"
echo "--------------------------------------------"
wrk -t2 -c10 -d30s -s wrk_script.lua $API_URL

sleep 10

# Test 2: 2000 requests, 20 connections, 4 threads, 30 seconds
echo ""
echo "Test 2: 2000 requests with 20 concurrent connections"
echo "--------------------------------------------"
wrk -t4 -c20 -d30s -s wrk_script.lua $API_URL

sleep 10

# Test 3: High load - 50 connections, 10 threads, 60 seconds
echo ""
echo "Test 3: High load - 50 concurrent connections"
echo "--------------------------------------------"
wrk -t10 -c50 -d60s -s wrk_script.lua $API_URL

echo ""
echo "============================================"
echo "Stress Tests Completed"
echo "============================================"
