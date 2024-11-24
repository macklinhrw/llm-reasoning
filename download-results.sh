#!/bin/bash

# Get pod name
POD_NAME=$(kubectl get pods -l app=macklin-dep -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "Error: Could not find pod with label app=macklin-dep"
    exit 1
fi

# Create local results directory if it doesn't exist
mkdir -p ./results

# Copy results from pod to local directory
echo "Downloading results from pod $POD_NAME..."
kubectl cp $POD_NAME:/workspace/results/. ./results/

echo "Results downloaded successfully!"
