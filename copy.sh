#!/bin/bash

# Copy files to the Kubernetes pod
POD_NAME=$(kubectl get pods -l app=macklin-dep -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "Error: Could not find pod with label app=macklin-dep"
    exit 1
fi

# Copy all notebook files and src files to the pod
echo "Copying files to pod $POD_NAME..."
kubectl cp ./notebooks/. $POD_NAME:/workspace/
kubectl cp ./src/. $POD_NAME:/workspace/src/
kubectl cp ./results/. $POD_NAME:/workspace/results/

echo "Files copied successfully!"
