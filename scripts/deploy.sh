#!/bin/bash

# Deployment script

echo "Deploying Fraud Detection System..."

# Build Docker image
docker build -t frauddetect:latest -f deployment/docker/Dockerfile .

# Start services
docker-compose -f deployment/docker/docker-compose.yml up -d

# Apply Kubernetes manifests (if kubectl available)
if command -v kubectl &> /dev/null; then
    kubectl apply -f deployment/kubernetes/
    echo "Kubernetes deployment completed!"
fi

echo "Deployment completed!"
