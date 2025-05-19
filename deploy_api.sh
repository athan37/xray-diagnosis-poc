#!/bin/bash

# X-Ray Diagnosis API Deployment Script

# Build the Docker image
echo "Building Docker image: xray-diagnosis-api..."
docker build -t xray-diagnosis-api .

# Check if model file exists
if [ ! -f "deployment/best_model.pth" ]; then
    echo "WARNING: Model file not found at deployment/best_model.pth"
    echo "You need to download the model file before running the API."
    echo "Please see the README for instructions on how to obtain the model file."
    exit 1
fi

# Run the container
echo "Starting X-Ray Diagnosis API container..."
docker run -d \
    --name xray-diagnosis-api \
    -p 5000:5000 \
    --restart unless-stopped \
    xray-diagnosis-api

echo "API is now running at http://localhost:5000"
echo "You can view logs with: docker logs xray-diagnosis-api"
echo "Stop the API with: docker stop xray-diagnosis-api" 