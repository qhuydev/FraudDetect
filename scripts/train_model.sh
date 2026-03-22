#!/bin/bash

# ML Pipeline setup and training script

echo "Setting up ML Pipeline..."

cd ml_pipeline

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
python src/data/preprocess.py

# Train model
python src/pipelines/training_pipeline.py

# Evaluate model
python src/models/evaluate.py

echo "ML Pipeline training completed!"
