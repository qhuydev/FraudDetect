#!/bin/bash

# Backend setup script

echo "Setting up Fraud Detection Backend..."

# Install dependencies
pip install -r requirements.txt

# Database migrations
cd backend
alembic upgrade head

# Start application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
