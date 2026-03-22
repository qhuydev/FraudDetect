<<<<<<< HEAD
# FraudDetect
Hệ thống phát hiện gian lận giao dịch
=======
# Fraud Detection System - Comprehensive README

## Overview
Fraud Detection System is an AI-powered Fintech solution combining:
- **Real-time Risk Scoring**: Instant fraud probability assessment
- **Explainable AI (XAI)**: Transparent decision explanations
- **Advanced ML Pipeline**: Feature engineering & model training
- **Full-stack Architecture**: FastAPI backend + React/React Native frontend
- **Enterprise Monitoring**: Prometheus, Grafana, Loki, Tempo
- **Cloud-native Deployment**: Docker & Kubernetes ready

## Project Structure

### Backend (FastAPI)
- `api/`: REST endpoints for transactions, alerts, users
- `services/`: Business logic (fraud detection, risk scoring, XAI)
- `core/`: ML model integration & inference engine
- `schemas/`: Pydantic data models
- `db/`: Database configuration & ORM
- `logger.py`: Centralized logging configuration

### ML Pipeline
- `data/`: Data preprocessing & loading
- `features/`: Feature engineering modules
- `models/`: Training, prediction, evaluation
- `pipelines/`: Training & real-time inference pipelines

### Frontend
- `web_dashboard/`: React web application with risk visualizations
- `mobile_app/`: React Native mobile app

### Monitoring (OTEL Stack)
- `prometheus/`: Metrics collection
- `grafana/`: Dashboard visualization
- `loki/`: Log aggregation
- `tempo/`: Distributed tracing

### Deployment
- `docker/`: Docker configurations
- `kubernetes/`: K8s manifests

## Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Kubernetes & kubectl (optional)

### Setup

1. **Clone repository**
   \`\`\`bash
   git clone <repo-url>
   cd FraudDetect
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Configure environment**
   \`\`\`bash
   cp .env.example .env
   # Edit .env with your settings
   \`\`\`

4. **Run backend**
   \`\`\`bash
   cd backend
   uvicorn app.main:app --reload
   \`\`\`

## API Endpoints

### Transactions
- `POST /api/transactions/score` - Get fraud risk score
- `GET /api/transactions/{id}` - Get transaction details
- `POST /api/transactions/{id}/explain` - Get XAI explanation

### Alerts
- `GET /api/alerts` - List fraud alerts
- `POST /api/alerts/{id}/resolve` - Mark alert as resolved

### Users
- `GET /api/users/{id}/profile` - User profile & risk history
- `POST /api/users/{id}/whitelist` - Add to whitelist

## ML Pipeline

### Training
\`\`\`bash
cd ml_pipeline
python src/pipelines/training_pipeline.py
\`\`\`

### Real-time Inference
- Integrated via FastAPI endpoint
- Supports batch & streaming predictions
- ~100ms latency for scoring

## Monitoring

### Start monitoring stack
\`\`\`bash
docker-compose -f deployment/docker/docker-compose.yml up
\`\`\`

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Loki: http://localhost:3100

## Deployment

### Docker
\`\`\`bash
docker build -t frauddetect:latest -f deployment/docker/Dockerfile .
docker run -p 8000:8000 frauddetect:latest
\`\`\`

### Kubernetes
\`\`\`bash
kubectl apply -f deployment/kubernetes/
\`\`\`

## Testing

\`\`\`bash
pytest backend/tests/ -v --cov
\`\`\`

## Contributing
Follow the contribution guidelines in CONTRIBUTING.md

## License
MIT License
>>>>>>> 2a39a52 (Init project with current code)
