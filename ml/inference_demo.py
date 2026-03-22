# Fraud Detection Inference Demo
# Author: Generated for FraudDetect Project
# Description: Demo script for real-time fraud detection inference

import json
from train import FraudDetectionModel

def demo_inference():
    """Demo inference with sample transactions"""

    # Initialize model
    model = FraudDetectionModel()

    # Sample transactions (some legitimate, some fraudulent)
    sample_transactions = [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 900.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 600.0
        },
        {
            "step": 2,
            "type": "TRANSFER",
            "amount": 50000.0,
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 50000.0
        },
        {
            "step": 3,
            "type": "CASH_OUT",
            "amount": 100000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 10000.0,
            "newbalanceDest": 110000.0
        }
    ]

    print("=== FRAUD DETECTION INFERENCE DEMO ===\n")

    for i, transaction in enumerate(sample_transactions, 1):
        print(f"Transaction {i}:")
        print(json.dumps(transaction, indent=2))

        result = model.predict_with_explanation(transaction)
        print("Prediction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 50)

if __name__ == "__main__":
    demo_inference()