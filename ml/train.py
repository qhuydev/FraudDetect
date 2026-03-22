# Fraud Detection Model Training and Evaluation
# Author: Generated for FraudDetect Project

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import joblib
import os
from typing import Dict, List, Any, Tuple
from preprocess import FraudDetectionPreprocessor, calculate_risk_score, explain_prediction
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Fraud Detection Model with training, evaluation, and prediction capabilities"""

    def __init__(self, model_dir: str = "../models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]):
        """Train the fraud detection model"""
        print("=== TRAINING MODEL ===")

        # Use Random Forest with regularization to prevent overfitting
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,  # Reduced from 10 to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',  # Handle imbalance without SMOTE
            n_jobs=-1
        )

        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names

        # Save model
        self.save_model()

    def cross_validate_model(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Cross-validate the model to check for overfitting"""
        print("=== CROSS-VALIDATING MODEL ===")

        # Create model for CV
        cv_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        # Stratified K-Fold CV
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Cross-validation scores
        cv_scores_precision = cross_val_score(cv_model, X_train, y_train, cv=cv_strategy, scoring='precision')
        cv_scores_recall = cross_val_score(cv_model, X_train, y_train, cv=cv_strategy, scoring='recall')
        cv_scores_f1 = cross_val_score(cv_model, X_train, y_train, cv=cv_strategy, scoring='f1')
        cv_scores_auc = cross_val_score(cv_model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')

        cv_results = {
            'cv_precision_mean': cv_scores_precision.mean(),
            'cv_precision_std': cv_scores_precision.std(),
            'cv_recall_mean': cv_scores_recall.mean(),
            'cv_recall_std': cv_scores_recall.std(),
            'cv_f1_mean': cv_scores_f1.mean(),
            'cv_f1_std': cv_scores_f1.std(),
            'cv_auc_mean': cv_scores_auc.mean(),
            'cv_auc_std': cv_scores_auc.std()
        }

        print("Cross-validation results:")
        for metric, value in cv_results.items():
            print(".4f")

        return cv_results

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        print("=== EVALUATING MODEL ===")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)

        evaluation_results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Confusion Matrix:\n{cm}")

        return evaluation_results

    def predict_with_explanation(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud with explanation"""
        print("=== PREDICTING WITH EXPLANATION ===")

        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])

        # Preprocess for inference
        preprocessor = FraudDetectionPreprocessor()
        X_processed = preprocessor.preprocess_inference(df)

        # Load model if not loaded
        if self.model is None:
            self.load_model()

        # Predict
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)

        # Calculate risk score
        risk_score = calculate_risk_score(probabilities)

        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Get feature values for explanation
        feature_dict = dict(zip(self.feature_names, X_processed[0]))

        # Explain prediction
        reasons = explain_prediction(feature_dict, prediction, risk_score)

        result = {
            "fraud": bool(prediction),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": reasons
        }

        print(f"Prediction: {result}")
        return result

    def save_model(self):
        """Save trained model"""
        path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def load_model(self):
        """Load trained model"""
        path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")

        # Load feature names if available
        feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
        if os.path.exists(feature_path):
            self.feature_names = joblib.load(feature_path)

def save_feature_names(feature_names: List[str], model_dir: str = "../models"):
    """Save feature names for later use"""
    path = os.path.join(model_dir, 'feature_names.pkl')
    joblib.dump(feature_names, path)
    print(f"Feature names saved to: {path}")

# Example usage and demo
if __name__ == "__main__":
    # Initialize components
    preprocessor = FraudDetectionPreprocessor()
    model_trainer = FraudDetectionModel()

    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_training()

    # Save feature names
    save_feature_names(feature_names)

    # Train model
    model = model_trainer.train_model(X_train, y_train, feature_names)

    # Cross-validate to check for overfitting
    cv_results = model_trainer.cross_validate_model(X_train, y_train)

    # Evaluate model
    evaluation = model_trainer.evaluate_model(X_test, y_test)

    # Demo prediction
    sample_transaction = {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 100000.0,
        'oldbalanceOrg': 100000.0,
        'newbalanceOrig': 0.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 100000.0
    }

    result = model_trainer.predict_with_explanation(sample_transaction)
    print("Sample prediction result:")
    print(result)