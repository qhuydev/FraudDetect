# Fraud Detection Preprocessing Module
# Author: Generated for FraudDetect Project
# Description: Complete preprocessing pipeline for PaySim dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPreprocessor:
    """
    Preprocessing pipeline for Fraud Detection using PaySim dataset.
    Supports both training and inference modes.
    """

    def __init__(self, data_path: str = "../data/paysim.csv", model_dir: str = "../models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.scaler = None
        self.encoder = None
        self.feature_names = None

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load PaySim dataset"""
        print("Loading data from:", self.data_path)
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # For demo purposes, sample a smaller subset to avoid memory issues
        # Sample fraud cases and 10x non-fraud cases
        fraud_cases = df[df['isFraud'] == 1]
        n_fraud = len(fraud_cases)
        non_fraud_cases = df[df['isFraud'] == 0].sample(n=min(n_fraud * 10, len(df[df['isFraud'] == 0])), random_state=42)
        df = pd.concat([fraud_cases, non_fraud_cases]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset shape: {df.shape}")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        print("Handling missing values...")
        # Check for missing values
        missing = df.isnull().sum()
        print(f"Missing values:\n{missing[missing > 0]}")

        # For this dataset, assume no missing values, but handle if any
        df = df.dropna()  # Simple drop for now
        print(f"After handling missing values: {df.shape}")
        return df

    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID columns and unnecessary features"""
        print("Removing unnecessary columns...")
        # Remove ID columns
        columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        print(f"Remaining columns: {list(df.columns)}")
        return df

    def feature_engineering(self, df: pd.DataFrame, is_training: bool = True, amount_threshold: float = None) -> pd.DataFrame:
        """Create new features for fraud detection"""
        print("Performing feature engineering...")

        # Balance differences (more interpretable than error_balance)
        df['balance_diff_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

        # Remove error_balance features as they are redundant (amount - balance_diff)
        # In fraud cases: balance_diff ≈ amount, so error_balance ≈ 0
        # In legitimate cases: also often error_balance ≈ 0

        # Large amount indicator (top 5% of amounts)
        if is_training and amount_threshold is None:
            amount_threshold = df['amount'].quantile(0.95)
            print(f"Amount threshold (training): {amount_threshold}")
        elif amount_threshold is not None:
            print(f"Using provided amount threshold: {amount_threshold}")
        else:
            # For inference, use a reasonable default or load from saved model
            amount_threshold = 100000  # Conservative default
            print(f"Using default amount threshold: {amount_threshold}")

        df['is_large_amount'] = (df['amount'] > amount_threshold).astype(int)

        # Transaction frequency features (simplified)
        df['step_hour'] = df['step'] % 24
        df['step_day'] = df['step'] // 24

        print(f"New features created: balance_diff_org, balance_diff_dest, is_large_amount, step_hour, step_day")
        return df, amount_threshold

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target"""
        # Target variable
        y = df['isFraud']

        # Features (exclude target)
        X = df.drop('isFraud', axis=1)

        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Target distribution: {y.value_counts()}")
        return X, y

    def get_preprocessor_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        # Categorical columns
        categorical_cols = ['type']

        # Numerical columns (exclude categorical)
        numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                         'oldbalanceDest', 'newbalanceDest', 'balance_diff_org',
                         'balance_diff_dest', 'is_large_amount', 'step_hour', 'step_day']

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])

        return preprocessor, numerical_cols + categorical_cols

    def preprocess_training(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Full preprocessing for training mode"""
        print("=== TRAINING MODE PREPROCESSING ===")

        # Load and clean data
        df = self.load_data()
        df = self.handle_missing_values(df)
        df = self.remove_unnecessary_columns(df)

        # TIME-BASED SPLIT: Sort by step (time) and split chronologically
        df = df.sort_values('step').reset_index(drop=True)
        split_idx = int(len(df) * 0.8)  # 80% train, 20% test

        train_df = df[:split_idx]
        test_df = df[split_idx:]

        print(f"Time-based split: Train steps {train_df['step'].min()}-{train_df['step'].max()}, Test steps {test_df['step'].min()}-{test_df['step'].max()}")

        # Separate X, y
        X_train = train_df.drop('isFraud', axis=1)
        y_train = train_df['isFraud']
        X_test = test_df.drop('isFraud', axis=1)
        y_test = test_df['isFraud']

        # Feature engineering on train data only
        X_train, amount_threshold = self.feature_engineering(X_train, is_training=True)
        X_test, _ = self.feature_engineering(X_test, is_training=False, amount_threshold=amount_threshold)

        print(f"Train target distribution: {y_train.value_counts()}")
        print(f"Test target distribution: {y_test.value_counts()}")

        # Get preprocessor
        preprocessor, feature_cols = self.get_preprocessor_pipeline()

        # Fit on training data
        X_train_processed = preprocessor.fit_transform(X_train[feature_cols])
        X_test_processed = preprocessor.transform(X_test[feature_cols])

        # No SMOTE - using class_weight in the model instead
        X_train_balanced = X_train_processed
        y_train_balanced = y_train

        print(f"Train shape: {X_train_balanced.shape}")
        print(f"Balanced target distribution: {pd.Series(y_train_balanced).value_counts()}")

        # Save preprocessor and threshold
        self.save_preprocessor(preprocessor)
        self.save_threshold(amount_threshold)
        self.save_threshold(amount_threshold)

        # Get feature names
        feature_names = self.get_feature_names(preprocessor, feature_cols)

        return X_train_balanced, X_test_processed, y_train_balanced, y_test, feature_names

    def preprocess_inference(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocessing for inference mode"""
        print("=== INFERENCE MODE PREPROCESSING ===")

        # Load preprocessor and threshold
        preprocessor = self.load_preprocessor()
        amount_threshold = self.load_threshold()

        # Clean and engineer features
        data = self.handle_missing_values(data)
        data = self.remove_unnecessary_columns(data)
        data, _ = self.feature_engineering(data, is_training=False, amount_threshold=amount_threshold)

        # Get feature columns
        _, feature_cols = self.get_preprocessor_pipeline()

        # Transform
        X_processed = preprocessor.transform(data[feature_cols])

        return X_processed

    def save_preprocessor(self, preprocessor: ColumnTransformer):
        """Save preprocessor to disk"""
        path = os.path.join(self.model_dir, 'preprocessor.pkl')
        joblib.dump(preprocessor, path)
        print(f"Preprocessor saved to: {path}")

    def save_threshold(self, threshold: float):
        """Save amount threshold to disk"""
        path = os.path.join(self.model_dir, 'amount_threshold.pkl')
        joblib.dump(threshold, path)
        print(f"Amount threshold saved to: {path}")

    def load_preprocessor(self) -> ColumnTransformer:
        """Load preprocessor from disk"""
        path = os.path.join(self.model_dir, 'preprocessor.pkl')
        preprocessor = joblib.load(path)
        print(f"Preprocessor loaded from: {path}")
        return preprocessor

    def load_threshold(self) -> float:
        """Load amount threshold from disk"""
        path = os.path.join(self.model_dir, 'amount_threshold.pkl')
        threshold = joblib.load(path)
        print(f"Amount threshold loaded from: {path}: {threshold}")
        return threshold

    def get_feature_names(self, preprocessor: ColumnTransformer, feature_cols: List[str]) -> List[str]:
        """Get feature names after preprocessing"""
        # Get numerical features
        num_features = feature_cols[:11]  # Adjust based on your numerical cols

        # Get categorical features
        cat_features = []
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['type']).tolist()

        feature_names = num_features + cat_features
        return feature_names

def calculate_risk_score(probabilities: np.ndarray) -> float:
    """Calculate risk score from prediction probabilities"""
    # Assuming probabilities[:, 1] is fraud probability
    return float(probabilities[0][1]) if len(probabilities.shape) > 1 else float(probabilities[1])

def explain_prediction(features: Dict[str, float], prediction: int, risk_score: float) -> List[str]:
    """Improved rule-based explanation for fraud prediction with behavior rules"""
    reasons = []

    amount = float(features.get('amount', 0.0))
    obs = float(features.get('oldbalanceOrg', 0.0))
    nbs = float(features.get('newbalanceOrig', 0.0))
    bdiff_org = float(features.get('balance_diff_org', 0.0))
    bdiff_dest = float(features.get('balance_diff_dest', 0.0))

    # 1. Hành vi rút hết tiền (withdraw-all)
    if obs > 0 and nbs == 0:
        reasons.append("Tài khoản gốc đã về 0 sau giao dịch")

    # 2. Chuyển toàn bộ số dư (transfer entire balance)
    if obs > 0 and abs(amount - obs) / obs >= 0.95:
        reasons.append("Giao dịch bằng hoặc gần bằng toàn bộ số dư gốc")

    # 3. Biến động số dư lớn so với amount
    if amount > 0 and abs(bdiff_org - amount) / amount >= 0.2:
        reasons.append("Mức chênh lệch số dư gốc không phù hợp với amount")
    if amount > 0 and abs(bdiff_dest - amount) / amount >= 0.2:
        reasons.append("Mức chênh lệch số dư đích không phù hợp với amount")

    # 4. Transaction type
    if features.get('type_CASH_OUT', 0) > 0:
        reasons.append("Loại giao dịch là CASH_OUT (rút tiền mặt), thường liên quan fraud")
    if features.get('type_TRANSFER', 0) > 0:
        reasons.append("Loại giao dịch là TRANSFER (chuyển khoản), rủi ro cao")

    # Nếu quá nhiều lý do, giữ 3 lý do quan trọng nhất
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    if prediction == 0 and not unique_reasons:
        return ["Không phát hiện dấu hiệu bất thường"]

    return unique_reasons[:3]

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = FraudDetectionPreprocessor()

    # For training
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_training()

    print(f"Training data ready: {X_train.shape}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10