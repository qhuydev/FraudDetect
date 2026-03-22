# Fraud Detection Pipeline - Đề Xuất Cải Tiến

## 1. Làm Model Thực Tế Hơn: Từ Proof-of-Concept Sang Production-Ready

### Vấn đề Hiện tại
Mô hình hiện tại đạt độ chính xác cao (Precision ~0.92, Recall ~0.98) trên tập test với time-based split, nhưng vẫn mang tính "academic" hơn là "production-ready". Các vấn đề chính bao gồm:

- **Overfitting tiềm ẩn**: Model học quá tốt trên patterns lịch sử mà không khái quát hóa cho dữ liệu thực tế
- **Thiếu robustness**: Không xử lý được outliers, missing values phức tạp, hoặc concept drift
- **Scalability hạn chế**: RandomForest với 100 cây có thể chậm trên dữ liệu lớn
- **Cold start problem**: Model cần lượng dữ liệu training lớn, không phù hợp cho startup fintech

### Giải Pháp Cụ Thể

#### a) Chuyển từ RandomForest sang XGBoost với Fine-tuning
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# XGBoost với scale_pos_weight để handle imbalance
model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # ~16.4
    max_depth=4,  # Giảm từ 6 để tránh overfitting
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Hyperparameter tuning với early stopping
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### b) Thêm Feature Engineering Nâng Cao
```python
def advanced_feature_engineering(df):
    # Transaction velocity features
    df['hourly_transaction_count'] = df.groupby(['nameOrig', 'step_hour']).cumcount()
    df['daily_transaction_count'] = df.groupby(['nameOrig', 'step_day']).cumcount()

    # Amount-based risk indicators
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # Avoid division by zero
    df['unusual_amount_pattern'] = ((df['amount'] > df['oldbalanceOrg'] * 0.9) &
                                   (df['type'] == 'TRANSFER')).astype(int)

    # Time-based patterns
    df['is_business_hour'] = ((df['step_hour'] >= 9) & (df['step_hour'] <= 17)).astype(int)
    df['is_weekend'] = (df['step_day'] % 7 >= 5).astype(int)

    return df
```

#### c) Implement Online Learning với Incremental Training
```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import pickle

class IncrementalFraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001)
        self.is_fitted = False

    def partial_fit(self, X_batch, y_batch):
        if not self.is_fitted:
            X_batch_scaled = self.scaler.fit_transform(X_batch)
            self.is_fitted = True
        else:
            X_batch_scaled = self.scaler.transform(X_batch)

        self.model.partial_fit(X_batch_scaled, y_batch, classes=[0, 1])
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
```

### Lợi Ích Sau Khi Cải Thiện

- **Độ chính xác ổn định**: XGBoost thường outperform RandomForest trên tabular data
- **Scalability tốt hơn**: XGBoost hỗ trợ parallel processing và GPU acceleration
- **Adaptability cao**: Online learning cho phép model học từ dữ liệu mới realtime
- **Robustness tăng**: Xử lý tốt hơn với outliers và missing data
- **Cost-effective**: Giảm resource usage cho inference trong production

### Kết Luận
Chuyển sang XGBoost + online learning sẽ biến mô hình từ "academic prototype" thành "production system" có thể scale cho hàng triệu transactions/ngày trong môi trường fintech thực tế.

## 2. Cải Thiện Explainable AI: Từ Rule-based Sang Model-driven Explanations

### Vấn đề Hiện tại
Hệ thống XAI hiện tại chỉ sử dụng rule-based explanations đơn giản:
- "Số tiền giao dịch rất lớn"
- "Số dư tài khoản gốc thay đổi rất lớn"

Những giải thích này:
- **Thiếu độ sâu**: Không giải thích tại sao model dự đoán với confidence cao
- **Không personalized**: Cùng một rule áp dụng cho tất cả cases
- **Thiếu context**: Không xem xét feature interactions
- **Không scalable**: Khó maintain khi thêm features mới

### Giải Pháp Cụ Thể

#### a) Implement SHAP (SHapley Additive exPlanations)
```python
import shap
import pandas as pd

class ExplainableFraudDetector:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit_explainer(self, X_background):
        """Fit SHAP explainer với background dataset"""
        self.explainer = shap.TreeExplainer(self.model, X_background)

    def explain_prediction(self, X_instance):
        """Giải thích prediction cho một instance cụ thể"""
        shap_values = self.explainer(X_instance)

        # Lấy top 5 features quan trọng nhất
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values.values[0],
            'feature_value': X_instance.iloc[0].values
        }).sort_values('shap_value', key=abs, ascending=False)

        return feature_importance.head(5)

    def generate_natural_language_explanation(self, feature_importance, prediction):
        """Chuyển SHAP values thành natural language"""
        explanations = []

        for _, row in feature_importance.iterrows():
            feature = row['feature']
            value = row['feature_value']
            impact = "tăng" if row['shap_value'] > 0 else "giảm"

            if feature == 'amount':
                explanations.append(f"Số tiền {value:,.0f} VND {impact} đáng kể khả năng fraud")
            elif feature == 'balance_diff_org':
                explanations.append(f"Thay đổi số dư tài khoản gốc {value:,.0f} VND {impact} rủi ro")
            elif feature == 'type_TRANSFER':
                explanations.append(f"Loại giao dịch chuyển khoản {impact} khả năng nghi ngờ")
            elif feature == 'is_large_amount':
                explanations.append(f"Giao dịch số tiền lớn {impact} độ tin cậy của dự đoán")

        return explanations
```

#### b) Local Feature Importance với LIME
```python
from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer:
    def __init__(self, X_train, feature_names, class_names=['Legitimate', 'Fraud']):
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

    def explain_instance(self, instance, predict_fn, num_features=5):
        """Giải thích một instance cụ thể"""
        explanation = self.explainer.explain_instance(
            instance.values[0],
            predict_fn,
            num_features=num_features
        )

        # Trả về dict với feature contributions
        feature_contributions = {}
        for feature, weight in explanation.as_list():
            feature_contributions[feature] = weight

        return feature_contributions
```

#### c) Counterfactual Explanations
```python
def generate_counterfactual_explanation(model, instance, feature_names):
    """Tạo 'what-if' scenarios"""
    original_pred = model.predict_proba(instance)[0][1]

    counterfactuals = []

    # Scenario 1: Giảm amount xuống mức bình thường
    modified_instance = instance.copy()
    modified_instance['amount'] = instance['amount'] * 0.1  # Giảm 90%
    new_pred = model.predict_proba(modified_instance)[0][1]
    counterfactuals.append({
        'scenario': 'Giảm số tiền xuống 10%',
        'original_risk': original_pred,
        'new_risk': new_pred,
        'risk_reduction': original_pred - new_pred
    })

    # Scenario 2: Thay đổi loại giao dịch
    if instance['type'].iloc[0] == 'TRANSFER':
        modified_instance = instance.copy()
        modified_instance['type'] = 'PAYMENT'
        new_pred = model.predict_proba(modified_instance)[0][1]
        counterfactuals.append({
            'scenario': 'Đổi từ TRANSFER sang PAYMENT',
            'original_risk': original_pred,
            'new_risk': new_pred,
            'risk_reduction': original_pred - new_pred
        })

    return counterfactuals
```

### Lợi Ích Sau Khi Cải Thiện

- **Giải thích chính xác hơn**: SHAP/LIME cho biết chính xác feature nào ảnh hưởng như thế nào
- **Trust tăng**: User hiểu được lý do đằng sau decision
- **Debugging tốt hơn**: Developer có thể identify model biases
- **Regulatory compliance**: Giúp đáp ứng yêu cầu transparency của ngân hàng trung ương
- **User experience**: Khách hàng có thể hiểu và contest decisions

### Kết Luận
SHAP + LIME + counterfactual explanations sẽ biến hệ thống XAI từ "black box" thành "glass box", tăng trust và compliance trong môi trường fintech quy định chặt chẽ.

## 3. Risk Scoring Nâng Cao: Từ Binary Classification Sang Risk Continuum

### Vấn đề Hiện tại
Hệ thống hiện tại chỉ output binary prediction (fraud/not fraud) với risk_score đơn giản từ predict_proba. Các vấn đề:

- **Thiếu granularity**: Chỉ có 2 levels thay vì risk spectrum
- **Không adaptive**: Risk score không thay đổi theo context
- **Thiếu confidence**: Không biết độ tin cậy của prediction
- **Không personalized**: Cùng risk score cho tất cả users

### Giải Pháp Cụ Thể

#### a) Multi-level Risk Scoring System
```python
class AdvancedRiskScorer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def calculate_comprehensive_risk_score(self, transaction, historical_data=None):
        """Tính risk score toàn diện"""
        # Base probability
        base_proba = self.model.predict_proba(transaction)[0][1]

        # Confidence adjustment
        confidence = self._calculate_prediction_confidence(transaction)

        # Context-based adjustment
        context_multiplier = self._calculate_context_multiplier(transaction, historical_data)

        # Behavioral adjustment
        behavioral_score = self._calculate_behavioral_score(transaction, historical_data)

        # Final risk score
        risk_score = (base_proba * confidence * context_multiplier + behavioral_score) / 2

        return {
            'base_probability': base_proba,
            'confidence': confidence,
            'context_multiplier': context_multiplier,
            'behavioral_score': behavioral_score,
            'final_risk_score': min(risk_score, 1.0),
            'risk_level': self._classify_risk_level(risk_score)
        }

    def _calculate_prediction_confidence(self, transaction):
        """Tính confidence của prediction dựa trên feature values"""
        # Sử dụng prediction variance hoặc distance to decision boundary
        # Với tree models, có thể dùng leaf purity hoặc path length

        # Simplified version: confidence based on feature extremity
        confidence = 1.0
        features = transaction.iloc[0]

        # High amount + unusual balance changes = high confidence
        if features['is_large_amount'] > 0 and abs(features['balance_diff_org']) > features['amount'] * 0.8:
            confidence *= 1.2  # Boost confidence
        elif features['amount'] < features['oldbalanceOrg'] * 0.1:
            confidence *= 0.8  # Reduce confidence for small amounts

        return min(confidence, 1.0)

    def _calculate_context_multiplier(self, transaction, historical_data):
        """Điều chỉnh dựa trên context (time, location, etc.)"""
        multiplier = 1.0
        features = transaction.iloc[0]

        # Time-based adjustments
        if features['step_hour'] in [2, 3, 4, 5]:  # Late night hours
            multiplier *= 1.3  # Higher risk

        if features['is_weekend'] > 0:
            multiplier *= 1.1  # Slightly higher risk

        # Transaction type adjustments
        if features.get('type_TRANSFER', 0) > 0:
            multiplier *= 1.4  # TRANSFER is riskier than PAYMENT

        return multiplier

    def _calculate_behavioral_score(self, transaction, historical_data):
        """Tính điểm dựa trên behavioral patterns"""
        if historical_data is None:
            return 0.0

        features = transaction.iloc[0]
        user_id = features.get('nameOrig', 'unknown')

        # Calculate deviation from user's normal behavior
        user_history = historical_data[historical_data['nameOrig'] == user_id]

        if len(user_history) < 5:
            return 0.1  # New user, slight risk

        # Amount deviation
        avg_amount = user_history['amount'].mean()
        amount_deviation = abs(features['amount'] - avg_amount) / avg_amount

        # Time pattern deviation
        usual_hours = user_history['step_hour'].mode().iloc[0] if len(user_history['step_hour'].mode()) > 0 else 12
        time_deviation = abs(features['step_hour'] - usual_hours) / 24

        behavioral_score = (amount_deviation + time_deviation) / 2
        return min(behavioral_score, 0.5)  # Cap at 0.5

    def _classify_risk_level(self, risk_score):
        """Classify risk into levels"""
        if risk_score >= 0.8:
            return "Critical"
        elif risk_score >= 0.6:
            return "High"
        elif risk_score >= 0.4:
            return "Medium"
        elif risk_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
```

#### b) Dynamic Thresholding
```python
class DynamicThresholdAdjuster:
    def __init__(self):
        self.business_rules = {
            'max_false_positive_rate': 0.05,  # 5% false positives
            'min_recall': 0.95,  # 95% fraud detection
            'cost_false_positive': 10,  # Cost of investigating legitimate transaction
            'cost_false_negative': 1000  # Cost of missing fraud
        }

    def optimize_threshold(self, y_true, y_proba):
        """Tìm threshold tối ưu dựa trên business costs"""
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate cost for each threshold
        costs = []
        for i, threshold in enumerate(thresholds):
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate confusion matrix elements
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            # Calculate total cost
            total_cost = fp * self.business_rules['cost_false_positive'] + \
                        fn * self.business_rules['cost_false_negative']

            costs.append(total_cost)

        # Find threshold with minimum cost
        optimal_idx = costs.index(min(costs))
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold, costs[optimal_idx]
```

### Lợi Ích Sau Khi Cải Thiện

- **Decision making tốt hơn**: Risk levels thay vì binary classification
- **Cost optimization**: Dynamic thresholds cân bằng false positives/negatives
- **Personalization**: Behavioral scoring cho từng user
- **Confidence assessment**: Biết được độ tin cậy của predictions
- **Business value**: Giúp prioritize investigations và resource allocation

### Kết Luận
Multi-level risk scoring với context awareness sẽ biến hệ thống từ "binary classifier" thành "risk management platform" thông minh, tối ưu hóa cả accuracy và business costs.

## 4. Validation Nâng Cao: Từ Holdout Sang Comprehensive Testing

### Vấn đề Hiện tại
Validation hiện tại chỉ dùng time-based split + cross-validation cơ bản. Các vấn đề:

- **Thiếu robustness testing**: Không test với adversarial inputs
- **Không evaluate temporal stability**: Model có thể degraded over time
- **Thiếu fairness assessment**: Không check bias across user groups
- **Không có stress testing**: Không test với extreme scenarios

### Giải Pháp Cụ Thể

#### a) Rolling Window Validation
```python
import numpy as np
from sklearn.metrics import classification_report

def rolling_window_validation(X, y, model, window_size=1000, step_size=500):
    """
    Validate model trên rolling windows để simulate real-world deployment
    """
    results = []

    for start_idx in range(0, len(X) - window_size, step_size):
        end_idx = start_idx + window_size

        # Training window
        X_train_window = X[:end_idx]
        y_train_window = y[:end_idx]

        # Test window (future data)
        X_test_window = X[end_idx:end_idx + step_size]
        y_test_window = y[end_idx:end_idx + step_size]

        if len(X_test_window) == 0:
            break

        # Train model on historical data
        model.fit(X_train_window, y_train_window)

        # Test on future data
        y_pred = model.predict(X_test_window)
        y_proba = model.predict_proba(X_test_window)[:, 1]

        # Calculate metrics
        report = classification_report(y_test_window, y_pred, output_dict=True)

        window_result = {
            'window_start': start_idx,
            'window_end': end_idx,
            'test_size': len(X_test_window),
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1'],
            'auc': roc_auc_score(y_test_window, y_proba)
        }

        results.append(window_result)

    return pd.DataFrame(results)
```

#### b) Adversarial Testing
```python
def adversarial_testing(model, X_test, y_test, feature_names):
    """
    Test model robustness với adversarial examples
    """
    adversarial_results = []

    # Perturbation strategies
    perturbations = {
        'amount_perturbation': lambda x: x * np.random.uniform(0.9, 1.1),  # ±10%
        'balance_noise': lambda x: x + np.random.normal(0, x * 0.05),  # 5% noise
        'type_switch': lambda x: 'TRANSFER' if x == 'PAYMENT' else 'PAYMENT'
    }

    for perturbation_name, perturbation_func in perturbations.items():
        X_perturbed = X_test.copy()

        # Apply perturbation
        if perturbation_name == 'amount_perturbation':
            X_perturbed['amount'] = X_perturbed['amount'].apply(perturbation_func)
        elif perturbation_name == 'balance_noise':
            X_perturbed['oldbalanceOrg'] = X_perturbed['oldbalanceOrg'].apply(perturbation_func)
            X_perturbed['newbalanceOrig'] = X_perturbed['newbalanceOrig'].apply(perturbation_func)
        elif perturbation_name == 'type_switch':
            X_perturbed['type'] = X_perturbed['type'].apply(perturbation_func)

        # Retransform features
        preprocessor = load_preprocessor()  # Assume function exists
        X_perturbed_processed = preprocessor.transform(X_perturbed)

        # Test model
        y_pred_original = model.predict(preprocessor.transform(X_test))
        y_pred_perturbed = model.predict(X_perturbed_processed)

        # Calculate robustness
        prediction_changes = (y_pred_original != y_pred_perturbed).sum()
        robustness_score = 1 - (prediction_changes / len(y_test))

        adversarial_results.append({
            'perturbation': perturbation_name,
            'prediction_changes': prediction_changes,
            'robustness_score': robustness_score
        })

    return pd.DataFrame(adversarial_results)
```

#### c) Fairness Assessment
```python
def fairness_analysis(model, X_test, y_test, sensitive_features):
    """
    Analyze fairness across different user groups
    """
    fairness_results = {}

    for sensitive_feature in sensitive_features:
        groups = X_test[sensitive_feature].unique()

        for group in groups:
            mask = X_test[sensitive_feature] == group
            if mask.sum() < 50:  # Skip small groups
                continue

            X_group = X_test[mask]
            y_group = y_test[mask]

            y_pred = model.predict(X_group)
            y_proba = model.predict_proba(X_group)[:, 1]

            # Calculate metrics for this group
            precision = precision_score(y_group, y_pred)
            recall = recall_score(y_group, y_pred)
            auc = roc_auc_score(y_group, y_proba)

            fairness_results[f"{sensitive_feature}_{group}"] = {
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'sample_size': len(X_group)
            }

    return fairness_results
```

#### d) Stress Testing với Extreme Scenarios
```python
def stress_testing(model, X_test, y_test):
    """
    Test model performance under extreme conditions
    """
    stress_scenarios = {
        'high_amount_only': X_test[X_test['amount'] > X_test['amount'].quantile(0.95)],
        'low_balance_only': X_test[X_test['oldbalanceOrg'] < X_test['oldbalanceOrg'].quantile(0.05)],
        'transfer_only': X_test[X_test['type'] == 'TRANSFER'],
        'late_night_only': X_test[X_test['step_hour'].isin([0, 1, 2, 3, 4, 5])]
    }

    stress_results = {}

    for scenario_name, X_scenario in stress_scenarios.items():
        if len(X_scenario) < 100:  # Skip small scenarios
            continue

        y_scenario = y_test.loc[X_scenario.index]

        y_pred = model.predict(X_scenario)
        y_proba = model.predict_proba(X_scenario)[:, 1]

        # Calculate fraud rate in scenario
        fraud_rate = y_scenario.mean()

        # Calculate detection rate
        detection_rate = precision_score(y_scenario, y_pred, pos_label=1)

        stress_results[scenario_name] = {
            'sample_size': len(X_scenario),
            'fraud_rate': fraud_rate,
            'precision': precision_score(y_scenario, y_pred),
            'recall': recall_score(y_scenario, y_pred),
            'auc': roc_auc_score(y_scenario, y_proba)
        }

    return pd.DataFrame.from_dict(stress_results, orient='index')
```

### Lợi Ích Sau Khi Cải Thiện

- **Reliability cao hơn**: Rolling validation detect model degradation over time
- **Security tốt hơn**: Adversarial testing identify vulnerabilities
- **Fairness đảm bảo**: Fairness analysis prevent discriminatory decisions
- **Risk management**: Stress testing prepare for extreme scenarios
- **Confidence tăng**: Comprehensive validation build trust với stakeholders

### Kết Luận
Comprehensive validation framework sẽ biến mô hình từ "lab prototype" thành "enterprise-grade system" có thể deploy an toàn trong môi trường fintech với hàng tỷ dollars giao dịch.