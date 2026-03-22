# Fraud Detection Data Preprocessing Pipeline - Quy Trình Chuẩn

## Tổng Quan Pipeline

Preprocessing cho Fraud Detection là một quy trình có thứ tự nghiêm ngặt. **Thứ tự rất quan trọng** vì sai thứ tự có thể gây data leakage, làm model overfit và không khái quát hóa được.

```
Raw Data → Inspection → Cleaning → Feature Engineering → Split → Scaling/Encoding → Imbalance Handling → Model Ready
```

## 1. Data Loading (Bước Đầu Tiên)

### Mục đích
- Đọc dữ liệu thô từ source
- Tạo DataFrame ban đầu
- Kiểm tra basic integrity

### Code Example
```python
import pandas as pd

def load_data(data_path):
    """Load PaySim dataset"""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")
    return df

# Usage
df = load_data("../data/paysim.csv")
```

### Tại sao làm trước?
- Cần dữ liệu thô để inspect
- Không thể làm gì nếu chưa có data

### Lưu ý với PaySim
- Dataset ~6M rows, cần đủ RAM
- Columns: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud

## 2. Initial Data Inspection (Ngay Sau Loading)

### Mục đích
- Hiểu structure của data
- Phát hiện issues cơ bản
- Lên kế hoạch preprocessing

### Các việc cần làm
```python
def inspect_data(df):
    print("=== DATA INSPECTION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicated rows: {df.duplicated().sum()}")
    print(f"Target distribution:\n{df['isFraud'].value_counts(normalize=True)}")

    # Check for data quality issues
    print(f"Negative amounts: {(df['amount'] < 0).sum()}")
    print(f"Zero amounts: {(df['amount'] == 0).sum()}")
    print(f"Negative balances: {(df[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] < 0).any(axis=1).sum()}")

inspect_data(df)
```

### Tại sao làm ngay sau loading?
- Cần hiểu data trước khi xử lý
- Phát hiện missing values, outliers sớm
- Quyết định strategy cleaning

### Với PaySim
- Không có missing values (theo design)
- Imbalanced: ~99.8% legitimate, 0.2% fraud
- Có thể có negative balances (thiết kế của dataset)

## 3. Data Cleaning (Sau Inspection)

### Mục đích
- Xử lý missing values, duplicates, outliers
- Đảm bảo data quality
- Chuẩn bị cho feature engineering

### Thứ tự xử lý
```python
def clean_data(df):
    """Clean PaySim data"""
    print("=== DATA CLEANING ===")

    # 1. Handle missing values (PaySim thường không có)
    df = df.dropna()  # Simple drop, có thể interpolate nếu cần

    # 2. Remove duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicates")

    # 3. Handle invalid values
    # Negative amounts - có thể là errors
    df = df[df['amount'] >= 0]

    # 4. Remove unnecessary columns
    # nameOrig, nameDest là IDs, không dùng cho ML
    # isFlaggedFraud là engineered feature, có thể redundant
    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 5. Handle outliers (optional, careful với fraud detection)
    # Có thể remove extreme outliers nhưng giữ fraud cases
    fraud_cases = df[df['isFraud'] == 1]
    legitimate_cases = df[df['isFraud'] == 0]

    # Remove extreme outliers only from legitimate cases
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig']:
        if col in legitimate_cases.columns:
            q99 = legitimate_cases[col].quantile(0.99)
            legitimate_cases = legitimate_cases[legitimate_cases[col] <= q99]

    df = pd.concat([fraud_cases, legitimate_cases])

    print(f"After cleaning: {df.shape}")
    return df

df_clean = clean_data(df)
```

### Tại sao làm trước feature engineering?
- Feature engineering cần data sạch
- Outliers có thể làm distortion statistics
- Missing values làm fail calculations

### Lỗi phổ biến
- **Xóa fraud cases khi remove outliers**: Fraud thường có extreme values
- **Fill missing với global mean**: Leakage nếu fill trước split

## 4. Feature Engineering (Sau Cleaning, Trước Split)

### Mục đích
- Tạo features mới từ data có sẵn
- Capture domain knowledge
- Tăng predictive power

### Thứ tự quan trọng
```python
def engineer_features(df):
    """Create fraud detection features"""
    print("=== FEATURE ENGINEERING ===")

    # 1. Balance difference features
    df['balance_diff_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # 2. Transaction amount patterns
    # Large amount indicator (cần threshold từ toàn bộ data cho consistency)
    amount_threshold = df['amount'].quantile(0.95)
    df['is_large_amount'] = (df['amount'] > amount_threshold).astype(int)

    # 3. Time-based features
    df['step_hour'] = df['step'] % 24
    df['step_day'] = df['step'] // 24

    # 4. Business logic features
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)
    df['suspicious_balance_change'] = (
        (df['balance_diff_org'] < -df['amount'] * 0.9) |  # Large reduction
        (df['balance_diff_dest'] > df['amount'] * 0.9)     # Large increase
    ).astype(int)

    print(f"Created {len(df.columns) - 8} new features")  # 8 original columns
    return df

df_featured = engineer_features(df_clean)
```

### Tại sao làm trước split?
- **Nếu dùng global statistics**: Threshold như quantile phải tính từ toàn bộ data
- **Domain features**: Logic business áp dụng cho tất cả data
- **Tránh leakage**: Nếu split trước, train/test có threshold khác nhau

### Với PaySim
- `balance_diff_org`: Thường = amount trong fraud (newbalanceOrig = 0)
- `is_large_amount`: Fraud thường có amount lớn
- Time features: Fraud có thể cluster theo giờ

### Lỗi phổ biến
- **Tính threshold sau split**: Train/test có threshold khác nhau → inconsistency
- **Dùng future data**: Trong time series, dùng thông tin từ tương lai

## 5. Train/Test Split (Sau Feature Engineering)

### Mục đích
- Tạo tập train để học patterns
- Tạo tập test để đánh giá generalization
- Simulate real-world deployment

### Cách đúng cho Fraud Detection
```python
def split_data(df, test_size=0.2):
    """Time-based split for temporal fraud data"""
    print("=== TRAIN/TEST SPLIT ===")

    # Sort by time (step) để đảm bảo temporal order
    df = df.sort_values('step').reset_index(drop=True)

    # Split chronologically
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    print(f"Train period: steps {train_df['step'].min()} - {train_df['step'].max()}")
    print(f"Test period: steps {test_df['step'].min()} - {test_df['step'].max()}")
    print(f"Train fraud rate: {train_df['isFraud'].mean():.4f}")
    print(f"Test fraud rate: {test_df['isFraud'].mean():.4f}")

    return train_df, test_df

train_df, test_df = split_data(df_featured)
```

### Tại sao làm sau feature engineering?
- Features đã sẵn sàng
- Đảm bảo train/test có cùng feature space

### Tại sao time-based split?
- Fraud patterns thay đổi theo thời gian
- Simulate predict future frauds
- Tránh data leakage từ temporal dependencies

### Lỗi phổ biến
- **Random split**: Model học patterns từ future, không thực tế
- **Stratified split**: Có thể leak temporal information
- **Split sau scaling**: Scaling parameters từ test data

## 6. Feature Scaling & Encoding (Sau Split)

### Mục đích
- Chuẩn hóa numerical features
- Encode categorical features
- Đảm bảo model convergence

### Quy trình đúng
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def scale_and_encode(train_df, test_df):
    """Fit on train, transform on both"""
    print("=== FEATURE SCALING & ENCODING ===")

    # Define feature types
    categorical_cols = ['type']
    numerical_cols = [col for col in train_df.columns
                     if col not in categorical_cols + ['isFraud']]

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])

    # Fit on training data ONLY
    X_train = preprocessor.fit_transform(train_df.drop('isFraud', axis=1))
    X_test = preprocessor.transform(test_df.drop('isFraud', axis=1))

    # Get feature names
    num_features = numerical_cols
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = num_features + list(cat_features)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature names: {len(feature_names)}")

    return X_train, X_test, preprocessor, feature_names

X_train, X_test, preprocessor, feature_names = scale_and_encode(train_df, test_df)
```

### Tại sao fit chỉ trên train?
- Tránh data leakage từ test statistics
- Simulate real deployment (không biết test data)

### Lỗi phổ biến
- **Fit trên toàn bộ data**: Leakage, model biết test distribution
- **Scale trước split**: Cùng vấn đề
- **Use different scalers**: Inconsistency

## 7. Handle Class Imbalance (Sau Scaling, Chỉ Train)

### Mục đích
- Balance classes để model học tốt minority class
- Cải thiện recall cho fraud detection

### Cách đúng
```python
from imblearn.over_sampling import SMOTE

def handle_imbalance(X_train, y_train):
    """Apply SMOTE only on training data"""
    print("=== CLASS IMBALANCE HANDLING ===")

    print(f"Original class distribution: {pd.Series(y_train).value_counts()}")

    # SMOTE to oversample minority class
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts()}")
    print(f"Balanced training shape: {X_train_balanced.shape}")

    return X_train_balanced, y_train_balanced

X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
y_test = test_df['isFraud'].values  # Keep original test labels
```

### Tại sao chỉ trên train?
- Test phải reflect real distribution
- SMOTE tạo synthetic data, không nên dùng cho evaluation

### Với PaySim
- Fraud rate ~0.2%, SMOTE tăng lên 50%
- Cải thiện model sensitivity

## 8. Final Data Preparation (Cuối Cùng)

### Mục đích
- Chuẩn bị final X, y cho model
- Save preprocessors cho inference

### Code
```python
import joblib

def prepare_final_data(X_train_balanced, X_test, y_train_balanced, y_test, preprocessor, feature_names):
    """Final preparation"""
    print("=== FINAL DATA PREPARATION ===")

    # Save preprocessor for inference
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    joblib.dump(feature_names, '../models/feature_names.pkl')

    print("Data ready for model training!")
    print(f"Train: {X_train_balanced.shape}, Test: {X_test.shape}")

    return X_train_balanced, X_test, y_train_balanced, y_test

X_train_final, X_test_final, y_train_final, y_test_final = prepare_final_data(
    X_train_balanced, X_test, y_train_balanced, y_test, preprocessor, feature_names
)
```

## Pipeline Hoàn Chỉnh

### Training Pipeline
```
Raw Data → Inspect → Clean → Feature Engineering → Time Split → Scale/Encode → SMOTE → Model Training
```

### Inference Pipeline
```
Raw Transaction → Load Preprocessor → Clean → Feature Engineering → Scale/Encode → Predict
```

### Code Inference
```python
def preprocess_inference(transaction_data):
    """Inference preprocessing"""
    # Load saved preprocessor
    preprocessor = joblib.load('../models/preprocessor.pkl')
    feature_names = joblib.load('../models/feature_names.pkl')

    # Convert to DataFrame
    df = pd.DataFrame([transaction_data])

    # Apply same cleaning and feature engineering
    df = clean_data(df)
    df = engineer_features(df)

    # Transform using saved preprocessor
    X_processed = preprocessor.transform(df)

    return X_processed, feature_names
```

## Các Lỗi Phổ Biến & Tại Sao

### 1. Data Leakage từ Feature Engineering
```python
# SAI: Tính threshold từ toàn bộ data sau split
train_df, test_df = train_test_split(df)
threshold = df['amount'].quantile(0.95)  # Leakage!
```

**Tại sao sai**: Model biết test distribution thông qua threshold.

### 2. Scaling Trước Split
```python
# SAI
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Leakage!
X_train, X_test = train_test_split(df_scaled)
```

**Tại sao sai**: Scaler học từ test data.

### 3. SMOTE Trên Toàn Bộ Data
```python
# SAI
X_full, y_full = SMOTE().fit_resample(X, y)
X_train, X_test = train_test_split(X_full, y_full)
```

**Tại sao sai**: Test data bị synthetic, không reflect real distribution.

### 4. Random Split Với Temporal Data
```python
# SAI cho fraud detection
X_train, X_test = train_test_split(df, random_state=42)
```

**Tại sao sai**: Model predict "past" frauds thay vì future.

## Kết Luận

Thứ tự preprocessing chuẩn cho Fraud Detection:

1. **Load** → 2. **Inspect** → 3. **Clean** → 4. **Feature Engineering** → 5. **Split** → 6. **Scale/Encode** → 7. **Imbalance Handling** → 8. **Model Ready**

**Logic chính**: Xử lý data càng sớm càng tốt, nhưng transformations (scaling, encoding) phải fit chỉ trên train. Feature engineering có thể dùng global stats nếu không leak temporal information.

**Với PaySim**: Time-based split rất quan trọng vì dataset có temporal patterns. Fraud detection luôn cần simulate real-world deployment scenarios.