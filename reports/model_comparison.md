# Model Comparison Report

## Data Preparation
- **Raw data:** (284807, 31)
- **After feature engineering:** (284807, 33)
- **New features:** Amount_log, Time_fraction
- **Scaling:** StandardScaler applied to all features

## Dataset for Training
- **Train set:** 199,364 samples
- **Test set:** 85,443 samples
- **Features:** 32
- **Target balance (train):** 199,020 legítimas, 344 fraudes
- **Target balance (test):** 85,295 legítimas, 148 fraudes

## Model Performance

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.9714 | 0.1192 | 0.0640 | 0.8716 |
| Random Forest | 0.9591 | 0.7914 | 0.8462 | 0.7432 |
| XGBoost | 0.9745 | 0.8357 | 0.8864 | 0.7905 |

## Winner
**XGBoost**
- ROC-AUC: 0.9745
- Status: ✅ TARGET ALCANZADO (> 0.95)

## Files Saved
- **Models:** `models/lr_model.pkl`, `models/rf_model.pkl`, `models/xgb_model.pkl`
- **Scaler:** `models/scaler.pkl`
- **Processed data:** `data/training/features_processed.csv`
- **Train/Test splits:** `data/training/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- **Visualizations:** `reports/resources/images/07_*.png`, `08_*.png`, `09_*.png`
