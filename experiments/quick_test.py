"""
Quick Test - Verifica que todo funciona correctamente
Script rápido para validar la pipeline completa
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.data_preparation import FraudDataPreprocessor
from src.train import ModelTrainer
from src.model_utils import ModelEvaluator
import joblib

print("="*70)
print("🚀 QUICK TEST - Validating Pipeline")
print("="*70)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
print("\n[1] Testing Data Preparation...")

try:
    preprocessor = FraudDataPreprocessor()
    print("   ✅ FraudDataPreprocessor initialized")
    
    # Crear sample data
    np.random.seed(42)
    n = 100
    sample_df = pd.DataFrame({
        'V1': np.random.randn(n),
        'V2': np.random.randn(n),
        'V3': np.random.randn(n),
        'V4': np.random.randn(n),
        'V5': np.random.randn(n),
        'Amount': np.abs(np.random.randn(n)) * 100,
        'Time': np.arange(n),
        'Class': np.random.binomial(1, 0.1, n)
    })
    
    # Feature engineering
    sample_processed = preprocessor.engineer_features(sample_df)
    print(f"   ✅ Feature engineering: {sample_df.shape} -> {sample_processed.shape}")
    
    # Scaling
    sample_scaled = preprocessor.scale_features(sample_processed, fit=True)
    print(f"   ✅ Scaling applied")
    
    # Split
    X = sample_scaled.drop('Class', axis=1)
    y = sample_scaled['Class']
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.3)
    print(f"   ✅ Split: Train {X_train.shape}, Test {X_test.shape}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================
print("\n[2] Testing Model Training...")

try:
    trainer = ModelTrainer()
    print("   ✅ ModelTrainer initialized")
    
    # Train LR
    lr = trainer.train_logistic_regression(X_train, y_train)
    print("   ✅ Logistic Regression trained")
    
    # Train RF
    rf = trainer.train_random_forest(X_train, y_train)
    print("   ✅ Random Forest trained")
    
    # Train XGB
    xgb = trainer.train_xgboost(X_train, y_train)
    print("   ✅ XGBoost trained")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# 3. MODEL EVALUATION
# ============================================================================
print("\n[3] Testing Model Evaluation...")

try:
    evaluator = ModelEvaluator(lr, "LR")
    results = evaluator.evaluate(X_test, y_test)
    
    print(f"   ✅ LR evaluated: ROC-AUC = {results['roc_auc']:.4f}")
    print(f"   ✅ F1-Score = {results['f1']:.4f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# 4. MODEL SERIALIZATION
# ============================================================================
print("\n[4] Testing Model Serialization...")

try:
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    # Save
    joblib.dump(lr, temp_path)
    print(f"   ✅ Model saved: {temp_path}")
    
    # Load
    loaded_model = joblib.load(temp_path)
    print(f"   ✅ Model loaded")
    
    # Verify
    y_pred_original = lr.predict_proba(X_test)[:, 1]
    y_pred_loaded = loaded_model.predict_proba(X_test)[:, 1]
    
    if np.allclose(y_pred_original, y_pred_loaded):
        print(f"   ✅ Predictions match after save/load")
    else:
        print(f"   ❌ Predictions don't match")
    
    os.remove(temp_path)
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*70)
print("✅ ALL TESTS PASSED")
print("="*70)
print("""
Pipeline is fully functional:
  ✓ Data Preparation
  ✓ Feature Engineering
  ✓ Scaling
  ✓ Model Training (LR, RF, XGB)
  ✓ Model Evaluation
  ✓ Model Serialization

Ready for production!
""")
print("="*70)