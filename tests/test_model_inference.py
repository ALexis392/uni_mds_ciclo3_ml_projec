"""
Tests para inferencia de modelos
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import ModelEvaluator, find_optimal_threshold, calculate_detailed_metrics


class TestModelEvaluator:
    """Tests para ModelEvaluator"""
    
    @pytest.fixture
    def sample_data(self):
        """Crea sample data"""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'V1': np.random.randn(n_samples),
            'V2': np.random.randn(n_samples),
            'V3': np.random.randn(n_samples),
            'Amount': np.abs(np.random.randn(n_samples)) * 100,
            'Time': np.arange(n_samples)
        })
        
        y = np.random.binomial(1, 0.1, n_samples)
        
        return X, y
    
    @pytest.fixture
    def mock_model(self):
        """Crea un modelo mock"""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.binomial(1, 0.1, 100)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        return model
    
    def test_evaluator_initialization(self, mock_model):
        """Test de inicialización"""
        evaluator = ModelEvaluator(mock_model, "Test Model")
        
        assert evaluator.model_name == "Test Model"
        assert evaluator.model is not None
    
    def test_evaluate(self, mock_model, sample_data):
        """Test de evaluación"""
        X, y = sample_data
        evaluator = ModelEvaluator(mock_model, "Test")
        
        results = evaluator.evaluate(X, y, threshold=0.5)
        
        # Verificar que retorna diccionario con métricas
        assert 'roc_auc' in results
        assert 'f1' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'y_pred_proba' in results
        assert 'y_pred' in results
        
        # Verificar rangos
        assert 0 <= results['roc_auc'] <= 1
        assert 0 <= results['f1'] <= 1


class TestThresholdOptimization:
    """Tests para threshold optimization"""
    
    def test_find_optimal_threshold(self):
        """Test de threshold óptimo"""
        # Crear datos ficticios
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        threshold = find_optimal_threshold(y_true, y_proba, metric='f1')
        
        # Threshold debe estar entre 0 y 1
        assert 0 <= threshold <= 1


class TestDetailedMetrics:
    """Tests para cálculo de métricas detalladas"""
    
    def test_calculate_detailed_metrics(self):
        """Test de métricas detalladas"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        
        metrics = calculate_detailed_metrics(y_true, y_pred)
        
        # Verificar que todas las métricas están presentes
        required_keys = ['TP', 'TN', 'FP', 'FN', 'Accuracy', 
                        'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
        
        for key in required_keys:
            assert key in metrics
        
        # Verificar que los valores están en rangos válidos
        for key in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']:
            assert 0 <= metrics[key] <= 1 or np.isnan(metrics[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])