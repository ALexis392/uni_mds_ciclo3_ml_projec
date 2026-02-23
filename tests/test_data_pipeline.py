"""
Tests para el módulo de preparación de datos
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preparation import FraudDataPreprocessor


class TestFraudDataPreprocessor:
    """Tests para FraudDataPreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Crea instancia de preprocessor"""
        return FraudDataPreprocessor(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Crea sample data para tests - CON SUFICIENTES FRAUDES"""
        np.random.seed(42)
        n_samples = 200
        
        # Crear 150 legítimas y 50 fraudes (para stratified split)
        legit = {
            'V1': np.random.randn(150),
            'V2': np.random.randn(150),
            'V3': np.random.randn(150),
            'V4': np.random.randn(150),
            'V5': np.random.randn(150),
            'Amount': np.abs(np.random.randn(150)) * 100,
            'Time': np.arange(150),
            'Class': np.zeros(150, dtype=int)
        }
        
        fraud = {
            'V1': np.random.randn(50),
            'V2': np.random.randn(50),
            'V3': np.random.randn(50),
            'V4': np.random.randn(50),
            'V5': np.random.randn(50),
            'Amount': np.abs(np.random.randn(50)) * 150,
            'Time': np.arange(150, 200),
            'Class': np.ones(50, dtype=int)
        }
        
        legit_df = pd.DataFrame(legit)
        fraud_df = pd.DataFrame(fraud)
        
        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def test_initialization(self, preprocessor):
        """Test de inicialización"""
        assert preprocessor.random_state == 42
        assert preprocessor.scaler is not None
    
    def test_engineer_features(self, preprocessor, sample_data):
        """Test de feature engineering"""
        df_engineered = preprocessor.engineer_features(sample_data)
        
        # Verificar que se crearon nuevas columnas
        assert 'Amount_log' in df_engineered.columns
        assert 'Time_fraction' in df_engineered.columns
        
        # Verificar que Amount_log es correcto
        assert np.allclose(
            df_engineered['Amount_log'].values,
            np.log1p(sample_data['Amount'].values)
        )
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test de scaling"""
        df_engineered = preprocessor.engineer_features(sample_data)
        df_scaled = preprocessor.scale_features(df_engineered, fit=True)
        
        # Verificar que se escaló
        assert df_scaled.shape == df_engineered.shape
        
        # Features numéricos deben tener media ~0 y std ~1
        numeric_cols = [col for col in df_scaled.columns if col != 'Class']
        means = df_scaled[numeric_cols].mean()
        stds = df_scaled[numeric_cols].std()
        
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=0.1)
    
    def test_split_data(self, preprocessor, sample_data):
        """Test de split"""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.3
        )
        
        # Verificar proporciones
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.3) < 0.1  # ~30%
        
        # Verificar que y tiene mismo tamaño
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_process_pipeline(self, preprocessor, sample_data, tmp_path):
        """Test del pipeline completo"""
        # Guardar sample data temporalmente
        sample_data.to_csv(tmp_path / 'test_data.csv', index=False)
        
        X_train, X_test, y_train, y_test, scaler = preprocessor.process(
            str(tmp_path / 'test_data.csv'),
            test_size=0.3,
            save_path=None
        )
        
        # Verificar shapes
        assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Verificar que scaler existe
        assert scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])