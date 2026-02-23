# ============================================================================
# src/data_preparation.py - Pipeline de datos
# ============================================================================

"""
Data Preparation Module
Preparación y transformación de datos para el modelo de fraude
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataPreprocessor:
    """Preprocesamiento de datos para fraude detection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_data(self, filepath):
        """Carga datos raw"""
        logger.info(f"Cargando datos desde {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Shape: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """Feature engineering"""
        logger.info("Realizando feature engineering...")
        df_processed = df.copy()
        
        # 1. Log transform de Amount
        df_processed['Amount_log'] = np.log1p(df_processed['Amount'])
        
        # 2. Time fraction (fracción del día)
        df_processed['Time_fraction'] = (df_processed['Time'] % 86400) / 86400
        
        logger.info(f"Features creados: Amount_log, Time_fraction")
        return df_processed
    
    def scale_features(self, df, fit=True):
        """Escalado de features"""
        logger.info("Escalando features...")
        
        # Columnas a escalar (todo excepto Class)
        columns_to_scale = [col for col in df.columns if col != 'Class']
        self.feature_columns = columns_to_scale
        
        if fit:
            df_scaled = df.copy()
            df_scaled[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        else:
            df_scaled = df.copy()
            df_scaled[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        logger.info(f"Escaladas {len(columns_to_scale)} features")
        return df_scaled
    
    def split_data(self, X, y, test_size=0.3):
        """Split train/test"""
        logger.info(f"Splitting datos (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def process(self, filepath, test_size=0.3, save_path=None):
        """Pipeline completo"""
        logger.info("Iniciando pipeline de preparación...")
        
        # Cargar
        df = self.load_data(filepath)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Scaling
        df = self.scale_features(df, fit=True)
        
        # Separar X, y
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Guardar si se especifica
        if save_path:
            logger.info(f"Guardando datos en {save_path}...")
            X_train.to_csv(f'{save_path}/X_train.csv', index=False)
            X_test.to_csv(f'{save_path}/X_test.csv', index=False)
            y_train.to_csv(f'{save_path}/y_train.csv', index=False)
            y_test.to_csv(f'{save_path}/y_test.csv', index=False)
            
            # Guardar scaler
            joblib.dump(self.scaler, f'{save_path}/../scaler.pkl')
            logger.info("Scaler guardado")
        
        logger.info("Pipeline completado")
        return X_train, X_test, y_train, y_test, self.scaler
    
    def transform_new_data(self, df):
        """Transformar nuevos datos (sin fit)"""
        df_processed = self.engineer_features(df)
        df_scaled = self.scale_features(df_processed, fit=False)
        return df_scaled.drop('Class', axis=1) if 'Class' in df_scaled.columns else df_scaled


if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor()
    X_train, X_test, y_train, y_test, scaler = preprocessor.process(
        'data/raw/creditcard.csv',
        test_size=0.3,
        save_path='data/training'
    )
    print("✅ Procesamiento completado")
