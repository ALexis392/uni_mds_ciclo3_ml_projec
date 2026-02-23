"""
Train Module
Entrenamiento de modelos
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Entrenador de modelos"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_logistic_regression(self, X_train, y_train):
        """Entrena Logistic Regression"""
        logger.info("Entrenando Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        
        self.models['lr'] = model
        logger.info("✅ LR entrenado")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Entrena Random Forest"""
        logger.info("Entrenando Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        self.models['rf'] = model
        logger.info("✅ RF entrenado")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Entrena XGBoost"""
        logger.info("Entrenando XGBoost...")
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        self.models['xgb'] = model
        logger.info("✅ XGB entrenado")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evalúa un modelo"""
        logger.info(f"Evaluando {model_name}...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        self.results[model_name] = results
        
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        
        return results
    
    def save_model(self, model_name, filepath):
        """Guarda un modelo"""
        if model_name not in self.models:
            logger.error(f"Modelo {model_name} no entrenado")
            return
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Modelo guardado: {filepath}")
    
    def get_best_model(self):
        """Retorna mejor modelo según ROC-AUC"""
        if not self.results:
            logger.error("Sin resultados para evaluar")
            return None
        
        best_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        return best_name, self.models[best_name]
    
    def train_all(self, X_train, y_train):
        """Entrena todos los modelos"""
        logger.info("="*70)
        logger.info("ENTRENANDO TODOS LOS MODELOS")
        logger.info("="*70)
        
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        logger.info("✅ Todos los modelos entrenados")
    
    def compare_models(self, X_test, y_test):
        """Compara todos los modelos"""
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN DE MODELOS")
        logger.info("="*70)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model, X_test, y_test)
        
        # Imprimir tabla comparativa
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        
        results_df = pd.DataFrame([
            {
                'Model': self.results[name]['model_name'],
                'ROC-AUC': self.results[name]['roc_auc'],
                'F1-Score': self.results[name]['f1'],
                'Precision': self.results[name]['precision'],
                'Recall': self.results[name]['recall']
            }
            for name in self.results.keys()
        ])
        
        print("\n" + results_df.to_string(index=False))
        
        best_name, best_model = self.get_best_model()
        print(f"\n🏆 MEJOR MODELO: {best_name.upper()}")
        print(f"   ROC-AUC: {self.results[best_name]['roc_auc']:.4f}")
        
        return results_df


if __name__ == "__main__":
    # Ejemplo de uso
    from data_preparation import FraudDataPreprocessor
    
    # Preparar datos
    preprocessor = FraudDataPreprocessor()
    X_train, X_test, y_train, y_test, scaler = preprocessor.process(
        '../data/raw/creditcard.csv',
        test_size=0.3,
        save_path='../data/training'
    )
    
    # Entrenar modelos
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)
    
    # Comparar
    trainer.compare_models(X_test, y_test)
    
    # Mejor modelo
    best_name, best_model = trainer.get_best_model()
    print(f"\n✅ Mejor modelo: {best_name}")
    
    # Guardar
    trainer.save_model(best_name, f'../models/{best_name}_model.pkl')