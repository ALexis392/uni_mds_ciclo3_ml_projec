# ============================================================================
# src/model_utils.py - Evaluación y métricas
# ============================================================================

"""
Model Utilities Module
Funciones auxiliares para evaluación y análisis de modelos
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluador de modelos de clasificación"""
    
    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.results = {}
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evalúa el modelo en test set"""
        logger.info(f"Evaluando {self.model_name}...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        self.results = {
            'model_name': self.model_name,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'threshold': threshold,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        logger.info(f"  ROC-AUC: {self.results['roc_auc']:.4f}")
        logger.info(f"  F1-Score: {self.results['f1']:.4f}")
        
        return self.results
    
    def get_confusion_matrix(self, y_test):
        """Retorna matriz de confusión"""
        if 'y_pred' not in self.results:
            raise ValueError("Debe llamar evaluate() primero")
        return confusion_matrix(y_test, self.results['y_pred'])
    
    def get_roc_curve(self, y_test):
        """Retorna curva ROC"""
        if 'y_pred_proba' not in self.results:
            raise ValueError("Debe llamar evaluate() primero")
        return roc_curve(y_test, self.results['y_pred_proba'])
    
    def get_precision_recall_curve(self, y_test):
        """Retorna curva Precision-Recall"""
        if 'y_pred_proba' not in self.results:
            raise ValueError("Debe llamar evaluate() primero")
        return precision_recall_curve(y_test, self.results['y_pred_proba'])
    
    def print_classification_report(self, y_test):
        """Imprime classification report"""
        if 'y_pred' not in self.results:
            raise ValueError("Debe llamar evaluate() primero")
        
        print(f"\n{'='*70}")
        print(f"Classification Report - {self.model_name}")
        print(f"{'='*70}\n")
        print(classification_report(y_test, self.results['y_pred'], 
                                   target_names=['Legitimate', 'Fraud']))


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Encuentra el threshold óptimo basado en métrica"""
    thresholds = np.arange(0, 1.01, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if (y_pred == 1).sum() == 0:
            scores.append(0)
        else:
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            scores.append(score)
    
    optimal_threshold = thresholds[np.argmax(scores)]
    return optimal_threshold


def calculate_detailed_metrics(y_true, y_pred):
    """Calcula métricas detalladas"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    
    return metrics


if __name__ == "__main__":
    print("✅ model_utils importado correctamente")