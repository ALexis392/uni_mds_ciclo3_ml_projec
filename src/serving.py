"""
Serving Module
API REST para servir el modelo de fraude
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="API para detectar fraude en transacciones de tarjeta de crédito",
    version="1.0.0"
)

# Servir archivos estáticos
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Cargar modelo y scaler
try:
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("✅ Modelo y scaler cargados")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    model = None
    scaler = None


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class TransactionFeatures(BaseModel):
    """Features de una transacción"""
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float
    Amount_log: float = None
    Time_fraction: float = None
    
    class Config:
        schema_extra = {
            "example": {
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536346,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.024798,
                "V19": 0.794992,
                "V20": 0.528960,
                "V21": -0.037606,
                "V22": 0.261057,
                "V23": 0.003725,
                "V24": 0.000788,
                "V25": -0.002137,
                "V26": -0.000727,
                "V27": -0.000627,
                "V28": -0.000216,
                "Amount": 149.62,
                "Time": 0
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    fraud_probability: float
    prediction: str
    risk_level: str
    confidence: float


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Página principal con interfaz web"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
def health_check():
    """Verifica que la API esté operativa"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Fraud Detection API is running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    """Predice si una transacción es fraudulenta"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        # Crear diccionario EN EL ORDEN EXACTO QUE ESPERA EL SCALER
        data_dict = {
            'Time': transaction.Time,
            'V1': transaction.V1,
            'V2': transaction.V2,
            'V3': transaction.V3,
            'V4': transaction.V4,
            'V5': transaction.V5,
            'V6': transaction.V6,
            'V7': transaction.V7,
            'V8': transaction.V8,
            'V9': transaction.V9,
            'V10': transaction.V10,
            'V11': transaction.V11,
            'V12': transaction.V12,
            'V13': transaction.V13,
            'V14': transaction.V14,
            'V15': transaction.V15,
            'V16': transaction.V16,
            'V17': transaction.V17,
            'V18': transaction.V18,
            'V19': transaction.V19,
            'V20': transaction.V20,
            'V21': transaction.V21,
            'V22': transaction.V22,
            'V23': transaction.V23,
            'V24': transaction.V24,
            'V25': transaction.V25,
            'V26': transaction.V26,
            'V27': transaction.V27,
            'V28': transaction.V28,
            'Amount': transaction.Amount,
            'Amount_log': np.log1p(transaction.Amount),
            'Time_fraction': (transaction.Time % 86400) / 86400
        }
        
        # Crear DataFrame con el orden correcto
        X = pd.DataFrame([data_dict])
        
        # Asegurar que el orden coincide exactamente
        col_order = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 
                     'Amount_log', 'Time_fraction']
        X = X[col_order]
        
        # Escalar
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=col_order
        )
        
        # Predicción
        fraud_prob = float(model.predict_proba(X_scaled)[0, 1])
        
        is_fraud = fraud_prob >= 0.5
        prediction = "Fraud" if is_fraud else "Legitimate"
        risk_level = "high" if fraud_prob >= 0.8 else ("medium" if fraud_prob >= 0.5 else ("low" if fraud_prob < 0.2 else "medium"))
        confidence = max(fraud_prob, 1 - fraud_prob)
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            prediction=prediction,
            risk_level=risk_level,
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(transactions: list[TransactionFeatures]):
    """
    Predice múltiples transacciones (batch)
    
    **Retorna:**
    - Lista de predicciones
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    results = []
    for transaction in transactions:
        result = predict(transaction)
        results.append(result)
    
    return {"predictions": results, "count": len(results)}


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 INICIANDO API")
    print("="*70)
    print("\n📍 URL: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("📊 ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )