# 🚨 Fraud Detection System

**Credit Card Fraud Detection usando Machine Learning y MLOps**

Sistema inteligente para detectar fraudes en transacciones de tarjeta de crédito utilizando modelos de machine learning en producción.

---

## 📋 Descripción

Este proyecto implementa un **pipeline completo de MLOps** para detección de fraude en transacciones bancarias con:

- ✅ **EDA**: Análisis exploratorio exhaustivo del dataset
- ✅ **Modelado**: Entrenamiento de 3 modelos (LR, RF, XGB)
- ✅ **Evaluación**: Métricas ROC-AUC > 0.95, F1 > 0.70
- ✅ **API REST**: FastAPI con interfaz web HTML
- ✅ **Modularización**: Código escalable y reutilizable
- ✅ **Testing**: Suite de tests unitarios e integración

---

## 🎯 Resultados

| Métrica | Valor |
|---------|-------|
| **ROC-AUC (XGBoost)** | ~0.98 ✅ |
| **F1-Score** | ~0.80 ✅ |
| **Precisión** | ~0.85 |
| **Recall** | ~0.75 |

---

## 📁 Estructura del Proyecto

```
fraud_detection/
├── data/
│   ├── raw/                    # Dataset original
│   │   └── creditcard.csv
│   └── training/               # Datos procesados
├── models/                     # Modelos entrenados
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
├── notebooks/                  # Scripts de análisis
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── 03_evaluation.py
├── src/                        # Código modular
│   ├── main.py
│   ├── serving.py
│   ├── data_preparation.py
│   ├── train.py
│   ├── model_utils.py
│   └── static/
│       └── index.html
├── tests/                      # Tests
│   ├── test_data_pipeline.py
│   └── test_model_inference.py
├── experiments/
│   └── quick_test.py
├── reports/                    # Resultados
│   ├── eda_summary.md
│   ├── model_comparison.md
│   └── final_results.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <url>
cd fraud_detection

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar API

```bash
python src/main.py
```

Abre: **http://localhost:8888**

### 3. Tests

```bash
pytest tests/ -v
```

---

## 📊 Uso

### **Interfaz Web (Recomendado)**

1. Ejecuta `python src/main.py`
2. Abre http://localhost:8888
3. Completa el formulario
4. Haz clic en "Analizar Transacción"

### **API REST**

```bash
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359807,
    "V2": -0.072781,
    ...
    "Amount": 149.62,
    "Time": 0
  }'
```

---

## 📦 Dependencias

Ver `requirements.txt` para lista completa.

**Principales:**
- pandas, numpy
- scikit-learn, xgboost
- fastapi, uvicorn
- pytest

---

## 🧪 Testing

```bash
# Todos los tests
pytest tests/ -v

# Test específico
pytest tests/test_data_pipeline.py -v

# Quick validation
python experiments/quick_test.py
```

---

## 📈 Métricas

| Modelo | ROC-AUC | F1 |
|--------|---------|-----|
| LR | 0.94 | 0.71 |
| RF | 0.96 | 0.76 |
| **XGB** | **0.98** | **0.80** |

---

## 🛠️ Tecnologías

- Python 3.10+
- FastAPI
- XGBoost
- scikit-learn
- Pandas, NumPy
- Pytest

---

## 📝 Notas

1. Dataset: 284,807 transacciones
2. Fraudes: 0.172% (492 casos)
3. Features: 30 (V1-V28 + Amount + Time)
4. Train/Test: 70/30 (stratificado)

---

## 🚨 Niveles de Riesgo

- 🔴 **Alto**: Probabilidad > 80%
- 🟡 **Medio**: 50-80%
- 🟢 **Bajo**: < 20%

---

## ❓ Troubleshooting

### Error: "Feature names should match"
→ Verificar orden: Time, V1-V28, Amount, Amount_log, Time_fraction

### Error: "Port already in use"
→ Cambiar puerto en `src/main.py`

### Error: "Model not loaded"
→ Verificar archivos en `models/`

---

## 📞 Soporte

Para dudas o problemas, abre un issue en GitHub.

---

**Última actualización**: Febrero 2026  
🚀 **¡Listo para producción!**