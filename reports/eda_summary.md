# EDA Summary - Credit Card Fraud Detection

## Dataset Overview
- Shape: (284807, 31)
- Registros: 284,807
- Features: 31
- Missing values: 0
- Duplicados: 1081

## Target Distribution
- Legítimas: 284,315 (99.828%)
- Fraudes: 492 (0.172%)
- Ratio: 1 fraude por cada 577 legítimas

## Amount Statistics
- Mean (Legítimas): $88.29
- Mean (Fraudes): $122.21
- Max: $25691.16
- Min: $0.00

## Top 10 Features Correlated with Fraud
V11       0.154876
V4        0.133447
V2        0.091289
V21       0.040413
V19       0.034783
V20       0.020090
V8        0.019875
V27       0.017580
V28       0.009536
Amount    0.005632

## Key Findings
1. Severo desbalance de clases (0.17% fraudes)
2. Sin missing values
3. Amount es diferente entre clases
4. Features PCA correlacionadas con fraude
5. No hay patrón temporal claro
6. No hay duplicados

## Decisiones para Modeling
- Usar todas las 31 features
- Class weights en modelos
- Log transform de Amount
- Split 70-15-15 stratified
- ROC-AUC > 0.95 como target
