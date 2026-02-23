# Final Evaluation Report

## Best Model: XGBoost

### Performance Metrics (Test Set)
- **ROC-AUC:** 0.9745
- **F1-Score:** 0.8519
- **Precision:** 0.9426
- **Recall:** 0.7770
- **Accuracy:** 0.9995

### Threshold Optimization
- **Optimal Threshold:** 0.88
- **F1-Score at Optimal:** 0.8519

### Confusion Matrix
| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| Actual Legitimate | 85288 | 7 |
| Actual Fraud | 33 | 115 |

### Detailed Metrics
- **True Negatives:** 85288
- **True Positives:** 115
- **False Positives:** 7
- **False Negatives:** 33
- **Sensitivity (TPR):** 0.7770
- **Specificity (TNR):** 0.9999
- **False Positive Rate:** 0.0001
- **False Negative Rate:** 0.2230

### Model Comparison (ROC-AUC on Test Set)
| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.9714 | 0.1192 | 0.0640 | 0.8716 |
| Random Forest | 0.9591 | 0.7914 | 0.8462 | 0.7432 |
| XGBoost | 0.9745 | 0.8357 | 0.8864 | 0.7905 |

### Artifacts Generated
- Visualizations: `reports/resources/images/10_evaluation_complete.png`
- This report: `reports/final_results.md`

### Conclusions
- Target ROC-AUC (> 0.95): ACHIEVED
- Best threshold found: 0.88
- Model is ready for deployment
