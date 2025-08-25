# XGBoost Training Pipeline

Una pipeline completa para entrenar y evaluar modelos XGBoost en la clasificación de literatura médica.

## Estructura del Proyecto

# Medical AI Dashboard - XGBoost Literature Classification

![Dashboard Preview](https://img.shields.io/badge/Status-Ready_for_V0-success)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![API](https://img.shields.io/badge/API-Flask%2FNext.js-blue)

## 🎯 Resumen Ejecutivo

Dashboard profesional para clasificación automática de literatura médica usando XGBoost. Sistema completo con API en tiempo real, visualizaciones interactivas y datos reales del modelo entrenado.

### ✨ Características Principales
- **4 Categorías Médicas:** Cardiovascular, Neurológico, Hepatorenal, Oncológico
- **Predicciones en Tiempo Real** con API Flask/Next.js
- **Visualizaciones Interactivas** con métricas de rendimiento
- **Datos Reales** del modelo entrenado (3,565 muestras)
- **Deployment Ready** para Vercel/GitHub Pages

## 📊 Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 85.47% |
| **Precision** | 82.34% |
| **Recall** | 78.91% |
| **F1-Score** | 80.58% |
| **Muestras** | 3,565 |

## Estructura del proyecto [WIP]

```
├── config/
│   └── xgboost_config.py          # Configuraciones del modelo
├── src/
│   ├── training/
│   │   └── xgboost_trainer.py     # Pipeline de entrenamiento
│   ├── models/
│   │   └── enhanced_xgboost.py    # Modelo XGBoost mejorado
│   └── evaluation/
│       └── xgboost_evaluator.py   # Evaluación y métricas
├── models/                        # Modelos entrenados
├── results/                       # Resultados y visualizaciones
├── run_xgboost_pipeline.py        # Script principal
└── requirements_xgboost.txt       # Dependencias
```

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements_xgboost.txt
```

2. Verificar instalación de XGBoost:
```bash
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
```

## Uso

### 1. Entrenar el Modelo

#### Entrenamiento básico:
```bash
python run_xgboost_pipeline.py train
```

#### Con configuración específica:
```bash
python run_xgboost_pipeline.py train --config high_performance
```

#### Con datos personalizados:
```bash
python run_xgboost_pipeline.py train --data-path data/mi_dataset.csv
```

#### Búsqueda de hiperparámetros:
```bash
python run_xgboost_pipeline.py train --hyperparameter-search
```

### 2. Evaluar el Modelo

```bash
python run_xgboost_pipeline.py evaluate --model-path models/xgboost_model.pkl
```

Con comparación baseline:
```bash
python run_xgboost_pipeline.py evaluate --model-path models/xgboost_model.pkl --baseline-results results/baseline_results.json
```

### 3. Analizar el Modelo

#### Análisis de importancia de características:
```bash
python run_xgboost_pipeline.py analyze --model-path models/xgboost_model.pkl
```

#### Explicar predicción específica:
```bash
python run_xgboost_pipeline.py analyze --model-path models/xgboost_model.pkl --text "Patient presents with cardiovascular symptoms..."
```

#### Modo interactivo:
```bash
python run_xgboost_pipeline.py analyze --model-path models/xgboost_model.pkl --interactive
```

### 4. Comparar Configuraciones

```bash
python run_xgboost_pipeline.py compare-configs
```

## Configuraciones Disponibles

### `default`
- **Propósito**: Configuración balanceada para uso general
- **Estimadores**: 100
- **Profundidad**: 6
- **Learning Rate**: 0.1

### `fast_training`
- **Propósito**: Entrenamiento rápido para prototipado
- **Estimadores**: 50
- **Profundidad**: 4
- **Learning Rate**: 0.2

### `high_performance`
- **Propósito**: Máximo rendimiento (más lento)
- **Estimadores**: 200
- **Profundidad**: 8
- **Learning Rate**: 0.05
- **Early Stopping**: 15 rondas

### `regularized`
- **Propósito**: Control de overfitting
- **Regularización L1**: 0.5
- **Regularización L2**: 2.0
- **Gamma**: 0.1

## Características del Modelo XGBoost

```python
if XGBOOST_AVAILABLE:
    self.models['XGBoost'] = OneVsRestClassifier(
        XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss'
        )
    )
```

### Ventajas de XGBoost:
- **Alto rendimiento**: Optimizado para velocidad y precisión
- **Regularización**: Control automático de overfitting
- **Manejo de missing values**: Automático
- **Feature importance**: Análisis detallado de características
- **Early stopping**: Previene sobreentrenamiento

## Salidas y Resultados

### Archivos generados durante entrenamiento:
- `models/xgboost_model.pkl`: Modelo entrenado completo
- `results/xgboost_results.json`: Métricas detalladas
- `results/xgboost_training_YYYYMMDD_HHMMSS.log`: Log de entrenamiento

### Archivos generados durante evaluación:
- `results/xgboost_comprehensive_evaluation.png`: Visualizaciones
- `results/xgboost_detailed_evaluation.json`: Métricas detalladas
- `results/xgboost_evaluation_summary.txt`: Resumen legible
- `results/xgboost_feature_importance.png`: Importancia de características

### Archivos generados durante análisis:
- `results/shap_explanation_[class].png`: Explicaciones SHAP
- `results/xgboost_confusion_matrices.png`: Matrices de confusión

## Métricas Evaluadas

### Métricas básicas:
- **Accuracy**: Precisión general
- **F1-Score**: Macro, Micro, Weighted
- **Precision/Recall**: Macro, Micro
- **Hamming Loss**: Error en multi-label
- **Jaccard Score**: Similitud de conjuntos

### Métricas por clase:
- **AUC-ROC**: Área bajo curva ROC
- **Average Precision**: Área bajo curva PR
- **Support**: Número de muestras por clase

### Análisis multi-label:
- **Exact Match Ratio**: Predicciones perfectas
- **Label Ranking Loss**: Error de ranking
- **Frecuencia de etiquetas**: Distribución real vs predicha

## Configuración Avanzada

### Personalizar hiperparámetros:
```python
from config.xgboost_config import XGBoostConfig

custom_config = XGBoostConfig(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    reg_alpha=0.1,
    reg_lambda=1.5,
    subsample=0.8,
    colsample_bytree=0.8
)

trainer = XGBoostTrainer(custom_config)
```

### Búsqueda de hiperparámetros personalizada:
```python
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [6, 8, 10],
    'estimator__learning_rate': [0.03, 0.1, 0.2],
    'estimator__reg_alpha': [0, 0.1, 0.5],
    'estimator__reg_lambda': [1, 1.5, 2]
}

search_results = trainer.hyperparameter_search(X_train, y_train, param_grid)
```

## Interpretabilidad del Modelo

### Análisis SHAP (requiere `pip install shap`):
```python
# Explicar predicción específica
explanation = model.explain_prediction_shap(
    text="Patient presents with neurological symptoms",
    plot=True
)

# Ver características más importantes
print("Top contributing features:")
for class_name, data in explanation.items():
    print(f"{class_name}: {data['top_features'][:5]}")
```

### Importancia de características:
```python
# Analizar importancia global
importance_results = model.analyze_feature_importance(top_n=30)

# Ver características más importantes por clase
for class_name, data in importance_results.items():
    print(f"\nTop features for {class_name}:")
    for feature, importance in zip(data['features'][:10], data['importances'][:10]):
        print(f"  {feature}: {importance:.4f}")
```

## Monitoreo y Logs

Los logs se generan automáticamente durante el entrenamiento y incluyen:
- Progreso del entrenamiento
- Métricas de validación
- Tiempo de ejecución
- Errores y advertencias

Ubicación: `results/xgboost_training_YYYYMMDD_HHMMSS.log`

## Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para detalles.
