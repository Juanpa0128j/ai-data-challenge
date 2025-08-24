# XGBoost Training Pipeline

Una pipeline completa para entrenar y evaluar modelos XGBoost en la clasificación de literatura médica.

## Estructura del Proyecto

```
ai-data-challenge/
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

## API Programática

### Entrenamiento básico:
```python
from src.training.xgboost_trainer import XGBoostTrainer
from config.xgboost_config import get_config

# Cargar configuración
config = get_config('high_performance')

# Entrenar modelo
trainer = XGBoostTrainer(config)
results = trainer.run_full_pipeline(
    data_path='data/mi_dataset.csv',
    perform_cv=True
)

print(f"F1-Score: {results['test_metrics']['f1_macro']:.4f}")
```

### Análisis del modelo:
```python
from src.models.enhanced_xgboost import EnhancedXGBoostModel

# Cargar modelo entrenado
model = EnhancedXGBoostModel('models/xgboost_model.pkl')

# Hacer predicción
result = model.predict_single("Patient shows cardiovascular symptoms")
print(f"Predicted labels: {result['predicted_labels']}")

# Analizar importancia de características
importance = model.analyze_feature_importance(top_n=20)
```

### Evaluación comprensiva:
```python
from src.evaluation.xgboost_evaluator import XGBoostEvaluator

# Evaluar modelo
evaluator = XGBoostEvaluator('models/xgboost_model.pkl')
results = evaluator.comprehensive_evaluation(X_test, y_test)

# Generar reporte
report = evaluator.generate_model_report()
print(report)
```

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

## Extensiones Posibles

### 1. Validación cruzada estratificada:
```python
from sklearn.model_selection import StratifiedKFold

# Implementar en trainer personalizado
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = trainer.cross_validate(X, y, cv=skf)
```

### 2. Optimización bayesiana:
```python
# Requiere: pip install optuna
import optuna

def objective(trial):
    config = XGBoostConfig(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
    )
    # ... entrenamiento y evaluación
    return f1_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 3. Ensemble con otros modelos:
```python
from sklearn.ensemble import VotingClassifier

# Combinar XGBoost con otros modelos
ensemble = VotingClassifier([
    ('xgb', xgboost_model),
    ('rf', random_forest_model),
    ('lr', logistic_regression_model)
], voting='soft')
```

## Troubleshooting

### Problemas comunes:

1. **XGBoost no se instala**:
```bash
# En Windows
conda install -c conda-forge xgboost

# En Linux/Mac
pip install --upgrade pip
pip install xgboost
```

2. **Error de memoria**:
```python
# Reducir max_features en configuración
config.max_features = 2000

# O usar procesamiento por lotes
# Implementar en trainer personalizado
```

3. **SHAP muy lento**:
```python
# Usar muestras más pequeñas
explainer = shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X_sample[:100])
```

4. **Convergencia lenta**:
```python
# Aumentar learning_rate
config.learning_rate = 0.2
# O reducir regularización
config.reg_alpha = 0.0
config.reg_lambda = 1.0
```

## Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para detalles.
