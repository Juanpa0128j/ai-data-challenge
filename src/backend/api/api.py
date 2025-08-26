"""
API Flask para predicciones en tiempo real
Para integración con V0 dashboard
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import sys
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Importar EnhancedXGBoostModel
from ..models.enhanced_xgboost import EnhancedXGBoostModel

app = Flask(__name__)
CORS(app)  # Permitir requests desde cualquier origen para desarrollo

# Variables globales para el modelo
enhanced_model = None
category_names = ['cardiovascular', 'neurological', 'hepatorenal', 'oncological']

def load_models():
    """Cargar modelo y componentes guardados"""
    global enhanced_model
    
    try:
        # Cargar modelo usando EnhancedXGBoostModel
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
        if os.path.exists(model_path):
            enhanced_model = EnhancedXGBoostModel(model_path)
            print("Modelo XGBoost y componentes cargados exitosamente")
            return True
        else:
            print("No se encontró el archivo de modelo guardado.")
            return False
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar estado de la API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': enhanced_model is not None,
        'vectorizer_loaded': enhanced_model.vectorizer is not None if enhanced_model else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Realizar predicción para texto médico utilizando EnhancedXGBoostModel"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Se requiere campo "text"'}), 400
        
        text = data['text']
        
        if not enhanced_model:
            return jsonify({'error': 'Modelo no está cargado'}), 500
        
        # Usar el método predict_single de EnhancedXGBoostModel
        prediction_result = enhanced_model.predict_single(text)
        
        # Formatear resultados para la API
        results = []
        for category in category_names:
            probability = prediction_result['class_probabilities'].get(category, 0.0)
            predicted = category in prediction_result['predicted_labels']
            results.append({
                'category': category,
                'probability': float(probability),
                'predicted': predicted,
                'confidence': 'Alta' if probability > 0.8 else 'Media' if probability > 0.5 else 'Baja'
            })
        
        # Obtener características más importantes si está disponible
        feature_importance = []
        if hasattr(enhanced_model.model, 'feature_importances_'):
            # Vectorizar el texto para obtener características
            text_vectorized = enhanced_model.vectorizer.transform([text])
            text_sparse = text_vectorized.toarray()[0]
            non_zero_indices = np.nonzero(text_sparse)[0]
            
            for idx in non_zero_indices[:10]:  # Top 10 features
                if idx < len(enhanced_model.feature_names):
                    importance = float(enhanced_model.model.feature_importances_[idx]) if idx < len(enhanced_model.model.feature_importances_) else 0.0
                    feature_importance.append({
                        'feature': enhanced_model.feature_names[idx],
                        'value': float(text_sparse[idx]),
                        'importance': importance
                    })
            
            # Ordenar por importancia
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
        
        response = {
            'text': text,
            'predictions': results,
            'feature_importance': feature_importance[:5],  # Top 5 features
            'prediction_summary': {
                'total_categories': len([r for r in results if r['predicted']]),
                'max_probability': max([r['probability'] for r in results]),
                'primary_category': max(results, key=lambda x: x['probability'])['category']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Obtener información sobre el modelo"""
    try:
        if not enhanced_model:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        # Usar el método model_summary de EnhancedXGBoostModel
        summary = enhanced_model.model_summary()
        
        # Get model params and filter for JSON serializable values
        raw_params = getattr(enhanced_model.model, 'get_params', lambda: {})()
        serializable_types = (str, int, float, bool, type(None))
        model_params = {k: v if isinstance(v, serializable_types) else str(v) for k, v in raw_params.items()}

        info = {
            'model_type': str(type(enhanced_model.model).__name__),
            'categories': category_names,
            'n_features': len(enhanced_model.feature_names) if enhanced_model.feature_names is not None else 0,
            'model_params': model_params,
            'training_info': {
                'algorithm': 'XGBoost',
                'task_type': 'Multi-label Classification',
                'text_processing': 'TF-IDF Vectorization'
            }
        }

        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Error obteniendo info: {str(e)}'}), 500

@app.route('/api/demo-examples', methods=['GET'])
def demo_examples():
    """Obtener ejemplos de textos para demo"""
    examples = [
        {
            'title': 'Hypertensive response during dobutamine stress echocardiography.',
            'text': 'Among 3,129 dobutamine stress echocardiographic studies, a hypertensive response, defined as systolic blood pressure (BP) > or = 220 mm Hg and/or diastolic BP > or = 110 mm Hg, occurred in 30 patients (1%). Patients with this response more often had a history of hypertension and had higher resting systolic and diastolic BP before dobutamine infusion.',
            'expected_categories': ['cardiovascular']
        },
        {
            'title': 'The interpeduncular nucleus regulates nicotine\'s effects on free-field activity.',
            'text': 'Partial lesions were made with kainic acid in the interpeduncular nucleus of the ventral midbrain of the rat. Compared with sham-operated controls, lesions significantly (p < 0.25) blunted the early (<60 min) free-field locomotor hypoactivity caused by nicotine (0.5 mg kg(-1), i.m.), enhanced the later (60-120 min) nicotine-induced hyperactivity, and raised spontaneous nocturnal activity. Lesions reduced the extent of immunohistological staining for choline acetyltransferase in the interpeduncular nucleus (p <0.025), but not for tyrosine hydroxylase in the surrounding catecholaminergic A10 region. We conclude that the interpeduncular nucleus mediates nicotinic depression of locomotor activity and dampens nicotinic arousal mechanisms located elsewhere in the brain.',
            'expected_categories': ['neurologico']
        },
        {
            'title': 'Effects of suprofen on the isolated perfused rat kidney.',
            'text': 'Although suprofen has been associated with the development of acute renal failure in greater than 100 subjects, the mechanism of damage remains unclear. The direct nephrotoxic effects of a single dose of 15 mg of suprofen were compared in the recirculating isolated rat kidney perfused with cell-free buffer with or without the addition of 5 mg/dL of uric acid. There were no significant differences in renal sodium excretion, oxygen consumption, or urinary flow rates in kidneys perfused with suprofen compared with the drug-free control groups. In contrast, a significant decline in glomerular filtration rate was found after the introduction of suprofen to the kidney perfused with uric acid; no changes were found with suprofen in the absence of uric acid. A significant decrease in the baseline excretion rate of uric acid was found in rats given suprofen, compared with drug-free controls. However, the fractional excretion of uric acid was unchanged between the groups over the experimental period. In summary, suprofen causes acute declines in renal function, most likely by directly altering the intrarenal distribution of uric acid.',
            'expected_categories': ['hepatorenal']
        },
        {
            'title': 'Haplotype and phenotype analysis of six recurrent BRCA1 mutations in 61 families: results of an international study.',
            'text': 'Several BRCA1 mutations have now been found to occur in geographically diverse breast and ovarian cancer families. To investigate mutation origin and mutation-specific phenotypes due to BRCA1, we constructed a haplotype of nine polymorphic markers within or immediately flanking the BRCA1 locus in a set of 61 breast/ovarian cancer families selected for having one of six recurrent BRCA1 mutations. Tests of both mutations and family-specific differences in age at diagnosis were not significant. A comparison of the six mutations in the relative proportions of cases of breast and ovarian cancer was suggestive of an effect (P = .069), with 57% of women presumed affected because of the 1294 del 40 BRCA1 mutation having ovarian cancer, compared with 14% of affected women with the splice-site mutation in intron 5 of BRCA1. For the BRCA1 mutations studied here, the individual mutations are estimated to have arisen 9-170 generations ago. In general, a high degree of haplotype conservation across the region was observed, with haplotype differences most often due to mutations in the short-tandem-repeat markers, although some likely instances of recombination also were observed. For several of the instances, there was evidence for multiple, independent, BRCA1 mutational events.',
            'expected_categories': ['oncologico']
        },
        {
            'title': 'Fluoxetine improves the memory deficits caused by the chemotherapy agent 5-fluorouracil.',
            'text': 'Cancer patients who have been treated with systemic adjuvant chemotherapy have described experiencing deteriorations in cognition. A widely used chemotherapeutic agent, 5-fluorouracil (5-FU), readily crosses the blood-brain barrier and so could have a direct effect on brain function. In particular this anti mitotic drug could reduce cell proliferation in the neurogenic regions of the adult brain. In contrast reports indicate that hippocampal dependent neurogenesis and cognition are enhanced by the SSRI antidepressant Fluoxetine. In this investigation the behavioural effects of chronic (two week) treatment with 5-FU and (three weeks) with Fluoxetine either separately or in combination with 5-FU were tested on adult Lister hooded rats. Behavioural effects were tested using a context dependent conditioned emotional response test (CER) which showed that animals treated with 5-FU had a significant reduction in freezing time compared to controls. A separate group of animals was tested using a hippocampal dependent spatial working memory test, the object location recognition test (OLR). Animals treated only with 5-FU showed significant deficits in their ability to carry out the OLR task but co administration of Fluoxetine improved their performance. 5-FU chemotherapy caused a significant reduction in the number of proliferating cells in the sub granular zone of the dentate gyrus compared to controls. This reduction was eliminated when Fluoxetine was co administered with 5-FU. Fluoxetine on its own had no effect on proliferating cell number or behaviour. These findings suggest that 5-FU can negatively affect both cell proliferation and hippocampal dependent working memory and that these deficits can be reversed by the simultaneous administration of the antidepressant Fluoxetine.',
            'expected_categories': ['neurologico', 'oncologico']
        },
        {
            'title': 'Role of activation of bradykinin B2 receptors in disruption of the blood-brain barrier during acute hypertension.',
            'text': 'Cellular mechanisms which account for disruption the blood-brain barrier during acute hypertension are not clear. The goal of this study was to determine the role of synthesis/release of bradykinin to activate B2 receptors in disruption of the blood-brain barrier during acute hypertension. Permeability of the blood-brain barrier was quantitated by clearance of fluorescent-labeled dextran before and during phenylephrine-induced acute hypertension in rats treated with vehicle and Hoe-140 (0.1 microM). Phenylephrine infusion increased arterial pressure, arteriolar diameter and clearance of fluorescent dextran by a similar magnitude in both groups. These findings suggest that disruption of the blood-brain barrier during acute hypertension is not related to the synthesis/release of bradykinin to activate B2 receptors.',
            'expected_categories': ['neurologico', 'cardiovascular']
        }
    ]
    
    return jsonify({'examples': examples})

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Obtener estadísticas del modelo y datos de entrenamiento"""
    try:
        # Cargar estadísticas desde xgboost_results.json
        stats_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'xgboost_results.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            # Puedes ajustar aquí qué campos mostrar
            response = {
                'config': results_data.get('config', {}),
                'data_shape': results_data.get('data_shape', {}),
                'test_metrics': results_data.get('test_metrics', {}),
                'cv_results': results_data.get('cv_results', {}),
                'training_time': results_data.get('training_time', None),
                'timestamp': results_data.get('timestamp', None)
            }
            return jsonify(response)
        else:
            # Mensaje si no se encuentra el archivo de resultados
            return jsonify({
                'error': 'No se encontró el archivo de resultados xgboost_results.json.'
            }), 404
    except Exception as e:
        return jsonify({'error': f'Error obteniendo estadísticas: {str(e)}'}), 500

print("🚀 Iniciando API Flask para XGBoost...")
    
# Cargar modelos al iniciar
if load_models():
    print("✅ Modelos cargados correctamente")
else:
    print("⚠️ Algunos modelos no pudieron cargarse")

print("🌐 API disponible en:")
print("   - Health check: http://localhost:5000/api/health")
print("   - Predicción: http://localhost:5000/api/predict")
print("   - Info modelo: http://localhost:5000/api/model-info")
print("   - Ejemplos demo: http://localhost:5000/api/demo-examples")
print("   - Estadísticas: http://localhost:5000/api/statistics")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)