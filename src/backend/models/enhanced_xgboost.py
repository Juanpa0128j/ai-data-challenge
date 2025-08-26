"""
Enhanced XGBoost Model for Medical Literature Classification
===========================================================

Enhanced XGBoost model implementation with advanced features like
feature importance analysis, model interpretation, and prediction utilities.

Authors: Juan Pablo Mejía, Samuel Castaño, Mateo Builes
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from sklearn.metrics import confusion_matrix

# Try to import XGBoost and related libraries
try:
    from xgboost import XGBClassifier
    import shap
    XGBOOST_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    SHAP_AVAILABLE = False
    print("XGBoost or SHAP not available. Install with: pip install xgboost shap")

from config.xgboost_config import XGBoostConfig


class EnhancedXGBoostModel:
    """
    Enhanced XGBoost model with additional analysis and interpretation capabilities.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the enhanced XGBoost model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to a saved model file
        """
        self.model = None
        self.vectorizer = None
        self.label_binarizer = None
        self.config = None
        self.classes_ = None
        self.feature_names = None
        self.feature_importances = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        """
        print(f"Loading model from: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_binarizer = model_data['label_binarizer']
        self.config = model_data['config']
        self.classes_ = model_data['classes']
        
        # Get feature names from vectorizer
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            self.feature_names = [f'feature_{i}' for i in range(self.model.estimators_[0].n_features_in_)]
        
        print("Model loaded successfully!")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to classify
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted labels and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Vectorize texts
        X = self.vectorizer.transform(texts).toarray()
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make a prediction on a single text with detailed output.
        
        Parameters:
        -----------
        text : str
            Text to classify
        
        Returns:
        --------
        Dict[str, Any]
            Detailed prediction results
        """
        predictions, probabilities = self.predict([text])
        
        # Convert to readable format
        predicted_labels = self.label_binarizer.inverse_transform(predictions)[0]
        
        # Get probabilities for each class
        class_probabilities = {}
        for i, class_name in enumerate(self.classes_):
            class_probabilities[class_name] = float(probabilities[0][i])
        
        result = {
            'text': text,
            'predicted_labels': list(predicted_labels) if predicted_labels else [],
            'class_probabilities': class_probabilities,
            'confidence_scores': {
                label: class_probabilities[label] 
                for label in predicted_labels if predicted_labels
            }
        }
        
        return result
    
    def analyze_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Analyze and visualize feature importance across all classifiers.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show
        
        Returns:
        --------
        Dict[str, Any]
            Feature importance analysis results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Analyzing feature importance...")
        
        # Get feature importances from each classifier
        importance_dict = {}
        
        for (class_name, estimator) in zip(self.classes_, self.model.estimators_):
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                
                # Get top features for this class
                feature_importance_pairs = list(zip(self.feature_names, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                importance_dict[class_name] = {
                    'features': [pair[0] for pair in feature_importance_pairs[:top_n]],
                    'importances': [pair[1] for pair in feature_importance_pairs[:top_n]]
                }
        
        # Create visualization
        self._plot_feature_importance(importance_dict, top_n)
        
        return importance_dict
    
    def _plot_feature_importance(self, importance_dict: Dict[str, Any], top_n: int) -> None:
        """Plot feature importance for each class."""
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (class_name, importance_data) in enumerate(importance_dict.items()):
            ax = axes[i]
            
            features = importance_data['features']
            importances = importance_data['importances']
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color=f'C{i}')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Features - {class_name.title()}')
            ax.invert_yaxis()  # Top feature at the top
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature importance plot saved to: results/xgboost_feature_importance.png")
    
    def explain_prediction_shap(self, text: str, plot: bool = True) -> Dict[str, Any]:
        """
        Use SHAP to explain a prediction.
        
        Parameters:
        -----------
        text : str
            Text to explain
        plot : bool
            Whether to create SHAP plots
        
        Returns:
        --------
        Dict[str, Any]
            SHAP explanation results
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Generating SHAP explanations...")
        
        # Vectorize text
        X = self.vectorizer.transform([text]).toarray()
        
        explanations = {}
        
        for i, (class_name, estimator) in enumerate(zip(self.classes_, self.model.estimators_)):
            # Create SHAP explainer
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X)
            
            # Get top contributing features
            shap_importance = np.abs(shap_values[0])
            top_indices = np.argsort(shap_importance)[-10:][::-1]
            
            explanations[class_name] = {
                'shap_values': shap_values[0].tolist(),
                'top_features': [self.feature_names[idx] for idx in top_indices],
                'top_shap_values': [shap_values[0][idx] for idx in top_indices],
                'prediction': float(estimator.predict_proba(X)[0][1])  # Probability for positive class
            }
            
            if plot:
                # Create SHAP waterfall plot
                shap.waterfall_plot(
                    explainer.expected_value[1], 
                    shap_values[0], 
                    X[0],
                    feature_names=self.feature_names,
                    max_display=10,
                    show=False
                )
                
                plt.title(f'SHAP Explanation - {class_name.title()}')
                plt.tight_layout()
                
                # Save plot
                plot_path = f'results/shap_explanation_{class_name}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"SHAP plot saved to: {plot_path}")
        
        return explanations
    
    def generate_confusion_matrices(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate confusion matrices for each class.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Confusion matrices for each class
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Generating confusion matrices...")
        
        y_pred = self.model.predict(X_test)
        confusion_matrices = {}
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.classes_):
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            
            cm = confusion_matrix(y_true_class, y_pred_class)
            confusion_matrices[class_name] = cm
            
            # Plot confusion matrix
            ax = axes[i]
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=ax,
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes']
            )
            ax.set_title(f'Confusion Matrix - {class_name.title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('results/xgboost_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Confusion matrices plot saved to: results/xgboost_confusion_matrices.png")
        
        return confusion_matrices
    
    def batch_predict_from_csv(
        self, 
        csv_path: str, 
        text_column: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions on a CSV file and save results.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        text_column : str
            Name of the column containing text
        output_path : str, optional
            Path to save results
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        print(f"Making predictions on {len(df)} samples...")
        
        texts = df[text_column].fillna('').astype(str).tolist()
        predictions, probabilities = self.predict(texts)
        
        # Add predictions to dataframe
        for i, class_name in enumerate(self.classes_):
            df[f'pred_{class_name}'] = predictions[:, i]
            df[f'prob_{class_name}'] = probabilities[:, i]
        
        # Add predicted labels as text
        predicted_labels = []
        for pred in predictions:
            labels = self.label_binarizer.inverse_transform([pred])[0]
            predicted_labels.append(', '.join(labels) if labels else 'none')
        
        df['predicted_labels'] = predicted_labels
        
        # Save results
        if output_path is None:
            output_path = csv_path.replace('.csv', '_predictions.csv')
        
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        return df
    
    def model_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive model summary.
        
        Returns:
        --------
        Dict[str, Any]
            Model summary information
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        summary = {
            'model_type': 'XGBoost with OneVsRestClassifier',
            'classes': self.classes_,
            'num_classes': len(self.classes_),
            'num_features': len(self.feature_names),
            'configuration': self.config,
            'estimators_info': []
        }
        
        # Get information about each estimator
        for i, (class_name, estimator) in enumerate(zip(self.classes_, self.model.estimators_)):
            estimator_info = {
                'class': class_name,
                'n_features': estimator.n_features_in_ if hasattr(estimator, 'n_features_in_') else 'Unknown',
                'n_estimators': getattr(estimator, 'n_estimators', 'Unknown'),
                'max_depth': getattr(estimator, 'max_depth', 'Unknown')
            }
            summary['estimators_info'].append(estimator_info)
        
        return summary
    
    def interactive_prediction_interface(self):
        """
        Simple interactive interface for making predictions.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("\n" + "="*60)
        print("XGBoost Medical Literature Classifier - Interactive Mode")
        print("="*60)
        print("Enter medical text to classify (type 'quit' to exit)")
        print("-"*60)
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            try:
                result = self.predict_single(text)
                
                print(f"\nPredicted Categories: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None'}")
                print("\nClass Probabilities:")
                for class_name, prob in result['class_probabilities'].items():
                    print(f"  {class_name.title()}: {prob:.4f}")
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}")


def main():
    """Main function for testing the enhanced model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced XGBoost Model Interface')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved model file')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive prediction interface')
    parser.add_argument('--analyze-features', action='store_true',
                       help='Analyze and plot feature importance')
    parser.add_argument('--predict-csv', type=str,
                       help='Path to CSV file for batch predictions')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Name of text column in CSV (default: text)')
    parser.add_argument('--explain-text', type=str,
                       help='Text to explain using SHAP')
    
    args = parser.parse_args()
    
    # Initialize model
    model = EnhancedXGBoostModel(args.model_path)
    
    # Show model summary
    summary = model.model_summary()
    print("\nModel Summary:")
    print("-" * 40)
    print(f"Classes: {', '.join(summary['classes'])}")
    print(f"Number of features: {summary['num_features']}")
    
    # Perform requested actions
    if args.analyze_features:
        model.analyze_feature_importance()
    
    if args.predict_csv:
        model.batch_predict_from_csv(args.predict_csv, args.text_column)
    
    if args.explain_text:
        model.explain_prediction_shap(args.explain_text)
    
    if args.interactive:
        model.interactive_prediction_interface()


if __name__ == "__main__":
    main()
