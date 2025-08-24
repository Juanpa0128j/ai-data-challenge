"""
XGBoost Model Evaluation Tools
==============================

Comprehensive evaluation tools for XGBoost model performance analysis,
including detailed metrics, visualizations, and comparison utilities.

Authors: Juan Pablo Mejía, Samuel Castaño, Mateo Builes
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    hamming_loss, jaccard_score
)

from src.models.enhanced_xgboost import EnhancedXGBoostModel


class XGBoostEvaluator:
    """
    Comprehensive evaluation toolkit for XGBoost model analysis.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the evaluator with a trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved XGBoost model
        """
        self.model = EnhancedXGBoostModel(model_path)
        self.evaluation_results = {}
    
    def comprehensive_evaluation(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        save_results : bool
            Whether to save results to file
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation results
        """
        print("Starting comprehensive XGBoost model evaluation...")
        
        # Make predictions
        y_pred = self.model.model.predict(X_test)
        y_pred_proba = self.model.model.predict_proba(X_test)
        
        # Calculate various metrics
        results = {
            'basic_metrics': self._calculate_basic_metrics(y_test, y_pred),
            'per_class_metrics': self._calculate_per_class_metrics(y_test, y_pred, y_pred_proba),
            'multi_label_metrics': self._calculate_multilabel_metrics(y_test, y_pred),
            'confusion_matrices': self._calculate_confusion_matrices(y_test, y_pred),
            'roc_analysis': self._calculate_roc_metrics(y_test, y_pred_proba),
            'pr_analysis': self._calculate_pr_metrics(y_test, y_pred_proba),
            'timestamp': datetime.now().isoformat()
        }
        
        self.evaluation_results = results
        
        # Generate visualizations
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba)
        
        # Save results
        if save_results:
            self._save_evaluation_results(results)
        
        print("Comprehensive evaluation completed!")
        return results
    
    def _calculate_basic_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'hamming_loss': hamming_loss(y_test, y_pred),
            'jaccard_score_macro': jaccard_score(y_test, y_pred, average='macro'),
            'jaccard_score_micro': jaccard_score(y_test, y_pred, average='micro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_micro': precision_score(y_test, y_pred, average='micro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_micro': recall_score(y_test, y_pred, average='micro')
        }
    
    def _calculate_per_class_metrics(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            y_prob_class = y_pred_proba[:, i]
            
            per_class_metrics[class_name] = {
                'accuracy': accuracy_score(y_true_class, y_pred_class),
                'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                'f1_score': f1_score(y_true_class, y_pred_class, zero_division=0),
                'auc_roc': auc(*roc_curve(y_true_class, y_prob_class)[:2]) if len(np.unique(y_true_class)) > 1 else 0.0,
                'average_precision': average_precision_score(y_true_class, y_prob_class) if len(np.unique(y_true_class)) > 1 else 0.0,
                'support': int(np.sum(y_true_class))
            }
        
        return per_class_metrics
    
    def _calculate_multilabel_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate multi-label specific metrics."""
        # Exact match ratio
        exact_match = np.all(y_test == y_pred, axis=1).mean()
        
        # Label-based metrics
        label_ranking_loss = 0  # Simplified for binary case
        
        # Calculate label frequency
        label_frequency = np.mean(y_test, axis=0)
        pred_frequency = np.mean(y_pred, axis=0)
        
        return {
            'exact_match_ratio': exact_match,
            'label_ranking_loss': label_ranking_loss,
            'true_label_frequency': label_frequency.tolist(),
            'predicted_label_frequency': pred_frequency.tolist(),
            'label_frequency_difference': (pred_frequency - label_frequency).tolist()
        }
    
    def _calculate_confusion_matrices(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, List[List[int]]]:
        """Calculate confusion matrices for each class."""
        confusion_matrices = {}
        
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            cm = confusion_matrix(y_true_class, y_pred_class)
            confusion_matrices[class_name] = cm.tolist()
        
        return confusion_matrices
    
    def _calculate_roc_metrics(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC curve metrics."""
        roc_data = {}
        
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_prob_class = y_pred_proba[:, i]
            
            if len(np.unique(y_true_class)) > 1:
                fpr, tpr, thresholds = roc_curve(y_true_class, y_prob_class)
                roc_auc = auc(fpr, tpr)
                
                roc_data[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': roc_auc
                }
            else:
                roc_data[class_name] = {
                    'fpr': [],
                    'tpr': [],
                    'thresholds': [],
                    'auc': 0.0
                }
        
        return roc_data
    
    def _calculate_pr_metrics(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate Precision-Recall curve metrics."""
        pr_data = {}
        
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_prob_class = y_pred_proba[:, i]
            
            if len(np.unique(y_true_class)) > 1:
                precision, recall, thresholds = precision_recall_curve(y_true_class, y_prob_class)
                avg_precision = average_precision_score(y_true_class, y_prob_class)
                
                pr_data[class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': thresholds.tolist(),
                    'average_precision': avg_precision
                }
            else:
                pr_data[class_name] = {
                    'precision': [],
                    'recall': [],
                    'thresholds': [],
                    'average_precision': 0.0
                }
        
        return pr_data
    
    def _create_evaluation_plots(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> None:
        """Create comprehensive evaluation plots."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrices
        for i, class_name in enumerate(self.model.classes_):
            plt.subplot(4, 4, i + 1)
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            cm = confusion_matrix(y_true_class, y_pred_class)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title(f'Confusion Matrix - {class_name.title()}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        # 5. ROC Curves
        plt.subplot(4, 4, 5)
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_prob_class = y_pred_proba[:, i]
            
            if len(np.unique(y_true_class)) > 1:
                fpr, tpr, _ = roc_curve(y_true_class, y_prob_class)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(fontsize=8)
        
        # 6. Precision-Recall Curves
        plt.subplot(4, 4, 6)
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_prob_class = y_pred_proba[:, i]
            
            if len(np.unique(y_true_class)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_class, y_prob_class)
                avg_precision = average_precision_score(y_true_class, y_prob_class)
                plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(fontsize=8)
        
        # 7. Class Distribution (True vs Predicted)
        plt.subplot(4, 4, 7)
        true_counts = np.sum(y_test, axis=0)
        pred_counts = np.sum(y_pred, axis=0)
        
        x = np.arange(len(self.model.classes_))
        width = 0.35
        
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class Distribution: True vs Predicted')
        plt.xticks(x, [c.title() for c in self.model.classes_], rotation=45)
        plt.legend()
        
        # 8. Per-Class F1 Scores
        plt.subplot(4, 4, 8)
        f1_scores = []
        for i, class_name in enumerate(self.model.classes_):
            y_true_class = y_test[:, i]
            y_pred_class = y_pred[:, i]
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            f1_scores.append(f1)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.model.classes_)))
        bars = plt.bar(range(len(self.model.classes_)), f1_scores, color=colors)
        plt.xlabel('Classes')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.xticks(range(len(self.model.classes_)), [c.title() for c in self.model.classes_], rotation=45)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 9. Prediction Probability Distribution
        plt.subplot(4, 4, 9)
        for i, class_name in enumerate(self.model.classes_):
            y_prob_class = y_pred_proba[:, i]
            plt.hist(y_prob_class, bins=20, alpha=0.6, label=class_name, density=True)
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend(fontsize=8)
        
        # 10. Classification Report Heatmap
        plt.subplot(4, 4, 10)
        report = classification_report(y_test, y_pred, target_names=self.model.classes_, output_dict=True)
        
        # Extract metrics for heatmap
        metrics_data = []
        classes_for_heatmap = []
        for class_name in self.model.classes_:
            if class_name in report:
                metrics_data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
                classes_for_heatmap.append(class_name.title())
        
        if metrics_data:
            metrics_array = np.array(metrics_data)
            sns.heatmap(metrics_array, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=['Precision', 'Recall', 'F1-Score'],
                       yticklabels=classes_for_heatmap)
            plt.title('Classification Metrics Heatmap')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/xgboost_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comprehensive evaluation plots saved to: results/xgboost_comprehensive_evaluation.png")
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON file."""
        os.makedirs('results', exist_ok=True)
        
        # Save detailed results
        with open('results/xgboost_detailed_evaluation.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary = self._create_summary_report(results)
        with open('results/xgboost_evaluation_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print("Evaluation results saved to:")
        print("  - results/xgboost_detailed_evaluation.json")
        print("  - results/xgboost_evaluation_summary.txt")
    
    def _create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary report."""
        report = []
        report.append("XGBoost Model Evaluation Summary")
        report.append("=" * 50)
        report.append(f"Evaluation Date: {results['timestamp']}")
        report.append("")
        
        # Basic metrics
        report.append("Overall Performance Metrics:")
        report.append("-" * 30)
        basic_metrics = results['basic_metrics']
        report.append(f"Accuracy: {basic_metrics['accuracy']:.4f}")
        report.append(f"F1 Score (Macro): {basic_metrics['f1_macro']:.4f}")
        report.append(f"F1 Score (Micro): {basic_metrics['f1_micro']:.4f}")
        report.append(f"Precision (Macro): {basic_metrics['precision_macro']:.4f}")
        report.append(f"Recall (Macro): {basic_metrics['recall_macro']:.4f}")
        report.append(f"Hamming Loss: {basic_metrics['hamming_loss']:.4f}")
        report.append(f"Jaccard Score (Macro): {basic_metrics['jaccard_score_macro']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("Per-Class Performance:")
        report.append("-" * 30)
        per_class = results['per_class_metrics']
        
        for class_name, metrics in per_class.items():
            report.append(f"\n{class_name.upper()}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            report.append(f"  Average Precision: {metrics['average_precision']:.4f}")
            report.append(f"  Support: {metrics['support']}")
        
        # Multi-label metrics
        report.append("\nMulti-label Specific Metrics:")
        report.append("-" * 30)
        ml_metrics = results['multi_label_metrics']
        report.append(f"Exact Match Ratio: {ml_metrics['exact_match_ratio']:.4f}")
        
        # Label frequency analysis
        report.append("\nLabel Frequency Analysis:")
        report.append("-" * 30)
        for i, class_name in enumerate(self.model.classes_):
            true_freq = ml_metrics['true_label_frequency'][i]
            pred_freq = ml_metrics['predicted_label_frequency'][i]
            diff = ml_metrics['label_frequency_difference'][i]
            report.append(f"{class_name}: True={true_freq:.4f}, Pred={pred_freq:.4f}, Diff={diff:+.4f}")
        
        return "\n".join(report)
    
    def compare_with_baseline(
        self, 
        baseline_results: Dict[str, Any],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compare current model results with baseline.
        
        Parameters:
        -----------
        baseline_results : Dict[str, Any]
            Baseline model evaluation results
        save_comparison : bool
            Whether to save comparison results
        
        Returns:
        --------
        Dict[str, Any]
            Comparison results
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run comprehensive_evaluation() first.")
        
        comparison = {
            'xgboost_metrics': self.evaluation_results['basic_metrics'],
            'baseline_metrics': baseline_results.get('basic_metrics', {}),
            'improvements': {},
            'per_class_comparison': {}
        }
        
        # Calculate improvements in basic metrics
        for metric in self.evaluation_results['basic_metrics']:
            if metric in baseline_results.get('basic_metrics', {}):
                xgb_value = self.evaluation_results['basic_metrics'][metric]
                baseline_value = baseline_results['basic_metrics'][metric]
                improvement = xgb_value - baseline_value
                comparison['improvements'][metric] = {
                    'absolute': improvement,
                    'relative': improvement / baseline_value if baseline_value != 0 else 0
                }
        
        # Compare per-class metrics
        if 'per_class_metrics' in baseline_results:
            for class_name in self.model.classes_:
                if class_name in baseline_results['per_class_metrics']:
                    xgb_class_metrics = self.evaluation_results['per_class_metrics'][class_name]
                    baseline_class_metrics = baseline_results['per_class_metrics'][class_name]
                    
                    class_comparison = {}
                    for metric in ['precision', 'recall', 'f1_score', 'auc_roc']:
                        if metric in xgb_class_metrics and metric in baseline_class_metrics:
                            xgb_value = xgb_class_metrics[metric]
                            baseline_value = baseline_class_metrics[metric]
                            improvement = xgb_value - baseline_value
                            class_comparison[metric] = {
                                'xgboost': xgb_value,
                                'baseline': baseline_value,
                                'improvement': improvement
                            }
                    
                    comparison['per_class_comparison'][class_name] = class_comparison
        
        if save_comparison:
            os.makedirs('results', exist_ok=True)
            with open('results/xgboost_baseline_comparison.json', 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, default=str)
            print("Comparison results saved to: results/xgboost_baseline_comparison.json")
        
        return comparison
    
    def generate_model_report(self) -> str:
        """
        Generate a comprehensive model report.
        
        Returns:
        --------
        str
            Formatted model report
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run comprehensive_evaluation() first.")
        
        model_summary = self.model.model_summary()
        
        report = []
        report.append("XGBoost Model Comprehensive Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model configuration
        report.append("Model Configuration:")
        report.append("-" * 30)
        report.append(f"Model Type: {model_summary['model_type']}")
        report.append(f"Number of Classes: {model_summary['num_classes']}")
        report.append(f"Classes: {', '.join(model_summary['classes'])}")
        report.append(f"Number of Features: {model_summary['num_features']}")
        report.append("")
        
        # Add detailed configuration if available
        if model_summary['configuration']:
            report.append("Hyperparameters:")
            config = model_summary['configuration']
            for key, value in config.items():
                if key in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']:
                    report.append(f"  {key}: {value}")
        report.append("")
        
        # Performance summary
        basic_metrics = self.evaluation_results['basic_metrics']
        report.append("Performance Summary:")
        report.append("-" * 30)
        report.append(f"Overall Accuracy: {basic_metrics['accuracy']:.4f}")
        report.append(f"Macro F1-Score: {basic_metrics['f1_macro']:.4f}")
        report.append(f"Micro F1-Score: {basic_metrics['f1_micro']:.4f}")
        report.append(f"Hamming Loss: {basic_metrics['hamming_loss']:.4f}")
        report.append("")
        
        # Best and worst performing classes
        per_class_metrics = self.evaluation_results['per_class_metrics']
        f1_scores = {class_name: metrics['f1_score'] for class_name, metrics in per_class_metrics.items()}
        
        best_class = max(f1_scores.keys(), key=lambda x: f1_scores[x])
        worst_class = min(f1_scores.keys(), key=lambda x: f1_scores[x])
        
        report.append("Class Performance Analysis:")
        report.append("-" * 30)
        report.append(f"Best performing class: {best_class.title()} (F1: {f1_scores[best_class]:.4f})")
        report.append(f"Worst performing class: {worst_class.title()} (F1: {f1_scores[worst_class]:.4f})")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 30)
        
        if basic_metrics['accuracy'] > 0.85:
            report.append("✓ Model shows strong overall performance")
        elif basic_metrics['accuracy'] > 0.75:
            report.append("• Model shows good performance but has room for improvement")
        else:
            report.append("⚠ Model performance needs improvement")
        
        if basic_metrics['hamming_loss'] < 0.15:
            report.append("✓ Low prediction error rate")
        else:
            report.append("⚠ Consider addressing prediction error rate")
        
        # Check class imbalance issues
        ml_metrics = self.evaluation_results['multi_label_metrics']
        max_freq_diff = max([abs(x) for x in ml_metrics['label_frequency_difference']])
        if max_freq_diff > 0.1:
            report.append("⚠ Consider addressing class imbalance issues")
        
        if f1_scores[worst_class] < 0.5:
            report.append(f"⚠ Focus on improving {worst_class} classification")
        
        report.append("")
        report.append("Files Generated:")
        report.append("-" * 30)
        report.append("• results/xgboost_comprehensive_evaluation.png")
        report.append("• results/xgboost_detailed_evaluation.json")
        report.append("• results/xgboost_evaluation_summary.txt")
        
        return "\n".join(report)


def main():
    """Main function for running XGBoost evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost Model Evaluation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained XGBoost model')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--baseline-results', type=str,
                       help='Path to baseline results JSON for comparison')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive model report')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = XGBoostEvaluator(args.model_path)
    
    # Load test data (you would need to implement data loading logic)
    print(f"Loading test data from: {args.test_data}")
    # This is a placeholder - you would implement actual data loading
    # X_test, y_test = load_test_data(args.test_data)
    
    # For demonstration, create dummy data
    print("Note: Using placeholder data for demonstration")
    X_test = np.random.random((100, 5000))
    y_test = np.random.randint(0, 2, (100, 4))
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    # Compare with baseline if provided
    if args.baseline_results:
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        evaluator.compare_with_baseline(baseline_results)
    
    # Generate comprehensive report
    if args.generate_report:
        report = evaluator.generate_model_report()
        print("\n" + report)
        
        with open('results/xgboost_comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("\nComprehensive report saved to: results/xgboost_comprehensive_report.txt")


if __name__ == "__main__":
    main()
