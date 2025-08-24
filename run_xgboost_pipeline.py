"""
XGBoost Training Pipeline Runner
================================

Main script to run the complete XGBoost training pipeline for medical 
literature classification. Provides a command-line interface with various
options for training, evaluation, and model analysis.

Authors: Juan Pablo Mejía, Samuel Castaño, Mateo Builes
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.training.xgboost_trainer import XGBoostTrainer
    from src.models.enhanced_xgboost import EnhancedXGBoostModel
    from src.evaluation.xgboost_evaluator import XGBoostEvaluator
    from config.xgboost_config import get_config, CONFIGS
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available and XGBoost is installed.")
    print("Install with: pip install xgboost scikit-learn pandas numpy matplotlib seaborn")
    sys.exit(1)


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'results',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def train_model(config_name: str, data_path: Optional[str] = None, 
                perform_cv: bool = True, hyperparameter_search: bool = False) -> Dict[str, Any]:
    """
    Train the XGBoost model with specified configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration to use
    data_path : str, optional
        Path to the training data
    perform_cv : bool
        Whether to perform cross-validation
    hyperparameter_search : bool
        Whether to perform hyperparameter search
    
    Returns:
    --------
    Dict[str, Any]
        Training results
    """
    print(f"\n{'='*60}")
    print(f"STARTING XGBOOST TRAINING - {config_name.upper()} CONFIG")
    print(f"{'='*60}")
    
    # Load configuration
    config = get_config(config_name)
    if data_path:
        config.data_path = data_path
    
    print(f"Using configuration: {config_name}")
    print(f"Data path: {config.data_path}")
    print(f"Cross-validation: {perform_cv}")
    print(f"Hyperparameter search: {hyperparameter_search}")
    
    # Initialize trainer
    trainer = XGBoostTrainer(config)
    
    # Perform hyperparameter search if requested
    if hyperparameter_search:
        print("\n" + "-"*40)
        print("PERFORMING HYPERPARAMETER SEARCH")
        print("-"*40)
        
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        search_results = trainer.hyperparameter_search(X_train, y_train)
        
        print("\nHyperparameter Search Results:")
        print(f"Best parameters: {search_results['best_params']}")
        print(f"Best CV score: {search_results['best_score']:.4f}")
        
        # Save search results
        with open(f'results/hyperparameter_search_{config_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(search_results, f, indent=2, default=str)
        
        return search_results
    
    # Run full training pipeline
    results = trainer.run_full_pipeline(
        data_path=data_path,
        perform_cv=perform_cv,
        save_results=True
    )
    
    print("\n" + "-"*40)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("-"*40)
    print(f"Final Results:")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  F1-Macro: {results['test_metrics']['f1_macro']:.4f}")
    print(f"  F1-Micro: {results['test_metrics']['f1_micro']:.4f}")
    
    if results['cv_results']:
        print(f"  CV F1-Macro: {results['cv_results']['f1_macro_mean']:.4f} (±{results['cv_results']['f1_macro_std']:.4f})")
    
    print(f"  Training time: {results['training_time']}")
    print(f"\nModel saved to: {config.model_save_path}")
    print(f"Results saved to: {config.results_path}")
    
    return results


def evaluate_model(model_path: str, test_data_path: Optional[str] = None,
                  baseline_results_path: Optional[str] = None,
                  generate_report: bool = True) -> Dict[str, Any]:
    """
    Evaluate a trained XGBoost model.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    test_data_path : str, optional
        Path to test data
    baseline_results_path : str, optional
        Path to baseline results for comparison
    generate_report : bool
        Whether to generate comprehensive report
    
    Returns:
    --------
    Dict[str, Any]
        Evaluation results
    """
    print(f"\n{'='*60}")
    print("STARTING XGBOOST MODEL EVALUATION")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path or 'Will use data from training'}")
    
    # Initialize evaluator
    evaluator = XGBoostEvaluator(model_path)
    
    # Load and process the actual data for evaluation
    print("\nLoading actual data for evaluation...")
    
    # Load the model to get configuration and preprocessors
    model = EnhancedXGBoostModel(model_path)
    
    # Determine data path - use test_data_path if provided, otherwise use training data
    if test_data_path and os.path.exists(test_data_path):
        data_path = test_data_path
        print(f"Using test data: {data_path}")
    else:
        # Use the same data as training (we'll split it for evaluation)
        data_path = model.config.get('data_path', 'data/challenge_data-18-ago.csv')
        print(f"Using training data for evaluation: {data_path}")
    
    # Load and preprocess data using the same logic as training
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Load data with proper separator
        try:
            df = pd.read_csv(data_path, sep=';')
        except:
            df = pd.read_csv(data_path, sep=',')
        
        print(f"Loaded data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Combine title and abstract for text processing
        if 'title' in df.columns and 'abstract' in df.columns:
            texts = (df['title'].fillna('') + ' ' + df['abstract'].fillna('')).astype(str)
        elif 'title_and_abstract' in df.columns:
            texts = df['title_and_abstract'].fillna('').astype(str)
        else:
            raise ValueError("No suitable text columns found")
        
        # Transform text using the saved vectorizer
        X = model.vectorizer.transform(texts).toarray()
        
        # Process labels
        classes = ['cardiovascular', 'neurological', 'hepatorenal', 'oncological']
        labels = []
        
        if 'group' in df.columns:
            # Process group column with pipe-separated values
            for _, row in df.iterrows():
                row_labels = []
                if pd.notna(row['group']):
                    group_labels = [label.strip() for label in str(row['group']).split('|')]
                    for label in group_labels:
                        if label in classes:
                            row_labels.append(label)
                labels.append(row_labels)
        else:
            # Process separate binary columns
            for _, row in df.iterrows():
                row_labels = []
                for class_name in classes:
                    if class_name in df.columns and row[class_name] == 1:
                        row_labels.append(class_name)
                labels.append(row_labels)
        
        # Transform labels using the saved label binarizer
        y = model.label_binarizer.transform(labels)
        
        # If using training data, split to get test portion
        if test_data_path is None or test_data_path == data_path:
            # Split the same way as training to get the test set
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Using test split: {X_test.shape[0]} samples")
        else:
            X_test, y_test = X, y
            print(f"Using full dataset: {X_test.shape[0]} samples")
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Label distribution: {dict(zip(classes, y_test.sum(axis=0)))}")
        
    except Exception as e:
        print(f"Error loading real data: {str(e)}")
        print("Falling back to placeholder data for demo...")
        import numpy as np
        X_test = np.random.random((200, 5000))  # 200 samples, 5000 features
        y_test = np.random.randint(0, 2, (200, 4))  # 4 classes
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(X_test, y_test, save_results=True)
    
    # Compare with baseline if provided
    if baseline_results_path and os.path.exists(baseline_results_path):
        print(f"\nComparing with baseline: {baseline_results_path}")
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        comparison = evaluator.compare_with_baseline(baseline_results)
        
        print("\nComparison Summary:")
        for metric, improvement in comparison['improvements'].items():
            print(f"  {metric}: {improvement['absolute']:+.4f} ({improvement['relative']:+.2%})")
    
    # Generate comprehensive report
    if generate_report:
        print("\nGenerating comprehensive model report...")
        report = evaluator.generate_model_report()
        
        report_path = f'results/xgboost_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        print("\nReport Summary:")
        print("-" * 40)
        basic_metrics = results['test_metrics']
        print(f"Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"F1-Macro: {basic_metrics['f1_macro']:.4f}")
        print(f"Hamming Loss: {basic_metrics['hamming_loss']:.4f}")
    
    return results


def analyze_model(model_path: str, text: Optional[str] = None,
                 analyze_features: bool = True, interactive: bool = False) -> None:
    """
    Perform model analysis and interpretation.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    text : str, optional
        Text to analyze/explain
    analyze_features : bool
        Whether to analyze feature importance
    interactive : bool
        Whether to start interactive mode
    """
    print(f"\n{'='*60}")
    print("STARTING XGBOOST MODEL ANALYSIS")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize enhanced model
    model = EnhancedXGBoostModel(model_path)
    
    # Show model summary
    summary = model.model_summary()
    print("\nModel Summary:")
    print("-" * 30)
    print(f"Classes: {', '.join(summary['classes'])}")
    print(f"Features: {summary['num_features']}")
    print(f"Configuration: {summary['configuration']['n_estimators']} estimators, "
          f"depth {summary['configuration']['max_depth']}, "
          f"lr {summary['configuration']['learning_rate']}")
    
    # Analyze feature importance
    if analyze_features:
        print("\nAnalyzing feature importance...")
        importance_results = model.analyze_feature_importance(top_n=20)
        print("Feature importance analysis completed and plots saved!")
    
    # Analyze specific text
    if text:
        print(f"\nAnalyzing text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Make prediction
        result = model.predict_single(text)
        print(f"\nPrediction Results:")
        print(f"Predicted labels: {', '.join(result['predicted_labels']) if result['predicted_labels'] else 'None'}")
        print("\nClass probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name.title()}: {prob:.4f}")
        
        # Try SHAP explanation if available
        try:
            print("\nGenerating SHAP explanation...")
            shap_results = model.explain_prediction_shap(text)
            print("SHAP explanation completed and plots saved!")
        except ImportError:
            print("SHAP not available for detailed explanation. Install with: pip install shap")
        except Exception as e:
            print(f"Could not generate SHAP explanation: {e}")
    
    # Start interactive mode
    if interactive:
        print("\nStarting interactive mode...")
        model.interactive_prediction_interface()


def compare_configurations() -> None:
    """Compare different XGBoost configurations."""
    print(f"\n{'='*60}")
    print("COMPARING XGBOOST CONFIGURATIONS")
    print(f"{'='*60}")
    
    print("Available configurations:")
    for config_name, config in CONFIGS.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Estimators: {config.n_estimators}")
        print(f"  Max depth: {config.max_depth}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Features: {config.max_features}")
        
        if hasattr(config, 'reg_alpha') and config.reg_alpha > 0:
            print(f"  L1 regularization: {config.reg_alpha}")
        if hasattr(config, 'reg_lambda') and config.reg_lambda > 1:
            print(f"  L2 regularization: {config.reg_lambda}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='XGBoost Training Pipeline for Medical Literature Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python run_xgboost_pipeline.py train

  # Train with high performance configuration
  python run_xgboost_pipeline.py train --config high_performance

  # Train with custom data path and hyperparameter search
  python run_xgboost_pipeline.py train --data-path data/my_data.csv --hyperparameter-search

  # Evaluate a trained model
  python run_xgboost_pipeline.py evaluate --model-path models/xgboost_model.pkl

  # Analyze model with specific text
  python run_xgboost_pipeline.py analyze --model-path models/xgboost_model.pkl --text "Patient shows symptoms..."

  # Compare configurations
  python run_xgboost_pipeline.py compare-configs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train XGBoost model')
    train_parser.add_argument('--config', type=str, default='default',
                             choices=list(CONFIGS.keys()),
                             help='Configuration to use')
    train_parser.add_argument('--data-path', type=str,
                             help='Path to training data (overrides config)')
    train_parser.add_argument('--no-cv', action='store_true',
                             help='Skip cross-validation')
    train_parser.add_argument('--hyperparameter-search', action='store_true',
                             help='Perform hyperparameter search instead of training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--test-data', type=str,
                            help='Path to test data')
    eval_parser.add_argument('--baseline-results', type=str,
                            help='Path to baseline results for comparison')
    eval_parser.add_argument('--no-report', action='store_true',
                            help='Skip comprehensive report generation')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze and interpret model')
    analyze_parser.add_argument('--model-path', type=str, required=True,
                               help='Path to trained model')
    analyze_parser.add_argument('--text', type=str,
                               help='Text to analyze and explain')
    analyze_parser.add_argument('--no-features', action='store_true',
                               help='Skip feature importance analysis')
    analyze_parser.add_argument('--interactive', action='store_true',
                               help='Start interactive prediction mode')
    
    # Compare configurations command
    subparsers.add_parser('compare-configs', help='Compare available configurations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup directories
    setup_directories()
    
    try:
        if args.command == 'train':
            train_model(
                config_name=args.config,
                data_path=args.data_path,
                perform_cv=not args.no_cv,
                hyperparameter_search=args.hyperparameter_search
            )
        
        elif args.command == 'evaluate':
            evaluate_model(
                model_path=args.model_path,
                test_data_path=args.test_data,
                baseline_results_path=args.baseline_results,
                generate_report=not args.no_report
            )
        
        elif args.command == 'analyze':
            analyze_model(
                model_path=args.model_path,
                text=args.text,
                analyze_features=not args.no_features,
                interactive=args.interactive
            )
        
        elif args.command == 'compare-configs':
            compare_configurations()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
