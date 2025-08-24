"""
XGBoost Configuration for Medical Literature Classification
==========================================================

Configuration file containing hyperparameters and settings for XGBoost model
training in the medical literature classification task.

Authors: Juan Pablo Mejía, Samuel Castaño, Mateo Builes
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class XGBoostConfig:
    """Configuration class for XGBoost model parameters."""
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = 'logloss'
    early_stopping_rounds: Optional[int] = 10
    verbose: bool = False
    
    # Data parameters
    max_features: int = 5000
    test_size: float = 0.2
    validation_split: float = 0.2
    
    # Paths
    data_path: str = "data/challenge_data-18-ago.csv"
    model_save_path: str = "models/xgboost_model.pkl"
    results_path: str = "results/xgboost_results.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters specifically for XGBoost model initialization."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'eval_metric': self.eval_metric,
            'verbose': self.verbose
        }


# Predefined configurations for different scenarios
CONFIGS = {
    'default': XGBoostConfig(),
    
    'fast_training': XGBoostConfig(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.2,
        max_features=2000
    ),
    
    'high_performance': XGBoostConfig(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=15
    ),
    
    'regularized': XGBoostConfig(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1
    )
}


def get_config(config_name: str = 'default') -> XGBoostConfig:
    """
    Get a predefined configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration ('default', 'fast_training', 'high_performance', 'regularized')
    
    Returns:
    --------
    XGBoostConfig
        Configuration object
    """
    if config_name not in CONFIGS:
        available_configs = list(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    return CONFIGS[config_name]


if __name__ == "__main__":
    # Example usage
    config = get_config('default')
    print("Default XGBoost Configuration:")
    print("-" * 40)
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")
