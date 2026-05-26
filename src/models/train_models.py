import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from src.utils.logging_utils import setup_logger
import joblib
from src.utils.paths import MODELS_DIR

logger = setup_logger("train_models")

def train_models(X_train: pd.DataFrame, y_train: np.ndarray, model_flags: List[str], random_seed: int = 42) -> Dict[str, Any]:
    """
    Trains specified models on the provided training feature matrix.
    No internal train_test_split is performed to maintain strict chronological blocks.
    """
    logger.info(f"Training models: {model_flags}")
    
    models = {}
    
    if 'logistic_regression' in model_flags:
        logger.info("Training Logistic Regression...")
        lr = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_seed,
                        max_iter=2000,
                        class_weight='balanced',
                    ),
                ),
            ]
        )
        lr.fit(X_train, y_train)
        models['logistic_regression'] = lr
        joblib.dump(lr, MODELS_DIR / "lr_model.pkl")
        
    if 'random_forest' in model_flags:
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight='balanced')
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
        
    if 'lightgbm' in model_flags:
        logger.info("Training LightGBM...")
        lgbm = lgb.LGBMClassifier(random_state=random_seed, class_weight='balanced', verbose=-1)
        lgbm.fit(X_train, y_train)
        models['lightgbm'] = lgbm
        joblib.dump(lgbm, MODELS_DIR / "lgbm_model.pkl")
        
    if 'ebm' in model_flags:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            logger.info("Training Explainable Boosting Machine...")
            ebm = ExplainableBoostingClassifier(random_state=random_seed)
            ebm.fit(X_train, y_train)
            models['ebm'] = ebm
            joblib.dump(ebm, MODELS_DIR / "ebm_model.pkl")
        except ImportError:
            logger.warning("interpretML not installed. Skipping EBM.")
            
    return {
        'models': models
    }
