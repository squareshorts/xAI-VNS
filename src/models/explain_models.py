import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.utils.logging_utils import setup_logger
from src.utils.paths import REPORTS_DIR, TABLES_DIR

logger = setup_logger("explain_models")

def get_feature_importances(model: Any, model_name: str, X: pd.DataFrame) -> pd.Series:
    """Extracts feature importances depending on the model type."""
    if model_name == 'logistic_regression':
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            coef = model.named_steps["classifier"].coef_[0]
            imp = np.abs(coef)
        else:
            std = X.std(axis=0)
            coef = model.coef_[0]
            imp = np.abs(coef * std)
        return pd.Series(imp, index=X.columns)
        
    elif hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=X.columns)
        
    elif model_name == 'ebm':
        # interpretML EBM
        ebm_global = model.explain_global()
        imp = dict(zip(ebm_global.data()['names'], ebm_global.data()['scores']))
        return pd.Series(imp)
        
    return pd.Series(np.zeros(len(X.columns)), index=X.columns)

def explain_models(train_results: Dict[str, Any], use_shap: bool = True) -> Dict[str, pd.Series]:
    """Generates global explanations and SHAP values if requested."""
    models = train_results['models']
    X_train = train_results['X_train']
    
    logger.info("Generating model explanations...")
    
    importances = {}
    top_features_list = []
    
    for name, model in models.items():
        imp = get_feature_importances(model, name, X_train)
        imp = imp.sort_values(ascending=False)
        importances[name] = imp
        
        # Get top 10
        top_10 = imp.head(10)
        for rank, (feat, score) in enumerate(top_10.items(), 1):
            top_features_list.append({
                'Model': name,
                'Rank': rank,
                'Feature': feat,
                'ImportanceScore': score
            })
            
    # Try SHAP
    shap_values_dict = {}
    if use_shap and 'lightgbm' in models:
        try:
            import shap
            logger.info("Computing SHAP values for LightGBM...")
            # For POC, compute SHAP on a subset to save time
            X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
            explainer = shap.TreeExplainer(models['lightgbm'])
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list): # depending on shap version for lightgbm
                shap_values = shap_values[1] 
            shap_values_dict['lightgbm'] = (explainer, shap_values, X_sample)
        except ImportError:
            logger.warning("SHAP not installed. Skipping SHAP analysis.")
            
    # Save Top Features Table
    df_top = pd.DataFrame(top_features_list)
    table_path = TABLES_DIR / "table_top_features_real_eeg.csv"
    df_top.to_csv(table_path, index=False)
    
    # Write XAI Summary
    summary_path = REPORTS_DIR / "xai_summary_real_eeg.md"
    with open(summary_path, "w") as f:
        f.write("# Explainable AI Summary\n\n")
        f.write("> **Disclaimer**: Simulated responsive stimulation-triggering policy — not evidence of therapeutic efficacy or biological causation.\n\n")
        f.write("The features listed below are *associated* with model-predicted seizure risk. They do not necessarily *cause* seizure onset.\n\n")
        
        for name in models.keys():
            f.write(f"## {name}\n")
            top = importances[name].head(5)
            f.write("Top 5 contributing features:\n")
            for feat, val in top.items():
                f.write(f"- **{feat}**: {val:.4f}\n")
            f.write("\n")
            
    logger.info(f"Saved XAI summary to {summary_path}")
            
    return {'importances': importances, 'shap': shap_values_dict}
