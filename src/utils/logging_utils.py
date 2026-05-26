import logging
import sys
import pkg_resources
from .paths import REPORTS_DIR, ensure_directories

def setup_logger(name: str = "vns_project", log_file: str = "run.log") -> logging.Logger:
    """Configures and returns a logger that writes to console and file."""
    ensure_directories()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_path = REPORTS_DIR / log_file
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def log_environment_info(logger: logging.Logger):
    """Logs the python version and installed package versions."""
    logger.info(f"Python Version: {sys.version}")
    packages_to_check = ['numpy', 'scipy', 'mne', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'shap', 'matplotlib', 'seaborn', 'pyyaml']
    
    for pkg in packages_to_check:
        try:
            version = pkg_resources.get_distribution(pkg).version
            logger.info(f"Package '{pkg}': {version}")
        except pkg_resources.DistributionNotFound:
            logger.warning(f"Package '{pkg}' is not installed.")
