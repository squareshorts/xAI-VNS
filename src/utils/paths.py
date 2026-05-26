import os
from pathlib import Path

# Repository root (2 levels up from src/utils/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = RESULTS_DIR / "models"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "configs"

def get_project_root() -> Path:
    return PROJECT_ROOT

def ensure_directories():
    """Ensure all required directories exist."""
    dirs_to_create = [
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        FIGURES_DIR, TABLES_DIR, REPORTS_DIR, MODELS_DIR
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
