import random
import os
import numpy as np

def set_global_seed(seed: int = 42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # Try to set sklearn random seed if it has any global effect, 
    # though usually sklearn relies on numpy's global state or explicit random_state parameters.
    # It's best practice to pass this seed to all sklearn estimators as `random_state=seed`.
    return seed
