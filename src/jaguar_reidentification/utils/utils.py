from pathlib import Path

import numpy as np
import pandas as pd


def json_safe(obj: object) -> object:
    # Convert numpy/pandas/Path types to JSON-serializable forms and normalize NaN -> None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        # keep NaN handling below
        val = float(obj)
        return None if pd.isna(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # Normalize pandas/np NaN/NaT
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj
