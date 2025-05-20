try:
    import cupy as xp
except Exception:  # pragma: no cover - cupy may not be installed
    import numpy as xp
