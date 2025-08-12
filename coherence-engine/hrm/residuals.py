from typing import Dict

def residual_blend(lower_repr: Dict[str, float], higher_hint: Dict[str, float], alpha: float = 0.25) -> Dict[str, float]:
    """Convex blend lower representation with higher-level hint."""
    out: Dict[str, float] = {}
    for k, v in lower_repr.items():
        h = higher_hint.get(k, v)
        out[k] = (1.0 - alpha) * v + alpha * h
    return out
