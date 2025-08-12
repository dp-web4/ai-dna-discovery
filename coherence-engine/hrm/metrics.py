from typing import Dict

def norm(e: Dict[str, float]) -> float:
    if not e: return 0.0
    return sum(abs(v) for v in e.values()) / max(1, len(e))

class CoherenceMetrics:
    def __init__(self):
        self.ema = 0.0

    def update(self, err: Dict[str, float], beta: float = 0.9) -> float:
        val = 1.0 - min(1.0, norm(err))  # 0..1
        self.ema = beta * self.ema + (1.0 - beta) * val
        return self.ema
