from collections import defaultdict, deque
from datetime import datetime, timedelta
import math

class AttentionBudget:
    def __init__(self, total_budget: float = 1.0):
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.attention_records = defaultdict(lambda: deque(maxlen=100))  # time-weighted
        self.last_decay = datetime.now()

    def register_stimulus(self, source: str, weight: float, confidence: float):
        """
        Registers a stimulus from a given source.
        The actual cost is weight * (1 - confidence), encouraging high-confidence signals.
        """
        now = datetime.now()
        self._decay_budget(now)

        cost = weight * (1 - confidence)
        if cost > self.remaining_budget:
            return False  # Budget exceeded

        self.remaining_budget -= cost
        self.attention_records[source].append((now, cost))
        return True

    def _decay_budget(self, now):
        """Decay budget over time to simulate renewal of attention."""
        elapsed = (now - self.last_decay).total_seconds()
        decay_rate = 0.1  # budget units per second
        replenished = elapsed * decay_rate
        self.remaining_budget = min(self.total_budget, self.remaining_budget + replenished)
        self.last_decay = now

    def attention_score(self, source: str) -> float:
        """Computes cumulative attention spent on a source."""
        now = datetime.now()
        return sum(math.exp(-(now - t).total_seconds() / 10.0) * c for t, c in self.attention_records[source])

    def reset(self):
        self.remaining_budget = self.total_budget
        self.attention_records.clear()
        self.last_decay = datetime.now()


# Example Usage:
if __name__ == "__main__":
    ab = AttentionBudget()
    print(ab.register_stimulus("camera", weight=0.3, confidence=0.8))
    print(ab.register_stimulus("audio", weight=0.5, confidence=0.4))
    print(ab.attention_score("camera"))
    print(ab.remaining_budget)
