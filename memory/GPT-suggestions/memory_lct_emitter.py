from datetime import datetime, timedelta
from typing import List, Optional
from collections import deque
from confidence_framework import MemoryConfidence
from confidence_framework import ConfidenceThresholds
from memory_integration_bridge import MemoryEvent


class TemporalConfidenceBuffer:
    def __init__(self, window_seconds=10):
        self.buffer = deque()
        self.window = timedelta(seconds=window_seconds)

    def add(self, confidence: MemoryConfidence):
        now = datetime.now()
        self.buffer.append((now, confidence))
        self._trim()

    def _trim(self):
        cutoff = datetime.now() - self.window
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def average_confidence(self) -> float:
        if not self.buffer:
            return 0.0
        return sum(c.value for _, c in self.buffer) / len(self.buffer)

    def recent_stability(self, tolerance: float = 0.1) -> bool:
        if len(self.buffer) < 3:
            return False
        values = [c.value for _, c in self.buffer]
        avg = sum(values) / len(values)
        return all(abs(v - avg) < tolerance for v in values)


class MemoryLCTEmitter:
    def __init__(self, thresholds: ConfidenceThresholds):
        self.buffer = TemporalConfidenceBuffer()
        self.thresholds = thresholds
        self.last_emitted: Optional[datetime] = None
        self.emit_interval = timedelta(seconds=5)  # Rate-limit emission

    def process_event(self, event: MemoryEvent) -> Optional[str]:
        self.buffer.add(event.confidence)

        avg = self.buffer.average_confidence()
        stable = self.buffer.recent_stability()
        now = datetime.now()

        if (
            avg >= self.thresholds.emit_threshold
            and stable
            and (self.last_emitted is None or now - self.last_emitted > self.emit_interval)
        ):
            self.last_emitted = now
            return self._emit_lct(event)
        return None

    def _emit_lct(self, event: MemoryEvent) -> str:
        # Placeholder LCT structure
        lct_payload = {
            "memory_id": event.memory_id,
            "sensor": event.sensor_name,
            "time": datetime.now().isoformat(),
            "confidence": event.confidence.value,
            "context": event.context_snapshot,
        }
        return f"LCT-EMITTED::{lct_payload}"


# Example usage:
# emitter = MemoryLCTEmitter(thresholds=ConfidenceThresholds(emit_threshold=0.85))
# lct = emitter.process_event(event)
# if lct:
#     print(lct)
