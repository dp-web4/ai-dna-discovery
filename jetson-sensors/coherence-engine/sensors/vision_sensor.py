from dataclasses import dataclass
import math, random

@dataclass
class VisionSensor:
    id: str = "vision"
    noise: float = 0.02
    brightness: float = 0.5  # normalized

    def read(self, *, tick: int) -> float:
        drift = 0.1 * math.sin(tick / 10.0)
        val = self.brightness + drift + random.uniform(-self.noise, self.noise)
        return max(0.0, min(1.0, val))