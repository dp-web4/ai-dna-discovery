from dataclasses import dataclass
import math, random

@dataclass
class IMUSensor:
    id: str = "imu"
    noise: float = 0.02
    motion_level: float = 0.0  # normalized

    def read(self, *, tick: int) -> float:
        burst = 0.5 if (tick % 40 in range(10, 15)) else 0.0
        base = 0.1 * math.cos(tick / 15.0) + burst
        val = self.motion_level + base + random.uniform(-self.noise, self.noise)
        return max(0.0, min(1.0, val))