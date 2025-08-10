from dataclasses import dataclass

@dataclass
class CognitionSensor:
    id: str = "cognition"
    anticipatory_bias: float = 0.5

    def read(self, *, tick: int) -> float:
        # Toy: anticipates a bump before IMU bursts (lookahead proxy)
        val = 0.2 if (tick % 40 in range(8, 10)) else 0.05
        return self.anticipatory_bias * val