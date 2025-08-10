from typing import Protocol

class BaseSensor(Protocol):
    id: str
    def read(self, *, tick: int) -> float: ...