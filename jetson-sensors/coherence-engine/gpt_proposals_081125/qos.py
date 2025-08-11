"""
Bridge QoS — prioritized, TTL-aware bounded queues with backpressure.
"""
from __future__ import annotations
import time, threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ClassPolicy:
    priority: int
    ttl_ms: int
    drop_policy: str
    queue_max: int

@dataclass
class Message:
    cls: str
    payload: Any
    t: float = field(default_factory=lambda: time.time())
    priority: int = 0
    ttl_ms: int = 0

class QoSBridge:
    def __init__(self, policies: Dict[str, ClassPolicy], housekeeper_interval_ms: int = 200):
        self.policies = policies
        self.queues: Dict[str, List[Message]] = {k: [] for k in policies}
        self.hk_ms = housekeeper_interval_ms
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._housekeeper, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=1.0)

    def enqueue(self, cls: str, payload: Any) -> bool:
        pol = self.policies[cls]
        msg = Message(cls=cls, payload=payload, priority=pol.priority, ttl_ms=pol.ttl_ms)
        with self._lock:
            q = self.queues[cls]
            # expire oldest if needed
            self._expire_locked(cls)
            if len(q) >= pol.queue_max:
                if pol.drop_policy == "drop_oldest":
                    q.pop(0)
                elif pol.drop_policy == "drop_new":
                    return False
                elif pol.drop_policy == "block":
                    # simple spin-wait with timeout
                    t0 = time.time()
                    while len(q) >= pol.queue_max and time.time() - t0 < 0.5:
                        time.sleep(0.005)
                    if len(q) >= pol.queue_max:
                        return False
            q.append(msg)
            return True

    def dequeue(self) -> Optional[Message]:
        # pick highest priority non-empty class
        with self._lock:
            best_cls = None
            best_pri = -1
            for cls, pol in self.policies.items():
                if self.queues[cls] and pol.priority > best_pri:
                    best_pri = pol.priority
                    best_cls = cls
            if best_cls is None:
                return None
            self._expire_locked(best_cls)
            if not self.queues[best_cls]:
                return None
            return self.queues[best_cls].pop(0)

    def _expire_locked(self, cls: str):
        pol = self.policies[cls]
        now = time.time()
        q = self.queues[cls]
        self.queues[cls] = [m for m in q if (now - m.t) * 1000 <= pol.ttl_ms]

    def _housekeeper(self):
        while self._running:
            time.sleep(self.hk_ms / 1000.0)
            with self._lock:
                for cls in list(self.queues.keys()):
                    self._expire_locked(cls)
