from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple


# ---------- Logging ----------

logger = logging.getLogger("coherence_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------- Sensor Protocols ----------

class Sensor(Protocol):
    """Abstract sensor protocol. Implementations must be deterministic w.r.t. given context tick."""
    id: str

    def read(self, *, tick: int) -> float:
        """
        Return a normalized reading in [0, 1] for the current tick.
        Implementations are responsible for their own normalization.
        """
        ...


# ---------- Context & State ----------

class ContextState(Enum):
    STABLE = auto()
    MOVING = auto()
    UNSTABLE = auto()
    NOVEL = auto()


@dataclass
class ContextSnapshot:
    tick: int
    state: ContextState
    relevance: Dict[str, float]
    trust: Dict[str, float]
    field_value: float
    trigger: Optional[str] = None
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrustModel:
    """Tracks trust per sensor, optionally per state. Simple EWMA-style updates."""
    base: Dict[str, float] = field(default_factory=dict)  # global fallback in [0,1]
    per_state: Dict[Tuple[ContextState, str], float] = field(default_factory=dict)
    lr: float = 0.05  # learning rate for trust updates

    def get(self, sensor_id: str, state: ContextState) -> float:
        return self.per_state.get((state, sensor_id), self.base.get(sensor_id, 0.5))

    def update(self, sensor_id: str, state: ContextState, delta: float) -> None:
        key = (state, sensor_id)
        prev = self.get(sensor_id, state)
        new = max(0.0, min(1.0, prev + self.lr * delta))
        self.per_state[key] = new


@dataclass
class RelevanceModel:
    """Computes task/scene fit per sensor given context state; tunable via per-state priors."""
    priors: Dict[Tuple[ContextState, str], float] = field(default_factory=dict)

    def get(self, sensor_id: str, state: ContextState) -> float:
        return max(0.0, min(1.0, self.priors.get((state, sensor_id), 0.5)))


@dataclass
class AttentionPolicy:
    """Defines when to trigger reweighting / context shifts."""
    surprise_threshold: float = 0.25      # relative change threshold
    conflict_threshold: float = 0.35      # simple conflict heuristic
    confidence_floor: float = 0.15        # if field magnitude dips, re-evaluate
    cooldown_ticks: int = 5               # hysteresis

    _last_trigger_tick: int = -10

    def should_trigger(
        self,
        *,
        tick: int,
        field_value: float,
        prev_field_value: Optional[float],
        raw: Dict[str, float],
        weights: Dict[str, Tuple[float, float]],  # (relevance, trust)
    ) -> Optional[str]:
        """Return trigger reason or None."""
        if tick - self._last_trigger_tick < self.cooldown_ticks:
            return None

        # Surprise: relative delta in field value
        if prev_field_value is not None:
            denom = max(1e-6, abs(prev_field_value))
            rel_change = abs(field_value - prev_field_value) / denom
            if rel_change > self.surprise_threshold:
                self._last_trigger_tick = tick
                return f"surprise:{rel_change:.2f}"

        # Conflict: variance among weighted raw contributions
        contribs = []
        for sid, val in raw.items():
            r, t = weights[sid]
            contribs.append(val * r * t)
        if contribs:
            mean = sum(contribs) / len(contribs)
            var = sum((c - mean) ** 2 for c in contribs) / max(1, len(contribs) - 1)
            if var > self.conflict_threshold:
                self._last_trigger_tick = tick
                return f"conflict:{var:.2f}"

        # Confidence floor
        if field_value < self.confidence_floor:
            self._last_trigger_tick = tick
            return f"low_confidence:{field_value:.2f}"

        return None


@dataclass
class Context:
    """Holds current state, models, and observability."""
    state: ContextState = ContextState.STABLE
    trust: TrustModel = field(default_factory=TrustModel)
    relevance: RelevanceModel = field(default_factory=RelevanceModel)
    attention: AttentionPolicy = field(default_factory=AttentionPolicy)
    history: List[ContextSnapshot] = field(default_factory=list)
    prev_field_value: Optional[float] = None

    def compute_relevance_weights(self, sensor_ids: Iterable[str]) -> Dict[str, float]:
        return {sid: self.relevance.get(sid, self.state) for sid in sensor_ids}

    def compute_trust_weights(self, sensor_ids: Iterable[str]) -> Dict[str, float]:
        return {sid: self.trust.get(sid, self.state) for sid in sensor_ids}

    def shift(self, *, trigger: str, field_value: float, raw: Dict[str, float]) -> None:
        """Simple state machine with hysteresis-friendly transitions."""
        old = self.state
        # Heuristic transitions based on trigger type
        if trigger.startswith("surprise"):
            self.state = ContextState.UNSTABLE
        elif trigger.startswith("conflict"):
            self.state = ContextState.NOVEL
        elif trigger.startswith("low_confidence"):
            self.state = ContextState.MOVING
        else:
            # default: toggle to UNSTABLE to force reevaluation
            self.state = ContextState.UNSTABLE

        logger.info(f"context shift: {old.name} -> {self.state.name} (trigger={trigger})")

        # Record snapshot
        snap = ContextSnapshot(
            tick=len(self.history),
            state=self.state,
            relevance=self.compute_relevance_weights(raw.keys()),
            trust=self.compute_trust_weights(raw.keys()),
            field_value=field_value,
            trigger=trigger,
        )
        self.history.append(snap)

    def log_snapshot(self, *, field_value: float, raw: Dict[str, float], trigger: Optional[str] = None) -> None:
        snap = ContextSnapshot(
            tick=len(self.history),
            state=self.state,
            relevance=self.compute_relevance_weights(raw.keys()),
            trust=self.compute_trust_weights(raw.keys()),
            field_value=field_value,
            trigger=trigger,
        )
        self.history.append(snap)


# ---------- Coherence Engine ----------

@dataclass
class CoherenceEngine:
    sensors: List[Sensor]
    context: Context

    def step(self, *, tick: int) -> float:
        # Read
        raw = {s.id: s.read(tick=tick) for s in self.sensors}

        # Weights
        rel = self.context.compute_relevance_weights(raw.keys())
        tru = self.context.compute_trust_weights(raw.keys())
        weights = {sid: (rel[sid], tru[sid]) for sid in raw.keys()}

        # Fuse
        field_val = sum(raw[sid] * rel[sid] * tru[sid] for sid in raw.keys())

        # Attention
        reason = self.context.attention.should_trigger(
            tick=tick,
            field_value=field_val,
            prev_field_value=self.context.prev_field_value,
            raw=raw,
            weights=weights,
        )

        if reason:
            self.context.shift(trigger=reason, field_value=field_val, raw=raw)
            # Recompute weights after shift
            rel = self.context.compute_relevance_weights(raw.keys())
            tru = self.context.compute_trust_weights(raw.keys())
            field_val = sum(raw[sid] * rel[sid] * tru[sid] for sid in raw.keys())
            self.context.log_snapshot(field_value=field_val, raw=raw, trigger="post_shift")
        else:
            self.context.log_snapshot(field_value=field_val, raw=raw, trigger=None)

        # Trust updates (toy heuristic: reward sensors that align with fused field sign/magnitude)
        for sid in raw.keys():
            aligned = 1.0 - abs((raw[sid] * rel[sid] * tru[sid]) - field_val)
            delta = (aligned - 0.5)  # in [-0.5, 0.5]
            self.context.trust.update(sid, self.context.state, delta)

        self.context.prev_field_value = field_val
        return field_val

    # ---- Observability helpers ----

    def export_history(self, path: Path) -> None:
        data = [
            {
                "tick": snap.tick,
                "state": snap.state.name,
                "relevance": snap.relevance,
                "trust": snap.trust,
                "field_value": snap.field_value,
                "trigger": snap.trigger,
                "notes": snap.notes,
            }
            for snap in self.context.history
        ]
        path.write_text(json.dumps(data, indent=2))


# ---------- Example Sensor Implementations (stubs) ----------

@dataclass
class VisionSensor:
    id: str = "vision"
    noise: float = 0.02
    brightness: float = 0.5  # normalized

    def read(self, *, tick: int) -> float:
        # Simple periodic change to simulate lighting / occlusion
        import math, random
        drift = 0.1 * math.sin(tick / 10.0)
        val = max(0.0, min(1.0, self.brightness + drift + random.uniform(-self.noise, self.noise)))
        return val


@dataclass
class IMUSensor:
    id: str = "imu"
    noise: float = 0.02
    motion_level: float = 0.0  # normalized

    def read(self, *, tick: int) -> float:
        # Simulate motion bursts
        import math, random
        burst = 0.5 if (tick % 40 in range(10, 15)) else 0.0
        base = 0.1 * math.cos(tick / 15.0) + burst
        val = max(0.0, min(1.0, self.motion_level + base + random.uniform(-self.noise, self.noise)))
        return val


@dataclass
class MemorySensor:
    id: str = "memory"
    recency_bias: float = 0.6  # how strongly recent snapshots influence output [0..1]
    window: int = 20
    _buffer: List[float] = field(default_factory=list)

    def read(self, *, tick: int) -> float:
        # Emits a "confidence" based on pattern stability in recent fused fields
        if not self._buffer:
            return 0.3
        n = min(len(self._buffer), self.window)
        recent = self._buffer[-n:]
        # Stability proxy: inverse of variance
        mean = sum(recent) / n
        var = sum((x - mean) ** 2 for x in recent) / max(1, n - 1)
        stability = 1.0 / (1.0 + var)  # in (0,1]
        return self.recency_bias * stability + (1 - self.recency_bias) * (mean if mean >= 0 else 0.0)

    def observe_fused_value(self, fused: float) -> None:
        self._buffer.append(fused)
        if len(self._buffer) > 1024:
            self._buffer.pop(0)


@dataclass
class CognitionSensor:
    id: str = "cognition"
    anticipatory_bias: float = 0.5
    # In a real system, this would call out to local/remote models.

    def read(self, *, tick: int) -> float:
        # Toy: anticipates a bump before IMU bursts (lookahead)
        val = 0.2 if (tick % 40 in range(8, 10)) else 0.05
        return self.anticipatory_bias * val


# ---------- Minimal Demo ----------

def demo_run(ticks: int = 120, export_path: Optional[Path] = None) -> None:
    # Initialize sensors
    vision = VisionSensor()
    imu = IMUSensor()
    memory = MemorySensor()
    cognition = CognitionSensor()

    # Context models
    trust = TrustModel(base={s.id: 0.5 for s in (vision, imu, memory, cognition)})
    # Relevance priors per state
    rel = RelevanceModel(priors={
        (ContextState.STABLE, "vision"): 0.8,
        (ContextState.STABLE, "imu"): 0.4,
        (ContextState.STABLE, "memory"): 0.3,
        (ContextState.STABLE, "cognition"): 0.2,

        (ContextState.MOVING, "vision"): 0.7,
        (ContextState.MOVING, "imu"): 0.9,
        (ContextState.MOVING, "memory"): 0.25,
        (ContextState.MOVING, "cognition"): 0.25,

        (ContextState.UNSTABLE, "vision"): 0.5,
        (ContextState.UNSTABLE, "imu"): 0.6,
        (ContextState.UNSTABLE, "memory"): 0.6,
        (ContextState.UNSTABLE, "cognition"): 0.7,

        (ContextState.NOVEL, "vision"): 0.4,
        (ContextState.NOVEL, "imu"): 0.5,
        (ContextState.NOVEL, "memory"): 0.8,
        (ContextState.NOVEL, "cognition"): 0.9,
    })
    ctx = Context(state=ContextState.STABLE, trust=trust, relevance=rel)

    engine = CoherenceEngine(sensors=[vision, imu, memory, cognition], context=ctx)

    logger.info("Starting demo run...")
    for tick in range(ticks):
        fused = engine.step(tick=tick)
        # Let memory observe the fused field to evolve
        memory.observe_fused_value(fused)
        time.sleep(0.005)  # gentle pacing

    logger.info("Demo run complete.")
    if export_path:
        engine.export_history(export_path)
        logger.info(f"Exported history to {export_path}")


# ---------- CLI ----------

if __name__ == "__main__":
    out = Path("coherence_history.json")
    demo_run(ticks=150, export_path=out)
