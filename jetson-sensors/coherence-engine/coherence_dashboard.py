
"""
Stdlib + matplotlib dashboard for Coherence Engine.
- Runs an in-process simulation (no external deps beyond matplotlib).
- Two separate figures (no subplots): field over time, and current sensor contributions.
- Does not set explicit colors (per instructions).
"""
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local imports
from coherence_engine import (
    CoherenceEngine, Context, ContextState,
    TrustModel, RelevanceModel,
)
from sensors import VisionSensor, IMUSensor, MemorySensor, CognitionSensor


class EngineRunner(threading.Thread):
    def __init__(self, ticks:int=300, fps:float=30.0):
        super().__init__(daemon=True)
        self.ticks = ticks
        self.fps = fps
        self.q: "queue.Queue[Tuple[int, float, Dict[str, float], Dict[str, Tuple[float,float]]]]" = queue.Queue()
        self._stop = threading.Event()

    def run(self):
        # Build engine
        vision = VisionSensor()
        imu = IMUSensor()
        memory = MemorySensor()
        cognition = CognitionSensor()

        trust = TrustModel(base={s.id: 0.5 for s in (vision, imu, memory, cognition)})
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

        # Run loop
        for tick in range(self.ticks):
            if self._stop.is_set():
                break

            field_val = engine.step(tick=tick)

            # compute contributions for display
            raw = {
                "vision": vision.read(tick=tick),
                "imu": imu.read(tick=tick),
                "memory": memory.read(tick=tick),
                "cognition": cognition.read(tick=tick),
            }
            rel_w = ctx.compute_relevance_weights(raw.keys())
            tru_w = ctx.compute_trust_weights(raw.keys())
            weights = {sid: (rel_w[sid], tru_w[sid]) for sid in raw.keys()}

            # push a snapshot
            self.q.put((tick, field_val, raw, weights))

            # memory observes fused field
            memory.observe_fused_value(field_val)

            time.sleep(1.0 / self.fps)

    def stop(self):
        self._stop.set()


def main():
    runner = EngineRunner(ticks=400, fps=30.0)
    runner.start()

    # Figure 1: field value over time
    fig1 = plt.figure()
    fig1.canvas.manager.set_window_title("Reality Field — Over Time")
    xs: List[int] = []
    ys: List[float] = []
    line, = plt.plot([], [])
    plt.xlabel("tick")
    plt.ylabel("field value")
    plt.title("Reality Field (fused)")

    def update_line(_):
        # drain queue, keep last N
        drained = 0
        while True:
            try:
                tick, field_val, raw, weights = runner.q.get_nowait()
                xs.append(tick)
                ys.append(field_val)
                if len(xs) > 500:
                    xs.pop(0); ys.pop(0)
                drained += 1
            except Exception:
                break
        if drained:
            line.set_data(xs, ys)
            plt.xlim(max(0, xs[0] if xs else 0), (xs[-1] if xs else 1) + 1)
            ymin = min(ys) if ys else 0.0
            ymax = max(ys) if ys else 1.0
            pad = (ymax - ymin) * 0.1 + 1e-3
            plt.ylim(ymin - pad, ymax + pad)
        return line,

    ani1 = FuncAnimation(fig1, update_line, interval=100)

    # Figure 2: sensor contributions (current)
    fig2 = plt.figure()
    fig2.canvas.manager.set_window_title("Sensor Contributions — Current")
    bars = plt.bar(["vision","imu","memory","cognition"], [0,0,0,0])
    plt.xlabel("sensor")
    plt.ylabel("weighted contribution")
    plt.title("Current Sensor Contributions")

    # use closure to hold last snapshot
    last: Dict[str, float] = {"vision":0,"imu":0,"memory":0,"cognition":0}
    last_weights: Dict[str, Tuple[float,float]] = {k:(0.5,0.5) for k in last.keys()}

    def update_bars(_):
        # non-blocking peek: get last available
        got_any = False
        while True:
            try:
                tick, field_val, raw, weights = runner.q.get_nowait()
                # recompute contributions using the weights snapshot
                for i, sid in enumerate(["vision","imu","memory","cognition"]):
                    r, t = weights[sid]
                    contrib = raw[sid] * r * t
                    last[sid] = contrib
                    last_weights[sid] = (r, t)
                got_any = True
            except Exception:
                break

        if got_any:
            for i, sid in enumerate(["vision","imu","memory","cognition"]):
                bars[i].set_height(last[sid])
        return bars

    ani2 = FuncAnimation(fig2, update_bars, interval=150)

    try:
        plt.show()
    finally:
        runner.stop()

if __name__ == "__main__":
    main()
