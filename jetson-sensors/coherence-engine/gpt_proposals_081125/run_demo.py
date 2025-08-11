import time, threading, random
from collections import deque

# Import from our packaged modules
from coherence_engine.runtime.latency_watchdog import LatencyWatchdog, Budgets
from coherence_engine.bridge.qos import QoSBridge, ClassPolicy

def make_bridge():
    policies = {
        "state": ClassPolicy(priority=1, ttl_ms=1000, drop_policy="drop_oldest", queue_max=128),
        "event": ClassPolicy(priority=2, ttl_ms=5000, drop_policy="drop_new", queue_max=256),
        "experience_batch": ClassPolicy(priority=0, ttl_ms=15000, drop_policy="drop_oldest", queue_max=32),
        "pattern_diff": ClassPolicy(priority=3, ttl_ms=2000, drop_policy="block", queue_max=64),
    }
    br = QoSBridge(policies, housekeeper_interval_ms=100)
    br.start()
    return br

def phase(latencies, duration_s, bridge, wd):
    """
    latencies: dict with keys 's2f','f2d','d2e' specifying mean ms (jitter will be added)
    """
    t_end = time.time() + duration_s
    last_state = wd.state
    drops = 0
    processed = 0

    while time.time() < t_end:
        # Producer side — push some messages (vary rate with jitter)
        for cls in ("event", "state", "experience_batch", "pattern_diff"):
            ok = bridge.enqueue(cls, {"t": time.time(), "cls": cls})
            if not ok:
                drops += 1

        # Simulate processing — dequeue one (highest priority first)
        msg = bridge.dequeue()
        if msg:
            # derive jittered timings for each stage
            s2f = max(1, random.gauss(latencies["s2f"], latencies["s2f"] * 0.15))
            f2d = max(1, random.gauss(latencies["f2d"], latencies["f2d"] * 0.15))
            d2e = max(1, random.gauss(latencies["d2e"], latencies["d2e"] * 0.15))
            # "spend" the total latency budget as sleep (scaled down to be fast)
            sleep_ms = (s2f + f2d + d2e) * 0.001  # scale for demo speed
            time.sleep(min(0.02, sleep_ms))
            wd.record(s2f, f2d, d2e)
            processed += 1

        # Print state flips
        if wd.state != last_state:
            print(f"[{time.strftime('%H:%M:%S')}] Watchdog state -> {wd.state}")
            last_state = wd.state

        # Control loop rate
        time.sleep(0.005)

    return processed, drops

def main():
    # Budgets roughly aligned with docs (in ms)
    wd = LatencyWatchdog(Budgets(s2f=50, f2d=30, d2e=20, e2e=80, violation_window=5, recovery_window=10, hysteresis_ms=5),
                         window=50, log_path="logs/latency_watchdog.jsonl")
    br = make_bridge()

    print("Phase 1: Under budget (should stay RUNNING)")
    p1, d1 = phase({"s2f": 20, "f2d": 10, "d2e": 10}, 5, br, wd)

    print("Phase 2: Overload (should flip to DEGRADED)")
    p2, d2 = phase({"s2f": 80, "f2d": 60, "d2e": 40}, 6, br, wd)

    print("Phase 3: Recovery (should return to RUNNING)")
    p3, d3 = phase({"s2f": 25, "f2d": 15, "d2e": 10}, 6, br, wd)

    br.stop()
    print("\nSummary:")
    print(f"  processed: {p1+p2+p3}")
    print(f"  drops:     {d1+d2+d3}")
    print("  See logs/latency_watchdog.jsonl for p95 metrics and state transitions.")

if __name__ == "__main__":
    main()
