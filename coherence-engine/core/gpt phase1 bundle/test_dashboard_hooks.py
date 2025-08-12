from coherence_engine.dashboard.hooks import CoherenceDashboard
import time

def test_dashboard_snapshot():
    d = CoherenceDashboard()
    t = time.time()
    d.record("coherence_ema", t, 0.8)
    snap = d.snapshot()
    assert "coherence_ema" in snap and 0 <= snap["coherence_ema"] <= 1
