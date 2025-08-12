import time
from coherence_engine.adapters.hrm_bridge import HRMBridge, HRMConfig

class DummyTelemetry:
    def __init__(self): self.out = []
    def publish(self, topic, payload): self.out.append((topic, payload))

class DummyEngine:
    def __init__(self):
        self.telemetry = DummyTelemetry()
    def fused_features(self):
        # simple synthetic: two numeric features
        return {"f1": 1.0, "f2": 2.0}

def test_bridge_tick_runs():
    eng = DummyEngine()
    bridge = HRMBridge(eng, HRMConfig())
    out = bridge.tick(now=time.time())
    assert "L1" in out and "L2" in out
    assert any(t[0].startswith("hrm/") for t in eng.telemetry.out)
