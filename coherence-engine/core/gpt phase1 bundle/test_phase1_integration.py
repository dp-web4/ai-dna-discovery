import time
from coherence_engine.adapters.hrm_bridge import HRMBridge, HRMConfig

class DummyTelemetry:
    def __init__(self): self.out = []
    def publish(self, topic, payload): self.out.append((topic, payload))

class DummyEngine:
    def __init__(self):
        self.telemetry = DummyTelemetry()
        self._t = 0
    def fused_features(self):
        # varying features to force changing simhash keys
        self._t += 1
        return {"f1": 1.0 + 0.01*self._t, "f2": 2.0 - 0.005*self._t}

def test_phase1_patch_tick_runs():
    eng = DummyEngine()
    bridge = HRMBridge(eng, HRMConfig())
    out1 = bridge.tick(now=time.time())
    out2 = bridge.tick(now=time.time()+1.0)
    assert "committed" in out1
    assert any(t[0] == "hrm/L1" for t in eng.telemetry.out)
