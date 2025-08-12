import time
from coherence_engine.hrm.sidecar import AffectGate, FastWeights

def test_affect_gate_and_fastweights():
    g = AffectGate(thresh=0.5, cooldown=0.2, refractory=0.1)
    fw = FastWeights()
    now = time.time()
    assert g.should_commit(0.6, now) is True
    assert g.should_commit(0.9, now) is False  # within refractory+cooldown
    later = now + 0.5
    assert g.should_commit(0.6, later) is True
    fw.update("k", {"x": 1.0}, lr=0.5)
    assert abs(fw.recall("k")["x"] - 1.0) < 1e-6
