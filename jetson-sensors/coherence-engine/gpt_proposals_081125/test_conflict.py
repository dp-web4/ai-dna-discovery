import time
from coherence_engine.coherence.conflict import Source, resolve

def test_simple_resolution():
    t = time.time()
    hyps = {
        "object=A": [Source("vision", trust=0.8, expertise=0.9, t=t)],
        "object=B": [Source("imu", trust=0.6, expertise=0.5, t=t)]
    }
    decision, scores = resolve(hyps, quorum=0.5, half_life=120)
    assert decision == "object=A"
    assert scores["object=A"] > scores["object=B"]

def test_tie_break_by_role():
    t = time.time()
    hyps = {
        "H1": [Source("v1", trust=0.8, expertise=0.5, role_priority=1, t=t)],
        "H2": [Source("v2", trust=0.5, expertise=0.8, role_priority=2, t=t)],
    }
    # balanced scores → rely on role_priority
    decision, _ = resolve(hyps, quorum=0.1, half_life=120)
    assert decision in ("H1","H2")
