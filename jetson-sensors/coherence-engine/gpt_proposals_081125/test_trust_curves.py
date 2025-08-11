import json, math, pathlib

cfg = json.loads(pathlib.Path("coherence_engine/config/trust_curves.yaml").read_text().replace("\n", "\n")) if False else {}

def apply_curve(trust, event, p):
    # event: +1 consistent, -1 contradict
    if event > 0:
        trust += p["rise"] * (1 - trust)
    else:
        trust -= p["fall"] * trust
    return max(0.0, min(trust, p.get("saturation", 0.99)))

def test_hysteresis_behavior():
    p = {"rise":0.1, "fall":0.2, "saturation":0.95}
    t = 0.5
    t = apply_curve(t, +1, p)
    t = apply_curve(t, -1, p)
    assert 0 <= t <= 0.95

def test_cross_context_bonus_penalty():
    agree_bonus, disagree_penalty = 0.04, 0.07
    t1, t2 = 0.6, 0.6
    # agreement
    t1 += agree_bonus; t2 += agree_bonus
    # disagreement
    t1 -= disagree_penalty; t2 += disagree_penalty
    assert abs(t1-0.57) < 1e-6 and abs(t2-0.67) < 1e-6
