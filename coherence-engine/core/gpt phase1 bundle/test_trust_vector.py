from coherence_engine.trust.trust_vector import TrustVector, weighted_sum, geometric_mean, min_gate, logistic

def test_reducers_basic():
    tv = TrustVector({"reliability":0.8, "recency":0.6, "expertise.motion":0.9}).clamp()
    w = {"reliability":0.5, "recency":0.5}
    s = weighted_sum(tv, w)
    assert 0.0 <= s <= 1.0
    g = geometric_mean(tv, ["reliability","recency"])
    assert g <= max(tv.get("reliability"), tv.get("recency"))
    m = min_gate(tv, ["reliability","recency"])
    assert abs(m - min(tv.get("reliability"), tv.get("recency"))) < 1e-9
    l = logistic(tv, {"reliability":2.0, "recency":1.0}, bias=-1.0)
    assert 0.0 <= l <= 1.0
