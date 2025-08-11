from coherence_engine.patterns.lifecycle import Lifecycle, Policy

def test_promotion_path():
    lc = Lifecycle(Policy(support_min=2, promotion_score=0.7))
    it = lc.observe("motion-left", True)
    assert it.state in ("CANDIDATE","STABLE")
    it = lc.observe("motion-left", True)
    assert it.state in ("STABLE","PROMOTED")
    it = lc.observe("motion-left", True)
    assert it.state in ("PROMOTED","STABLE")

def test_deprecation_on_contradiction():
    lc = Lifecycle(Policy(support_min=1, promotion_score=0.0, contradiction_limit=1))
    it = lc.observe("face-present", True)
    assert it.state in ("STABLE","PROMOTED")
    it = lc.observe("face-present", False)
    assert it.state == "DEPRECATED"
