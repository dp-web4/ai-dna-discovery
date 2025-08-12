from coherence_engine.hrm.levels import BaseLevel

def test_encode_predict_update_roundtrip():
    lvl = BaseLevel("L1", {"learn_rate": 0.1})
    x = {"a": 1.0, "b": 2.0, "note": "skip"}
    enc = lvl.encode(x)
    pred = lvl.predict(None)
    assert set(enc) == set(pred) == {"a","b"}
    # apply an error and ensure features move in right direction
    err = {"a": 1.0, "b": -2.0}
    lvl.update(err)
    assert lvl.state.features["a"] > enc["a"]
    assert lvl.state.features["b"] < enc["b"]
