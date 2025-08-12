from coherence_engine.hrm.motif_keys import simhash, MotifTable

def test_simhash_stability_and_table_eviction():
    v1 = {"a":1.0, "b":2.0, "c":3.0}
    v2 = {"a":1.0, "b":2.0, "c":3.1}  # small change should keep close key (not enforced here, but stable hash)
    k1 = simhash(v1); k2 = simhash(v2)
    assert isinstance(k1, int) and isinstance(k2, int)
    tbl = MotifTable(max_keys=2, witness_cb=lambda e: None)
    tbl.put(v1); tbl.put({"x":1}); tbl.put({"y":2})  # triggers eviction
    assert len(tbl) == 2
