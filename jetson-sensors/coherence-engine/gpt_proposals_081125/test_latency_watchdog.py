from coherence_engine.runtime.latency_watchdog import LatencyWatchdog, Budgets

def test_flip_to_degraded_and_recover(tmp_path, monkeypatch):
    log = tmp_path / "watchdog.jsonl"
    b = Budgets(s2f=10, f2d=10, d2e=10, e2e=40, violation_window=3, recovery_window=3, hysteresis_ms=0)
    wd = LatencyWatchdog(budgets=b, window=10, log_path=str(log))

    # Start under budget
    for _ in range(5):
        wd.record(5,5,5)
    assert wd.state == "RUNNING"

    # Exceed budgets to trigger DEGRADED
    for _ in range(3):
        wd.record(20,20,5)
    assert wd.state == "DEGRADED"

    # Recover under budget
    for _ in range(3):
        wd.record(5,5,5)
    assert wd.state == "RUNNING"
