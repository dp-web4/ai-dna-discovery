import time
from coherence_engine.bridge.qos import QoSBridge, ClassPolicy

def mkbridge():
    pol = {
        "state": ClassPolicy(priority=1, ttl_ms=1000, drop_policy="drop_oldest", queue_max=2),
        "event": ClassPolicy(priority=2, ttl_ms=5000, drop_policy="drop_new", queue_max=2),
        "pattern_diff": ClassPolicy(priority=3, ttl_ms=2000, drop_policy="block", queue_max=1),
    }
    br = QoSBridge(pol, housekeeper_interval_ms=50)
    br.start()
    return br

def test_priority_order():
    br = mkbridge()
    br.enqueue("state", 1)
    br.enqueue("event", 2)
    br.enqueue("pattern_diff", 3)
    m = br.dequeue()
    assert m.cls == "pattern_diff"  # highest priority first
    br.stop()

def test_drop_policies_and_ttl():
    br = mkbridge()
    # drop_oldest
    assert br.enqueue("state", "a")
    assert br.enqueue("state", "b")
    assert br.enqueue("state", "c")  # should drop "a"
    m = br.dequeue(); assert m.payload == "b"
    # drop_new
    assert br.enqueue("event", "e1")
    assert br.enqueue("event", "e2")
    assert not br.enqueue("event", "e3")  # rejected
    # ttl expiry
    time.sleep(1.2)  # exceed state ttl
    assert br.dequeue() is not None  # should still have event
    br.stop()
