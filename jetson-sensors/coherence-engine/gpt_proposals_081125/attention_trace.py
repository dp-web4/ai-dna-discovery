"""
Attention Trace — lightweight introspection for policy decisions.

Set environment variable ATTENTION_TRACE=1 to enable.
Emits JSONL events with fields:
  t, policy, features_increase, features_decrease, reason, weights_pre, weights_post
"""
import json, os, time
from typing import Dict, Any

TRACE_ENABLED = os.getenv("ATTENTION_TRACE", "0") == "1"
TRACE_PATH = os.getenv("ATTENTION_TRACE_PATH", "logs/attention_trace.jsonl")

def _ensure_path(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def emit(policy: str, features_up: Dict[str, float], features_down: Dict[str, float],
         reason: str, weights_pre: Dict[str, float], weights_post: Dict[str, float]) -> None:
    if not TRACE_ENABLED:
        return
    _ensure_path(TRACE_PATH)
    evt = {
        "t": time.time(),
        "policy": policy,
        "features_up": features_up,
        "features_down": features_down,
        "reason": reason,
        "weights_pre": weights_pre,
        "weights_post": weights_post,
        "version": 1
    }
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")
