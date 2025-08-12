"""
Motif keying utilities:
- SimHash for high-dimensional vectors → fixed-size binary fingerprint
- MinHash-lite for set-like features
Includes LRU+age eviction policy with optional witness callback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Callable, Optional, List
import time, hashlib, random

def _hash_bytes(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)

def simhash(vec: Dict[str, float], bits: int = 64) -> int:
    acc = [0.0] * bits
    for k, v in vec.items():
        h = _hash_bytes(k)
        for i in range(bits):
            acc[i] += v if (h >> i) & 1 else -v
    out = 0
    for i, a in enumerate(acc):
        if a >= 0:
            out |= (1 << i)
    return out

def minhash(features: Iterable[str], bands: int = 4, rows: int = 8) -> Tuple[int, ...]:
    # Simple L permutations via seeded hashes; (bands * rows) signatures, grouped into bands
    sig: List[int] = []
    feats = list(features)
    for i in range(bands * rows):
        seed = f"mh{i}"
        hv = min(_hash_bytes(seed + f) for f in feats) if feats else 0
        sig.append(hv)
    # Reduce each band to a single int via xor
    bands_out = []
    for b in range(bands):
        x = 0
        for r in range(rows):
            x ^= sig[b * rows + r]
        bands_out.append(x)
    return tuple(bands_out)

@dataclass
class MotifTable:
    max_keys: int = 4096
    witness_cb: Optional[Callable[[Dict], None]] = None
    _store: Dict[int, Tuple[float, Dict[str, float]]] = field(default_factory=dict)  # key -> (last_ts, vec)

    def put(self, vec: Dict[str, float]) -> int:
        key = simhash(vec)
        now = time.time()
        if key in self._store:
            self._store[key] = (now, vec)
        else:
            if len(self._store) >= self.max_keys:
                # Evict LRU
                k_evict = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                last_ts, old_vec = self._store.pop(k_evict)
                if self.witness_cb:
                    self.witness_cb({
                        "type": "motif_eviction",
                        "key": int(k_evict),
                        "ts": now,
                        "last_seen": last_ts,
                        "size": len(old_vec)
                    })
            self._store[key] = (now, vec)
        return key

    def get(self, key: int) -> Optional[Dict[str, float]]:
        item = self._store.get(key)
        if not item: return None
        ts, vec = item
        self._store[key] = (time.time(), vec)  # touch
        return dict(vec)

    def __len__(self) -> int:
        return len(self._store)
