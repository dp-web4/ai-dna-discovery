# Motif Keys (SimHash / MinHash-lite)

- **SimHash** over feature vectors → 64-bit fingerprint (`simhash(vec)`).
- **MinHash-lite** for set features → banded signature (`minhash(features, bands, rows)`).
- **MotifTable**: LRU with eviction witness via callback.

Integration target: L1 motifs keyed by SimHash; cap with `max_keys`; emit evictions to witness/lightchain.
