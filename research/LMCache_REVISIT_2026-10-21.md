# Revisiting the 2025 KV-cache experiments: LMCache
Date: 2026-10-21 | Research cross-reference, not an implementation

AI-DNA Discovery's `archive/MEMORY_SYSTEM_PAPER.md` (July 16, 2025) demonstrated external SQLite/context-injection memory with Ollama context tokens as portable KV *proxies*; direct KV persistence was a future direction. `archive/HIDDEN_STATE_ANALYSIS.md`, `archive/GPU_MODEL_STATE_INTERIM_REPORT.md`, and `archive/MEMORY_INVESTIGATION_SUMMARY.md` record the research trajectory. Subsequently, SAGE implemented a functioning Hugging Face `past_key_values` save/resume prototype at `forum/nova/archive/persistent-kv-demo/`.

LMCache (https://lmcache.ai/en/ ; https://github.com/LMCache/LMCache) offers a modern opportunity to test practical KV persistence, offload and reuse via supported serving engines, without reviving our experimental serialization code as production infrastructure.

Key distinction: cached attention computations are implementation-bound, disposable optimizations. They are not the being's durable memory, identity or a reliable cross-model continuation mechanism.

Next experiment belongs in SAGE (resident wake/restart economics) with a dev-SAGE SGLang baseline; see umbrella record `dp-web4/private-context/research/lmcache-kv-persistence-evaluation-2026-10-21.md`. Maintain clear historical attribution, measure benefits over native prefix caching, and do not rewrite 2025 conclusions as proof of LMCache capabilities.
