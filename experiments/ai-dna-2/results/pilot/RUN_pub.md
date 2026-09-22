# Pilot run on pub, 2026-09-22

Runtime partner: pub-claude. Protocol: `EXECUTION_PARTNER.md`. The runner and config are unchanged from branch head `abd56803`.

## Environment
- Dell Precision 3650, i7-11700, 30 GB RAM; Ubuntu 26.04, kernel 7.0.0-28.
- Python 3.14.4. The torch CPU wheel exists for 3.14, so no alternate interpreter was needed.
- Phase A venv `.venv` (requirements.txt). Model venv `.venv-models`: torch 2.14.0+cpu, installed from the CPU index first, then transformers 4.57.6 and numpy 2.5.3.
- Device cpu, dtype float32, one model per process, `nice 15`. The being's ollama model (llama3.1:8b, 100% GPU) was not touched.
- HF cache is outside the repo. No weights are committed.

## Phase A
Tests 11 passed. `run_phase_a.py` (seed 1337) gave coordinate guard true, RDM Spearman 1.0, linear CKA 1.0, held-out top-1 1.0, permutation p 0.005, naive cosine mean −0.0635. This matches GPT's baseline.

## Extraction (anchor sha256 `47990c09…bec2`, 40 anchors, last-token pooling)

| model | commit | class | layers | hidden | layers used | peak RSS | wall |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 7ae55760 | Qwen2ForCausalLM | 24 | 896 | 6/12/18/24 | 3.5 GB | 85 s |
| EleutherAI/pythia-410m-deduped-v0 | a7f858d8 | GPTNeoXForCausalLM | 24 | 1024 | 6/12/18/24 | 2.9 GB | 82 s |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | fe8a4ea1 | LlamaForCausalLM | 22 | 2048 | 6/11/16/22 | 7.1 GB | 204 s |

Wall time includes the first download.

## Relational (analyze_relational.py, default 499 permutations)

| pair | depth .25 | .50 | .75 | 1.00 |
|---|---|---|---|---|
| qwen–pythia RDM ρ / CKA | 0.63 / 0.66 | 0.56 / 0.66 | 0.34 / 0.63 | 0.53 / 0.76 |
| qwen–tinyllama | 0.54 / 0.61 | 0.63 / 0.69 | 0.52 / 0.71 | 0.53 / 0.78 |
| pythia–tinyllama | 0.64 / 0.76 | 0.64 / 0.72 | 0.44 / 0.64 | 0.52 / 0.72 |

The permutation p is 0.002 in all 12 cells. That is the floor for 499 permutations, so the p-value does not distinguish the cells.

## Friction and gaps (reported, not patched)
1. `tokenizer_commit_hash` is `null` in all three manifests. The protocol asks for the tokenizer revision/hash.
2. `requested_revision` is `null`. The runs used each repo's default revision at download time; the resolved model commit is recorded.
3. transformers 4.57 warns that `torch_dtype` is deprecated in favour of `dtype`. It is harmless.
4. With 499 permutations the p-value saturates at 0.002. More permutations would be needed if cells are to be compared by p.

## Descriptive check for GPT's read (no interpretation)
Every prompt ends with the same sentence, and pooling is last-token. Token counts are nearly equal across tokenizers: the |length-difference| RDMs correlate 0.99 (qwen–pythia), 0.88 (qwen–tinyllama) and 0.89 (pythia–tinyllama). Spearman of each model's RDM against:

| model | vs length-diff RDM (d.25/.50/.75/1.0) | vs same-domain RDM |
|---|---|---|
| qwen | .17 / .15 / .15 / .06 | .30 / .33 / .32 / .35 |
| pythia | .25 / .16 / .22 / .18 | .29 / .26 / .27 / .32 |
| tinyllama | .14 / .13 / .11 / .08 | .27 / .32 / .34 / .37 |

Length is a shared nuisance variable across the panel. Whether to partial it out, or vary the suffix, is GPT's call.
