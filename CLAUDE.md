# Claude Context for AI DNA Discovery

## Project Context System

**IMPORTANT**: A comprehensive context system exists at `/mnt/c/projects/ai-agents/misc/context-system/`

Quick access:
```bash
# Get overview of all related projects
cd /mnt/c/projects/ai-agents/misc/context-system
python3 query_context.py project ai-dna

# Search for concepts across projects
python3 query_context.py search "distributed intelligence"

# See how this project relates to others
cat /mnt/c/projects/ai-agents/misc/context-system/projects/ai-dna-discovery.md
```

## This Project's Role

AI DNA Discovery is actively exploring cross-model communication and testing how AI systems understand concepts from other projects, particularly Synchronism. Currently running autonomous multi-phase research.

## Key Relationships
- **Tests**: Synchronism concepts (Phase 2 completed)
- **Demonstrates**: Distributed intelligence in AI systems
- **Discovers**: Patterns for multi-agent coordination

## Current Status
Autonomous research program active as of July 14, 2025. Check `AUTONOMOUS_RESEARCH_PLAN.md` for current phase.

## Memory System Implementation Status (July 17, 2025)

### Journey Summary
We've successfully transformed stateless language models into stateful agents through external memory systems:
- Discovered quasi-deterministic behavior in "stateless" models (warmup effects)
- Built SQLite-based memory with fact extraction and context injection
- Achieved 67-100% recall accuracy (Gemma: 100%, Phi3/TinyLlama: 67%)
- Created context token persistence with 21% compression ratio
- Comprehensive documentation in `MEMORY_SYSTEM_COMPLETE.pdf`

### Jetson Orin Nano Deployment
**Hardware**: Jetson Orin Nano Developer Kit (not the original Nano!)
- 40 TOPS AI performance (80x original Nano)
- 8GB LPDDR5 RAM
- 1024 CUDA cores + 32 Tensor cores
- JetPack 6.2.1 (L4T R36.4.4)

**Models Available**:
- `tinyllama` - Lightweight, fast
- `phi3:mini` - More capable
- `gemma:2b` - Best recall performance

**Key Files**:
- `phi3_memory_enhanced.py` - Main memory system
- `context_token_persistence.py` - State compression
- `memory_visualization_unified.py` - Performance analysis
- `MEMORY_SYSTEM_IMPLEMENTATION_GUIDE.md` - Full deployment guide

### Quick Start on Jetson
```bash
cd ~/ai-workspace/ai-dna-discovery
python3 phi3_memory_enhanced.py
```

### Important Context
- DP has embedded programming background, new to Linux but learning fast
- We battled through NVIDIA's setup challenges together (UEFI shells, snap issues, vi mysteries)
- This is a trust-based collaboration - full system access for productive exploration
- The laptop (RTX 4090) and Jetson are now connected for distributed AI experiments