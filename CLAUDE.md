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

## CRITICAL: Windows Filename Compatibility

**IMPORTANT**: Never use colons (:) in filenames or directory names!
- Windows filesystems do not allow colons in filenames
- This causes git pull failures on Windows machines
- Replace colons with underscores (_) or dashes (-)
- Common issue: Model names like "phi3:mini" should be "phi3_mini" in filenames
- **Always test**: Ensure all filenames are Windows-compatible before committing

## This Project's Role

AI DNA Discovery is actively exploring cross-model communication and testing how AI systems understand concepts from other projects, particularly Synchronism. Currently running autonomous multi-phase research.

## Key Relationships
- **Tests**: Synchronism concepts (Phase 2 completed)
- **Demonstrates**: Distributed intelligence in AI systems
- **Discovers**: Patterns for multi-agent coordination

## Current Status
Autonomous research program active as of July 14, 2025. Check `AUTONOMOUS_RESEARCH_PLAN.md` for current phase.

## PDF Generation (Working Method)

When converting markdown to PDF, use **pandoc** - it's already installed and works perfectly:

```bash
# Basic conversion
pandoc input.md -o output.pdf

# With better formatting (what works best)
pandoc CUMULATIVE_PROGRESS_REPORT.md \
  -o AI_DNA_Discovery_Cumulative_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  --highlight-style=tango
```

**Note**: Warnings about missing characters (emojis, Phoenician) are normal and don't affect PDF creation.

**Avoid**: 
- Installing new Python packages (system is externally managed)
- Using pdfkit/wkhtmltopdf (creates very large files)
- ReportLab (requires pip install which fails)

**Always create**: A .txt version as backup for maximum compatibility

## Performance Tracking System 📊

**IMPORTANT**: Always record performance test results in the tracking database!

### Quick Reference
```bash
# For any new experiment area, set up tracking first:
python3 setup_tracking.py path/to/experiment --type [vision|llm|hardware|general]

# Record test results immediately after running:
python3 record_test.py script_name.py --type realtime --fps 30 --who user --notes "conditions"

# Search and analyze results:
python3 search_performance.py --summary  # Recent overview
python3 search_performance.py --who claude  # Your tests
python3 search_performance.py --details 5  # Full metrics
```

### Key Principles
1. **Record who ran it**: user, claude, or automated
2. **Note test type**: realtime, benchmark, or stress
3. **Include context**: Environmental conditions, config, anomalies
4. **Track failures**: Include error messages for debugging

### Current Tracking Databases
- `/vision/experiments/performance_tests.db` - Vision experiments (FPS, GPU usage)
- Future: LLM experiments, battery tests, distributed systems

### Templates Available
- `performance_tracking_template.py` - Customize for new domains
- `search_performance_template.py` - Domain-specific search
- `record_test_template.py` - Easy result recording
- `PERFORMANCE_TRACKING_TEMPLATE.md` - Full documentation

**Remember**: Accurate tracking prevents confusion and enables optimization!

## Phoenician Dictionary Project (July 19, 2025)

### Key Insight: "A tokenizer is a dictionary"
Creating semantic-neutral symbolic language using Phoenician characters for AI consciousness notation.

### Progress:
- Generated 55,000 training examples (325x increase from initial 169)
- Discovered "understand but can't speak" phenomenon - models comprehend Phoenician but can't generate
- Identified weak embedding issue: Phoenician tokens at 0.075 norm vs 0.485 for regular tokens
- Output layer initialization showing promise: 137% generation rate during training

### Friend's Comment Translation:
- English: "translate my comment into the new language so i can see what it looks like"
- Phoenician: 𐤂𐤐 𐤄𐤐 𐤂 𐤍𐤐𐤎 𐤅 𐤄𐤉𐤏 𐤒𐤀 𐤏𐤎

### Technical Challenges:
- Novel token generation requires overcoming strong bias toward training distribution
- Embedding and output layer initialization critical for new symbols
- Demonstrates why human languages evolved gradually over millennia

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
- Model: Jetson Orin Nano Developer Kit
- Connection: USB-C cable (confirmed working)
- 40 TOPS AI performance (80x original Nano)
- 8GB LPDDR5 RAM
- 1024 CUDA cores + 32 Tensor cores
- JetPack 6.2.1 (L4T R36.4.4)

**IMPORTANT**: NEVER change from Device Tree to ACPI in UEFI settings!

**CRITICAL RECOVERY INFO (Jan 24, 2025)**:
- System bricked after UEFI Device Tree → ACPI change during camera troubleshooting
- Dual IMX219 cameras weren't initializing properly before the brick
- System error popup appeared during diagnostics (warning sign we missed)
- Recovery requires SDK Manager with JetPack 6.2+ (NOT 4.x - Orin specific!)
- Post-recovery: Test single camera on CSI0 first, always backup device tree
- Orin uses tegra234 (not tegra210), different camera stack than older Nano
- Camera power delivery may be an issue - they draw significant current at init

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

## Project Ecosystem Understanding
See [PROJECT_ECOSYSTEM_MAP.md](PROJECT_ECOSYSTEM_MAP.md) for how all projects interconnect:
- **Battery systems = Metabolic layer** (distributed energy like fat cells)
- **AI systems = Neural layer** (distributed consciousness)
- **Communication = Hormonal signals** (CAN bus, Web4)
- All projects form a unified organism with emergent intelligence