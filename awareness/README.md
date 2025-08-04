# HRM-CPTE Awareness System

## Overview

This directory contains the design and implementation of an integrated Hierarchical Reasoning Model (HRM) with Contextual Pretrained Experts (CPTEs) for situational awareness and context management.

## Architecture

The system integrates four key components:

1. **Sensor Fusion** - Trust-weighted context inputs (where we are)
2. **Memory System** - Temporal context (how we got here)  
3. **HRM Reasoning** - Decision making (what do we do and why)
4. **CPTE Routing** - Knowledge access (do we know what we need to know)

## Key Documents

### Planning & Design
- [`HRM_CPTE_OPTIMIZATION_PLAN.md`](HRM_CPTE_OPTIMIZATION_PLAN.md) - Comprehensive architecture and optimization strategy
- [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) - Detailed development plan with code templates

### Specifications
- [`TRAINING_DATA_SPECIFICATION.md`](TRAINING_DATA_SPECIFICATION.md) - Training data requirements and generation pipeline
- [`EVALUATION_BENCHMARKS.md`](EVALUATION_BENCHMARKS.md) - Comprehensive evaluation metrics and protocols

## Core Concepts

### Hierarchical Processing
- **High-level module**: Slow, deliberative reasoning (Ψ consciousness)
- **Low-level module**: Fast, reactive processing (θ immediate thought)
- **Bidirectional communication**: Modules influence each other

### Trust-Weighted Sensor Fusion
- Dynamic trust weights learned from experience
- Graceful handling of sensor failures
- Context-aware fusion strategies

### CPTE Integration
- Internal knowledge with confidence assessment
- External expert routing when confidence is low
- Efficient knowledge lifecycle management

### Memory Hierarchy
- Sensory buffer (100ms)
- Working memory (10s)
- Episodic memory (minutes)
- Semantic markers (permanent)

## Target Specifications

- **Model Size**: ~30M parameters
- **Inference**: <100ms on Jetson Orin Nano
- **Memory**: <500MB runtime footprint
- **Accuracy**: >90% situational awareness

## Development Status

**Current Phase**: Planning and design refinement

### Completed ✅
- Comprehensive optimization plan
- Implementation roadmap with templates
- Training data specification
- Evaluation benchmark suite

### Next Steps 📋
1. Review and refine plans based on feedback
2. Set up development environment
3. Implement core modules
4. Generate initial training data
5. Establish baseline metrics

## Quick Start

```bash
# Set up development environment (when ready)
cd awareness/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests (coming soon)
python test_sensor_fusion.py
python test_memory_integration.py
python test_cpte_routing.py
```

## Integration Points

### Existing Systems
- Enhanced memory system (`../memory/`)
- Consciousness HRM (`../src/consciousness_hrm.py`)
- Sensor confidence framework (`../sensor_confidence/`)
- Binocular vision system (`../vision/experiments/`)

### External Resources
- MCP servers for CPTE access
- Distributed Jetson network
- Web4 value routing

## Key Insights

1. **HRM validates hierarchical consciousness** - Multi-timescale processing creates emergent reasoning
2. **CPTEs solve the knowledge scaling problem** - Not everything needs to be known internally
3. **Trust weights enable robust fusion** - Systems can learn which sensors to trust when
4. **Edge deployment is achievable** - 30M parameters can deliver superhuman awareness

## Research Questions

1. What's the optimal fast/slow processing ratio?
2. Can trust weights be learned end-to-end?
3. How to dynamically adjust CPTE routing confidence?
4. How to merge awareness across distributed devices?

## Success Metrics

- Match or exceed standard HRM on reasoning tasks
- >90% accuracy on situational assessment
- <20% external CPTE calls for common tasks
- >70% performance maintained with 50% sensor failure
- Run on Jetson within target constraints

---

*"True awareness isn't just knowing where you are, but understanding how you got there, why you're there, and what you need to know next."*