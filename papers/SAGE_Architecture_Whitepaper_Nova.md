# SAGE Architecture Whitepaper

**Status: Anchor Document for Future Development**\
**Date: September 2025**

------------------------------------------------------------------------

## 1. Introduction

SAGE (**Sentient Agentic Generative Engine**) is being developed as a
modular, resource-aware cognitive architecture. The objective is to
combine efficient abstract reasoning, adaptive sensor/effector
management, and scalable multi-type memory into a coherent loop. This
document summarizes the current design discussions around HRM
(Hierarchical Reasoning Model), TinyVAE components, and IRP (Iterative
Refinement Primitives).

This whitepaper serves as both a status snapshot ("here is where we
are") and a roadmap ("here is where we are going").

------------------------------------------------------------------------

## 2. HRM: The Awareness Engine

The **Hierarchical Reasoning Model (HRM)** sits at the core of SAGE. It
functions as the *awareness engine*, responsible for: - Monitoring all
active sensors (including temporal streams). - Framing the system's
world-state as **existential puzzles**. - Inferring abstract rules
governing current dynamics. - Deciding which sensors are
trustworthy/needed/ignored. - Commanding effectors with appropriate
actions.

### Key Properties

-   **Dual-loop architecture:**

    -   **H-loop (high/strategic):** abstract reasoning ("what rule
        applies here?").\
    -   **L-loop (low/tactical):** concrete execution of
        transformations.\
    -   **Adaptive computation cycles:** HRM "thinks harder" when
        puzzles are complex.

-   **Pattern-oriented learning:**\
    HRM is trained on ARC-style puzzles, emphasizing invariance
    detection, few-shot rule inference, and generalization.

-   **Compact efficiency:**\
    With only \~5--6M parameters, HRM achieves performance on par with
    humans in certain puzzle benchmarks, while remaining light enough
    for edge deployment (Jetson-class hardware).

------------------------------------------------------------------------

## 3. Sensor & Effector Economy with IRPs

Raw world signals are mediated by **IRPs (Iterative Refinement
Primitives)**, which act as flexible plugin mechanisms for both sensors
and effectors. These allow SAGE to integrate cameras, microphones, IMUs,
text streams, and actuators.\
The critical challenge is *resource allocation*: some IRPs (like LLMs or
diffusion engines) are computationally heavy and must only be engaged
when necessary.

### Sensor Hierarchy

1.  **TinyVAE Layer (always-on, cheap):**
    -   Compresses raw sensor streams into compact latent codes.\
    -   Acts as the **baseline filter**.\
    -   Detects novelty/anomalies, escalates when needed.
2.  **Puzzle Mapping IRPs (specialized TinyVAE):**
    -   Convert multimodal latent codes into **existential puzzles**
        (structured grids or symbolic representations).\
    -   HRM operates on these puzzles to infer abstract rules.
3.  **LLM IRPs (Language Models):**
    -   Serve as **planning/detection primitives**.\
    -   Called when semantic/narrative reasoning is required.\
    -   Expensive; lightly loaded, event-driven.
4.  **Diffusion IRPs:**
    -   Serve as **perceptual imagers/clarifiers**.\
    -   Used when high-fidelity reconstruction is needed (e.g.,
        disambiguating visual blobs).\
    -   Expensive; event-driven.

### Effector Pathway

-   **Rule Actualizer IRP (specialized TinyVAE):** transforms HRM's
    abstract rule solutions into effector commands.\
-   **Effectors:** motors, speech, UI, communication channels.\
-   **Closed loop:** Actions feed back into sensors, creating new
    puzzles.

------------------------------------------------------------------------

## 4. The Existential Puzzle Paradigm

Instead of processing data streams directly, SAGE reframes reality as a
sequence of **existential puzzles**:\
- "Given these sensory inputs and changes, what rule best explains the
situation?"\
- "How should the rule be applied to guide action?"

**Example:**\
- Sensors detect a moving object.\
- TinyVAE compresses sensor data into a grid.\
- Puzzle Mapper frames it as "object motion puzzle" (state t vs. t+1).\
- HRM infers: "linear trajectory at velocity v."\
- Actualizer converts this to: "move 2m forward to intercept."

This paradigm creates symmetry between perception and action: both are
puzzles → rules → transformations.

------------------------------------------------------------------------

## 5. Memory in SAGE

SAGE does not rely on a single monolithic memory. Instead, it maintains
multiple specialized memory types, each serving a distinct role in
coherence and reasoning.

### Types of Memory

-   **SNARC (Surprise--Novelty--Arousal--Reward--Conflict):**\
    Highlights surprising, novel, emotionally arousing, rewarding, or
    conflicting experiences. Prioritized for attention and
    consolidation.

-   **Exact Recall:**\
    Stores verbatim records with timestamps for later reference. Useful
    for precise replay and audit.

-   **Compressed Wisdom:**\
    HRM-distilled abstractions and rules, representing the "wisdom" of
    past experiences. Enables efficient reuse without storing raw
    detail.

-   **Associative Memory:**\
    Links events, entities, and modalities across time and context.
    Supports analogy, inference, and flexible recall.

Together, these layers balance efficiency, precision, salience, and
context in SAGE's evolving world model.

------------------------------------------------------------------------

## 6. Comparison to Other Models

-   **vs. LLMs:**\
    LLMs excel at fluency and semantic planning, but lack true abstract
    rule induction. HRM fills that gap.

-   **vs. Diffusion Models:**\
    Diffusion excels at high-fidelity generative perception, but lacks
    symbolic reasoning. HRM focuses on discrete, abstract transformation
    rules.

-   **vs. Classic Robotics Pipelines:**\
    Traditional robotics separates perception → planning → control.\
    SAGE reframes the entire process as **puzzle → rule → action**,
    unifying perception, reasoning, and execution.

------------------------------------------------------------------------

## 7. Roadmap (Where We're Going)

Immediate steps:\
1. Finalize HRM's stable training regime (batch size, validation
cadence, checkpointing).\
2. Deploy HRM onto Jetson hardware for edge inference tests.\
3. Integrate specialized TinyVAE for puzzle building/actualization.\
4. Establish escalation thresholds for LLM/diffusion IRPs.

Near-term development:\
- Build the **meta-controller** to manage sensor/effector load.\
- Implement **trust evaluation** and memory consolidation loops (SNARC +
others).\
- Prototype real-world "existential puzzles" from multi-modal sensor
inputs.

Longer-term trajectory:\
- Scale puzzle representations beyond 30×30 grids to higher-dimensional
domains.\
- Extend HRM reasoning from visual-spatial puzzles into text, audio, and
sensor fusion.\
- Formalize SAGE as a general-purpose reasoning + awareness kernel for
Web4 and Synchronism frameworks.

------------------------------------------------------------------------

## 8. Conclusion

HRM provides a compact, efficient awareness engine capable of abstract
reasoning from minimal examples. By embedding it within SAGE, alongside
TinyVAEs, IRPs, LLMs, and diffusion models, we establish a coherent
architecture for adaptive, trust-aware cognition.\
The **existential puzzle paradigm** gives SAGE a unified way to frame
perception and action, enabling scalable intelligence on both edge
devices and larger distributed systems.

This whitepaper captures the present alignment of concepts and serves as
an anchor for iterative development.

------------------------------------------------------------------------
