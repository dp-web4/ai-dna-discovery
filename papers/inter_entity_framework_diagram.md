# Inter‑Entity Communication Framework — Summary Diagram (Mermaid)

```mermaid
flowchart TD

  %% ---------------- Core Principle ----------------
  CT[[Compression Trust<br/>Shared / Aligned Latent Fields]]:::core

  %% ---------------- VAE ----------------
  subgraph VAE[Variational Autoencoder (Mechanics)]
    direction LR
    V1[Input x] --> V2[Encoder → (μ, σ)]
    V2 --> V3[z ~ N(μ, σ)<br/>Ephemeral latent coordinate]
    V3 --> V4[Decoder → x̂]
    V5[(Weights: encoder/decoder)] --- V2
    V5 --- V4
  end

  %% ---------------- Synchronism ----------------
  subgraph SYN[Synchronism (Theory of Witnesses)]
    direction LR
    S1[Witness observes MRH] --> S2[Compression (resonance paths)]
    S2 --> S3[Ephemeral MRH coordinate]
    S3 --> S4[Expansion / Recall]
    S5[(Witness resonance memory)] --- S2
    S5 --- S4
  end

  %% ---------------- Web4 ----------------
  subgraph W4[Web4 (Infrastructure)]
    direction TB
    W1[Dictionary Entity<br/>(Shared Codebook / Embeddings)]
    W2[LCT Wrapper<br/>(Provenance • Trust • Alignment)]
    W3[Mapping Layer<br/>(Cross‑dictionary alignment)]
    W1 --> W2
    W1 --> W3
  end

  %% ---------------- AI DNA Discovery ----------------
  subgraph DNA[AI DNA Discovery (Empirics)]
    direction TB
    D1[Cross‑model Embedding Tests]
    D2[Universal Anchors<br/>(logic • math • cognition • computation)]
    D3[High Cosine Similarity ≈ 1.0<br/>across diverse models]
    D1 --> D2 --> D3
  end

  %% ---------------- SAGE / Memory ----------------
  subgraph SAGE[SAGE (Operation)]
    direction TB
    G1[IRPs: perceptual compression<br/>(VAE / VQ‑VAE tokens)]
    G2[SNARC Memory]
    G3[Vector DB<br/>(similarity search / recall)]
    G1 --> G2 --> G3
  end

  %% ---------------- Cross Links ----------------
  %% VAE & SYN connect to CT
  V3 -. governed by .-> CT
  S3 -. governed by .-> CT

  %% Web4 backs CT
  CT -. backed by .-> W1
  CT -. audited by .-> W2
  CT -. maintained by .-> W3

  %% DNA informs CT
  D3 -. empirical support .-> CT

  %% SAGE uses CT
  G1 -. produces latents .-> CT
  CT -. enables coherent recall .-> G3

  %% Web4 wraps artifacts
  W2 -. wraps .-> V3
  W2 -. wraps .-> S3
  W2 -. wraps .-> G2

  %% Styling
  classDef core fill:#222,color:#fff,stroke:#999,stroke-width:1px;
  class VAE,SYN,W4,DNA,SAGE,CT core;
```
